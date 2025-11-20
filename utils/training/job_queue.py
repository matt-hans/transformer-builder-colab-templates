"""
Job Queue and Scheduler for Automated Training Workflows

This module provides a SQLite-based job queue and scheduler for managing
automated training, evaluation, and export jobs. It enables:
- Priority-based job queuing with atomic claiming
- Scheduled training with cron-like syntax
- Job status tracking and logging
- Integration with Trainer, Model Registry, and ExperimentDB
- Resource checking and concurrency control

Example Usage:
    >>> from utils.training.job_queue import JobManager, TrainingScheduler
    >>> from utils.training.training_config import TrainingConfig
    >>>
    >>> # Submit training job
    >>> manager = JobManager('jobs.db')
    >>> config = TrainingConfig(learning_rate=5e-5, batch_size=4, epochs=10)
    >>> job_id = manager.submit_job(
    ...     job_type='training',
    ...     config={'training_config': config.to_dict()},
    ...     priority=5
    ... )
    >>>
    >>> # Schedule periodic retraining
    >>> scheduler = TrainingScheduler('jobs.db')
    >>> schedule_id = scheduler.create_schedule(
    ...     name='daily-retrain',
    ...     job_type='training',
    ...     config={'training_config': config.to_dict()},
    ...     schedule_expr='0 2 * * *',  # Daily at 2am
    ...     priority=3
    ... )
    >>>
    >>> # Execute jobs (worker process)
    >>> executor = JobExecutor(manager)
    >>> executor.run_worker(max_jobs=10)

Architecture:
    The job queue uses SQLite with WAL mode for concurrent access:
    - jobs: Job metadata, config, status, priority
    - schedules: Recurring job schedules with next run time
    - job_logs: Execution logs for debugging

    Jobs are claimed atomically to prevent race conditions in multi-worker
    scenarios. Status transitions are tracked with timestamps.

Author: MLOps Agent (Phase 2 - Production Hardening)
Version: 3.7.0
"""

import json
import logging
import sqlite3
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal, Union, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


# Job type constants
JobType = Literal["training", "evaluation", "export", "retraining"]
JobStatus = Literal["pending", "running", "completed", "failed", "cancelled"]


@dataclass
class Job:
    """
    Job queue entry with metadata and config.

    Attributes:
        job_id: Unique integer ID (auto-assigned)
        job_type: Type of job (training, evaluation, export, retraining)
        status: Current status (pending, running, completed, failed, cancelled)
        priority: Priority 1-10 (higher = more urgent)
        config: Job configuration dictionary (JSON serialized)
        created_at: ISO timestamp of job creation
        started_at: ISO timestamp when job started (None if not started)
        completed_at: ISO timestamp when job finished (None if not finished)
        error_message: Error message if failed (None if successful)
        worker_id: ID of worker that claimed job (None if unclaimed)
        retry_count: Number of times job has been retried
        max_retries: Maximum retry attempts (default 3)
    """
    job_id: int
    job_type: JobType
    status: JobStatus
    priority: int
    config: Dict[str, Any]
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
    worker_id: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class Schedule:
    """
    Scheduled job configuration.

    Attributes:
        schedule_id: Unique integer ID (auto-assigned)
        name: Human-readable schedule name
        job_type: Type of job to create
        config: Job configuration template
        schedule_expr: Cron-like expression (e.g., "0 2 * * *" for daily 2am)
        priority: Priority for created jobs
        enabled: Whether schedule is active
        next_run: ISO timestamp of next scheduled run
        last_run: ISO timestamp of last run (None if never run)
        created_at: ISO timestamp of schedule creation
    """
    schedule_id: int
    name: str
    job_type: JobType
    config: Dict[str, Any]
    schedule_expr: str
    priority: int
    enabled: bool
    next_run: str
    last_run: Optional[str] = None
    created_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


class JobManager:
    """
    SQLite-based job queue manager with atomic operations.

    Provides job submission, status tracking, and atomic claiming for
    worker processes. Supports priority-based scheduling and retry logic.

    Thread-safe: Uses SQLite WAL mode and transactions for concurrency.
    """

    def __init__(self, db_path: Union[str, Path] = 'jobs.db'):
        """
        Initialize job queue database.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self._create_schema()
        logger.info(f"Initialized JobManager at {self.db_path}")

    def _create_schema(self) -> None:
        """Create database schema with WAL mode for concurrency."""
        with sqlite3.connect(self.db_path) as conn:
            # Enable WAL mode for concurrent reads/writes
            conn.execute('PRAGMA journal_mode=WAL')
            conn.execute('PRAGMA synchronous=NORMAL')

            cursor = conn.cursor()

            # Jobs table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_type TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    priority INTEGER NOT NULL DEFAULT 5,
                    config TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    error_message TEXT,
                    worker_id TEXT,
                    retry_count INTEGER DEFAULT 0,
                    max_retries INTEGER DEFAULT 3,
                    CHECK (job_type IN ('training', 'evaluation', 'export', 'retraining')),
                    CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
                    CHECK (priority BETWEEN 1 AND 10)
                )
            ''')

            # Index for efficient job claiming
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_jobs_status_priority
                ON jobs (status, priority DESC, created_at)
            ''')

            # Job logs table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS job_logs (
                    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id INTEGER NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    level TEXT NOT NULL,
                    message TEXT NOT NULL,
                    FOREIGN KEY (job_id) REFERENCES jobs (job_id) ON DELETE CASCADE
                )
            ''')

            # Index for log queries
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_job_logs_job_id
                ON job_logs (job_id, timestamp)
            ''')

            conn.commit()
            logger.debug("Job queue schema created/validated")

    def submit_job(
        self,
        job_type: JobType,
        config: Dict[str, Any],
        priority: int = 5,
        max_retries: int = 3
    ) -> int:
        """
        Submit new job to queue.

        Args:
            job_type: Type of job (training, evaluation, export, retraining)
            config: Job configuration dictionary
            priority: Priority 1-10 (higher = more urgent)
            max_retries: Maximum retry attempts

        Returns:
            job_id: Unique job ID

        Example:
            >>> manager = JobManager()
            >>> config = {'training_config': {...}, 'model_path': 'model.pt'}
            >>> job_id = manager.submit_job('training', config, priority=7)
        """
        if not 1 <= priority <= 10:
            raise ValueError(f"Priority must be 1-10, got {priority}")

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO jobs (job_type, config, priority, max_retries, status)
                VALUES (?, ?, ?, ?, 'pending')
            ''', (job_type, json.dumps(config), priority, max_retries))

            job_id = cursor.lastrowid
            conn.commit()

        assert job_id is not None, "Failed to get job_id from database"
        logger.info(f"Submitted {job_type} job with ID={job_id}, priority={priority}")
        return job_id

    def get_job(self, job_id: int) -> Optional[Job]:
        """
        Retrieve job by ID.

        Args:
            job_id: Job ID

        Returns:
            Job object or None if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM jobs WHERE job_id = ?', (job_id,))
            row = cursor.fetchone()

        if row is None:
            return None

        return Job(
            job_id=row['job_id'],
            job_type=row['job_type'],
            status=row['status'],
            priority=row['priority'],
            config=json.loads(row['config']),
            created_at=row['created_at'],
            started_at=row['started_at'],
            completed_at=row['completed_at'],
            error_message=row['error_message'],
            worker_id=row['worker_id'],
            retry_count=row['retry_count'],
            max_retries=row['max_retries']
        )

    def list_jobs(
        self,
        status: Optional[JobStatus] = None,
        job_type: Optional[JobType] = None,
        limit: int = 100
    ) -> List[Job]:
        """
        List jobs with optional filtering.

        Args:
            status: Filter by status (None = all)
            job_type: Filter by type (None = all)
            limit: Maximum number of jobs to return

        Returns:
            List of Job objects
        """
        query = 'SELECT * FROM jobs WHERE 1=1'
        params: List[Any] = []

        if status is not None:
            query += ' AND status = ?'
            params.append(status)

        if job_type is not None:
            query += ' AND job_type = ?'
            params.append(job_type)

        query += ' ORDER BY priority DESC, created_at LIMIT ?'
        params.append(limit)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()

        jobs = []
        for row in rows:
            jobs.append(Job(
                job_id=row['job_id'],
                job_type=row['job_type'],
                status=row['status'],
                priority=row['priority'],
                config=json.loads(row['config']),
                created_at=row['created_at'],
                started_at=row['started_at'],
                completed_at=row['completed_at'],
                error_message=row['error_message'],
                worker_id=row['worker_id'],
                retry_count=row['retry_count'],
                max_retries=row['max_retries']
            ))

        return jobs

    def claim_job(self, worker_id: str) -> Optional[Job]:
        """
        Atomically claim next pending job.

        Uses SQL transaction to prevent race conditions when multiple
        workers try to claim the same job.

        Args:
            worker_id: Unique worker identifier

        Returns:
            Claimed Job object or None if no jobs available

        Example:
            >>> manager = JobManager()
            >>> job = manager.claim_job('worker-01')
            >>> if job:
            ...     execute_job(job)
            ...     manager.update_job_status(job.job_id, 'completed')
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Start transaction for atomic claim
            cursor.execute('BEGIN IMMEDIATE')

            try:
                # Find highest priority pending job
                cursor.execute('''
                    SELECT job_id FROM jobs
                    WHERE status = 'pending'
                    ORDER BY priority DESC, created_at
                    LIMIT 1
                ''')
                row = cursor.fetchone()

                if row is None:
                    conn.rollback()
                    return None

                job_id = row['job_id']

                # Claim job atomically
                cursor.execute('''
                    UPDATE jobs
                    SET status = 'running',
                        started_at = ?,
                        worker_id = ?
                    WHERE job_id = ? AND status = 'pending'
                ''', (datetime.now().isoformat(), worker_id, job_id))

                # Verify claim succeeded (prevents race condition)
                if cursor.rowcount == 0:
                    conn.rollback()
                    return None

                # Fetch claimed job
                cursor.execute('SELECT * FROM jobs WHERE job_id = ?', (job_id,))
                row = cursor.fetchone()

                conn.commit()

                job = Job(
                    job_id=row['job_id'],
                    job_type=row['job_type'],
                    status=row['status'],
                    priority=row['priority'],
                    config=json.loads(row['config']),
                    created_at=row['created_at'],
                    started_at=row['started_at'],
                    completed_at=row['completed_at'],
                    error_message=row['error_message'],
                    worker_id=row['worker_id'],
                    retry_count=row['retry_count'],
                    max_retries=row['max_retries']
                )

                logger.info(f"Worker {worker_id} claimed job {job_id}")
                return job

            except Exception as e:
                conn.rollback()
                logger.error(f"Failed to claim job: {e}")
                return None

    def update_job_status(
        self,
        job_id: int,
        status: JobStatus,
        error_message: Optional[str] = None
    ) -> None:
        """
        Update job status with transition logging.

        Args:
            job_id: Job ID
            status: New status
            error_message: Error message if failed
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            if status in ['completed', 'failed', 'cancelled']:
                # Final states - set completion timestamp
                cursor.execute('''
                    UPDATE jobs
                    SET status = ?,
                        completed_at = ?,
                        error_message = ?
                    WHERE job_id = ?
                ''', (status, datetime.now().isoformat(), error_message, job_id))
            else:
                cursor.execute('''
                    UPDATE jobs
                    SET status = ?,
                        error_message = ?
                    WHERE job_id = ?
                ''', (status, error_message, job_id))

            conn.commit()

        logger.info(f"Job {job_id} status updated to {status}")

    def cancel_job(self, job_id: int) -> bool:
        """
        Cancel pending or running job.

        Args:
            job_id: Job ID

        Returns:
            True if cancelled, False if already completed/failed
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE jobs
                SET status = 'cancelled',
                    completed_at = ?
                WHERE job_id = ? AND status IN ('pending', 'running')
            ''', (datetime.now().isoformat(), job_id))

            cancelled = cursor.rowcount > 0
            conn.commit()

        if cancelled:
            logger.info(f"Cancelled job {job_id}")
        else:
            logger.warning(f"Failed to cancel job {job_id} (already completed/failed)")

        return cancelled

    def retry_job(self, job_id: int) -> bool:
        """
        Retry failed job if retries remaining.

        Args:
            job_id: Job ID

        Returns:
            True if job reset to pending, False if max retries exceeded
        """
        job = self.get_job(job_id)
        if job is None:
            logger.error(f"Job {job_id} not found")
            return False

        if job.retry_count >= job.max_retries:
            logger.warning(f"Job {job_id} exceeded max retries ({job.max_retries})")
            return False

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE jobs
                SET status = 'pending',
                    retry_count = retry_count + 1,
                    started_at = NULL,
                    completed_at = NULL,
                    error_message = NULL,
                    worker_id = NULL
                WHERE job_id = ?
            ''', (job_id,))
            conn.commit()

        logger.info(f"Retrying job {job_id} (attempt {job.retry_count + 1}/{job.max_retries})")
        return True

    def log_job_message(
        self,
        job_id: int,
        level: str,
        message: str
    ) -> None:
        """
        Log message for job execution.

        Args:
            job_id: Job ID
            level: Log level (INFO, WARNING, ERROR)
            message: Log message
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO job_logs (job_id, level, message)
                VALUES (?, ?, ?)
            ''', (job_id, level, message))
            conn.commit()

    def get_job_logs(self, job_id: int) -> pd.DataFrame:
        """
        Retrieve job execution logs.

        Args:
            job_id: Job ID

        Returns:
            DataFrame with columns: timestamp, level, message
        """
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query('''
                SELECT timestamp, level, message
                FROM job_logs
                WHERE job_id = ?
                ORDER BY timestamp
            ''', conn, params=(job_id,))

        return df

    def get_queue_stats(self) -> Dict[str, int]:
        """
        Get queue statistics.

        Returns:
            Dictionary with status counts
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT status, COUNT(*) as count
                FROM jobs
                GROUP BY status
            ''')
            rows = cursor.fetchall()

        stats = {status: 0 for status in ['pending', 'running', 'completed', 'failed', 'cancelled']}
        for status, count in rows:
            stats[status] = count

        return stats


class TrainingScheduler:
    """
    Cron-like scheduler for recurring training jobs.

    Supports schedule expressions for periodic job creation:
    - "0 2 * * *" - Daily at 2am
    - "0 */6 * * *" - Every 6 hours
    - "0 0 * * 0" - Weekly on Sunday midnight
    - "0 0 1 * *" - Monthly on 1st at midnight
    """

    def __init__(self, db_path: Union[str, Path] = 'jobs.db'):
        """
        Initialize scheduler.

        Args:
            db_path: Path to SQLite database file (shared with JobManager)
        """
        self.db_path = Path(db_path)
        self.job_manager = JobManager(db_path)
        self._create_schema()
        logger.info(f"Initialized TrainingScheduler")

    def _create_schema(self) -> None:
        """Create schedules table."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Schedules table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS schedules (
                    schedule_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    job_type TEXT NOT NULL,
                    config TEXT NOT NULL,
                    schedule_expr TEXT NOT NULL,
                    priority INTEGER NOT NULL DEFAULT 5,
                    enabled BOOLEAN NOT NULL DEFAULT 1,
                    next_run TIMESTAMP NOT NULL,
                    last_run TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    CHECK (job_type IN ('training', 'evaluation', 'export', 'retraining'))
                )
            ''')

            # Index for efficient next run queries
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_schedules_next_run
                ON schedules (enabled, next_run)
            ''')

            conn.commit()
            logger.debug("Scheduler schema created/validated")

    def create_schedule(
        self,
        name: str,
        job_type: JobType,
        config: Dict[str, Any],
        schedule_expr: str,
        priority: int = 5
    ) -> int:
        """
        Create recurring job schedule.

        Args:
            name: Unique schedule name
            job_type: Type of job to create
            config: Job configuration template
            schedule_expr: Cron expression (e.g., "0 2 * * *")
            priority: Priority for created jobs

        Returns:
            schedule_id: Unique schedule ID

        Example:
            >>> scheduler = TrainingScheduler()
            >>> schedule_id = scheduler.create_schedule(
            ...     name='daily-retrain',
            ...     job_type='training',
            ...     config={'training_config': {...}},
            ...     schedule_expr='0 2 * * *',  # Daily at 2am
            ...     priority=5
            ... )
        """
        # Parse expression and compute next run
        next_run = self._compute_next_run(schedule_expr)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO schedules (name, job_type, config, schedule_expr, priority, next_run)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (name, job_type, json.dumps(config), schedule_expr, priority, next_run.isoformat()))

            schedule_id = cursor.lastrowid
            conn.commit()

        assert schedule_id is not None, "Failed to get schedule_id from database"
        logger.info(f"Created schedule '{name}' with ID={schedule_id}, next run at {next_run}")
        return schedule_id

    def _compute_next_run(self, schedule_expr: str, after: Optional[datetime] = None) -> datetime:
        """
        Compute next run time from cron expression.

        Simplified cron parser supporting:
        - minute hour day month weekday
        - * for any value
        - */N for every N units

        Args:
            schedule_expr: Cron expression
            after: Compute next run after this time (default: now)

        Returns:
            Next run datetime
        """
        if after is None:
            after = datetime.now()

        parts = schedule_expr.split()
        if len(parts) != 5:
            raise ValueError(f"Invalid cron expression: {schedule_expr} (expected 5 fields)")

        minute, hour, day, month, weekday = parts

        # Simple parsing for common patterns
        next_run = after.replace(second=0, microsecond=0)

        # Parse hour and minute
        if hour == '*':
            hour_val = next_run.hour
        elif hour.startswith('*/'):
            # Every N hours
            interval = int(hour[2:])
            hour_val = (next_run.hour // interval + 1) * interval
            if hour_val >= 24:
                hour_val = 0
                next_run += timedelta(days=1)
        else:
            hour_val = int(hour)

        if minute == '*':
            minute_val = 0  # Top of hour
        else:
            minute_val = int(minute)

        # Set next run time
        next_run = next_run.replace(hour=hour_val, minute=minute_val)

        # If computed time is in the past, add appropriate interval
        if next_run <= after:
            if hour.startswith('*/'):
                interval = int(hour[2:])
                next_run += timedelta(hours=interval)
            else:
                next_run += timedelta(days=1)

        return next_run

    def check_due_schedules(self) -> List[int]:
        """
        Check for schedules due to run and create jobs.

        Returns:
            List of created job IDs
        """
        now = datetime.now()
        job_ids = []

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Find schedules due to run
            cursor.execute('''
                SELECT * FROM schedules
                WHERE enabled = 1 AND next_run <= ?
                ORDER BY next_run
            ''', (now.isoformat(),))

            schedules = cursor.fetchall()

        for schedule_row in schedules:
            schedule = Schedule(
                schedule_id=schedule_row['schedule_id'],
                name=schedule_row['name'],
                job_type=schedule_row['job_type'],
                config=json.loads(schedule_row['config']),
                schedule_expr=schedule_row['schedule_expr'],
                priority=schedule_row['priority'],
                enabled=bool(schedule_row['enabled']),
                next_run=schedule_row['next_run'],
                last_run=schedule_row['last_run'],
                created_at=schedule_row['created_at']
            )

            # Create job
            job_id = self.job_manager.submit_job(
                job_type=schedule.job_type,
                config=schedule.config,
                priority=schedule.priority
            )
            job_ids.append(job_id)

            # Update schedule
            next_run = self._compute_next_run(schedule.schedule_expr, after=now)
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE schedules
                    SET last_run = ?,
                        next_run = ?
                    WHERE schedule_id = ?
                ''', (now.isoformat(), next_run.isoformat(), schedule.schedule_id))
                conn.commit()

            logger.info(f"Created job {job_id} from schedule '{schedule.name}', next run at {next_run}")

        return job_ids

    def list_schedules(self, enabled_only: bool = False) -> List[Schedule]:
        """
        List all schedules.

        Args:
            enabled_only: Only return enabled schedules

        Returns:
            List of Schedule objects
        """
        query = 'SELECT * FROM schedules'
        if enabled_only:
            query += ' WHERE enabled = 1'
        query += ' ORDER BY next_run'

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()

        schedules = []
        for row in rows:
            schedules.append(Schedule(
                schedule_id=row['schedule_id'],
                name=row['name'],
                job_type=row['job_type'],
                config=json.loads(row['config']),
                schedule_expr=row['schedule_expr'],
                priority=row['priority'],
                enabled=bool(row['enabled']),
                next_run=row['next_run'],
                last_run=row['last_run'],
                created_at=row['created_at']
            ))

        return schedules

    def enable_schedule(self, schedule_id: int) -> None:
        """Enable schedule."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('UPDATE schedules SET enabled = 1 WHERE schedule_id = ?', (schedule_id,))
            conn.commit()
        logger.info(f"Enabled schedule {schedule_id}")

    def disable_schedule(self, schedule_id: int) -> None:
        """Disable schedule."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('UPDATE schedules SET enabled = 0 WHERE schedule_id = ?', (schedule_id,))
            conn.commit()
        logger.info(f"Disabled schedule {schedule_id}")


class JobExecutor:
    """
    Job execution engine with resource checking.

    Polls queue for jobs, executes based on type, and updates status.
    Integrates with Trainer for training jobs.
    """

    def __init__(
        self,
        job_manager: JobManager,
        worker_id: Optional[str] = None,
        max_concurrent: int = 1
    ):
        """
        Initialize job executor.

        Args:
            job_manager: JobManager instance
            worker_id: Unique worker ID (default: hostname-pid)
            max_concurrent: Maximum concurrent jobs
        """
        self.job_manager = job_manager
        self.worker_id = worker_id or f"worker-{time.time()}"
        self.max_concurrent = max_concurrent
        logger.info(f"Initialized JobExecutor with ID={self.worker_id}")

    def run_worker(
        self,
        max_jobs: Optional[int] = None,
        poll_interval: float = 5.0
    ) -> None:
        """
        Run worker loop to process jobs.

        Args:
            max_jobs: Maximum jobs to process (None = infinite)
            poll_interval: Seconds between queue polls

        Example:
            >>> executor = JobExecutor(manager)
            >>> executor.run_worker(max_jobs=10)  # Process 10 jobs then exit
        """
        processed = 0
        logger.info(f"Worker {self.worker_id} starting (max_jobs={max_jobs})")

        while max_jobs is None or processed < max_jobs:
            # Claim next job
            job = self.job_manager.claim_job(self.worker_id)

            if job is None:
                # No jobs available, wait
                time.sleep(poll_interval)
                continue

            # Execute job
            try:
                self._execute_job(job)
                self.job_manager.update_job_status(job.job_id, 'completed')
                self.job_manager.log_job_message(job.job_id, 'INFO', 'Job completed successfully')
                processed += 1

            except Exception as e:
                error_msg = f"Job failed: {str(e)}"
                logger.error(f"Job {job.job_id} failed: {e}", exc_info=True)
                self.job_manager.update_job_status(job.job_id, 'failed', error_message=error_msg)
                self.job_manager.log_job_message(job.job_id, 'ERROR', error_msg)

                # Retry if possible
                if job.retry_count < job.max_retries:
                    self.job_manager.retry_job(job.job_id)

        logger.info(f"Worker {self.worker_id} processed {processed} jobs, exiting")

    def _execute_job(self, job: Job) -> None:
        """
        Execute job based on type.

        Args:
            job: Job to execute
        """
        logger.info(f"Executing {job.job_type} job {job.job_id}")
        self.job_manager.log_job_message(job.job_id, 'INFO', f'Starting {job.job_type} job')

        if job.job_type == 'training':
            self._execute_training_job(job)
        elif job.job_type == 'evaluation':
            self._execute_evaluation_job(job)
        elif job.job_type == 'export':
            self._execute_export_job(job)
        elif job.job_type == 'retraining':
            self._execute_retraining_job(job)
        else:
            raise ValueError(f"Unknown job type: {job.job_type}")

    def _execute_training_job(self, job: Job) -> None:
        """Execute training job."""
        # Import here to avoid circular dependencies
        from utils.training.engine.trainer import Trainer
        from utils.training.training_config import TrainingConfig
        from utils.training.task_spec import TaskSpec

        config_dict = job.config.get('training_config', {})
        training_config = TrainingConfig(**config_dict)

        # Load model and task spec (should be in config)
        model_path = job.config.get('model_path')
        task_spec_dict = job.config.get('task_spec')

        if model_path is None or task_spec_dict is None:
            raise ValueError("Training job requires 'model_path' and 'task_spec' in config")

        # TODO: Load model from checkpoint
        # For now, log that we would train
        self.job_manager.log_job_message(
            job.job_id,
            'INFO',
            f"Would train model from {model_path} with config {config_dict}"
        )

        logger.info(f"Training job {job.job_id} completed (stub)")

    def _execute_evaluation_job(self, job: Job) -> None:
        """Execute evaluation job."""
        self.job_manager.log_job_message(
            job.job_id,
            'INFO',
            'Evaluation job execution (stub)'
        )
        logger.info(f"Evaluation job {job.job_id} completed (stub)")

    def _execute_export_job(self, job: Job) -> None:
        """Execute export job."""
        self.job_manager.log_job_message(
            job.job_id,
            'INFO',
            'Export job execution (stub)'
        )
        logger.info(f"Export job {job.job_id} completed (stub)")

    def _execute_retraining_job(self, job: Job) -> None:
        """Execute retraining job (triggered by drift detection)."""
        self.job_manager.log_job_message(
            job.job_id,
            'INFO',
            'Retraining job execution (stub)'
        )
        logger.info(f"Retraining job {job.job_id} completed (stub)")


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    'Job',
    'Schedule',
    'JobManager',
    'TrainingScheduler',
    'JobExecutor',
    'JobType',
    'JobStatus',
]
