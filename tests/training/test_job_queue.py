"""
Tests for Job Queue and Scheduler

Tests cover:
- Job submission and retrieval
- Atomic job claiming (concurrent workers)
- Status transitions and logging
- Job retry logic
- Schedule creation and execution
- Queue statistics and cleanup

Author: MLOps Agent (Phase 2 - Production Hardening)
Version: 3.7.0
"""

import pytest
import tempfile
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import List
import threading

from utils.training.job_queue import (
    JobManager,
    TrainingScheduler,
    JobExecutor,
    Job,
    Schedule
)


@pytest.fixture
def temp_db() -> Path:
    """Create temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = Path(f.name)
    yield db_path
    if db_path.exists():
        db_path.unlink()


@pytest.fixture
def job_manager(temp_db: Path) -> JobManager:
    """Create JobManager instance."""
    return JobManager(temp_db)


@pytest.fixture
def scheduler(temp_db: Path) -> TrainingScheduler:
    """Create TrainingScheduler instance."""
    return TrainingScheduler(temp_db)


@pytest.fixture
def executor(job_manager: JobManager) -> JobExecutor:
    """Create JobExecutor instance."""
    return JobExecutor(job_manager, worker_id='test-worker')


class TestJobManager:
    """Tests for JobManager."""

    def test_submit_job(self, job_manager: JobManager):
        """Test job submission."""
        config = {'learning_rate': 5e-5, 'batch_size': 4}
        job_id = job_manager.submit_job('training', config, priority=7)

        assert job_id > 0

        job = job_manager.get_job(job_id)
        assert job is not None
        assert job.job_type == 'training'
        assert job.status == 'pending'
        assert job.priority == 7
        assert job.config == config

    def test_submit_job_invalid_priority(self, job_manager: JobManager):
        """Test job submission with invalid priority."""
        with pytest.raises(ValueError, match='Priority must be 1-10'):
            job_manager.submit_job('training', {}, priority=11)

    def test_get_nonexistent_job(self, job_manager: JobManager):
        """Test retrieving nonexistent job."""
        job = job_manager.get_job(999)
        assert job is None

    def test_list_jobs(self, job_manager: JobManager):
        """Test listing jobs with filters."""
        # Submit multiple jobs
        job_manager.submit_job('training', {}, priority=5)
        job_manager.submit_job('evaluation', {}, priority=7)
        job_manager.submit_job('training', {}, priority=3)

        # List all jobs
        all_jobs = job_manager.list_jobs()
        assert len(all_jobs) == 3

        # Filter by type
        training_jobs = job_manager.list_jobs(job_type='training')
        assert len(training_jobs) == 2
        assert all(j.job_type == 'training' for j in training_jobs)

        # Filter by status
        pending_jobs = job_manager.list_jobs(status='pending')
        assert len(pending_jobs) == 3

    def test_priority_ordering(self, job_manager: JobManager):
        """Test jobs are listed by priority."""
        job_manager.submit_job('training', {}, priority=3)
        job_manager.submit_job('training', {}, priority=9)
        job_manager.submit_job('training', {}, priority=5)

        jobs = job_manager.list_jobs()
        priorities = [j.priority for j in jobs]
        assert priorities == [9, 5, 3]  # Descending order

    def test_claim_job(self, job_manager: JobManager):
        """Test atomic job claiming."""
        job_id = job_manager.submit_job('training', {}, priority=5)

        # Claim job
        job = job_manager.claim_job('worker-1')
        assert job is not None
        assert job.job_id == job_id
        assert job.status == 'running'
        assert job.worker_id == 'worker-1'
        assert job.started_at is not None

        # Try to claim again (should fail)
        job2 = job_manager.claim_job('worker-2')
        assert job2 is None  # No more pending jobs

    def test_concurrent_claim(self, job_manager: JobManager):
        """Test atomic claiming with concurrent workers."""
        # Submit 10 jobs
        for i in range(10):
            job_manager.submit_job('training', {'job': i}, priority=5)

        # Claim jobs concurrently with 3 workers
        claimed_jobs: List[Job] = []
        lock = threading.Lock()

        def worker(worker_id: str):
            for _ in range(5):  # Try to claim 5 jobs per worker
                job = job_manager.claim_job(worker_id)
                if job:
                    with lock:
                        claimed_jobs.append(job)
                time.sleep(0.001)  # Small delay

        threads = [
            threading.Thread(target=worker, args=(f'worker-{i}',))
            for i in range(3)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify all 10 jobs claimed exactly once
        assert len(claimed_jobs) == 10
        job_ids = [j.job_id for j in claimed_jobs]
        assert len(set(job_ids)) == 10  # No duplicates

    def test_update_job_status(self, job_manager: JobManager):
        """Test status updates."""
        job_id = job_manager.submit_job('training', {}, priority=5)

        # Update to running
        job_manager.update_job_status(job_id, 'running')
        job = job_manager.get_job(job_id)
        assert job.status == 'running'

        # Update to completed
        job_manager.update_job_status(job_id, 'completed')
        job = job_manager.get_job(job_id)
        assert job.status == 'completed'
        assert job.completed_at is not None

    def test_update_job_status_with_error(self, job_manager: JobManager):
        """Test status update with error message."""
        job_id = job_manager.submit_job('training', {}, priority=5)
        job_manager.claim_job('worker-1')

        error_msg = 'CUDA out of memory'
        job_manager.update_job_status(job_id, 'failed', error_message=error_msg)

        job = job_manager.get_job(job_id)
        assert job.status == 'failed'
        assert job.error_message == error_msg

    def test_cancel_job(self, job_manager: JobManager):
        """Test job cancellation."""
        job_id = job_manager.submit_job('training', {}, priority=5)

        # Cancel pending job
        result = job_manager.cancel_job(job_id)
        assert result is True

        job = job_manager.get_job(job_id)
        assert job.status == 'cancelled'

    def test_cancel_completed_job(self, job_manager: JobManager):
        """Test cancelling completed job fails."""
        job_id = job_manager.submit_job('training', {}, priority=5)
        job_manager.update_job_status(job_id, 'completed')

        # Try to cancel completed job
        result = job_manager.cancel_job(job_id)
        assert result is False  # Cannot cancel completed job

    def test_retry_job(self, job_manager: JobManager):
        """Test job retry logic."""
        job_id = job_manager.submit_job('training', {}, priority=5, max_retries=2)

        # Claim and fail
        job_manager.claim_job('worker-1')
        job_manager.update_job_status(job_id, 'failed', error_message='Error 1')

        # Retry
        result = job_manager.retry_job(job_id)
        assert result is True

        job = job_manager.get_job(job_id)
        assert job.status == 'pending'
        assert job.retry_count == 1
        assert job.started_at is None
        assert job.error_message is None

    def test_retry_exceeds_max(self, job_manager: JobManager):
        """Test retry fails when max retries exceeded."""
        job_id = job_manager.submit_job('training', {}, priority=5, max_retries=1)

        # Fail twice
        for i in range(2):
            job_manager.claim_job('worker-1')
            job_manager.update_job_status(job_id, 'failed')
            if i == 0:
                job_manager.retry_job(job_id)

        # Try to retry again (should fail)
        result = job_manager.retry_job(job_id)
        assert result is False

    def test_job_logging(self, job_manager: JobManager):
        """Test job execution logging."""
        job_id = job_manager.submit_job('training', {}, priority=5)

        # Log messages
        job_manager.log_job_message(job_id, 'INFO', 'Starting training')
        job_manager.log_job_message(job_id, 'WARNING', 'Low memory')
        job_manager.log_job_message(job_id, 'ERROR', 'Training failed')

        # Retrieve logs
        logs = job_manager.get_job_logs(job_id)
        assert len(logs) == 3
        assert logs['level'].tolist() == ['INFO', 'WARNING', 'ERROR']
        assert 'Starting training' in logs['message'].tolist()

    def test_queue_stats(self, job_manager: JobManager):
        """Test queue statistics."""
        # Submit jobs with different statuses
        job1 = job_manager.submit_job('training', {}, priority=5)
        job2 = job_manager.submit_job('training', {}, priority=5)
        job3 = job_manager.submit_job('training', {}, priority=5)

        job_manager.claim_job('worker-1')  # job1 running
        job_manager.update_job_status(job2, 'completed')
        job_manager.cancel_job(job3)

        stats = job_manager.get_queue_stats()
        assert stats['running'] == 1
        assert stats['completed'] == 1
        assert stats['cancelled'] == 1
        assert stats['pending'] == 0


class TestTrainingScheduler:
    """Tests for TrainingScheduler."""

    def test_create_schedule(self, scheduler: TrainingScheduler):
        """Test schedule creation."""
        config = {'training_config': {'learning_rate': 5e-5}}
        schedule_id = scheduler.create_schedule(
            name='daily-retrain',
            job_type='training',
            config=config,
            schedule_expr='0 2 * * *',
            priority=5
        )

        assert schedule_id > 0

        schedules = scheduler.list_schedules()
        assert len(schedules) == 1
        assert schedules[0].name == 'daily-retrain'
        assert schedules[0].enabled is True

    def test_compute_next_run_hourly(self, scheduler: TrainingScheduler):
        """Test next run computation for hourly schedule."""
        # Every hour at minute 0
        after = datetime(2025, 1, 15, 10, 30)
        next_run = scheduler._compute_next_run('0 * * * *', after=after)

        # Should be 11:00
        assert next_run.hour == 11
        assert next_run.minute == 0

    def test_compute_next_run_daily(self, scheduler: TrainingScheduler):
        """Test next run computation for daily schedule."""
        # Daily at 2am
        after = datetime(2025, 1, 15, 10, 0)
        next_run = scheduler._compute_next_run('0 2 * * *', after=after)

        # Should be tomorrow at 2am (already past 2am today)
        assert next_run.day == 16
        assert next_run.hour == 2
        assert next_run.minute == 0

    def test_compute_next_run_every_6_hours(self, scheduler: TrainingScheduler):
        """Test next run computation for interval schedule."""
        # Every 6 hours
        after = datetime(2025, 1, 15, 10, 0)
        next_run = scheduler._compute_next_run('0 */6 * * *', after=after)

        # Should be 12:00 (next 6-hour mark)
        assert next_run.hour == 12
        assert next_run.minute == 0

    def test_check_due_schedules(self, scheduler: TrainingScheduler):
        """Test checking and executing due schedules."""
        # Create schedule due 1 hour ago
        past_time = datetime.now() - timedelta(hours=1)
        config = {'training_config': {'learning_rate': 5e-5}}

        # Manually insert schedule with past next_run
        import sqlite3
        import json
        with sqlite3.connect(scheduler.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO schedules (name, job_type, config, schedule_expr, priority, next_run)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', ('test-schedule', 'training', json.dumps(config), '0 * * * *', 5, past_time.isoformat()))
            conn.commit()

        # Check due schedules
        job_ids = scheduler.check_due_schedules()
        assert len(job_ids) == 1

        # Verify job created
        job = scheduler.job_manager.get_job(job_ids[0])
        assert job is not None
        assert job.job_type == 'training'
        assert job.status == 'pending'

    def test_enable_disable_schedule(self, scheduler: TrainingScheduler):
        """Test enabling and disabling schedules."""
        schedule_id = scheduler.create_schedule(
            name='test-schedule',
            job_type='training',
            config={},
            schedule_expr='0 2 * * *',
            priority=5
        )

        # Disable
        scheduler.disable_schedule(schedule_id)
        schedules = scheduler.list_schedules(enabled_only=True)
        assert len(schedules) == 0

        # Enable
        scheduler.enable_schedule(schedule_id)
        schedules = scheduler.list_schedules(enabled_only=True)
        assert len(schedules) == 1


class TestJobExecutor:
    """Tests for JobExecutor."""

    def test_execute_training_job(self, executor: JobExecutor, job_manager: JobManager):
        """Test training job execution (stub)."""
        config = {
            'training_config': {'learning_rate': 5e-5},
            'model_path': 'model.pt',
            'task_spec': {'modality': 'text'}
        }
        job_id = job_manager.submit_job('training', config, priority=5)

        # Claim and execute
        job = job_manager.claim_job(executor.worker_id)
        assert job is not None

        try:
            executor._execute_job(job)
            job_manager.update_job_status(job_id, 'completed')
        except Exception as e:
            job_manager.update_job_status(job_id, 'failed', error_message=str(e))

        # Verify status
        final_job = job_manager.get_job(job_id)
        assert final_job.status == 'completed'

    def test_worker_loop(self, executor: JobExecutor, job_manager: JobManager):
        """Test worker loop processes multiple jobs."""
        # Submit 3 jobs
        for i in range(3):
            job_manager.submit_job('evaluation', {'job': i}, priority=5)

        # Run worker (process 3 jobs then exit)
        executor.run_worker(max_jobs=3)

        # Verify all completed
        stats = job_manager.get_queue_stats()
        assert stats['completed'] == 3
        assert stats['pending'] == 0

    def test_worker_handles_failure(self, executor: JobExecutor, job_manager: JobManager):
        """Test worker handles job failures with retry."""
        # Submit job with invalid config (missing required fields)
        job_id = job_manager.submit_job('training', {}, priority=5, max_retries=1)

        # Process job (will fail)
        executor.run_worker(max_jobs=1)

        # Verify failed and retried
        job = job_manager.get_job(job_id)
        assert job.status == 'pending'  # Retried back to pending
        assert job.retry_count == 1


class TestIntegration:
    """Integration tests for complete workflow."""

    def test_full_workflow(self, temp_db: Path):
        """Test complete job queue workflow."""
        # Setup
        manager = JobManager(temp_db)
        scheduler = TrainingScheduler(temp_db)
        executor = JobExecutor(manager, worker_id='worker-1')

        # Submit manual job
        job1 = manager.submit_job('evaluation', {'test': 'data'}, priority=8)

        # Create schedule
        scheduler.create_schedule(
            name='hourly-check',
            job_type='training',
            config={'training_config': {}},
            schedule_expr='0 * * * *',
            priority=5
        )

        # Process jobs
        executor.run_worker(max_jobs=1)

        # Verify
        job = manager.get_job(job1)
        assert job.status == 'completed'

        stats = manager.get_queue_stats()
        assert stats['completed'] == 1

    def test_performance(self, temp_db: Path):
        """Test queue operations performance (<10ms)."""
        manager = JobManager(temp_db)

        # Submit job
        start = time.perf_counter()
        job_id = manager.submit_job('training', {}, priority=5)
        submit_time = (time.perf_counter() - start) * 1000
        assert submit_time < 10  # <10ms

        # Claim job
        start = time.perf_counter()
        job = manager.claim_job('worker-1')
        claim_time = (time.perf_counter() - start) * 1000
        assert claim_time < 10  # <10ms

        # Update status
        start = time.perf_counter()
        manager.update_job_status(job_id, 'completed')
        update_time = (time.perf_counter() - start) * 1000
        assert update_time < 10  # <10ms

        print(f"Performance: submit={submit_time:.2f}ms, claim={claim_time:.2f}ms, update={update_time:.2f}ms")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
