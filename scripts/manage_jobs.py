#!/usr/bin/env python3
"""
Job Queue Management CLI

Command-line tool for managing training job queue and schedules.

Usage:
    # Submit job
    python scripts/manage_jobs.py submit --type training --config config.json --priority 7

    # List jobs
    python scripts/manage_jobs.py list --status pending --limit 20

    # View job details
    python scripts/manage_jobs.py show 123

    # View job logs
    python scripts/manage_jobs.py logs 123

    # Cancel job
    python scripts/manage_jobs.py cancel 123

    # Retry failed job
    python scripts/manage_jobs.py retry 123

    # Create schedule
    python scripts/manage_jobs.py schedule create --name daily-train \
        --type training --expr "0 2 * * *" --config config.json

    # List schedules
    python scripts/manage_jobs.py schedule list

    # Enable/disable schedule
    python scripts/manage_jobs.py schedule enable 1
    python scripts/manage_jobs.py schedule disable 1

    # Run worker
    python scripts/manage_jobs.py worker --max-jobs 10

    # Queue statistics
    python scripts/manage_jobs.py stats

Author: MLOps Agent (Phase 2 - Production Hardening)
Version: 3.7.0
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.training.job_queue import (
    JobManager,
    TrainingScheduler,
    JobExecutor
)


def submit_job(args: argparse.Namespace) -> None:
    """Submit new job to queue."""
    manager = JobManager(args.db)

    # Load config from file or JSON string
    if args.config.endswith('.json'):
        with open(args.config) as f:
            config = json.load(f)
    else:
        config = json.loads(args.config)

    job_id = manager.submit_job(
        job_type=args.type,
        config=config,
        priority=args.priority,
        max_retries=args.max_retries
    )

    print(f"✅ Submitted {args.type} job with ID={job_id}")


def list_jobs(args: argparse.Namespace) -> None:
    """List jobs with optional filtering."""
    manager = JobManager(args.db)

    jobs = manager.list_jobs(
        status=args.status,
        job_type=args.type,
        limit=args.limit
    )

    if not jobs:
        print("No jobs found")
        return

    # Print table header
    print(f"{'ID':<6} {'Type':<12} {'Status':<10} {'Priority':<8} {'Created':<20} {'Worker':<15}")
    print("-" * 80)

    # Print jobs
    for job in jobs:
        created = job.created_at[:19] if job.created_at else 'N/A'
        worker = job.worker_id or '-'
        print(f"{job.job_id:<6} {job.job_type:<12} {job.status:<10} {job.priority:<8} {created:<20} {worker:<15}")

    print(f"\nTotal: {len(jobs)} jobs")


def show_job(args: argparse.Namespace) -> None:
    """Show detailed job information."""
    manager = JobManager(args.db)
    job = manager.get_job(args.job_id)

    if job is None:
        print(f"❌ Job {args.job_id} not found")
        return

    print(f"Job ID: {job.job_id}")
    print(f"Type: {job.job_type}")
    print(f"Status: {job.status}")
    print(f"Priority: {job.priority}")
    print(f"Created: {job.created_at}")
    print(f"Started: {job.started_at or 'Not started'}")
    print(f"Completed: {job.completed_at or 'Not completed'}")
    print(f"Worker: {job.worker_id or 'Not assigned'}")
    print(f"Retries: {job.retry_count}/{job.max_retries}")

    if job.error_message:
        print(f"\n❌ Error: {job.error_message}")

    print(f"\nConfiguration:")
    print(json.dumps(job.config, indent=2))


def show_logs(args: argparse.Namespace) -> None:
    """Show job execution logs."""
    manager = JobManager(args.db)
    logs = manager.get_job_logs(args.job_id)

    if logs.empty:
        print(f"No logs found for job {args.job_id}")
        return

    print(f"Logs for job {args.job_id}:")
    print("-" * 80)

    for _, row in logs.iterrows():
        timestamp = row['timestamp'][:19]
        level = row['level']
        message = row['message']
        print(f"[{timestamp}] {level}: {message}")


def cancel_job(args: argparse.Namespace) -> None:
    """Cancel pending or running job."""
    manager = JobManager(args.db)
    success = manager.cancel_job(args.job_id)

    if success:
        print(f"✅ Cancelled job {args.job_id}")
    else:
        print(f"❌ Failed to cancel job {args.job_id} (already completed/failed)")


def retry_job(args: argparse.Namespace) -> None:
    """Retry failed job."""
    manager = JobManager(args.db)
    success = manager.retry_job(args.job_id)

    if success:
        print(f"✅ Job {args.job_id} reset to pending (will retry)")
    else:
        print(f"❌ Failed to retry job {args.job_id} (max retries exceeded or not found)")


def create_schedule(args: argparse.Namespace) -> None:
    """Create recurring job schedule."""
    scheduler = TrainingScheduler(args.db)

    # Load config
    if args.config.endswith('.json'):
        with open(args.config) as f:
            config = json.load(f)
    else:
        config = json.loads(args.config)

    schedule_id = scheduler.create_schedule(
        name=args.name,
        job_type=args.type,
        config=config,
        schedule_expr=args.expr,
        priority=args.priority
    )

    print(f"✅ Created schedule '{args.name}' with ID={schedule_id}")
    print(f"   Expression: {args.expr}")
    print(f"   Next run: {scheduler.list_schedules()[0].next_run}")


def list_schedules(args: argparse.Namespace) -> None:
    """List all schedules."""
    scheduler = TrainingScheduler(args.db)
    schedules = scheduler.list_schedules(enabled_only=args.enabled_only)

    if not schedules:
        print("No schedules found")
        return

    # Print table header
    print(f"{'ID':<6} {'Name':<20} {'Type':<12} {'Expression':<15} {'Enabled':<8} {'Next Run':<20}")
    print("-" * 90)

    # Print schedules
    for schedule in schedules:
        enabled = '✓' if schedule.enabled else '✗'
        next_run = schedule.next_run[:19]
        print(f"{schedule.schedule_id:<6} {schedule.name:<20} {schedule.job_type:<12} "
              f"{schedule.schedule_expr:<15} {enabled:<8} {next_run:<20}")

    print(f"\nTotal: {len(schedules)} schedules")


def enable_schedule(args: argparse.Namespace) -> None:
    """Enable schedule."""
    scheduler = TrainingScheduler(args.db)
    scheduler.enable_schedule(args.schedule_id)
    print(f"✅ Enabled schedule {args.schedule_id}")


def disable_schedule(args: argparse.Namespace) -> None:
    """Disable schedule."""
    scheduler = TrainingScheduler(args.db)
    scheduler.disable_schedule(args.schedule_id)
    print(f"✅ Disabled schedule {args.schedule_id}")


def run_worker(args: argparse.Namespace) -> None:
    """Run job worker."""
    manager = JobManager(args.db)
    executor = JobExecutor(manager, worker_id=args.worker_id)

    print(f"Starting worker {args.worker_id}...")
    print(f"Max jobs: {args.max_jobs or 'unlimited'}")
    print(f"Poll interval: {args.poll_interval}s")
    print("Press Ctrl+C to stop")

    try:
        executor.run_worker(
            max_jobs=args.max_jobs,
            poll_interval=args.poll_interval
        )
    except KeyboardInterrupt:
        print("\n\nWorker stopped by user")


def show_stats(args: argparse.Namespace) -> None:
    """Show queue statistics."""
    manager = JobManager(args.db)
    stats = manager.get_queue_stats()

    print("Queue Statistics:")
    print("-" * 40)
    print(f"Pending:    {stats['pending']:>6}")
    print(f"Running:    {stats['running']:>6}")
    print(f"Completed:  {stats['completed']:>6}")
    print(f"Failed:     {stats['failed']:>6}")
    print(f"Cancelled:  {stats['cancelled']:>6}")
    print("-" * 40)
    print(f"Total:      {sum(stats.values()):>6}")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Job Queue Management CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--db',
        type=str,
        default='jobs.db',
        help='Path to job queue database (default: jobs.db)'
    )

    subparsers = parser.add_subparsers(dest='command', help='Command')

    # Submit job
    submit_parser = subparsers.add_parser('submit', help='Submit new job')
    submit_parser.add_argument('--type', required=True, choices=['training', 'evaluation', 'export', 'retraining'])
    submit_parser.add_argument('--config', required=True, help='Config JSON file or string')
    submit_parser.add_argument('--priority', type=int, default=5, help='Priority 1-10 (default: 5)')
    submit_parser.add_argument('--max-retries', type=int, default=3, help='Max retries (default: 3)')
    submit_parser.set_defaults(func=submit_job)

    # List jobs
    list_parser = subparsers.add_parser('list', help='List jobs')
    list_parser.add_argument('--status', choices=['pending', 'running', 'completed', 'failed', 'cancelled'])
    list_parser.add_argument('--type', choices=['training', 'evaluation', 'export', 'retraining'])
    list_parser.add_argument('--limit', type=int, default=100, help='Max jobs to show (default: 100)')
    list_parser.set_defaults(func=list_jobs)

    # Show job
    show_parser = subparsers.add_parser('show', help='Show job details')
    show_parser.add_argument('job_id', type=int, help='Job ID')
    show_parser.set_defaults(func=show_job)

    # Show logs
    logs_parser = subparsers.add_parser('logs', help='Show job logs')
    logs_parser.add_argument('job_id', type=int, help='Job ID')
    logs_parser.set_defaults(func=show_logs)

    # Cancel job
    cancel_parser = subparsers.add_parser('cancel', help='Cancel job')
    cancel_parser.add_argument('job_id', type=int, help='Job ID')
    cancel_parser.set_defaults(func=cancel_job)

    # Retry job
    retry_parser = subparsers.add_parser('retry', help='Retry failed job')
    retry_parser.add_argument('job_id', type=int, help='Job ID')
    retry_parser.set_defaults(func=retry_job)

    # Schedule commands
    schedule_parser = subparsers.add_parser('schedule', help='Manage schedules')
    schedule_subparsers = schedule_parser.add_subparsers(dest='schedule_command')

    # Create schedule
    schedule_create = schedule_subparsers.add_parser('create', help='Create schedule')
    schedule_create.add_argument('--name', required=True, help='Schedule name')
    schedule_create.add_argument('--type', required=True, choices=['training', 'evaluation', 'export', 'retraining'])
    schedule_create.add_argument('--config', required=True, help='Config JSON file or string')
    schedule_create.add_argument('--expr', required=True, help='Cron expression (e.g., "0 2 * * *")')
    schedule_create.add_argument('--priority', type=int, default=5, help='Priority (default: 5)')
    schedule_create.set_defaults(func=create_schedule)

    # List schedules
    schedule_list = schedule_subparsers.add_parser('list', help='List schedules')
    schedule_list.add_argument('--enabled-only', action='store_true', help='Show only enabled schedules')
    schedule_list.set_defaults(func=list_schedules)

    # Enable schedule
    schedule_enable = schedule_subparsers.add_parser('enable', help='Enable schedule')
    schedule_enable.add_argument('schedule_id', type=int, help='Schedule ID')
    schedule_enable.set_defaults(func=enable_schedule)

    # Disable schedule
    schedule_disable = schedule_subparsers.add_parser('disable', help='Disable schedule')
    schedule_disable.add_argument('schedule_id', type=int, help='Schedule ID')
    schedule_disable.set_defaults(func=disable_schedule)

    # Run worker
    worker_parser = subparsers.add_parser('worker', help='Run job worker')
    worker_parser.add_argument('--worker-id', default=None, help='Worker ID (default: auto-generated)')
    worker_parser.add_argument('--max-jobs', type=int, default=None, help='Max jobs to process (default: unlimited)')
    worker_parser.add_argument('--poll-interval', type=float, default=5.0, help='Poll interval in seconds (default: 5.0)')
    worker_parser.set_defaults(func=run_worker)

    # Queue stats
    stats_parser = subparsers.add_parser('stats', help='Show queue statistics')
    stats_parser.set_defaults(func=show_stats)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Execute command
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
