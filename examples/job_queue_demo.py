"""
Job Queue and Scheduler Demo

Demonstrates complete workflow:
1. Submit manual training jobs with priorities
2. Create recurring schedules (hourly, daily)
3. Run worker to process jobs
4. Monitor queue statistics
5. Handle job failures and retries

Usage:
    python examples/job_queue_demo.py

Author: MLOps Agent (Phase 2 - Production Hardening)
Version: 3.7.0
"""

import time
from pathlib import Path
from utils.training.job_queue import (
    JobManager,
    TrainingScheduler,
    JobExecutor
)
from utils.training.training_config import TrainingConfig


def demo_basic_workflow():
    """Demo 1: Basic job submission and execution."""
    print("\n" + "=" * 80)
    print("DEMO 1: Basic Job Submission and Execution")
    print("=" * 80)

    # Initialize
    db_path = Path('demo_jobs.db')
    if db_path.exists():
        db_path.unlink()  # Clean slate

    manager = JobManager(db_path)

    # Submit jobs with different priorities
    print("\n📋 Submitting jobs with different priorities...")

    config1 = TrainingConfig(learning_rate=1e-4, batch_size=8, epochs=5).to_dict()
    job1 = manager.submit_job('training', {'training_config': config1}, priority=3)
    print(f"   - Job {job1}: Training (priority 3)")

    config2 = TrainingConfig(learning_rate=5e-5, batch_size=4, epochs=10).to_dict()
    job2 = manager.submit_job('training', {'training_config': config2}, priority=9)
    print(f"   - Job {job2}: Training (priority 9) - URGENT")

    job3 = manager.submit_job('evaluation', {'dataset': 'test'}, priority=5)
    print(f"   - Job {job3}: Evaluation (priority 5)")

    # Show queue stats
    print("\n📊 Queue statistics:")
    stats = manager.get_queue_stats()
    print(f"   Pending: {stats['pending']}, Running: {stats['running']}, Completed: {stats['completed']}")

    # List jobs (should be ordered by priority)
    print("\n📝 Job queue (ordered by priority):")
    jobs = manager.list_jobs()
    for job in jobs:
        print(f"   {job.job_id}. [{job.priority}] {job.job_type} - {job.status}")

    # Cleanup
    print("\n✅ Demo 1 complete\n")


def demo_atomic_claiming():
    """Demo 2: Atomic job claiming with concurrent workers."""
    print("\n" + "=" * 80)
    print("DEMO 2: Atomic Job Claiming (Concurrent Workers)")
    print("=" * 80)

    db_path = Path('demo_jobs.db')
    manager = JobManager(db_path)

    # Submit 5 jobs
    print("\n📋 Submitting 5 jobs...")
    for i in range(5):
        manager.submit_job('evaluation', {'test': i}, priority=5)

    # Simulate 2 workers claiming jobs
    print("\n👷 Worker 1 claiming job...")
    job1 = manager.claim_job('worker-1')
    if job1:
        print(f"   Worker 1 claimed job {job1.job_id}")

    print("👷 Worker 2 claiming job...")
    job2 = manager.claim_job('worker-2')
    if job2:
        print(f"   Worker 2 claimed job {job2.job_id}")

    # Verify no overlap
    assert job1.job_id != job2.job_id, "Race condition detected!"
    print("\n✅ No race condition - jobs claimed atomically")

    # Complete jobs
    manager.update_job_status(job1.job_id, 'completed')
    manager.update_job_status(job2.job_id, 'completed')

    stats = manager.get_queue_stats()
    print(f"\n📊 Queue: Pending={stats['pending']}, Running={stats['running']}, Completed={stats['completed']}")

    print("\n✅ Demo 2 complete\n")


def demo_job_retry():
    """Demo 3: Job failure and retry logic."""
    print("\n" + "=" * 80)
    print("DEMO 3: Job Failure and Retry Logic")
    print("=" * 80)

    db_path = Path('demo_jobs.db')
    manager = JobManager(db_path)

    # Submit job with max 2 retries
    print("\n📋 Submitting job with max_retries=2...")
    job_id = manager.submit_job('training', {'config': 'test'}, priority=5, max_retries=2)

    # Simulate failure
    print(f"\n❌ Job {job_id} failed (attempt 1/3)")
    manager.claim_job('worker-1')
    manager.update_job_status(job_id, 'failed', error_message='CUDA out of memory')

    # Retry
    print(f"🔄 Retrying job {job_id}...")
    success = manager.retry_job(job_id)
    if success:
        job = manager.get_job(job_id)
        print(f"   ✅ Job reset to pending (retry_count={job.retry_count})")

    # Fail again
    print(f"\n❌ Job {job_id} failed (attempt 2/3)")
    manager.claim_job('worker-1')
    manager.update_job_status(job_id, 'failed', error_message='CUDA out of memory')

    # Retry again
    print(f"🔄 Retrying job {job_id}...")
    success = manager.retry_job(job_id)
    if success:
        job = manager.get_job(job_id)
        print(f"   ✅ Job reset to pending (retry_count={job.retry_count})")

    # Fail third time
    print(f"\n❌ Job {job_id} failed (attempt 3/3)")
    manager.claim_job('worker-1')
    manager.update_job_status(job_id, 'failed', error_message='CUDA out of memory')

    # Try to retry (should fail - max retries exceeded)
    print(f"🔄 Attempting to retry job {job_id}...")
    success = manager.retry_job(job_id)
    if not success:
        print(f"   ❌ Max retries exceeded - job permanently failed")

    print("\n✅ Demo 3 complete\n")


def demo_scheduler():
    """Demo 4: Recurring job schedules."""
    print("\n" + "=" * 80)
    print("DEMO 4: Recurring Job Schedules")
    print("=" * 80)

    db_path = Path('demo_jobs.db')
    scheduler = TrainingScheduler(db_path)

    # Create hourly schedule
    print("\n📅 Creating hourly schedule...")
    config = TrainingConfig(learning_rate=5e-5, batch_size=4, epochs=5).to_dict()
    schedule1 = scheduler.create_schedule(
        name='hourly-evaluation',
        job_type='evaluation',
        config={'training_config': config},
        schedule_expr='0 * * * *',  # Every hour
        priority=5
    )
    print(f"   ✅ Schedule {schedule1}: hourly-evaluation")

    # Create daily schedule
    print("\n📅 Creating daily schedule...")
    schedule2 = scheduler.create_schedule(
        name='daily-retrain',
        job_type='training',
        config={'training_config': config},
        schedule_expr='0 2 * * *',  # Daily at 2am
        priority=7
    )
    print(f"   ✅ Schedule {schedule2}: daily-retrain")

    # List schedules
    print("\n📋 Active schedules:")
    schedules = scheduler.list_schedules()
    for s in schedules:
        enabled = '✓' if s.enabled else '✗'
        print(f"   [{enabled}] {s.name} ({s.schedule_expr}) - Next: {s.next_run[:19]}")

    # Disable schedule
    print(f"\n⏸️  Disabling schedule {schedule1}...")
    scheduler.disable_schedule(schedule1)

    enabled_schedules = scheduler.list_schedules(enabled_only=True)
    print(f"   Active schedules: {len(enabled_schedules)}")

    print("\n✅ Demo 4 complete\n")


def demo_worker_execution():
    """Demo 5: Worker execution loop."""
    print("\n" + "=" * 80)
    print("DEMO 5: Worker Execution Loop")
    print("=" * 80)

    db_path = Path('demo_jobs.db')
    manager = JobManager(db_path)
    executor = JobExecutor(manager, worker_id='demo-worker')

    # Submit 3 jobs
    print("\n📋 Submitting 3 evaluation jobs...")
    for i in range(3):
        manager.submit_job('evaluation', {'test': i}, priority=5)

    # Run worker
    print("\n👷 Starting worker (will process 3 jobs)...")
    start = time.time()
    executor.run_worker(max_jobs=3)
    duration = time.time() - start

    print(f"\n✅ Worker completed 3 jobs in {duration:.2f}s")

    # Show final stats
    stats = manager.get_queue_stats()
    print(f"📊 Final queue stats: Completed={stats['completed']}, Pending={stats['pending']}")

    print("\n✅ Demo 5 complete\n")


def demo_job_logs():
    """Demo 6: Job execution logging."""
    print("\n" + "=" * 80)
    print("DEMO 6: Job Execution Logging")
    print("=" * 80)

    db_path = Path('demo_jobs.db')
    manager = JobManager(db_path)

    # Submit and claim job
    print("\n📋 Submitting job...")
    job_id = manager.submit_job('training', {'config': 'test'}, priority=5)
    manager.claim_job('worker-1')

    # Log execution progress
    print(f"\n📝 Logging execution for job {job_id}...")
    manager.log_job_message(job_id, 'INFO', 'Starting training')
    manager.log_job_message(job_id, 'INFO', 'Epoch 1/10 - loss: 0.45')
    manager.log_job_message(job_id, 'WARNING', 'GPU memory at 85%')
    manager.log_job_message(job_id, 'INFO', 'Epoch 2/10 - loss: 0.38')
    manager.log_job_message(job_id, 'INFO', 'Training completed successfully')

    manager.update_job_status(job_id, 'completed')

    # Retrieve logs
    print(f"\n📜 Job {job_id} logs:")
    logs = manager.get_job_logs(job_id)
    for _, row in logs.iterrows():
        timestamp = row['timestamp'][:19]
        level = row['level']
        message = row['message']
        print(f"   [{timestamp}] {level}: {message}")

    print("\n✅ Demo 6 complete\n")


def main():
    """Run all demos."""
    print("\n" + "=" * 80)
    print("JOB QUEUE AND SCHEDULER DEMO")
    print("=" * 80)

    try:
        demo_basic_workflow()
        demo_atomic_claiming()
        demo_job_retry()
        demo_scheduler()
        demo_worker_execution()
        demo_job_logs()

        print("\n" + "=" * 80)
        print("ALL DEMOS COMPLETE ✅")
        print("=" * 80)

        # Cleanup
        db_path = Path('demo_jobs.db')
        if db_path.exists():
            print(f"\nDemo database saved at: {db_path}")
            print("To inspect: sqlite3 demo_jobs.db")

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
