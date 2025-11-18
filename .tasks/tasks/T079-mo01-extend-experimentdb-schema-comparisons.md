---
id: T079
enhancement_id: MO-01
title: Extend ExperimentDB Schema for Baseline Comparisons and Run Metadata
status: pending
priority: 2
agent: backend
dependencies: []
blocked_by: []
created: 2025-11-18T00:00:00Z
updated: 2025-11-18T00:00:00Z
tags: [monitoring, tier5, experiment-tracking, database, enhancement1.0]

context_refs:
  - context/project.md

est_tokens: 10000
actual_tokens: null
---

## Description

Extend ExperimentDB schema with `models/runs` table (run metadata: task_name, modality, strategy, devices, artifact_paths), `run_metrics` table (per-epoch metrics), and `comparisons` table (baseline vs candidate run tracking).

Provides structured storage for experiment tracking, enabling regression testing and performance comparisons across runs.

## Business Context

**User Story**: As a researcher, I want to compare my new model (candidate) against the baseline, so I can quantify improvements and catch regressions.

**Why This Matters**: Enables data-driven model selection; prevents regressions in production

**What It Unblocks**: MO-02 (regression testing), MO-03 (drift detection), MO-04 (Tier 5 monitoring)

**Priority Justification**: Priority 2 - Foundation for monitoring tier

## Acceptance Criteria

- [ ] `models` or `runs` table added: `run_id, run_name, task_name, modality, strategy, devices, artifact_paths, created_at`
- [ ] `run_metrics` table added: `metric_id, run_id, split (train/val/test), metric_name, value, step, epoch, timestamp`
- [ ] `comparisons` table added (optional): `comparison_id, baseline_run_id, candidate_run_id, created_at, notes`
- [ ] Helper APIs: `register_run(run_info) -> run_id`, `log_metrics(run_id, metrics, split, step)`, `create_comparison(baseline, candidate)`
- [ ] `get_run_metrics(run_id, metric_name) -> DataFrame` for analysis
- [ ] Type hints for all DB methods
- [ ] Unit test: register run, log metrics, query metrics
- [ ] Migration script for existing ExperimentDB instances (if applicable)

## Test Scenarios

**Test Case 1: Register Run**
- Given: Run metadata: name="baseline-v1", task_name="lm_tiny", strategy="auto"
- When: db.register_run(run_info)
- Then: Returns run_id=1, row inserted into runs table

**Test Case 2: Log Metrics**
- Given: run_id=1, metrics={"train/loss": 0.45, "train/accuracy": 0.82}, epoch=5
- When: db.log_metrics(run_id, metrics, split="train", epoch=5)
- Then: 2 rows inserted into run_metrics table

**Test Case 3: Query Metrics**
- Given: run_id=1 has 10 epochs of val/loss logged
- When: db.get_run_metrics(run_id=1, metric_name="val/loss")
- Then: Returns DataFrame with 10 rows: [epoch, value]

**Test Case 4: Create Comparison**
- Given: baseline_run_id=1, candidate_run_id=2
- When: db.create_comparison(1, 2, notes="New architecture test")
- Then: Comparison row created with comparison_id=1

**Test Case 5: Artifact Paths**
- Given: Run has checkpoint_path="checkpoints/run_1/best.pt", export_path="exports/run_1/model.onnx"
- When: Stored in artifact_paths JSON column
- Then: Queryable as dict: `run["artifact_paths"]["checkpoint"]`

## Technical Implementation

```python
# utils/training/experiment_db.py (extend existing)
class ExperimentDB:
    def __init__(self, db_path: str = "experiments.db"):
        self.conn = sqlite3.connect(db_path)
        self._create_schema()

    def _create_schema(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_name TEXT NOT NULL,
                task_name TEXT,
                modality TEXT,
                strategy TEXT,
                devices TEXT,
                artifact_paths TEXT,  -- JSON blob
                created_at TEXT,
                updated_at TEXT,
                status TEXT DEFAULT 'running'
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS run_metrics (
                metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                split TEXT,  -- train/val/test
                metric_name TEXT NOT NULL,
                value REAL NOT NULL,
                step INTEGER,
                epoch INTEGER,
                timestamp TEXT,
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS comparisons (
                comparison_id INTEGER PRIMARY KEY AUTOINCREMENT,
                baseline_run_id INTEGER,
                candidate_run_id INTEGER,
                created_at TEXT,
                notes TEXT,
                FOREIGN KEY (baseline_run_id) REFERENCES runs(run_id),
                FOREIGN KEY (candidate_run_id) REFERENCES runs(run_id)
            )
        """)

    def register_run(self, run_info: dict) -> int:
        """Register new training run."""
        cursor = self.conn.execute("""
            INSERT INTO runs (run_name, task_name, modality, strategy, devices, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            run_info["run_name"],
            run_info.get("task_name"),
            run_info.get("modality"),
            run_info.get("strategy"),
            run_info.get("devices"),
            datetime.now().isoformat()
        ))
        self.conn.commit()
        return cursor.lastrowid

    def log_metrics(
        self, run_id: int, metrics: dict, split: str, step: int | None = None, epoch: int | None = None
    ):
        """Log metrics for a run."""
        timestamp = datetime.now().isoformat()
        for metric_name, value in metrics.items():
            self.conn.execute("""
                INSERT INTO run_metrics (run_id, split, metric_name, value, step, epoch, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (run_id, split, metric_name, value, step, epoch, timestamp))
        self.conn.commit()

    def get_run_metrics(self, run_id: int, metric_name: str | None = None) -> pd.DataFrame:
        """Query metrics for a run."""
        query = "SELECT * FROM run_metrics WHERE run_id = ?"
        params = [run_id]
        if metric_name:
            query += " AND metric_name = ?"
            params.append(metric_name)
        return pd.read_sql_query(query, self.conn, params=params)
```

## Dependencies

**Hard Dependencies**: None - Extends existing ExperimentDB

**External Dependencies**:
- sqlite3 (built-in)
- pandas (already in requirements)

## Design Decisions

**Decision 1: SQLite instead of cloud database**
- **Rationale**: Lightweight, no internet required, portable .db file
- **Trade-offs**: Not suitable for team collaboration (no concurrent writes)

**Decision 2: JSON blob for artifact_paths**
- **Rationale**: Flexible schema for evolving artifact types
- **Trade-offs**: Not queryable by SQL (but rarely needed)

**Decision 3: Separate run_metrics table (not embedded JSON)**
- **Rationale**: Efficient time-series queries, indexed by run_id + metric_name
- **Trade-offs**: More tables, but better performance

## Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Database corruption on Colab crash | H - Lost experiment history | L | Auto-backup to Google Drive every N minutes |
| SQLite locked during concurrent access | M - Write failures | L | Document single-writer constraint; use write-ahead logging (WAL) |

## Progress Log

### 2025-11-18 - Task Created

**Created By:** task-creator agent
**Reason:** First monitoring tier task (MO-01 from enhancement1.0.md)
**Dependencies:** None - extends existing ExperimentDB
**Estimated Complexity:** Standard (schema design + API methods)

## Completion Checklist

- [ ] Schema migration SQL created
- [ ] Helper methods implemented
- [ ] Unit tests for register_run, log_metrics, get_run_metrics
- [ ] All 8 acceptance criteria met
- [ ] All 5 test scenarios validated
- [ ] 3 design decisions documented
- [ ] 2 risks mitigated

**Definition of Done:** ExperimentDB schema extended, runs/metrics/comparisons tables functional, helper APIs working, unit tests passing.
