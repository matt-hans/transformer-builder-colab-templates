"""
Tests for automated retraining trigger system.

Tests cover:
- Individual trigger implementations (Drift, Performance, Time, DataVolume)
- Composite triggers (AND/OR logic)
- Trigger manager registration and evaluation
- Report generation and serialization
- Integration with MetricsEngine, ModelRegistry, ExperimentDB
"""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any

import pytest

from utils.training.retraining_triggers import (
    # Core classes
    RetrainingTriggerManager,
    RetrainingReport,
    TriggerDetail,

    # Trigger implementations
    DriftTrigger,
    PerformanceTrigger,
    TimeTrigger,
    DataVolumeTrigger,
    CompositeTrigger,

    # Configuration
    TriggerConfig,
    DriftTriggerConfig,
    PerformanceTriggerConfig,
    TimeTriggerConfig,
    DataVolumeTriggerConfig,

    # Example configs
    get_conservative_config,
    get_aggressive_config,
    get_balanced_config,
)


# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------

@pytest.fixture
def sample_drift_metrics() -> Dict[str, Any]:
    """Sample drift metrics from MetricsEngine."""
    return {
        'js_divergence': 0.18,
        'status': 'warning',
        'seq_length_js': 0.12,
        'token_overlap': 0.85,
        'brightness_js': 0.08
    }


@pytest.fixture
def sample_current_metrics() -> Dict[str, float]:
    """Sample current model performance metrics."""
    return {
        'val_loss': 0.45,
        'val_accuracy': 0.82,
        'val_perplexity': 1.57
    }


@pytest.fixture
def sample_baseline_metrics() -> Dict[str, float]:
    """Sample baseline model performance metrics."""
    return {
        'val_loss': 0.40,
        'val_accuracy': 0.85,
        'val_perplexity': 1.49
    }


@pytest.fixture
def sample_metadata() -> Dict[str, Any]:
    """Sample metadata for trigger evaluation."""
    last_training = datetime.now() - timedelta(days=8)
    return {
        'last_training_time': last_training.isoformat(),
        'current_count': 6000,
        'baseline_count': 5000,
        'model_id': 5,
        'run_id': 42
    }


# -------------------------------------------------------------------------
# Test Trigger Implementations
# -------------------------------------------------------------------------

class TestDriftTrigger:
    """Tests for DriftTrigger."""

    def test_trigger_fires_above_threshold(self, sample_drift_metrics):
        """Test trigger fires when drift exceeds threshold."""
        trigger = DriftTrigger(threshold=0.15, metric_name='js_divergence')
        result = trigger.evaluate(drift_metrics=sample_drift_metrics)

        assert result.triggered is True
        assert result.severity == 'warning'
        assert result.threshold == 0.15
        assert result.actual_value == 0.18
        assert 'exceeds threshold' in result.reason

    def test_trigger_not_fires_below_threshold(self, sample_drift_metrics):
        """Test trigger does not fire when drift below threshold."""
        trigger = DriftTrigger(threshold=0.20, metric_name='js_divergence')
        result = trigger.evaluate(drift_metrics=sample_drift_metrics)

        assert result.triggered is False
        assert 'within threshold' in result.reason

    def test_missing_drift_metrics(self):
        """Test graceful handling when drift metrics missing."""
        trigger = DriftTrigger(threshold=0.15)
        result = trigger.evaluate(drift_metrics=None)

        assert result.triggered is False
        assert 'No drift metrics provided' in result.reason

    def test_missing_metric_name(self, sample_drift_metrics):
        """Test graceful handling when specific metric not found."""
        trigger = DriftTrigger(threshold=0.15, metric_name='nonexistent_metric')
        result = trigger.evaluate(drift_metrics=sample_drift_metrics)

        assert result.triggered is False
        assert 'not found' in result.reason

    def test_custom_metric_name(self):
        """Test trigger with custom drift metric."""
        drift_metrics = {'brightness_js': 0.25}
        trigger = DriftTrigger(threshold=0.20, metric_name='brightness_js')
        result = trigger.evaluate(drift_metrics=drift_metrics)

        assert result.triggered is True
        assert result.actual_value == 0.25


class TestPerformanceTrigger:
    """Tests for PerformanceTrigger."""

    def test_trigger_fires_loss_increase(
        self, sample_current_metrics, sample_baseline_metrics
    ):
        """Test trigger fires when loss increases beyond threshold."""
        trigger = PerformanceTrigger(
            threshold=0.05,  # 5% threshold
            metric_name='val_loss',
            mode='min'
        )
        result = trigger.evaluate(
            current_metrics=sample_current_metrics,
            baseline_metrics=sample_baseline_metrics
        )

        # Loss increased from 0.40 to 0.45 (12.5% increase)
        assert result.triggered is True
        assert result.actual_value > 0.05
        assert 'degradation' in result.reason.lower()

    def test_trigger_fires_accuracy_decrease(
        self, sample_current_metrics, sample_baseline_metrics
    ):
        """Test trigger fires when accuracy decreases beyond threshold."""
        trigger = PerformanceTrigger(
            threshold=0.03,  # 3% threshold
            metric_name='val_accuracy',
            mode='max'
        )
        result = trigger.evaluate(
            current_metrics=sample_current_metrics,
            baseline_metrics=sample_baseline_metrics
        )

        # Accuracy decreased from 0.85 to 0.82 (3.5% decrease)
        assert result.triggered is True
        assert result.actual_value > 0.03

    def test_trigger_not_fires_within_threshold(
        self, sample_current_metrics, sample_baseline_metrics
    ):
        """Test trigger does not fire when performance within threshold."""
        trigger = PerformanceTrigger(
            threshold=0.15,  # 15% threshold (very high)
            metric_name='val_loss',
            mode='min'
        )
        result = trigger.evaluate(
            current_metrics=sample_current_metrics,
            baseline_metrics=sample_baseline_metrics
        )

        assert result.triggered is False
        assert 'stable' in result.reason.lower()

    def test_missing_metrics(self):
        """Test graceful handling when metrics missing."""
        trigger = PerformanceTrigger(threshold=0.05)
        result = trigger.evaluate(
            current_metrics=None,
            baseline_metrics=None
        )

        assert result.triggered is False
        assert 'Missing' in result.reason

    def test_missing_metric_name(
        self, sample_current_metrics, sample_baseline_metrics
    ):
        """Test graceful handling when specific metric not found."""
        trigger = PerformanceTrigger(
            threshold=0.05,
            metric_name='nonexistent_metric'
        )
        result = trigger.evaluate(
            current_metrics=sample_current_metrics,
            baseline_metrics=sample_baseline_metrics
        )

        assert result.triggered is False
        assert 'not found' in result.reason

    def test_zero_baseline_handling(self):
        """Test handling of zero baseline value."""
        trigger = PerformanceTrigger(threshold=0.05, metric_name='val_loss')
        result = trigger.evaluate(
            current_metrics={'val_loss': 0.5},
            baseline_metrics={'val_loss': 0.0}
        )

        # Should not crash, should detect change
        assert isinstance(result.triggered, bool)


class TestTimeTrigger:
    """Tests for TimeTrigger."""

    def test_trigger_fires_after_interval(self):
        """Test trigger fires when interval elapsed."""
        # Last training was 8 days ago
        last_training = datetime.now() - timedelta(days=8)
        metadata = {'last_training_time': last_training.isoformat()}

        trigger = TimeTrigger(interval_hours=168)  # 7 days
        result = trigger.evaluate(metadata=metadata)

        assert result.triggered is True
        assert result.actual_value > 168
        assert 'elapsed' in result.reason.lower()

    def test_trigger_not_fires_within_interval(self):
        """Test trigger does not fire within interval."""
        # Last training was 5 days ago
        last_training = datetime.now() - timedelta(days=5)
        metadata = {'last_training_time': last_training.isoformat()}

        trigger = TimeTrigger(interval_hours=168)  # 7 days
        result = trigger.evaluate(metadata=metadata)

        assert result.triggered is False
        assert 'remaining' in result.reason.lower()

    def test_missing_last_training_time(self):
        """Test graceful handling when last training time missing."""
        trigger = TimeTrigger(interval_hours=168)
        result = trigger.evaluate(metadata={})

        assert result.triggered is False
        assert 'No last training time' in result.reason

    def test_invalid_timestamp_format(self):
        """Test graceful handling of invalid timestamp."""
        metadata = {'last_training_time': 'invalid-timestamp'}
        trigger = TimeTrigger(interval_hours=168)
        result = trigger.evaluate(metadata=metadata)

        assert result.triggered is False
        assert 'Invalid timestamp' in result.reason

    def test_custom_interval(self):
        """Test trigger with custom time interval."""
        last_training = datetime.now() - timedelta(hours=49)
        metadata = {'last_training_time': last_training.isoformat()}

        trigger = TimeTrigger(interval_hours=48)  # 2 days
        result = trigger.evaluate(metadata=metadata)

        assert result.triggered is True


class TestDataVolumeTrigger:
    """Tests for DataVolumeTrigger."""

    def test_trigger_fires_samples_threshold(self):
        """Test trigger fires when sample threshold exceeded."""
        metadata = {'current_count': 6000, 'baseline_count': 5000}
        trigger = DataVolumeTrigger(threshold_samples=500)
        result = trigger.evaluate(metadata=metadata)

        # 1000 new samples > 500 threshold
        assert result.triggered is True
        assert result.actual_value == 1000

    def test_trigger_fires_percentage_threshold(self):
        """Test trigger fires when percentage threshold exceeded."""
        metadata = {'current_count': 6000, 'baseline_count': 5000}
        trigger = DataVolumeTrigger(threshold_percentage=0.15)
        result = trigger.evaluate(metadata=metadata)

        # 20% increase > 15% threshold
        assert result.triggered is True

    def test_trigger_fires_either_threshold(self):
        """Test trigger fires if either threshold exceeded (OR logic)."""
        metadata = {'current_count': 5400, 'baseline_count': 5000}

        # Only samples threshold met
        trigger = DataVolumeTrigger(
            threshold_samples=300,
            threshold_percentage=0.20
        )
        result = trigger.evaluate(metadata=metadata)

        # 400 new samples > 300, but 8% < 20%
        assert result.triggered is True

    def test_trigger_not_fires_within_thresholds(self):
        """Test trigger does not fire when within thresholds."""
        metadata = {'current_count': 5100, 'baseline_count': 5000}

        trigger = DataVolumeTrigger(
            threshold_samples=500,
            threshold_percentage=0.20
        )
        result = trigger.evaluate(metadata=metadata)

        # 100 new samples < 500 and 2% < 20%
        assert result.triggered is False

    def test_missing_metadata(self):
        """Test graceful handling when metadata missing."""
        trigger = DataVolumeTrigger(threshold_samples=500)
        result = trigger.evaluate(metadata={})

        assert result.triggered is False
        assert 'Missing' in result.reason

    def test_zero_baseline_handling(self):
        """Test handling of zero baseline count."""
        metadata = {'current_count': 1000, 'baseline_count': 0}
        trigger = DataVolumeTrigger(threshold_percentage=0.10)
        result = trigger.evaluate(metadata=metadata)

        # Should not crash
        assert isinstance(result.triggered, bool)

    def test_requires_at_least_one_threshold(self):
        """Test that at least one threshold must be specified."""
        with pytest.raises(ValueError, match='at least one'):
            DataVolumeTrigger(threshold_samples=None, threshold_percentage=None)


class TestCompositeTrigger:
    """Tests for CompositeTrigger."""

    def test_and_logic_all_triggered(
        self, sample_drift_metrics, sample_current_metrics, sample_baseline_metrics
    ):
        """Test AND logic when all triggers fire."""
        drift_trigger = DriftTrigger(threshold=0.15)
        perf_trigger = PerformanceTrigger(threshold=0.05, metric_name='val_loss')

        composite = CompositeTrigger(
            triggers=[drift_trigger, perf_trigger],
            logic='AND'
        )

        result = composite.evaluate(
            drift_metrics=sample_drift_metrics,
            current_metrics=sample_current_metrics,
            baseline_metrics=sample_baseline_metrics
        )

        assert result.triggered is True

    def test_and_logic_partial_triggered(
        self, sample_drift_metrics, sample_current_metrics, sample_baseline_metrics
    ):
        """Test AND logic when only some triggers fire."""
        drift_trigger = DriftTrigger(threshold=0.25)  # Will not fire (0.18 < 0.25)
        perf_trigger = PerformanceTrigger(threshold=0.05, metric_name='val_loss')

        composite = CompositeTrigger(
            triggers=[drift_trigger, perf_trigger],
            logic='AND'
        )

        result = composite.evaluate(
            drift_metrics=sample_drift_metrics,
            current_metrics=sample_current_metrics,
            baseline_metrics=sample_baseline_metrics
        )

        assert result.triggered is False

    def test_or_logic_any_triggered(
        self, sample_drift_metrics, sample_current_metrics, sample_baseline_metrics
    ):
        """Test OR logic when any trigger fires."""
        drift_trigger = DriftTrigger(threshold=0.25)  # Will not fire
        perf_trigger = PerformanceTrigger(threshold=0.05, metric_name='val_loss')

        composite = CompositeTrigger(
            triggers=[drift_trigger, perf_trigger],
            logic='OR'
        )

        result = composite.evaluate(
            drift_metrics=sample_drift_metrics,
            current_metrics=sample_current_metrics,
            baseline_metrics=sample_baseline_metrics
        )

        assert result.triggered is True

    def test_or_logic_none_triggered(self):
        """Test OR logic when no triggers fire."""
        drift_trigger = DriftTrigger(threshold=0.50)
        perf_trigger = PerformanceTrigger(threshold=0.50, metric_name='val_loss')

        composite = CompositeTrigger(
            triggers=[drift_trigger, perf_trigger],
            logic='OR'
        )

        result = composite.evaluate(
            drift_metrics={'js_divergence': 0.10},
            current_metrics={'val_loss': 0.40},
            baseline_metrics={'val_loss': 0.39}
        )

        assert result.triggered is False

    def test_nested_composite_triggers(self):
        """Test nested composite triggers (AND of ORs)."""
        drift_trigger = DriftTrigger(threshold=0.15)
        perf_trigger = PerformanceTrigger(threshold=0.05, metric_name='val_loss')
        time_trigger = TimeTrigger(interval_hours=1)  # Will fire (old timestamp)

        # Inner OR: drift OR performance
        inner_or = CompositeTrigger(
            triggers=[drift_trigger, perf_trigger],
            logic='OR',
            name='drift_or_perf'
        )

        # Outer AND: (drift OR performance) AND time
        outer_and = CompositeTrigger(
            triggers=[inner_or, time_trigger],
            logic='AND',
            name='composite_policy'
        )

        last_training = datetime.now() - timedelta(hours=2)
        result = outer_and.evaluate(
            drift_metrics={'js_divergence': 0.18},
            current_metrics={'val_loss': 0.45},
            baseline_metrics={'val_loss': 0.40},
            metadata={'last_training_time': last_training.isoformat()}
        )

        assert result.triggered is True


# -------------------------------------------------------------------------
# Test Trigger Manager
# -------------------------------------------------------------------------

class TestRetrainingTriggerManager:
    """Tests for RetrainingTriggerManager."""

    def test_register_trigger(self):
        """Test registering a trigger."""
        manager = RetrainingTriggerManager()
        trigger = DriftTrigger(threshold=0.15)

        manager.register_trigger('drift_monitor', trigger)

        assert 'drift_monitor' in manager.triggers
        assert manager.triggers['drift_monitor'] == trigger

    def test_register_convenience_methods(self):
        """Test convenience methods for registering triggers."""
        manager = RetrainingTriggerManager()

        manager.register_drift_trigger(threshold=0.15)
        manager.register_performance_trigger(threshold=0.05)
        manager.register_time_trigger(interval_hours=168)
        manager.register_data_volume_trigger(threshold_samples=1000)

        assert len(manager.triggers) == 4
        assert 'drift_trigger' in manager.triggers
        assert 'performance_trigger' in manager.triggers
        assert 'time_trigger' in manager.triggers
        assert 'data_volume_trigger' in manager.triggers

    def test_evaluate_all_triggers(
        self, sample_drift_metrics, sample_current_metrics,
        sample_baseline_metrics, sample_metadata
    ):
        """Test evaluating all registered triggers."""
        manager = RetrainingTriggerManager()

        manager.register_drift_trigger(threshold=0.15)
        manager.register_performance_trigger(threshold=0.05, metric_name='val_loss')
        manager.register_time_trigger(interval_hours=168)
        manager.register_data_volume_trigger(threshold_samples=500)

        report = manager.evaluate(
            drift_metrics=sample_drift_metrics,
            current_metrics=sample_current_metrics,
            baseline_metrics=sample_baseline_metrics,
            metadata=sample_metadata
        )

        assert isinstance(report, RetrainingReport)
        assert len(report.trigger_details) == 4
        assert report.triggered is True  # Multiple triggers fire
        assert report.severity in ['info', 'warning', 'critical']
        assert len(report.recommendations) > 0

    def test_evaluate_no_triggers_fired(self):
        """Test evaluation when no triggers fire."""
        manager = RetrainingTriggerManager()

        manager.register_drift_trigger(threshold=0.50)  # Very high
        manager.register_performance_trigger(threshold=0.50, metric_name='val_loss')

        report = manager.evaluate(
            drift_metrics={'js_divergence': 0.10},
            current_metrics={'val_loss': 0.40},
            baseline_metrics={'val_loss': 0.39}
        )

        assert report.triggered is False
        assert report.severity == 'info'

    def test_severity_priority(self):
        """Test that highest severity among triggered is used."""
        manager = RetrainingTriggerManager()

        # Register triggers with different severities
        manager.register_drift_trigger(threshold=0.15, severity='critical')
        manager.register_performance_trigger(
            threshold=0.50,  # Will not fire
            severity='warning'
        )
        manager.register_time_trigger(interval_hours=1, severity='info')

        last_training = datetime.now() - timedelta(hours=2)
        report = manager.evaluate(
            drift_metrics={'js_divergence': 0.18},
            current_metrics={'val_loss': 0.40},
            baseline_metrics={'val_loss': 0.39},
            metadata={'last_training_time': last_training.isoformat()}
        )

        # Drift (critical) and time (info) fire
        assert report.triggered is True
        assert report.severity == 'critical'  # Highest severity

    def test_trigger_history(self):
        """Test that trigger history is maintained."""
        manager = RetrainingTriggerManager()
        manager.register_drift_trigger(threshold=0.15)

        # First evaluation
        report1 = manager.evaluate(drift_metrics={'js_divergence': 0.18})

        # Second evaluation
        report2 = manager.evaluate(drift_metrics={'js_divergence': 0.12})

        history = manager.get_trigger_history()

        assert len(history) == 2
        assert history[0] == report1
        assert history[1] == report2

    def test_trigger_history_limit(self):
        """Test trigger history limit."""
        manager = RetrainingTriggerManager()
        manager.register_drift_trigger(threshold=0.15)

        # Generate multiple reports
        for i in range(15):
            manager.evaluate(drift_metrics={'js_divergence': 0.10 + i * 0.01})

        history = manager.get_trigger_history(limit=5)

        assert len(history) == 5
        assert history[-1] == manager.trigger_history[-1]


# -------------------------------------------------------------------------
# Test Report Serialization
# -------------------------------------------------------------------------

class TestRetrainingReport:
    """Tests for RetrainingReport serialization."""

    def test_to_dict(self):
        """Test converting report to dictionary."""
        detail = TriggerDetail(
            trigger_name='test_trigger',
            triggered=True,
            severity='warning',
            reason='Test reason',
            metrics={'metric1': 0.5},
            threshold=0.3,
            actual_value=0.5
        )

        report = RetrainingReport(
            triggered=True,
            trigger_details=[detail],
            recommendations=['Recommendation 1', 'Recommendation 2'],
            severity='warning',
            timestamp=datetime.now().isoformat(),
            metadata={'key': 'value'}
        )

        report_dict = report.to_dict()

        assert isinstance(report_dict, dict)
        assert report_dict['triggered'] is True
        assert report_dict['severity'] == 'warning'
        assert len(report_dict['trigger_details']) == 1
        assert len(report_dict['recommendations']) == 2
        assert 'metadata' in report_dict

    def test_to_json(self):
        """Test saving report to JSON file."""
        detail = TriggerDetail(
            trigger_name='test_trigger',
            triggered=True,
            severity='warning',
            reason='Test reason',
            metrics={},
            threshold=0.3
        )

        report = RetrainingReport(
            triggered=True,
            trigger_details=[detail],
            recommendations=['Recommendation'],
            severity='warning',
            timestamp=datetime.now().isoformat()
        )

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = Path(f.name)

        try:
            report.to_json(filepath)

            # Load and verify
            with open(filepath, 'r') as f:
                loaded = json.load(f)

            assert loaded['triggered'] is True
            assert loaded['severity'] == 'warning'
        finally:
            filepath.unlink()

    def test_to_markdown(self):
        """Test generating Markdown report."""
        detail1 = TriggerDetail(
            trigger_name='drift_trigger',
            triggered=True,
            severity='warning',
            reason='Drift detected',
            metrics={'js_divergence': 0.18},
            threshold=0.15,
            actual_value=0.18
        )

        detail2 = TriggerDetail(
            trigger_name='performance_trigger',
            triggered=False,
            severity='info',
            reason='Performance stable',
            metrics={'val_loss': 0.40},
            threshold=0.05
        )

        report = RetrainingReport(
            triggered=True,
            trigger_details=[detail1, detail2],
            recommendations=['Retrain the model', 'Review data sources'],
            severity='warning',
            timestamp=datetime.now().isoformat(),
            metadata={'model_id': 5}
        )

        markdown = report.to_markdown()

        assert '# Retraining Trigger Report' in markdown
        assert '🔴 TRIGGERED' in markdown
        assert '⚠️' in markdown
        assert 'drift_trigger' in markdown
        assert 'performance_trigger' in markdown
        assert 'Retrain the model' in markdown
        assert 'Review data sources' in markdown


# -------------------------------------------------------------------------
# Test Configuration
# -------------------------------------------------------------------------

class TestTriggerConfig:
    """Tests for TriggerConfig serialization."""

    def test_drift_config_to_dict(self):
        """Test converting drift config to dictionary."""
        config = DriftTriggerConfig(
            name='drift_test',
            threshold=0.15,
            metric_name='js_divergence',
            severity='warning'
        )

        config_dict = config.to_dict()

        assert config_dict['name'] == 'drift_test'
        assert config_dict['threshold'] == 0.15
        assert config_dict['metric_name'] == 'js_divergence'
        assert config_dict['severity'] == 'warning'

    def test_config_to_json(self):
        """Test saving config to JSON file."""
        config = PerformanceTriggerConfig(
            name='perf_test',
            threshold=0.05,
            metric_name='val_loss',
            mode='min'
        )

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = Path(f.name)

        try:
            config.to_json(filepath)

            # Load and verify
            loaded_config = PerformanceTriggerConfig.from_json(filepath)

            assert loaded_config.name == 'perf_test'
            assert loaded_config.threshold == 0.05
            assert loaded_config.metric_name == 'val_loss'
            assert loaded_config.mode == 'min'
        finally:
            filepath.unlink()

    def test_example_configs(self):
        """Test example configuration presets."""
        conservative = get_conservative_config()
        aggressive = get_aggressive_config()
        balanced = get_balanced_config()

        assert len(conservative) == 4
        assert len(aggressive) == 4
        assert len(balanced) == 4

        # Verify conservative thresholds are higher
        assert conservative['drift'].threshold > balanced['drift'].threshold
        assert aggressive['drift'].threshold < balanced['drift'].threshold


# -------------------------------------------------------------------------
# Integration Tests
# -------------------------------------------------------------------------

class TestIntegration:
    """Integration tests with MetricsEngine and ExperimentDB."""

    def test_integration_with_experiment_db(self):
        """Test integration with ExperimentDB for logging."""
        from utils.training.experiment_db import ExperimentDB

        with tempfile.NamedTemporaryFile(mode='w', suffix='.db', delete=False) as f:
            db_path = Path(f.name)

        try:
            db = ExperimentDB(db_path)
            manager = RetrainingTriggerManager(experiment_db=db)

            manager.register_drift_trigger(threshold=0.15)

            # Evaluate (triggers fire)
            report = manager.evaluate(drift_metrics={'js_divergence': 0.20})

            assert report.triggered is True

            # Verify logged to database
            runs = db.list_runs(limit=10)
            assert len(runs) > 0
            assert 'trigger_event' in runs.iloc[0]['run_name']

        finally:
            db_path.unlink()

    def test_end_to_end_workflow(self):
        """Test end-to-end workflow with all components."""
        from utils.training.experiment_db import ExperimentDB

        with tempfile.NamedTemporaryFile(mode='w', suffix='.db', delete=False) as f:
            db_path = Path(f.name)

        try:
            # Setup
            db = ExperimentDB(db_path)
            manager = RetrainingTriggerManager(experiment_db=db)

            # Register balanced triggers
            manager.register_drift_trigger(threshold=0.15)
            manager.register_performance_trigger(threshold=0.05, metric_name='val_loss')
            manager.register_time_trigger(interval_hours=168)

            # Create training run
            run_id = db.log_run(
                run_name='baseline_model',
                config={'learning_rate': 5e-5},
                notes='Initial baseline'
            )

            # Log metrics
            db.log_metric(run_id, 'val_loss', 0.40, epoch=0)
            db.log_metric(run_id, 'val_accuracy', 0.85, epoch=0)

            # Simulate time passing and performance degradation
            last_training = datetime.now() - timedelta(days=8)

            report = manager.evaluate(
                drift_metrics={'js_divergence': 0.18},
                current_metrics={'val_loss': 0.45, 'val_accuracy': 0.82},
                baseline_metrics={'val_loss': 0.40, 'val_accuracy': 0.85},
                metadata={'last_training_time': last_training.isoformat()}
            )

            # Verify results
            assert report.triggered is True
            assert len(report.trigger_details) == 3
            assert len(report.recommendations) > 0

            # Save report
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                report_path = Path(f.name)

            try:
                report.to_json(report_path)
                assert report_path.exists()
            finally:
                report_path.unlink()

        finally:
            db_path.unlink()


# -------------------------------------------------------------------------
# Property-Based Tests
# -------------------------------------------------------------------------

class TestTriggerProperties:
    """Property-based tests for trigger behavior."""

    def test_drift_trigger_monotonic(self):
        """Test drift trigger is monotonic (higher drift = more likely to fire)."""
        trigger = DriftTrigger(threshold=0.15)

        # Test increasing drift values
        drift_values = [0.10, 0.15, 0.20, 0.25, 0.30]
        results = []

        for drift in drift_values:
            result = trigger.evaluate(drift_metrics={'js_divergence': drift})
            results.append(result.triggered)

        # Once triggered, should stay triggered for higher values
        first_trigger_idx = next((i for i, r in enumerate(results) if r), None)
        if first_trigger_idx is not None:
            assert all(results[first_trigger_idx:])

    def test_performance_trigger_symmetric(self):
        """Test performance trigger symmetric for equal degradation."""
        trigger_loss = PerformanceTrigger(
            threshold=0.05,
            metric_name='val_loss',
            mode='min'
        )

        trigger_acc = PerformanceTrigger(
            threshold=0.05,
            metric_name='val_accuracy',
            mode='max'
        )

        # 10% increase in loss
        result_loss = trigger_loss.evaluate(
            current_metrics={'val_loss': 0.44},
            baseline_metrics={'val_loss': 0.40}
        )

        # 10% decrease in accuracy
        result_acc = trigger_acc.evaluate(
            current_metrics={'val_accuracy': 0.765},
            baseline_metrics={'val_accuracy': 0.85}
        )

        # Both should fire for 10% degradation with 5% threshold
        assert result_loss.triggered == result_acc.triggered

    def test_time_trigger_deterministic(self):
        """Test time trigger produces deterministic results."""
        trigger = TimeTrigger(interval_hours=168)

        last_training = datetime(2025, 1, 1, 0, 0, 0)
        metadata = {'last_training_time': last_training.isoformat()}

        # Multiple evaluations should produce same triggered result
        result1 = trigger.evaluate(metadata=metadata)
        result2 = trigger.evaluate(metadata=metadata)

        assert result1.triggered == result2.triggered
        # Actual values may differ slightly due to datetime.now() precision
        # but should be very close (within 1 second)
        assert abs(result1.actual_value - result2.actual_value) < 0.001


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
