class _DummyTracker:
    def __init__(self):
        self.logged = []
    def log_scalar(self, name, value, step=None):
        self.logged.append((name, value, step))


def test_log_gpu_metrics_cpu_only(monkeypatch):
    # Force CUDA unavailable
    import types
    import utils.tier3_training_utilities as t3

    class _CudaMock:
        @staticmethod
        def is_available():
            return False

    monkeypatch.setattr(t3, 'torch', types.SimpleNamespace(cuda=_CudaMock()))

    tracker = _DummyTracker()
    t3._log_gpu_metrics(tracker, step=0)

    # No entries expected on CPU
    assert tracker.logged == []

