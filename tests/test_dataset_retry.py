import types

def test_load_huggingface_retries(monkeypatch):
    from utils.training import dataset_utilities as du

    calls = {"n": 0}

    def fake_load_dataset(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise Exception("transient error")
        return "OK"  # sentinel

    monkeypatch.setattr(du, "load_dataset", fake_load_dataset)

    loader = du.DatasetLoader()
    out = loader.load_huggingface("dummy", "cfg", split=None)
    assert out == "OK"
    assert calls["n"] == 3

