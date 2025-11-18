import pytest


def test_format_results_basic():
    try:
        import pandas as pd
    except Exception:
        pytest.skip("pandas not available")

    from utils.tier3_training_utilities import _format_results

    df = pd.DataFrame({
        'epoch': [0, 1, 2],
        'train/loss': [1.0, 0.8, 0.6],
        'val/loss': [1.2, 0.9, 0.7],
        'val/perplexity': [3.3, 2.5, 2.0],
    })

    loss_hist = [1.0, 0.8, 0.6]
    res = _format_results(
        loss_history=loss_hist,
        training_time=12.0,
        metrics_summary=df,
        n_epochs=3,
        batch_size=4,
        train_dataset_size=100,
    )

    assert res['final_loss'] == pytest.approx(0.6)
    assert res['initial_loss'] == pytest.approx(1.0)
    assert res['best_epoch'] == df['val/loss'].idxmin()

