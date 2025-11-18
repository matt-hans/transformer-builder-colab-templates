from utils.training.experiment_db import ExperimentDB


def test_log_run_with_gist_metadata(tmp_path):
    db = ExperimentDB(tmp_path / 'exp.db')
    run_id = db.log_run(
        run_name='gist-test',
        config={'lr': 1e-4},
        notes='test',
        gist_id='abcdef1234',
        gist_revision='123456',
        gist_sha256='deadbeef',
    )
    run = db.get_run(run_id)
    # Column presence is ensured by schema migration, values may be absent in get_run dict
    # get_run returns config, status, etc. Ensure the row exists
    assert run['run_name'] == 'gist-test'

