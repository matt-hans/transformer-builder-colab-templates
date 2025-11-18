from utils.adapters.gist_loader import _parse_gist_id, load_gist_model


def test_parse_gist_id_from_url():
    url = 'https://gist.github.com/user/abcdef1234567890'
    gid = _parse_gist_id(url)
    assert gid == 'abcdef1234567890'


def test_load_gist_model_no_network(tmp_path):
    gid = 'abcdef1234567890'
    md = load_gist_model(gid, revision=None, download_dir=str(tmp_path))
    assert md.gist_id == gid
    assert md.revision is None
    # Without network, sha256 may be None
    assert md.sha256 is None or isinstance(md.sha256, str)

