import json
from pathlib import Path

import cross_event_verifier as api
from cross_event_verifier.vision import VisionAdapter, VisionTrack


def test_v1_exports_and_core_members_remain_available():
    manifest = json.loads(Path(__file__).with_name('public_api_v1.json').read_text())
    assert set(manifest['exports']) <= set(api.__all__)
    for name in manifest['exports']:
        assert getattr(api, name) is not None
    for name in manifest['core_members']:
        assert hasattr(api.CrossEventVerifier, name), name
    assert isinstance(api.CrossEventVerifier.formal_identities, property)
    assert callable(VisionAdapter.process) and callable(VisionAdapter.reset)
    assert VisionTrack.__name__ == 'VisionTrack'
