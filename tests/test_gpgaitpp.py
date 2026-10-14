import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from cross_event_verifier.gpgaitpp import GPGaitPPEncoder, prepare_pose
from cross_event_verifier.production_vision import ProductionVisionAdapter, ProductionVisionConfig


def _pose(length=60):
    values = np.zeros((length, 17, 3), np.float32)
    values[:, :, 0] = np.arange(17) * 4 + 100
    values[:, :, 1] = np.arange(17) * 12 + 50
    values[:, :, 2] = .9
    return values


def test_preprocessing_matches_frozen_official_values_and_owns_input():
    fixture = json.loads((Path(__file__).parent / 'fixtures/gpgaitpp_preprocess.json').read_text())
    pose = np.array(fixture['input'], np.float32)
    original = pose.copy()
    actual = prepare_pose(pose)
    np.testing.assert_allclose(actual, fixture['expected'], rtol=1e-6, atol=1e-5)
    np.testing.assert_array_equal(pose, original)
    np.testing.assert_array_equal(actual, prepare_pose(pose))
    assert actual.shape == (6, 3, 17, 1)
    assert actual.dtype == np.float32
    pose[:, :, 2] = .1
    np.testing.assert_array_equal(prepare_pose(pose), actual)


@pytest.mark.parametrize('bad', [np.zeros((60, 17, 3)), np.full((60, 17, 3), np.nan), np.zeros((60, 18, 3))])
def test_invalid_pose_is_rejected_before_official_scaling(bad):
    with pytest.raises(ValueError):
        prepare_pose(bad)


def test_bounded_batches_keep_invalid_slots_and_normalize_output():
    class Network:
        def __init__(self):
            self.sizes = []

        def __call__(self, inputs):
            batch = inputs[0][0]
            self.sizes.append(len(batch))
            return {'inference_feat': {'embeddings': torch.ones((len(batch), 256, 19))}}

    encoder = GPGaitPPEncoder.__new__(GPGaitPPEncoder)
    encoder.sequence_length = 60
    encoder.device = torch.device('cpu')
    encoder.model = Network()
    values = encoder.encode_batch([_pose()] * 5 + [_pose(30), np.zeros((60, 17, 3))])
    assert encoder.model.sizes == [4, 1]
    assert values[5:] == [None, None]
    for value in values[:5]:
        assert value.shape == (4864,) and value.dtype == np.float32
        assert np.isclose(np.linalg.norm(value), 1)


def test_real_checkpoint_loads_strictly_and_can_encode_without_training():
    checkpoint = Path(__file__).resolve().parents[1] / 'models/gpgaitpp_ccpg_state.pt'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    encoder = GPGaitPPEncoder(checkpoint, device=device)
    before = {name: value.clone() for name, value in encoder.model.named_buffers() if 'running_' in name}
    value = encoder.encode(_pose())
    assert value is not None and value.shape == (4864,)
    assert np.isfinite(value).all() and np.isclose(np.linalg.norm(value), 1)
    assert not encoder.model.training
    for name, expected in before.items():
        torch.testing.assert_close(dict(encoder.model.named_buffers())[name], expected, rtol=0, atol=0)


def test_production_contract_and_default_asset_are_new_model_specific():
    config = ProductionVisionConfig()
    assert config.gait_path.name == 'gpgaitpp_ccpg_state.pt'
    assert 'gpgaitpp' in ProductionVisionAdapter.model_version
    assert ProductionVisionAdapter.embedding_dimensions['gait'] == 4864
    assert ProductionVisionAdapter.tta_mode == 'none'
    assert 'affine' in ProductionVisionAdapter.feature_schema


def test_production_rejects_old_encoder_output_dimension():
    from cross_event_verifier.production_vision import ProductionVisionError
    class OldEncoder:
        output_dimension = 384
        def __init__(self, *args, **kwargs):
            pass
    with patch('cross_event_verifier.production_vision.production_readiness', return_value=(True, ())):
        adapter = ProductionVisionAdapter()
    with patch('cross_event_verifier.production_vision._YoloByteTracker'), patch('cross_event_verifier.production_vision._RtmposeEstimator'), patch('cross_event_verifier.production_vision._OsnetAppearanceExtractor', return_value=type('Appearance', (), {'output_dimension':512})()), patch('cross_event_verifier.production_vision.GPGaitPPEncoder', OldEncoder):
        with pytest.raises(ProductionVisionError, match='4864-dimensional'):
            adapter.preload()
    assert adapter.gait is None
