from dataclasses import MISSING, fields, replace
import importlib
import json
from pathlib import Path

import pytest
from cross_event_verifier import ActionEvent, ActionType, EmbeddingContract
from cross_event_verifier.smart_home import event_payload


def test_old_fields_enum_values_and_safe_additions():
    manifest = json.loads(Path(__file__).with_name('public_api_v1.json').read_text())
    for path, old_fields in manifest['fields'].items():
        module, name = path.split('.')
        cls = getattr(importlib.import_module('cross_event_verifier.' + module), name)
        current = {f.name: f for f in fields(cls)}
        assert set(old_fields) <= current.keys(), path
        # Positional legacy construction must retain its relative field order.
        assert [f.name for f in fields(cls) if f.name in old_fields] == old_fields
        assert [f.name for f in fields(cls) if not f.kw_only][:len(old_fields)] == old_fields
        for name in current.keys() - set(old_fields):
            f = current[name]
            assert f.default is not MISSING or f.default_factory is not MISSING
    for path, values in manifest['enum_values'].items():
        module, name = path.split('.')
        cls = getattr(importlib.import_module('cross_event_verifier.' + module), name)
        for key, value in values.items():
            assert cls[key].value == value


def test_action_wire_v1_legacy_meanings():
    event = ActionEvent('P1', 7, ActionType.WAVE, .9, 10., 'bedroom')
    payload = json.loads(event_payload(event))
    expected = dict(identity_id='P1', track_id=7, action='WAVE', confidence=.9,
                    timestamp=10., camera_id='bedroom', protocol='cevp.action.v1',
                    event_type='wave', event_id='bedroom:track-7:wave:10000',
                    area='bedroom', completed=True)
    assert expected.items() <= payload.items()


@pytest.mark.parametrize('key,value', [
    ('model_version', 'HyperGait'), ('feature_schema', 'hyper-v1'),
    ('artifact_sha256', 'other'), ('preprocess_version', 'v2'),
    ('joint_format', 'other'), ('sequence_length', 90), ('tta_mode', 'flip'),
    ('coordinate_contract', 'world'), ('dimensions', {'gait': 4}),
])
def test_embedding_compatibility_is_stricter_than_api_compatibility(key, value):
    contract = EmbeddingContract(model_version='GaitGraph2', dimensions={'gait': 2})
    assert not contract.compatible_with(replace(contract, **{key: value}))
    assert EmbeddingContract.from_dict(contract.to_dict()) == contract
