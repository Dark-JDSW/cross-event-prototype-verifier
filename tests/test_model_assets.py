"""外部模型引导的清单和张量指纹测试。"""

import unittest

from cross_event_verifier.model_assets import (
    MODEL_MANIFEST,
    _assets,
    _read_manifest,
    tensor_state_fingerprint,
)


class ModelAssetTests(unittest.TestCase):
    def test_manifest_has_https_sources_for_every_model(self) -> None:
        assets = _assets(_read_manifest(MODEL_MANIFEST))
        self.assertEqual(
            {asset.name for asset in assets},
            {
                "yolo11l.pt",
                "rtmpose-s.onnx",
                "osnet_ain_x1_0_dg.pth",
                "gaitgraph2_grew_state.pt",
            },
        )
        self.assertTrue(all(asset.url.startswith("https://") for asset in assets))
        self.assertTrue(
            all(asset.source_bytes > 0 and len(asset.source_sha256) == 64 for asset in assets)
        )

    def test_converted_assets_have_stable_tensor_fingerprints(self) -> None:
        manifest = _read_manifest(MODEL_MANIFEST)["models"]
        for name in ("osnet_ain_x1_0_dg.pth", "gaitgraph2_grew_state.pt"):
            path = MODEL_MANIFEST.parent / name
            expected = manifest[name]["tensor_sha256"]
            self.assertEqual(tensor_state_fingerprint(path), expected)


if __name__ == "__main__":
    unittest.main()
