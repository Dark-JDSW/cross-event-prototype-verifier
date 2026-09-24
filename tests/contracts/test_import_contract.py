import subprocess
import sys
import textwrap


def test_base_import_and_verification_without_optional_runtimes():
    code = textwrap.dedent('''
        import importlib.abc, sys
        blocked = {'cv2', 'torch', 'torchvision', 'ultralytics', 'onnxruntime',
                   'tensorrt', 'paho', 'bleak', 'homeassistant', 'hypergait',
                   'PyQt6', 'mmcv', 'mmaction', 'mmengine', 'faiss'}
        class Deny(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in blocked:
                    raise ModuleNotFoundError(fullname)
        sys.meta_path.insert(0, Deny())
        import cross_event_verifier as c
        from cross_event_verifier.vision import VisionTrack, VisionAdapter
        from cross_event_verifier.contexts import ContextComposer
        from cross_event_verifier.gui_presenter import GuiPresenter
        from cross_event_verifier.orchestration import TrackStateRegistry
        from cross_event_verifier.spatial import VisionMetadataSpatialProvider
        from cross_event_verifier.interaction import ContextRule
        v = c.CrossEventVerifier()
        assert v.verify(c.Observation()).identity_id is None
        assert v.verify_batch([]) == []
        v.close()
        assert not blocked.intersection(sys.modules)
    ''')
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
