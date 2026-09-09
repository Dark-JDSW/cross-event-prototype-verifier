"""Regression tests for the optional PyQt presentation adapter."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import unittest


class QtGuiAdapterTests(unittest.TestCase):
    def test_qt_adapter_reuses_pipeline_and_keeps_parameter_controls_visual_only(self) -> None:
        """Qt controls must drive the existing parameter transaction interface."""

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        script = r"""
from PyQt6.QtWidgets import QApplication
from cross_event_verifier.qt_gui import VerifierWindow

app = QApplication([])
window = VerifierWindow(":memory:", "demo")
assert window.pipeline is not None
assert window.worker is not None
assert window.preload_progress.isHidden()
assert window.preload_phase.isHidden()
assert window._preload_controls_layout.indexOf(window.preload_progress) == -1
assert window._preload_controls_layout.indexOf(window.preload_phase) == -1
assert window.notebook.count() == 3
assert window.notebook.tabText(2) == "03 / 地面控制点"
for point_id, image, world in (
    ("A", (0, 0), (0, 0)),
    ("B", (100, 0), (2, 0)),
    ("C", (100, 100), (2, 3)),
    ("D", (0, 100), (0, 3)),
):
    window.gcp_point_id.setText(point_id)
    window.gcp_image_u.setText(str(image[0]))
    window.gcp_image_v.setText(str(image[1]))
    window.gcp_world_x.setText(str(world[0]))
    window.gcp_world_y.setText(str(world[1]))
    window._gcp_upsert()
assert window._gcp_check_homography() is not None
assert window.gcp_table.rowCount() == 4
assert window.track_tree.columnCount() == 6
assert tuple(
    window.manual_action_combo.itemText(index)
    for index in range(window.manual_action_combo.count())
) == ("SQUAT", "SIT_DOWN", "STAND_UP", "HAND_UP", "WAVE")
assert tuple(
    window.transport_combo.itemText(index)
    for index in range(window.transport_combo.count())
) == ("MQTT", "BLE Light")
assert window.video_stack.currentWidget() is window.video_standby
key = "verifier.gait_novelty_threshold"
assert key in window.parameter_scales
assert key in window.parameter_entries
original = window.parameter_vars[key].text()
window.parameter_scales[key].setValue(500)
assert window.parameter_vars[key].text() != original
window.parameter_vars[key].setText("0.25")
app.processEvents()
spec = window.parameter_specs[key]
expected = window._scale_value(spec, 0.25)
assert abs(window.parameter_scales[key].value() - expected) <= 1
window.close()
app.processEvents()
app.quit()
"""
        environment = os.environ.copy()
        environment["QT_QPA_PLATFORM"] = "offscreen"
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if "No module named 'PyQt6'" in result.stderr:
            self.skipTest("PyQt6 is not installed")
        self.assertEqual(
            result.returncode,
            0,
            msg=f"Qt smoke test failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
        )

    def test_source_selection_indicator_and_sidebar_fit_at_minimum_window(self) -> None:
        """Selection chrome and the right control rail must survive a narrow window."""

        script = r"""
from PyQt6.QtWidgets import QApplication
from cross_event_verifier.qt_gui import VerifierWindow

app = QApplication([])
window = VerifierWindow(":memory:", "demo")
window.resize(980, 640)
window.show()
app.processEvents()
assert window.camera_radio.isChecked()
assert "QRadioButton::indicator:checked" in window.styleSheet()
content = window.side_canvas.widget()
assert content is not None
assert window.side_canvas.horizontalScrollBar().maximum() == 0
assert content.width() <= window.side_canvas.viewport().width()
window.close()
app.processEvents()
app.quit()
"""
        environment = os.environ.copy()
        environment["QT_QPA_PLATFORM"] = "offscreen"
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if "No module named 'PyQt6'" in result.stderr:
            self.skipTest("PyQt6 is not installed")
        self.assertEqual(
            result.returncode,
            0,
            msg=f"Qt layout regression failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
        )


if __name__ == "__main__":
    unittest.main()
