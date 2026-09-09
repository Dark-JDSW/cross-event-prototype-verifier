"""Local GUI for creating action-specific gait branch annotations.

The tool deliberately records only the fields needed to train an action gate and
separate gait branches: source video, source-local track, action, and frame span.
"""

from __future__ import annotations

import base64
import json
import os
import tempfile
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any


ACTION_LABELS = (
    "walking",
    "running",
    "carrying",
    "pushing_pulling",
    "standing",
    "sitting",
    "other",
    "unknown",
)
WALK_VIEW_LABELS = ("side", "oblique", "front", "back", "transition", "unknown")
WALK_DIRECTION_LABELS = (
    "left_to_right",
    "right_to_left",
    "approaching",
    "leaving",
    "transition",
    "unknown",
)
CONFIDENCE_VALUES = ("high", "medium", "low")
TRACK_DISPLAY_MODES = ("仅显示选中 Track", "显示所有候选追踪框")
VIDEO_SUFFIXES = {".avi", ".mp4", ".mov", ".mkv", ".mpeg", ".mpg"}
DEFAULT_OUTPUT = Path("E:/datasets/mevid-cross-event/action_branch_annotations.jsonl")


def branch_id_for_action(
    action_label: str,
    walking_view: str = "unknown",
    walking_direction: str = "unknown",
) -> str:
    """Return an isolated gait branch ID for an action, view, and direction."""
    action = action_label.strip().lower()
    if action == "unknown":
        return "unknown"
    if action not in ACTION_LABELS:
        raise ValueError(f"Unsupported action label: {action_label!r}")
    if action == "walking":
        view = walking_view.strip().lower()
        if view not in WALK_VIEW_LABELS:
            raise ValueError(f"Unsupported walking view: {walking_view!r}")
        direction = walking_direction.strip().lower()
        if direction not in WALK_DIRECTION_LABELS:
            raise ValueError(f"Unsupported walking direction: {walking_direction!r}")
        if {view, direction}.intersection({"transition", "unknown"}):
            return "unknown"
        return f"gait_walking_{view}_{direction}"
    return f"gait_{action}"


def annotation_key(
    source_video_id: str, track_id: int, gait_branch_id: str, start_frame: int, end_frame: int
) -> str:
    return f"{source_video_id}|track:{track_id}|branch:{gait_branch_id}|frames:{start_frame}-{end_frame}"


def tracks_to_display(
    candidate_tracks: list[dict[str, Any]], selected_track: str, show_all: bool
) -> list[dict[str, Any]]:
    if show_all:
        return candidate_tracks
    return [track for track in candidate_tracks if str(track.get("track_id")) == selected_track]


def track_observation_range(track: dict[str, Any]) -> tuple[int, int] | None:
    """Return the source-frame interval in which a local track was observed."""
    try:
        first = int(track["first_frame_index"])
        last = int(track["last_frame_index"])
    except (KeyError, TypeError, ValueError):
        frame_indices = [
            int(row["frame_index"])
            for row in track.get("preview", [])
            if "frame_index" in row
        ]
        if not frame_indices:
            return None
        first, last = min(frame_indices), max(frame_indices)
    return (first, last) if first <= last else None


def interpolated_box(
    preview: list[dict[str, Any]],
    frame_index: int,
    observation_range: tuple[int, int] | None = None,
) -> list[float] | None:
    points = [
        (int(row["frame_index"]), [float(value) for value in row["bbox_xyxy"]])
        for row in preview
        if isinstance(row.get("bbox_xyxy"), list) and len(row["bbox_xyxy"]) == 4
    ]
    if not points:
        return None
    points.sort(key=lambda point: point[0])
    first, last = observation_range or (points[0][0], points[-1][0])
    if not first <= frame_index <= last:
        return None
    if frame_index <= points[0][0]:
        return points[0][1]
    if frame_index >= points[-1][0]:
        return points[-1][1]
    for left, right in zip(points, points[1:]):
        if left[0] <= frame_index <= right[0]:
            if right[0] == left[0]:
                return left[1]
            ratio = (frame_index - left[0]) / (right[0] - left[0])
            return [start + ratio * (end - start) for start, end in zip(left[1], right[1])]
    return points[-1][1]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} 第 {line_number} 行不是有效 JSON。") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path} 第 {line_number} 行必须是 JSON 对象。")
        rows.append(row)
    return rows


def load_queue(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows = read_jsonl(path)
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("队列 JSON 必须是对象数组，或改用 JSONL 文件。")
        rows = [row for row in payload if isinstance(row, dict)]
    return [row for row in rows if row.get("source_video_path")]


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
    ) as handle:
        temp_path = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temp_path, path)


def preferred_saved_annotation(
    annotations: list[dict[str, Any]], source_video_id: str, source_video_path: str, track_id: str
) -> dict[str, Any] | None:
    """Return the most current saved annotation for one source-local track."""

    source_stem = Path(source_video_path).stem
    target_track = str(track_id).strip()
    candidates = [
        annotation
        for annotation in annotations
        if str(annotation.get("track_id", "")).strip() == target_track
        and (
            str(annotation.get("source_video_id", "")) == source_video_id
            or Path(str(annotation.get("source_video_path", ""))).stem == source_stem
        )
    ]
    if not candidates:
        return None

    def rank(annotation: dict[str, Any]) -> tuple[int, int, int, int]:
        version = {"mevid-action-branch-v1": 1, "mevid-action-branch-v2": 2, "mevid-action-branch-v3": 3}.get(
            str(annotation.get("schema_version", "")),
            0,
        )
        try:
            start = int(annotation.get("start_frame", -1))
            end = int(annotation.get("end_frame", -1))
        except (TypeError, ValueError):
            start, end = -1, -1
        return (version, int(annotation.get("annotation_status") == "human_confirmed"), end, start)

    return max(candidates, key=rank)


class ActionBranchGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("MEVID 动作分支标注工具")
        self.root.geometry("1420x860")

        self.queue_rows: list[dict[str, Any]] = []
        self.current_queue_row: dict[str, Any] | None = None
        self.current_path: Path | None = None
        self.capture: Any = None
        self.frame_count = 0
        self.fps = 0.0
        self.frame_index = 0
        self.setting_frame_scale = False
        self.photo: tk.PhotoImage | None = None
        self.playing = False
        self.annotations: dict[str, dict[str, Any]] = {}

        self.output_path = tk.StringVar(value=str(DEFAULT_OUTPUT))
        self.status = tk.StringVar(value="先载入快速筛选结果或选择视频目录。")
        self.source_video_id = tk.StringVar()
        self.source_video_path = tk.StringVar()
        self.track_id = tk.StringVar()
        self.action_label = tk.StringVar(value="unknown")
        self.walking_view = tk.StringVar(value="unknown")
        self.walking_direction = tk.StringVar(value="unknown")
        self.gait_branch_id = tk.StringVar(value="unknown")
        self.start_frame = tk.StringVar(value="0")
        self.end_frame = tk.StringVar(value="0")
        self.confidence = tk.StringVar(value="medium")
        self.note = tk.StringVar()
        self.confirmed = tk.BooleanVar(value=False)
        self.track_display_mode = tk.StringVar(value=TRACK_DISPLAY_MODES[0])

        self._build()
        self.action_label.trace_add("write", self._update_branch)
        self.walking_view.trace_add("write", self._update_branch)
        self.walking_direction.trace_add("write", self._update_branch)
        self._update_branch()
        self.track_id.trace_add("write", self._refresh_track_overlay)
        self.root.bind("<Control-s>", self._save_event)
        self._load_existing_annotations()

    def _build(self) -> None:
        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(fill=tk.X)
        ttk.Button(toolbar, text="载入快速筛选结果", command=self._choose_queue).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="选择视频目录", command=self._choose_video_directory).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(toolbar, text="选择输出 JSONL", command=self._choose_output).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Label(toolbar, textvariable=self.output_path).pack(side=tk.LEFT, padx=10)

        body = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        queue_panel = ttk.Frame(body, padding=4)
        body.add(queue_panel, weight=2)
        ttk.Label(queue_panel, text="视频 / 快速筛选记录").pack(anchor=tk.W)
        self.queue_list = tk.Listbox(queue_panel, exportselection=False, width=38)
        self.queue_list.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        self.queue_list.bind("<<ListboxSelect>>", self._on_queue_select)
        scrollbar = ttk.Scrollbar(queue_panel, orient=tk.VERTICAL, command=self.queue_list.yview)
        scrollbar.pack(fill=tk.Y, side=tk.RIGHT)
        self.queue_list.configure(yscrollcommand=scrollbar.set)

        preview_panel = ttk.Frame(body, padding=4)
        body.add(preview_panel, weight=5)
        self.preview = ttk.Label(preview_panel, text="选择一个视频后会显示画面和可用轨迹框。", anchor=tk.CENTER)
        self.preview.pack(fill=tk.BOTH, expand=True)
        controls = ttk.Frame(preview_panel, padding=(0, 8, 0, 0))
        controls.pack(fill=tk.X)
        ttk.Button(controls, text="上一帧", command=lambda: self._seek_relative(-1)).pack(side=tk.LEFT)
        self.play_button = ttk.Button(controls, text="播放", command=self._toggle_play)
        self.play_button.pack(side=tk.LEFT, padx=5)
        ttk.Button(controls, text="下一帧", command=lambda: self._seek_relative(1)).pack(side=tk.LEFT)
        ttk.Button(controls, text="设为开始帧", command=self._set_start).pack(side=tk.LEFT, padx=(16, 0))
        ttk.Button(controls, text="设为结束帧", command=self._set_end).pack(side=tk.LEFT, padx=5)
        ttk.Label(controls, text="追踪框显示").pack(side=tk.LEFT, padx=(12, 4))
        self.track_display = ttk.Combobox(
            controls,
            textvariable=self.track_display_mode,
            values=TRACK_DISPLAY_MODES,
            state="readonly",
            width=20,
        )
        self.track_display.pack(side=tk.LEFT)
        self.track_display.bind("<<ComboboxSelected>>", self._on_track_display_mode_changed)
        self.frame_scale = ttk.Scale(controls, from_=0, to=0, command=self._seek_scale)
        self.frame_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=12)
        self.frame_label = ttk.Label(controls, text="帧 0 / 0")
        self.frame_label.pack(side=tk.RIGHT)

        form = ttk.LabelFrame(body, text="动作分支标注", padding=10)
        body.add(form, weight=3)
        self._entry(form, 0, "视频 ID", self.source_video_id, readonly=True)
        self._entry(form, 1, "视频路径", self.source_video_path, readonly=True)
        self._entry(form, 2, "轨迹 ID（本视频内）", self.track_id)
        self._combo(form, 3, "动作", self.action_label, ACTION_LABELS)
        self.walking_view_combo = self._combo(
            form, 4, "行走方向 / 视角", self.walking_view, WALK_VIEW_LABELS
        )
        self.walking_direction_combo = self._combo(
            form, 5, "行进方向", self.walking_direction, WALK_DIRECTION_LABELS
        )
        self._entry(form, 6, "步态分支", self.gait_branch_id, readonly=True)
        self._entry(form, 7, "开始帧", self.start_frame)
        self._entry(form, 8, "结束帧", self.end_frame)
        self._combo(form, 9, "确认置信度", self.confidence, CONFIDENCE_VALUES)
        self._entry(form, 10, "备注（可选）", self.note)
        ttk.Checkbutton(
            form,
            text="我已确认该轨迹在这段帧区间内属于所选动作",
            variable=self.confirmed,
        ).grid(row=11, column=0, columnspan=2, sticky=tk.W, pady=(12, 4))
        ttk.Label(
            form,
            text=(
                "walking 必须同时选择身体视角和行进方向，二者共同决定独立分支。\n"
                "任一项为 transition 或 unknown 时只保存为隔离记录，不会写入正式步态分支。"
            ),
            foreground="#555555",
            justify=tk.LEFT,
        ).grid(row=12, column=0, columnspan=2, sticky=tk.W, pady=(0, 10))
        ttk.Button(form, text="保存当前片段  Ctrl+S", command=self._save_current).grid(
            row=13, column=0, columnspan=2, sticky=tk.EW, pady=3
        )
        ttk.Button(form, text="保存并下一个", command=self._save_and_next).grid(
            row=14, column=0, columnspan=2, sticky=tk.EW, pady=3
        )
        ttk.Button(form, text="跳过当前 / 下一个", command=self._next_item).grid(
            row=15, column=0, columnspan=2, sticky=tk.EW, pady=3
        )
        form.columnconfigure(1, weight=1)

        ttk.Label(self.root, textvariable=self.status, anchor=tk.W, padding=(10, 5)).pack(fill=tk.X)

    @staticmethod
    def _entry(parent: ttk.Widget, row: int, label: str, variable: tk.StringVar, readonly: bool = False) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, padx=(0, 8), pady=4)
        state = "readonly" if readonly else "normal"
        ttk.Entry(parent, textvariable=variable, state=state).grid(row=row, column=1, sticky=tk.EW, pady=4)

    @staticmethod
    def _combo(
        parent: ttk.Widget, row: int, label: str, variable: tk.StringVar, values: tuple[str, ...]
    ) -> ttk.Combobox:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, padx=(0, 8), pady=4)
        combo = ttk.Combobox(parent, textvariable=variable, values=values, state="readonly")
        combo.grid(
            row=row, column=1, sticky=tk.EW, pady=4
        )
        return combo

    def _choose_queue(self) -> None:
        path = filedialog.askopenfilename(
            title="选择 fast_screen.jsonl 或 fast_review_queue.jsonl",
            initialdir="E:/datasets/mevid-cross-event/fast_screen",
            filetypes=[("JSONL / JSON", "*.jsonl *.json"), ("所有文件", "*.*")],
        )
        if path:
            self._set_queue(load_queue(Path(path)))

    def _choose_video_directory(self) -> None:
        directory = filedialog.askdirectory(title="选择包含视频的目录", initialdir="E:/datasets/mevid-cross-event/raw/videos")
        if not directory:
            return
        rows = [
            {"source_video_id": path.stem, "source_video_path": str(path)}
            for path in sorted(Path(directory).iterdir())
            if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
        ]
        self._set_queue(rows)

    def _set_queue(self, rows: list[dict[str, Any]]) -> None:
        self.queue_rows = rows
        self.queue_list.delete(0, tk.END)
        for row in rows:
            source_id = str(row.get("source_video_id") or Path(str(row["source_video_path"])).stem)
            track = row.get("recommended_track_id")
            suffix = f"  track {track}" if track is not None else ""
            self.queue_list.insert(tk.END, f"{source_id}{suffix}")
        self.status.set(f"已载入 {len(rows)} 条记录。选择一条开始标注。")
        if rows:
            self.queue_list.selection_set(0)
            self._on_queue_select()

    def _choose_output(self) -> None:
        path = filedialog.asksaveasfilename(
            title="选择动作分支标注输出位置",
            initialdir=str(DEFAULT_OUTPUT.parent),
            initialfile=DEFAULT_OUTPUT.name,
            defaultextension=".jsonl",
            filetypes=[("JSONL", "*.jsonl")],
        )
        if path:
            self.output_path.set(path)
            self._load_existing_annotations()

    def _load_existing_annotations(self) -> None:
        try:
            rows = read_jsonl(Path(self.output_path.get()))
        except ValueError as exc:
            messagebox.showerror("无法读取已有标注", str(exc))
            return
        self.annotations = {str(row.get("annotation_key")): row for row in rows if row.get("annotation_key")}
        if rows:
            self.status.set(f"已载入 {len(self.annotations)} 条已有动作标注；相同片段保存时会更新。")

    def _on_queue_select(self, _event: object | None = None) -> None:
        selection = self.queue_list.curselection()
        if not selection:
            return
        self._open_row(self.queue_rows[selection[0]])

    def _open_row(self, row: dict[str, Any]) -> None:
        path = Path(str(row["source_video_path"]))
        if not path.exists():
            messagebox.showerror("视频不存在", f"找不到视频：\n{path}")
            return
        try:
            import cv2
        except ImportError:
            messagebox.showerror("缺少 OpenCV", "当前 Python 环境没有 cv2，无法播放视频。")
            return
        if self.capture is not None:
            self.capture.release()
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            messagebox.showerror("无法打开视频", str(path))
            return
        self.capture = capture
        self.current_queue_row = row
        self.current_path = path
        self.frame_count = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        self.fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        self.source_video_id.set(str(row.get("source_video_id") or path.stem))
        self.source_video_path.set(str(path))
        recommended_track = row.get("recommended_track_id")
        self.track_id.set("" if recommended_track is None else str(recommended_track))
        self.track_display_mode.set(TRACK_DISPLAY_MODES[0])
        selected = next(
            (
                track
                for track in row.get("candidate_tracks", [])
                if str(track.get("track_id")) == self.track_id.get()
            ),
            None,
        )
        observed = track_observation_range(selected) if selected is not None else None
        self.frame_index = observed[0] if observed is not None else 0
        self.action_label.set("unknown")
        self.walking_view.set("unknown")
        self.walking_direction.set("unknown")
        self.start_frame.set("0")
        self.end_frame.set(str(self.frame_count - 1))
        self.confidence.set("medium")
        self.note.set("")
        self.confirmed.set(False)
        self.frame_scale.configure(to=self.frame_count - 1)
        restored = self._restore_saved_annotation()
        if restored is not None:
            self.frame_index = max(0, min(self.frame_count - 1, int(restored["start_frame"])))
        self._show_frame()
        if restored is None:
            self.status.set(f"已打开 {path.name}；优先选择画面中对应的轨迹框并标注动作片段。")
        else:
            self.status.set(
                f"已恢复 Track {restored['track_id']} 的已保存标注："
                f"{restored.get('gait_branch_id', 'unknown')}，"
                f"帧 {restored['start_frame']}-{restored['end_frame']}。"
            )

    def _restore_saved_annotation(self) -> dict[str, Any] | None:
        """Fill the form from the saved annotation for the current source-local track."""

        saved = preferred_saved_annotation(
            list(self.annotations.values()),
            self.source_video_id.get(),
            self.source_video_path.get(),
            self.track_id.get(),
        )
        if saved is None:
            return None
        self.track_id.set(str(saved["track_id"]))
        self.action_label.set(str(saved.get("action_label", "unknown")).lower())
        self.walking_view.set(str(saved.get("walking_view", "unknown")).lower())
        self.walking_direction.set(str(saved.get("walking_direction", "unknown")).lower())
        self.start_frame.set(str(saved["start_frame"]))
        self.end_frame.set(str(saved["end_frame"]))
        self.confidence.set(str(saved.get("action_confidence", "medium")).lower())
        self.note.set(str(saved.get("annotation_note", "")))
        self.confirmed.set(saved.get("annotation_status") == "human_confirmed")
        return saved

    def _update_branch(self, *_args: object) -> None:
        action = self.action_label.get().strip().lower()
        if action != "walking" and self.walking_view.get() != "unknown":
            self.walking_view.set("unknown")
            return
        if action != "walking" and self.walking_direction.get() != "unknown":
            self.walking_direction.set("unknown")
            return
        self.walking_view_combo.configure(state="readonly" if action == "walking" else "disabled")
        self.walking_direction_combo.configure(state="readonly" if action == "walking" else "disabled")
        try:
            self.gait_branch_id.set(
                branch_id_for_action(action, self.walking_view.get(), self.walking_direction.get())
            )
        except ValueError:
            self.gait_branch_id.set("unknown")

    def _refresh_track_overlay(self, *_args: object) -> None:
        if self.capture is not None:
            self._show_frame()

    def _on_track_display_mode_changed(self, _event: object | None = None) -> None:
        if self.track_display_mode.get() == TRACK_DISPLAY_MODES[1]:
            self.status.set("已选择显示所有候选追踪框；当前未出现或已结束的 Track 不绘制。")
        else:
            self.status.set("已选择仅显示当前 Track。")
        self._show_frame()

    def _seek_relative(self, delta: int) -> None:
        self._seek(self.frame_index + delta)

    def _seek_scale(self, value: str) -> None:
        if self.capture is not None and not self.setting_frame_scale:
            self._seek(int(float(value)))

    def _seek(self, frame_index: int) -> None:
        if self.capture is None:
            return
        self.frame_index = max(0, min(self.frame_count - 1, frame_index))
        self._show_frame()

    def _show_frame(self) -> None:
        if self.capture is None:
            return
        import cv2

        self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.frame_index)
        ok, frame = self.capture.read()
        if not ok:
            self.status.set("无法读取当前帧。")
            return
        self._draw_tracks(frame)
        height, width = frame.shape[:2]
        max_width, max_height = 860, 600
        scale = min(max_width / width, max_height / height, 1.0)
        if scale < 1.0:
            frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ok, encoded = cv2.imencode(".png", rgb)
        if not ok:
            self.status.set("无法生成当前帧预览。")
            return
        self.photo = tk.PhotoImage(data=base64.b64encode(encoded.tobytes()).decode("ascii"))
        self.preview.configure(image=self.photo, text="")
        self.setting_frame_scale = True
        self.frame_scale.set(self.frame_index)
        self.setting_frame_scale = False
        range_text = ""
        if self.current_queue_row is not None:
            selected_track = self.track_id.get().strip()
            selected = next(
                (
                    track
                    for track in self.current_queue_row.get("candidate_tracks", [])
                    if str(track.get("track_id")) == selected_track
                ),
                None,
            )
            observed = track_observation_range(selected) if selected is not None else None
            if observed is not None:
                range_text = f" | 轨迹 {selected_track}: {observed[0]}-{observed[1]}"
        self.frame_label.configure(text=f"帧 {self.frame_index} / {self.frame_count - 1}{range_text}")

    def _draw_tracks(self, frame: Any) -> None:
        if not self.current_queue_row:
            return
        import cv2

        selected_track = self.track_id.get().strip()
        tracks = tracks_to_display(
            self.current_queue_row.get("candidate_tracks", []),
            selected_track,
            self.track_display_mode.get() == TRACK_DISPLAY_MODES[1],
        )
        for track in tracks:
            preview = track.get("preview", [])
            if not preview:
                continue
            box = interpolated_box(preview, self.frame_index, track_observation_range(track))
            if box is None:
                continue
            track_value = str(track.get("track_id", "?"))
            color = (0, 220, 0) if track_value == selected_track else (0, 180, 255)
            x1, y1, x2, y2 = (int(value) for value in box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"track {track_value}", (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    def _toggle_play(self) -> None:
        if self.capture is None:
            return
        self.playing = not self.playing
        self.play_button.configure(text="暂停" if self.playing else "播放")
        if self.playing:
            self._play_step()

    def _play_step(self) -> None:
        if not self.playing or self.capture is None:
            return
        if self.frame_index >= self.frame_count - 1:
            self.playing = False
            self.play_button.configure(text="播放")
            return
        self._seek_relative(1)
        delay = int(1000 / max(self.fps, 8.0))
        self.root.after(delay, self._play_step)

    def _set_start(self) -> None:
        self.start_frame.set(str(self.frame_index))

    def _set_end(self) -> None:
        self.end_frame.set(str(self.frame_index))

    def _build_annotation(self) -> dict[str, Any] | None:
        if self.current_path is None:
            messagebox.showerror("没有视频", "请先选择一个视频。")
            return None
        try:
            track_id = int(self.track_id.get())
            start_frame = int(self.start_frame.get())
            end_frame = int(self.end_frame.get())
        except ValueError:
            messagebox.showerror("字段格式错误", "轨迹 ID、开始帧和结束帧必须是整数。")
            return None
        if track_id < 0:
            messagebox.showerror("轨迹 ID 无效", "轨迹 ID 不能小于 0。")
            return None
        if not (0 <= start_frame <= end_frame < self.frame_count):
            messagebox.showerror("帧区间无效", f"帧区间必须位于 0 到 {self.frame_count - 1} 内，且开始帧不大于结束帧。")
            return None
        if not self.confirmed.get():
            messagebox.showerror("尚未确认", "请勾选“我已确认…”，再保存该动作片段。")
            return None
        action = self.action_label.get()
        walking_view = self.walking_view.get().strip().lower()
        walking_direction = self.walking_direction.get().strip().lower()
        branch = branch_id_for_action(action, walking_view, walking_direction)
        source_id = self.source_video_id.get()
        # ``fast_screen`` runs the tracker on a sampled frame stream.  Keep
        # that sampling contract with the annotation so extraction can replay
        # the same source-local track IDs instead of silently assigning IDs
        # from a different frame cadence.  Videos opened directly from a
        # directory do not have this field and therefore use every frame.
        source_frame_stride = 1
        source_image_size = 0
        source_fps: float | None = None
        if self.current_queue_row is not None:
            try:
                source_frame_stride = max(1, int(self.current_queue_row.get("frame_stride", 1)))
            except (TypeError, ValueError):
                source_frame_stride = 1
            try:
                source_image_size = max(0, int(self.current_queue_row.get("source_image_size", 0)))
            except (TypeError, ValueError):
                source_image_size = 0
            try:
                candidate_fps = float(self.current_queue_row.get("fps", 0.0))
                if candidate_fps > 0.0:
                    source_fps = candidate_fps
            except (TypeError, ValueError):
                source_fps = None
        record = {
            "schema_version": "mevid-action-branch-v3",
            "annotation_key": annotation_key(source_id, track_id, branch, start_frame, end_frame),
            "source_video_id": source_id,
            "source_video_path": str(self.current_path),
            "track_id": track_id,
            "action_label": action,
            "walking_view": walking_view,
            "walking_direction": walking_direction,
            "gait_branch_id": branch,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "action_confidence": self.confidence.get(),
            "annotation_status": "human_confirmed" if branch != "unknown" else "isolated_unknown",
            "annotation_note": self.note.get().strip(),
            "source_frame_stride": source_frame_stride,
        }
        if source_image_size:
            record["source_image_size"] = source_image_size
        if source_fps is not None:
            record["source_fps"] = source_fps
        return record

    def _save_current(self) -> bool:
        record = self._build_annotation()
        if record is None:
            return False
        self.annotations[record["annotation_key"]] = record
        output = Path(self.output_path.get())
        try:
            write_jsonl_atomic(output, [self.annotations[key] for key in sorted(self.annotations)])
        except OSError as exc:
            messagebox.showerror("保存失败", f"无法写入 {output}：\n{exc}")
            return False
        branch = record["gait_branch_id"]
        if branch == "unknown":
            self.status.set(f"已保存未知动作隔离记录；不会进入正式步态分支。共 {len(self.annotations)} 条。")
        else:
            self.status.set(f"已保存到 {branch}。共 {len(self.annotations)} 条动作分支标注。")
        return True

    def _save_and_next(self) -> None:
        if self._save_current():
            self._next_item()

    def _next_item(self) -> None:
        selection = self.queue_list.curselection()
        if not selection:
            return
        next_index = selection[0] + 1
        if next_index >= len(self.queue_rows):
            self.status.set("已到达队列末尾。")
            return
        self.queue_list.selection_clear(0, tk.END)
        self.queue_list.selection_set(next_index)
        self.queue_list.see(next_index)
        self._on_queue_select()

    def _save_event(self, _event: object) -> str:
        self._save_current()
        return "break"

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    app = ActionBranchGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.close)
    root.mainloop()


if __name__ == "__main__":
    main()
