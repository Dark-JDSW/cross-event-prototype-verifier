"""Small local GUI for reviewing production tracks and writing action spans.

The tool is intentionally annotation-only.  It creates no verifier, SQLite
gallery, identity, or gait-enrollment record.  Each saved JSONL row is accepted
by ``scripts/extract_action_branch_poses.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np

from cross_event_verifier.production_vision import ProductionVisionAdapter, ProductionVisionConfig


class ActionLabeler:
    def __init__(self, root: tk.Tk, *, output: Path, use_cuda: bool) -> None:
        self.root = root
        self.output = Path(output)
        self.use_cuda = use_cuda
        self.capture: cv2.VideoCapture | None = None
        self.adapter: ProductionVisionAdapter | None = None
        self.video_path: Path | None = None
        self.frame_count = 0
        self.fps = 0.0
        self.current_frame = -1
        self.next_frame = 0
        self.frame_image: tk.PhotoImage | None = None
        self.annotations: list[dict[str, object]] = []

        self.track_var = tk.StringVar(value="")
        self.action_var = tk.StringVar(value="WALK")
        self.start_var = tk.StringVar(value="")
        self.end_var = tk.StringVar(value="")
        self.seek_var = tk.StringVar(value="0")
        self.status_var = tk.StringVar(value="请选择视频")

        self.canvas = tk.Canvas(root, width=960, height=540, background="black")
        self.canvas.grid(row=0, column=0, columnspan=6, padx=8, pady=8)
        ttk.Button(root, text="打开视频", command=self.open_video).grid(row=1, column=0, sticky="ew")
        ttk.Button(root, text="上一帧", command=lambda: self.seek(self.current_frame - 1)).grid(row=1, column=1, sticky="ew")
        ttk.Button(root, text="下一帧", command=lambda: self.seek(self.current_frame + 1)).grid(row=1, column=2, sticky="ew")
        ttk.Button(root, text="播放/暂停", command=self.toggle_play).grid(row=1, column=3, sticky="ew")
        ttk.Label(root, text="跳转帧").grid(row=1, column=4)
        ttk.Entry(root, width=9, textvariable=self.seek_var).grid(row=1, column=5, sticky="ew")
        ttk.Button(root, text="跳转", command=self.seek_from_entry).grid(row=2, column=5, sticky="ew")

        ttk.Label(root, text="Track ID").grid(row=2, column=0)
        ttk.Entry(root, width=8, textvariable=self.track_var).grid(row=2, column=1)
        ttk.Label(root, text="动作").grid(row=2, column=2)
        ttk.Combobox(
            root,
            textvariable=self.action_var,
            values=("WALK", "UNKNOWN"),
            state="readonly",
            width=10,
        ).grid(row=2, column=3)
        ttk.Button(root, text="设为起点", command=self.mark_start).grid(row=3, column=0, sticky="ew")
        ttk.Button(root, text="设为终点", command=self.mark_end).grid(row=3, column=1, sticky="ew")
        ttk.Label(root, text="起点").grid(row=3, column=2)
        ttk.Label(root, textvariable=self.start_var, width=10).grid(row=3, column=3)
        ttk.Label(root, text="终点").grid(row=3, column=4)
        ttk.Label(root, textvariable=self.end_var, width=10).grid(row=3, column=5)
        ttk.Button(root, text="保存当前标注", command=self.save_annotation).grid(row=4, column=0, columnspan=2, sticky="ew")
        ttk.Button(root, text="写入 JSONL", command=self.write_output).grid(row=4, column=2, columnspan=2, sticky="ew")
        ttk.Label(root, textvariable=self.status_var).grid(row=4, column=4, columnspan=2, sticky="w")

        self.listbox = tk.Listbox(root, height=7, width=130)
        self.listbox.grid(row=5, column=0, columnspan=6, padx=8, pady=8, sticky="ew")
        self.playing = False
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def open_video(self) -> None:
        selected = filedialog.askopenfilename(
            title="选择 UCF-ARG 或其他视频",
            filetypes=[("视频", "*.avi *.mp4 *.mov *.mkv"), ("全部", "*.*")],
        )
        if not selected:
            return
        self.close_capture()
        path = Path(selected)
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            messagebox.showerror("打开失败", f"无法打开视频：{path}")
            return
        self.capture = capture
        self.video_path = path
        self.frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        self.adapter = ProductionVisionAdapter(
            ProductionVisionConfig(
                require_cuda=self.use_cuda,
                detector_inference_stride=1,
                low_light_threshold=0.0,
            )
        )
        self.status_var.set("正在加载 YOLO/RTMPose，请稍候…")
        self.root.update_idletasks()
        self.adapter.preload()
        self.adapter.warmup()
        self.annotations.clear()
        self.listbox.delete(0, tk.END)
        self.seek(0)
        self.status_var.set(f"{path.name} | {self.frame_count} 帧 | {self.fps:.2f} FPS")

    def close_capture(self) -> None:
        self.playing = False
        if self.capture is not None:
            self.capture.release()
        self.capture = None
        self.adapter = None

    def seek(self, frame_index: int) -> None:
        if self.capture is None or self.adapter is None or self.frame_count <= 0:
            return
        target = max(0, min(int(frame_index), self.frame_count - 1))
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.adapter.reset()
        frame = None
        tracks = ()
        for index in range(target + 1):
            ok, frame = self.capture.read()
            if not ok:
                break
            timestamp = index / self.fps if self.fps > 0.0 else float(index)
            tracks = self.adapter.process_at(frame, timestamp=timestamp)
        if frame is None:
            return
        self.current_frame = target
        self.next_frame = target + 1
        self.seek_var.set(str(target))
        self.render(frame, tracks)

    def seek_from_entry(self) -> None:
        try:
            frame = int(self.seek_var.get())
        except ValueError:
            messagebox.showwarning("帧号无效", "请输入整数帧号")
            return
        self.seek(frame)

    def render(self, frame, tracks) -> None:
        image = frame.copy()
        for track in tracks:
            x1, y1, x2, y2 = (int(value) for value in track.box)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                image,
                f"T{track.track_id}",
                (x1, max(18, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
            )
        height, width = image.shape[:2]
        scale = min(1.0, 960.0 / max(width, 1), 540.0 / max(height, 1))
        if scale < 0.999:
            image = cv2.resize(
                image,
                (max(1, int(width * scale)), max(1, int(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        ppm = f"P6\n{w} {h}\n255\n".encode("ascii") + np.asarray(rgb).tobytes()
        self.frame_image = tk.PhotoImage(master=self.root, data=ppm, format="PPM")
        self.canvas.delete("all")
        self.canvas.create_image(480, 270, image=self.frame_image)
        self.status_var.set(
            f"帧 {self.current_frame}/{max(self.frame_count - 1, 0)} | "
            f"检测到 Track: {', '.join(f'T{t.track_id}' for t in tracks) or '无'}"
        )

    def toggle_play(self) -> None:
        self.playing = not self.playing
        if self.playing:
            self.play_tick()

    def play_tick(self) -> None:
        if not self.playing or self.capture is None or self.adapter is None:
            return
        if self.next_frame >= self.frame_count:
            self.playing = False
            return
        ok, frame = self.capture.read()
        if not ok:
            self.playing = False
            return
        index = self.next_frame
        timestamp = index / self.fps if self.fps > 0.0 else float(index)
        tracks = self.adapter.process_at(frame, timestamp=timestamp)
        self.current_frame = index
        self.next_frame += 1
        self.seek_var.set(str(index))
        self.render(frame, tracks)
        delay = max(1, int(1000.0 / self.fps)) if self.fps > 0.0 else 33
        self.root.after(delay, self.play_tick)

    def mark_start(self) -> None:
        if self.current_frame >= 0:
            self.start_var.set(str(self.current_frame))

    def mark_end(self) -> None:
        if self.current_frame >= 0:
            self.end_var.set(str(self.current_frame))

    def save_annotation(self) -> None:
        if self.video_path is None:
            messagebox.showwarning("未选择视频", "先打开视频")
            return
        try:
            track_id = int(self.track_var.get())
            start = int(self.start_var.get())
            end = int(self.end_var.get())
        except ValueError:
            messagebox.showwarning("标注不完整", "请输入 Track ID，并设置起止帧")
            return
        if track_id < 0 or start < 0 or end < start:
            messagebox.showwarning("区间无效", "Track ID 和帧区间无效")
            return
        if end - start + 1 < 100:
            messagebox.showwarning("区间过短", "为了生成 T=100 窗口，动作区间至少需要 100 帧")
            return
        action = "walking" if self.action_var.get() == "WALK" else "unknown"
        row = {
            "annotation_key": f"{self.video_path.stem}_t{track_id}_{start}_{end}",
            "source_video_path": str(self.video_path),
            "source_video_id": self.video_path.stem,
            "track_id": track_id,
            "start_frame": start,
            "end_frame": end,
            "action_label": action,
            "source_frame_stride": 1,
            "source_image_size": 736,
            "walking_view": "unknown",
            "walking_direction": "unknown",
        }
        self.annotations.append(row)
        self.listbox.insert(tk.END, f"{action:7s} T{track_id} {start}-{end} {self.video_path.name}")
        self.start_var.set("")
        self.end_var.set("")

    def write_output(self) -> None:
        if not self.annotations:
            messagebox.showwarning("没有标注", "至少保存一条标注")
            return
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in self.annotations),
            encoding="utf-8",
        )
        messagebox.showinfo("已保存", f"已写入 {len(self.annotations)} 条标注：\n{self.output}")

    def close(self) -> None:
        self.close_capture()
        self.root.destroy()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(r"F:\dataset\ucf_arg\manifests\reviewed_action_annotations.jsonl"),
    )
    parser.add_argument("--cpu", action="store_true", help="允许使用 CPU 推理，速度较慢")
    args = parser.parse_args()
    root = tk.Tk()
    root.title("WALK/UNKNOWN 数据标注")
    root.geometry("1050x780")
    ActionLabeler(root, output=args.output, use_cuda=not args.cpu)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
