"""GPGait++ CCPG inference backend behind the existing vision boundary.

The official six-channel representation and BN-neck [256,19] output are
preserved. CEVP receives a C-order flattened, float32 L2-normalized vector.
Torch and the network are loaded only when constructing the optional encoder.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from .gpgaitpp_preprocessing import Affine, RescaleCenter, HOD_MultiInput, SkeletonInput


def prepare_pose(poses: np.ndarray) -> np.ndarray:
    """Apply the pinned official transforms without modifying caller arrays."""
    values = np.array(poses, dtype=np.float32, copy=True)
    if values.ndim != 3 or values.shape[1:] != (17, 3) or not len(values):
        raise ValueError("GPGait++ requires [T,17,3] full-frame COCO-17 XY/confidence")
    if not np.isfinite(values).all():
        raise ValueError("Non-finite pose")
    values = Affine()(values)
    if np.any(np.ptp(values[:, :, 1], axis=1) <= 1e-8):
        raise ValueError("Degenerate skeleton height")
    for transform in (RescaleCenter(), HOD_MultiInput(joint_format="coco"), SkeletonInput()):
        values = transform(values)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite official representation")
    return values.astype(np.float32)


class GPGaitPPEncoder:
    """Read-only tensor checkpoint loader and bounded-batch temporal encoder."""

    output_dimension = 4864

    def __init__(self, checkpoint_path: str | Path, *, device=None, sequence_length: int = 60) -> None:
        import torch
        from ._gpgaitpp_network import GPGait_PlusPlus

        self.path = Path(checkpoint_path)
        self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.sequence_length = max(25, int(sequence_length))
        self.model = GPGait_PlusPlus()
        state = torch.load(self.path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state.get("model", state), strict=True)
        self.model.to(self.device).eval()

    def encode(self, poses: Sequence[np.ndarray] | np.ndarray) -> np.ndarray | None:
        return self.encode_batch([poses])[0]

    def encode_batch(self, pose_sequences: Sequence[Sequence[np.ndarray] | np.ndarray]) -> list[np.ndarray | None]:
        import torch

        output: list[np.ndarray | None] = [None] * len(pose_sequences)
        prepared = []
        indexes = []
        for index, poses in enumerate(pose_sequences):
            try:
                sequence = np.asarray(poses, dtype=np.float32)
                if len(sequence) < self.sequence_length:
                    continue
                prepared.append(prepare_pose(sequence[-self.sequence_length:]))
                indexes.append(index)
            except (TypeError, ValueError):
                continue
        with torch.inference_mode():
            for start in range(0, len(prepared), 4):
                batch = torch.from_numpy(np.stack(prepared[start:start + 4])).to(self.device)
                # The official CCPG evaluator uses FP32; no implicit precision change.
                values = self.model(([batch], None, None, None, None))["inference_feat"]["embeddings"]
                if tuple(values.shape[1:]) != (256, 19):
                    raise RuntimeError("GPGait++ checkpoint output contract mismatch")
                flattened = values.float().cpu().numpy().reshape(len(batch), -1)
                for destination, vector in zip(indexes[start:start + 4], flattened):
                    norm = float(np.linalg.norm(vector))
                    if np.isfinite(vector).all() and np.isfinite(norm) and norm > 1e-8:
                        output[destination] = (vector / norm).astype(np.float32)
        return output

    def warmup(self) -> None:
        """Check runtime readiness with synthetic input, without identity evidence."""
        sequence = np.zeros((self.sequence_length, 17, 3), dtype=np.float32)
        sequence[:, :, 0] = np.arange(17) * 3 + 100
        sequence[:, :, 1] = np.arange(17) * 8 + 30
        sequence[:, :, 2] = 1
        if self.encode(sequence) is None:
            raise RuntimeError("GPGait++ warmup failed")
