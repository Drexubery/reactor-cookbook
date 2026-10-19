"""Plain, immutable step contracts and the stateful upstream model boundary."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WorldCrafterInput:
    """One immutable step: encoded anchor once, plus cumulative CPU poses.

    Both pose arrays are float32 ``(33 * (chunk_index + 1), 3, 4)`` c2w.
    ``local_poses`` restarts at each chunk's first frame; ``world_poses`` is
    relative to the initial camera and is used for memory retrieval.
    ``image`` is validated original PNG/JPEG/WebP bytes, not resized pixels.
    """
    world_id: int
    image: bytes | None
    prompt: str
    seed: int
    gravity_align: bool
    local_poses: np.ndarray
    world_poses: np.ndarray


@dataclass(frozen=True)
class WorldCrafterResult:
    """CPU RGB uint8 ``(33,384,640,3)`` in [0,255] and acknowledged world.

    ``camera_end`` is the last sampled c2w pose, shape ``(3,4)`` relative to
    the initial camera. ``world_up`` is a calibrated unit vector in that
    camera's coordinates, or None when disabled/unavailable.
    """
    world_id: int
    chunk_index: int
    frames: np.ndarray
    camera_end: list[list[float]]
    gravity_aligned: bool
    gpu_count: int
    world_up: list[float] | None


class WorldCrafterModel:
    def __init__(self):
        self.backend = None

    def load(self, settings):
        from upstream_backend import WorldCrafterBackend
        self.backend = WorldCrafterBackend(settings)

    def generate(self, input: WorldCrafterInput) -> WorldCrafterResult:
        return self.backend.generate(input)

    def reset(self):
        self.backend.reset()

    def close(self):
        if self.backend is not None:
            self.backend.close()
