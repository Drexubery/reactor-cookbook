"""Plain, immutable step contracts and the stateful upstream model boundary."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WorldCrafterInput:
    world_id: int
    image: bytes | None
    prompt: str
    seed: int
    gravity_align: bool
    action: str
    amount: float


@dataclass(frozen=True)
class WorldCrafterResult:
    world_id: int
    chunk_index: int
    frames: np.ndarray
    camera_end: list[list[float]]
    gravity_aligned: bool
    gpu_count: int
    memory_frames: int


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
