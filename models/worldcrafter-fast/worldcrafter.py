"""Reactor application for image-conditioned, camera-controlled WorldCrafter."""
from collections import deque
from pathlib import Path

from reactor_runtime import (
    ApplicationError, ClientInfo, CommandError, InputField, ReactorApp,
    StepOutcome, UploadedFile, connected, disconnected, event,
    session_started, session_ended, get_weights_path,
)
from worldcrafter_assets import prepare_runtime, read_settings
from worldcrafter_images import normalize_image
from worldcrafter_model import WorldCrafterInput, WorldCrafterModel, WorldCrafterResult
from worldcrafter_types import (
    ChunkGenerated, MotionQueued, PlaybackChanged, RolloutLimitReached,
    StateUpdate, WorldCrafterOutput, WorldCrafterState, WorldStarted,
)

ACTIONS = ("forward", "backward", "left", "right", "up", "down", "yaw_left",
           "yaw_right", "pitch_up", "pitch_down", "orbit_left", "orbit_right",
           "orbit_up", "orbit_down")


class WorldCrafter(ReactorApp):
    state: WorldCrafterState
    buffer_size = 33
    fps = 16

    def __init__(self):
        super().__init__()
        self._engine = None
        self._image = None
        self._prompt = ""
        self._seed = 0
        self._gravity = True
        self._queue = deque()
        self._paused = False
        self._completed = 0
        self._inflight = None
        self._max_chunks = 50

    def load(self, config_path: Path | None):
        settings = read_settings(config_path, weights_path=get_weights_path())
        prepare_runtime(settings)
        self._max_chunks = settings.max_chunks
        self._engine = WorldCrafterModel()
        self._engine.load(settings)

    @session_started
    def on_session_started(self):
        self.state._world_id = 0
        self.state._applied_world_id = None
        self._image = None
        self._prompt = ""
        self._queue.clear()
        self._completed = 0
        self._inflight = None
        self._paused = False

    @session_ended
    def on_session_ended(self):
        self._engine.reset()
        self._image = None
        self._prompt = ""
        self._queue.clear()
        self._inflight = None
        self._completed = 0

    @connected
    async def on_connected(self, client: ClientInfo):
        await client.send(self._snapshot())

    @disconnected
    async def on_disconnected(self):
        self._queue.clear()
        self._paused = True
        await self.send(self._snapshot())

    @event(name="start_world", description="Start a fresh world from an image and scene prompt; generates one stationary chunk. Returns `world_started` and broadcasts `state_update`. Fails with `invalid_image` or `prompt_required`.")
    async def start_world(
        self,
        image: UploadedFile = InputField(moderate=True, description="PNG, JPEG or WebP anchor image, at most 20 MiB."),
        prompt: str = InputField(default="", max_length=4096, moderate=True, description="Scene description, including first- or third-person perspective."),
        seed: int = InputField(default=0, ge=0, le=2147483647, description="Reproducible generation seed."),
        gravity_align: bool = InputField(default=True, description="Align yaw and horizontal orbit to image gravity when calibration succeeds."),
    ) -> WorldStarted:
        if not prompt.strip():
            raise CommandError("prompt_required", "Provide a scene description.")
        normalized = normalize_image(image)
        self._image, self._prompt = normalized, prompt.strip()
        self._seed, self._gravity = seed, gravity_align
        self._fresh_world()
        await self.send(self._snapshot())
        return WorldStarted(world_id=self.state._world_id, seed=self._seed)

    @event(name="move", description="Queue camera motion for 1–8 chunks. Translation amount is 0.1–5 units; rotation is 0.1–45 degrees per chunk. Returns `motion_queued` and broadcasts `state_update`. Fails with `image_required`, `invalid_motion`, `queue_full`, or `rollout_limit`.")
    async def move(
        self,
        action: str = InputField(choices=list(ACTIONS), description="Camera motion direction."),
        amount: float = InputField(default=1.0, ge=0.1, le=45.0, description="Distance or rotation degrees per chunk."),
        chunks: int = InputField(default=1, ge=1, le=8, description="Number of consecutive chunks."),
    ) -> MotionQueued:
        import math
        self._require_image()
        if action not in ACTIONS or not math.isfinite(amount) or not 0.1 <= amount <= 45 or not 1 <= chunks <= 8:
            raise CommandError("invalid_motion", "Use a supported direction and bounded amount and chunk count.")
        if action in ACTIONS[:6] and amount > 5:
            raise CommandError("invalid_motion", "Translation must not exceed 5 units per chunk.")
        if len(self._queue) + chunks > 16:
            raise CommandError("queue_full", "At most 16 camera chunks may be queued.")
        in_progress = int(self._inflight == self.state._world_id)
        if self._completed + in_progress + len(self._queue) + chunks > self._max_chunks:
            raise CommandError("rollout_limit", "Reset or start a new world before adding more chunks.")
        self._queue.extend([(action, amount)] * chunks)
        await self.send(self._snapshot())
        return MotionQueued(action=action, chunks=chunks)

    @event(name="pause", description="Pause before the next chunk; an in-flight chunk may finish. Returns `playback_changed` and broadcasts `state_update`.")
    async def pause(self) -> PlaybackChanged:
        self._paused = True
        await self.send(self._snapshot())
        return PlaybackChanged(paused=True)

    @event(name="resume", description="Resume queued camera motion. Returns `playback_changed` and broadcasts `state_update`. Fails with `image_required`.")
    async def resume(self) -> PlaybackChanged:
        self._require_image()
        self._paused = False
        await self.send(self._snapshot())
        return PlaybackChanged(paused=False)

    @event(name="reset", description="Restart from the selected image, prompt and seed, discarding exploration history. Returns `world_started` and broadcasts `state_update`. Fails with `image_required`.")
    async def reset(self) -> WorldStarted:
        self._require_image()
        self._fresh_world()
        await self.send(self._snapshot())
        return WorldStarted(world_id=self.state._world_id, seed=self._seed)

    async def process_input(self) -> WorldCrafterInput:
        if self._image is None or self._paused or not self._queue:
            raise ApplicationError("Waiting for an image or camera motion.")
        action, amount = self._queue.popleft()
        world_id = self.state._world_id
        self._inflight = world_id
        return WorldCrafterInput(world_id, self._image if world_id != self.state._applied_world_id else None,
                                 self._prompt, self._seed, self._gravity, action, amount)

    def generate(self, input: WorldCrafterInput) -> WorldCrafterResult:
        return self._engine.generate(input)

    async def process_output(self, outcome: StepOutcome) -> WorldCrafterOutput | None:
        self._inflight = None
        if outcome.error is not None:
            raise outcome.error
        result: WorldCrafterResult = outcome.result
        if result.world_id != self.state._world_id:
            return None
        self.state._applied_world_id = result.world_id
        self._completed = result.chunk_index + 1
        await self.send(ChunkGenerated(world_id=result.world_id, chunk_index=result.chunk_index,
                                      frames=len(result.frames), generation_seconds=outcome.elapsed,
                                      camera_end=result.camera_end, gravity_aligned=result.gravity_aligned,
                                      gpu_count=result.gpu_count))
        if self._completed >= self._max_chunks:
            self._queue.clear()
            await self.send(RolloutLimitReached(completed_chunks=self._completed))
        await self.send(self._snapshot())
        return WorldCrafterOutput(main_video=result.frames)

    def _fresh_world(self):
        self.output.flush()
        self.state._world_id += 1
        self._completed = 0
        self._paused = False
        self._queue.clear()
        self._queue.append(("static", 0.0))

    def _require_image(self):
        if self._image is None:
            raise CommandError("image_required", "Start a world with an image first.")

    def _snapshot(self):
        return StateUpdate(world_id=self.state._world_id, image_selected=self._image is not None,
                           paused=self._paused, completed_chunks=self._completed,
                           queued_chunks=len(self._queue), max_chunks=self._max_chunks)
