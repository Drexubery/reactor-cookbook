"""Exercise the Reactor contract and world transitions without GPU weights."""
import asyncio
from dataclasses import FrozenInstanceError
import io
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
from reactor_runtime import ApplicationError, CommandError, StepOutcome
from reactor_runtime.interface.model.contract import ModelContract

from worldcrafter import WorldCrafter
from worldcrafter_model import WorldCrafterInput, WorldCrafterModel, WorldCrafterResult
from worldcrafter_types import WorldCrafterState, ChunkGenerated
from worker import make_action


def run(awaitable):
    return asyncio.run(awaitable)


def ready():
    app = WorldCrafter()
    app.state = WorldCrafterState()
    messages = []

    async def send(message):
        messages.append(message)

    app.send = send
    app.on_session_started()
    return app, messages


def image():
    buffer = io.BytesIO()
    Image.new("RGB", (128, 128), (20, 80, 120)).save(buffer, format="PNG")
    return SimpleNamespace(data=buffer.getvalue(), name="scene.png")


def start(app):
    return run(app.start_world(image(), "A stone courtyard, first person.", 42, True))


def result(world=1, chunk=0):
    return WorldCrafterResult(world, chunk, np.zeros((33, 384, 640, 3), dtype=np.uint8),
                             np.eye(4)[:3].tolist(), True, 8, (chunk + 1) * 33)


def test_schema_surface():
    contract = ModelContract.of(WorldCrafter)
    assert set(contract.commands) == {"start_world", "move", "pause", "resume", "reset"}


def test_refused_step_never_calls_model():
    app, messages = ready()
    with pytest.raises(ApplicationError):
        run(app.process_input())
    assert not messages and app._engine is None


def test_refused_command_has_no_success_message():
    app, messages = ready()
    with pytest.raises(CommandError) as error:
        run(app.move("forward", 1, 1))
    assert error.value.code == "image_required"
    assert messages == []


def test_anchor_only_until_ack_and_runtime_elapsed():
    app, messages = ready()
    start(app)
    first = run(app.process_input())
    assert first.image and first.world_id == 1 and first.action == "static"
    with pytest.raises(FrozenInstanceError):
        first.seed = 12
    output = run(app.process_output(StepOutcome(result=result(), elapsed=1.25)))
    assert output.main_video.shape == (33, 384, 640, 3)
    event = next(message for message in messages if isinstance(message, ChunkGenerated))
    assert event.generation_seconds == 1.25 and event.gpu_count == 8
    run(app.move("yaw_left", 20, 1))
    second = run(app.process_input())
    assert second.image is None and second.action == "yaw_left" and second.amount == 20


def test_reset_rejects_stale_result_and_restarts_camera():
    app, messages = ready()
    start(app)
    run(app.process_input())
    run(app.reset())
    count = len(messages)
    assert run(app.process_output(StepOutcome(result=result()))) is None
    assert len(messages) == count and app._completed == 0
    fresh = run(app.process_input())
    assert fresh.world_id == 2 and fresh.image and fresh.action == "static"


def test_model_failure_is_not_a_completed_chunk():
    app, messages = ready()
    start(app)
    run(app.process_input())
    count = len(messages)
    with pytest.raises(RuntimeError, match="GPU failed"):
        run(app.process_output(StepOutcome(error=RuntimeError("GPU failed"))))
    assert len(messages) == count and app._completed == 0


@pytest.mark.parametrize("action,amount", [("forward", 6), ("yaw_left", float("nan")), ("roll", 30)])
def test_invalid_motion_does_not_enter_queue(action, amount):
    app, messages = ready()
    start(app)
    before = list(app._queue)
    with pytest.raises(CommandError):
        run(app.move(action, amount, 1))
    assert list(app._queue) == before


def test_queue_limit_pause_resume_disconnect():
    app, _ = ready()
    start(app)
    run(app.move("forward", 1, 8))
    with pytest.raises(CommandError) as error:
        run(app.move("forward", 1, 8))
    assert error.value.code == "queue_full"
    run(app.pause())
    with pytest.raises(ApplicationError):
        run(app.process_input())
    run(app.resume())
    assert run(app.process_input()).action == "static"
    run(app.on_disconnected())
    assert not app._queue and app._paused


def test_lifecycle_clears_image_and_resets_model():
    app, _ = ready()
    calls = []
    app._engine = SimpleNamespace(reset=lambda: calls.append("reset"))
    start(app)
    app.on_session_ended()
    app.state = WorldCrafterState()
    app.on_session_started()
    assert calls == ["reset"]
    assert app._image is None and not app._queue and app._completed == 0


def test_generate_uses_only_snapshot():
    model = WorldCrafterModel()
    calls = []
    expected = result()
    model.backend = SimpleNamespace(generate=lambda value: calls.append(value) or expected)
    snapshot = WorldCrafterInput(1, b"image", "scene", 0, True, "forward", 1)
    assert model.generate(snapshot) is expected and calls == [snapshot]


@pytest.mark.parametrize("action,axis,sign", [
    ("forward", "forward", 1), ("backward", "forward", -1),
    ("yaw_left", "yaw", -1), ("yaw_right", "yaw", 1),
    ("orbit_left", "yaw", 1), ("orbit_right", "yaw", -1),
])
def test_camera_sign_convention(action, axis, sign):
    class Action:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        def normalized(self):
            return self

    value = make_action(action, 2, Action)
    assert getattr(value, axis) == sign * (2 if axis == "yaw" else 1)
    assert value.orbit == action.startswith("orbit_")


def test_invalid_image_does_not_replace_world():
    app, _ = ready()
    start(app)
    old = app._image
    with pytest.raises(CommandError):
        run(app.start_world(SimpleNamespace(data=b"not image"), "scene", 0, True))
    assert app._image is old and app.state._world_id == 1
