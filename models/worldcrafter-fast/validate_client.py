"""Receive real video through the Reactor SDK and verify camera/reset behavior."""
import argparse
import asyncio
import json
from pathlib import Path
import time

import av
import numpy as np
from reactor_sdk import Reactor


async def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    messages = []
    frames = []
    boundary_frames = 0
    errors = []
    client = Reactor("worldcrafter-fast", local=True, api_url=args.url)

    def message(value):
        messages.append(value)
        print("MESSAGE " + json.dumps(value), flush=True)

    client.on("message", message)
    client.on("error", lambda error: errors.append(str(error)))

    @client.track("main_video").on_frame
    def on_frame(frame):
        nonlocal boundary_frames
        # Reactor emits a black boundary frame before media and after flush.
        # Its initial size is 1280x720, independent of the model output size.
        if (frame.shape == (720, 1280, 3) and not frames) or (float(frame.std()) < 1 and float(frame.mean()) < 20):
            boundary_frames += 1
            return
        if frame.shape != (384, 640, 3):
            errors.append(f"Unexpected decoded frame shape: {frame.shape}")
            return
        frames.append(frame.copy())

    async def wait_chunk(world_id, index):
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            for value in messages:
                payload = value.get("data", value)
                if (payload.get("world_id") == world_id and payload.get("chunk_index") == index
                        and "camera_end" in payload):
                    await asyncio.sleep(3)
                    return payload
            if errors:
                raise RuntimeError(errors[-1])
            await asyncio.sleep(0.25)
        raise TimeoutError(f"No completed chunk {world_id}/{index}")

    try:
        await asyncio.wait_for(client.connect(), 90)
        upload = await client.upload_file(args.image)
        response = await client.send_command("start_world", dict(image=upload,
                prompt=args.prompt.read_text().strip(), seed=42, gravity_align=True))
        print("START " + json.dumps(response), flush=True)
        first = await wait_chunk(1, 0)
        route = [("forward", 1.0), ("yaw_left", 20.0), ("yaw_right", 20.0)] if args.full else [("yaw_left", 20.0)]
        chunks = [first]
        for index, (action, amount) in enumerate(route, 1):
            await client.send_command("move", dict(action=action, amount=amount, chunks=1))
            chunks.append(await wait_chunk(1, index))
        await client.send_command("reset", {})
        reset = await wait_chunk(2, 0)
        assert np.allclose(first["camera_end"], reset["camera_end"], atol=1e-6)
        assert not np.allclose(first["camera_end"], chunks[-1]["camera_end"], atol=1e-4)
        assert all(item["gpu_count"] == args.gpus for item in chunks + [reset])
        await asyncio.sleep(4)
        print("FRAME_COUNT", len(frames), "SHAPES", sorted(set(str(x.shape) for x in frames)), flush=True)
        print("STATS", await client.get_stats(), flush=True)
        assert len(frames) >= 33 and frames[0].shape == (384, 640, 3)
        assert frames[0].dtype == np.uint8
        assert float(np.std(frames[-1])) > 5
        video = args.output / "reactor-client.mp4"
        with av.open(str(video), "w") as container:
            stream = container.add_stream("libx264", rate=16)
            stream.width, stream.height, stream.pix_fmt = 640, 384, "yuv420p"
            for frame in frames:
                for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        with av.open(str(video)) as container:
            decoded = sum(1 for _ in container.decode(video=0))
        assert decoded == len(frames)
        from PIL import Image
        for name, frame in (("first", frames[0]), ("last", frames[-1])):
            Image.fromarray(frame).save(args.output / f"{name}.png")
        report = dict(status="passed", gpu_count=args.gpus, received_frames=len(frames), boundary_frames=boundary_frames,
                      decoded_frames=decoded, chunks=chunks, reset=reset,
                      camera_changes_verified=True, reset_pose_verified=True,
                      transport="Reactor SDK 1.10.0 / WebRTC", video=str(video))
        (args.output / "validation.json").write_text(json.dumps(report, indent=2))
        print("VALIDATION " + json.dumps(report), flush=True)
    finally:
        (args.output / "messages.json").write_text(json.dumps(messages, indent=2))
        await client.disconnect()
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8090")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", type=int, required=True)
    parser.add_argument("--timeout", type=float, default=1200)
    parser.add_argument("--full", action="store_true")
    asyncio.run(main(parser.parse_args()))
