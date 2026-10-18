"""Python 3.11 inference worker; rank zero serves bounded local RPC requests."""
import argparse
import json
import os
from pathlib import Path
import socket
import sys
import threading
import traceback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--max-chunks", type=int, required=True)
    args = parser.parse_args()
    sys.path.insert(0, args.source)
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    size = int(os.environ.get("WORLD_SIZE", "1"))
    if size not in (1, 8):
        raise ValueError("Expected one or eight inference ranks")
    if size == 8:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7").split(",")
        if len(visible) != 8:
            raise ValueError("Eight visible GPUs are required")
        # Rank zero's VAE pipeline spans all GPUs; peer DiTs each own cuda:0.
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(visible) if rank == 0 else visible[rank]
        for key in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"):
            os.environ[key] = str(Path(os.environ.get(key, args.root / key.lower())) / f"rank_{rank}")
        if rank:
            from demo.multigpu.serving import serve_peer
            serve_peer()
            return

    import numpy as np
    from demo.camera import Camera
    from worldcrafter.camera import Action
    if size == 8:
        from demo.multigpu.serving import ContextParallelEngine
        engine = ContextParallelEngine()
    else:
        from demo.engine import WorldCrafterEngine
        engine = WorldCrafterEngine()
    world_id = None
    gravity_aligned = False
    cancel = threading.Event()
    connection = socket.socket(socket.AF_UNIX)
    connection.connect(args.socket)
    stream = connection.makefile("rwb")

    def respond(value):
        stream.write((json.dumps(value) + "\n").encode())
        stream.flush()

    respond(dict(gpu_count=size, engine=engine.info))
    try:
        for line in stream:
            request = json.loads(line)
            identity = request["id"]
            try:
                if request["command"] == "reset":
                    engine.close_session()
                    world_id = None
                    gravity_aligned = False
                    respond(dict(id=identity))
                    continue
                if request["command"] != "generate":
                    raise ValueError("Unknown worker command")
                if request["world_id"] != world_id:
                    if not request["has_image"]:
                        raise ValueError("A new world requires an anchor image")
                    up = None
                    if request["gravity_align"]:
                        from worldcrafter.calibration import calibrate_image
                        up = calibrate_image(args.root / "anchor.png")["world_up"]
                    engine.initialize_session(str(args.root / "anchor.png"), request["prompt"],
                                              request["seed"], args.max_chunks, cancel)
                    engine.camera = Camera(world_up=up)
                    world_id = request["world_id"]
                    gravity_aligned = up is not None
                if engine.index >= args.max_chunks:
                    raise ValueError("World chunk limit reached")
                action = make_action(request["action"], request["amount"], Action)
                result = engine.next_chunk(action)
                np.save(args.root / "frames.npy", result["frames"], allow_pickle=False)
                reply = dict(id=identity, world_id=world_id, chunk_index=result["chunk_index"],
                             camera_end=result["global_pose"][-1].tolist(),
                             gravity_aligned=gravity_aligned, gpu_count=size,
                             memory_frames=engine.index * 33)
                print("REACTOR_CHUNK " + json.dumps(dict(reply, action=request["action"],
                      engine_mode=engine.info["gpu_mode"], rgb_sha256=result["rgb_sha256"],
                      memory=result["memory"])), flush=True)
                respond(reply)
            except Exception as exc:
                traceback.print_exc()
                respond(dict(id=identity, error=f"{type(exc).__name__}: {exc}"))
                break
    finally:
        engine.close_session()
        if size == 8:
            engine.shutdown()
        stream.close()
        connection.close()


def make_action(name, amount, action_type):
    if name == "static":
        return action_type()
    mapping = {"forward": ("forward", 1), "backward": ("forward", -1),
               "left": ("right", -1), "right": ("right", 1),
               "up": ("up", 1), "down": ("up", -1),
               "yaw_left": ("yaw", -1), "yaw_right": ("yaw", 1),
               "pitch_up": ("pitch", 1), "pitch_down": ("pitch", -1),
               "orbit_left": ("yaw", 1), "orbit_right": ("yaw", -1),
               "orbit_up": ("pitch", -1), "orbit_down": ("pitch", 1)}
    axis, sign = mapping[name]
    rotating = axis in ("yaw", "pitch")
    return action_type(**{axis: sign * amount if rotating else sign},
                       speed=1.0 if rotating else amount, orbit=name.startswith("orbit_"),
                       orbit_radius=2.0 if name.startswith("orbit_") else 0.0).normalized()


if __name__ == "__main__":
    main()
