# WorldCrafter-Fast on Reactor

Explore an uploaded scene with image-conditioned video, camera controls and
persistent scene memory. This adapter streams native 33-frame chunks through
Reactor's `main_video` track and supports one GPU or eight GPUs on one host.
The eight-GPU configuration uses WorldCrafter's context-parallel DiT, fused
KV/RoPE path, geometry cache and eight-GPU VAE pipeline.

The adapter uses Reactor Runtime **3.8.3**. WorldCrafter runs in a separate
Python **3.11** / NumPy **1** environment because the Runtime requires Python
**3.12** / NumPy **2**. Communication stays on a private local Unix socket;
frames cross as uint8 RGB NumPy arrays without pickle. An eight-GPU worker
uses the pinned upstream `torchrun` topology; it is not nested inside a
second distributed launcher.

## Versions and preparation

- Source: `TencentARC/WorldCrafter@11712fe3a3f42cc448c10713a0b0ff79066e5935`.
- Weights: `TencentARC/WorldCrafter-Fast@51320c3eb58a6313ccd6021a1da6d723d53e1a58`.
- The upstream `uvenv/uv.lock` defines the inference environment, including CUDA 12.8 PyTorch.
- One generation occupies the configured GPU group; eight GPUs do not imply eight concurrent worlds.

Use a Linux GPU host with Docker, NVIDIA Container Toolkit and the
[Reactor CLI](https://docs.reactor.inc/deploy/development/local-testing).
Reserve at least 200 GB for the pinned weights and additional space for the
image, compilation cache and recordings. From this folder:

```bash
reactor validate
reactor build
reactor run --gpus all --port 8090
```

The image contains the pinned source and its frozen inference environment at
`/opt/worldcrafter`. `prepare.sh` installs them during the build; it does not
download weights. Reactor resolves the pinned `hf://` URI in `reactor.yaml`,
mounts the weights separately and exposes them through `get_weights_path()`.
To reuse a complete local checkpoint:

```bash
reactor run --gpus all --port 8090 --weights /absolute/path/WorldCrafter-Fast
```

The adapter checks every required file against `weights-manifest.json`, whose
sizes and SHA-256 hashes correspond to the pinned upstream snapshot. Every
Runtime load hashes the complete checkpoint and writes an audit receipt to
local scratch after success. Missing, truncated or corrupted files fail before
GPU loading; repair partial checkpoints before retrying. Sessions reuse the
resident model without loading or hashing the weights again.
The checkpoint directory is never modified by validation.

Eight GPUs are the default. For a single-GPU smoke test set
`-e WORLDCRAFTER_GPUS=1` on `reactor run`. Supported counts are
one and eight. Compilation defaults on; `WORLDCRAFTER_COMPILE=0` disables it
for diagnosis. `/health` must report `state: available` before connecting.
Caches and scratch use `/local-ssd`; provide a writable local disk there.

`reactor.yaml` requests eight H200 GPUs for hosted deployment. Local execution
uses the GPUs visible to the container. Hosted access and resource allocation
must be arranged with Reactor; a local build does not provision a hosted model.

### Direct Runtime development

For development without Docker, create a Python 3.12 environment on local SSD,
install `reactor-runtime==3.8.3`, `requirements.txt` and `uv==0.11.19`, then:

```bash
export WORLDCRAFTER_ROOT=/local-ssd/worldcrafter-reactor
export WORLDCRAFTER_SCRATCH=/local-ssd/wc-reactor
export UV_CACHE_DIR=/local-ssd/cache/uv
export UV_PYTHON_INSTALL_DIR=/local-ssd/python
export HF_HOME=/local-ssd/cache/huggingface
export TORCH_HOME=/local-ssd/cache/torch
export XDG_CACHE_HOME=/local-ssd/cache
export TORCHINDUCTOR_CACHE_DIR=/local-ssd/cache/inductor
export TRITON_CACHE_DIR=/local-ssd/cache/triton
export PYTHONPYCACHEPREFIX=/local-ssd/cache/pycache
export REACTOR_WEIGHTS_PATH=/absolute/path/WorldCrafter-Fast
export WORLDCRAFTER_GPUS=8
export OMP_NUM_THREADS=8
export TORCHINDUCTOR_COMPILE_THREADS=2
bash prepare.sh
HOST=127.0.0.1 PORT=8090 python -m reactor_runtime.serve
```

`prepare.sh` stages the pinned public source atomically and runs
`uv sync --frozen`. It refuses to overwrite an unrecognized nonempty source
directory. `WORLDCRAFTER_SOURCE` and `WORLDCRAFTER_PYTHON` can select prepared
absolute paths; `WORLDCRAFTER_WEIGHTS` explicitly overrides the Runtime's
weight path. Use standard Hugging Face authentication when needed; never put
credentials in the model folder.

## Commands

| Command | Input | Behavior |
| --- | --- | --- |
| `start_world` | Uploaded image, nonempty scene prompt, seed, `gravity_align` | Starts a fresh world and emits one stationary chunk. |
| `move` | `action`, `amount`, `chunks` | Queues one to eight camera chunks. |
| `pause` | None | Pauses before the next chunk; the in-flight chunk may finish. |
| `resume` | None | Continues queued motion. |
| `reset` | None | Clears exploration history and restarts the selected image and seed. |

Translate with `forward`, `backward`, `left`, `right`, `up`, `down` (0.1–5
units per chunk). Rotate with `yaw_left`, `yaw_right`, `pitch_up`, `pitch_down`,
`orbit_left`, `orbit_right`, `orbit_up`, `orbit_down` (0.1–45 degrees per chunk).
Orbit radius is two world units. Commands are sampled at chunk boundaries.
At most 16 chunks may be queued; a world is limited to 50 chunks.

Include the desired first- or third-person perspective in the scene prompt.
Gravity alignment defaults on for each new image: GeoCalib estimates the
full gravity vector, yaw and horizontal orbit use that axis, and calibration
failure falls back to unaligned controls. No uncertainty threshold blocks
exploration. Session end clears image, trajectory and model history; loaded
weights remain resident for the next session.

Viewers share one rollout and motion queue. Disconnecting one viewer does not
pause it or discard accepted actions. Runtime suspends generation when no
viewers remain and owns session teardown. Upload validation runs off the event
loop; the original encoded image crosses to the worker, where full-resolution
EXIF-corrected RGB is calibrated before native generation preprocessing.

`process_input()` plans cumulative camera poses on the CPU. Immutable step
inputs carry float32 local and world c2w arrays to the inference worker; the
worker executes those poses without interpreting actions. Gravity is returned
with the stationary anchor and applied before planning subsequent motion.

Accepted commands return their typed result. `state_update` reports queue and
world state. `chunk_generated` reports the chunk, runtime-measured duration,
camera endpoint and participating GPU count. Video plays at 16 FPS; playback
rate and generation throughput are different quantities.
Each model frame carries `world_id`, `chunk_index` and zero-based `frame_index`
metadata. These identify frames across resets and distinguish repeated pacer
frames from new model output, including completely dark frames.

## Test with an actual client

Install the pinned client and test dependencies in a Python 3.12 environment
with `pip install -r requirements-dev.txt`, then:

```bash
python validate_client.py --url http://127.0.0.1:8090 \
  --image /absolute/path/image.png --prompt /absolute/path/prompt.txt \
  --gpus 8 --full --multi-viewer --output /absolute/path/validation
```

The client uploads the image through Reactor, receives WebRTC frames, explores
forward and turns left/right, resets the world, checks camera poses, and saves
a decoded MP4 and a JSON report. It requires all 33 tagged frames for every
announced chunk, including the reset world; missing or unexpected frames fail
validation. Untagged Runtime connection/flush frames are excluded by identity,
not brightness. Repeated tagged pacer frames are counted once. `--multi-viewer`
also connects a second viewer, queues four chunks while paused, resumes and
disconnects that viewer during the rollout. The remaining viewer must receive
all four chunks without another resume. `--media-timeout` bounds delivery wait
after completion; failures save a report listing the missing frame identities.
A remote browser needs working WebRTC connectivity (host candidates or an
appropriate TURN relay); forwarding the HTTP port alone does not establish
video connectivity.

Contract tests do not require model weights:

```bash
python -m reactor_runtime.schema --path . --out /tmp/worldcrafter-schema.json
PYTHONPATH=. python -m pytest tests/ -q
```

## License

WorldCrafter code and weights retain their upstream academic-use license and
third-party notices. The source is included in the built image and weights are mounted
separately; neither is relicensed by this adapter. See [the upstream license](https://github.com/TencentARC/WorldCrafter/blob/11712fe3a3f42cc448c10713a0b0ff79066e5935/LICENSE.txt).

[Project](https://drexubery.github.io/WorldCrafter/) ·
[Source](https://github.com/TencentARC/WorldCrafter) ·
[Weights](https://huggingface.co/TencentARC/WorldCrafter-Fast) ·
[Reactor authoring guidelines](https://github.com/reactor-team/reactor-cookbook/blob/main/GUIDELINES.md)
