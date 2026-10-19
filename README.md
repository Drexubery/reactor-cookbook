# WorldCrafter-Fast PR #75 validation evidence

Tested adapter commit: [`47d447bc8a9d4d9be11d1c0cf66159698e416acb`](https://github.com/Drexubery/reactor-cookbook/commit/47d447bc8a9d4d9be11d1c0cf66159698e416acb).
Runtime 3.8.3; Python SDK 1.10.0; source and weight revisions are pinned in the adapter.
This branch contains validation artifacts only; it is not part of the model PR diff.

- [Actual eight-H200 Reactor client video](reactor-client-h200.mp4): 297 unique frames, 640×384, 16 FPS. The route is a stationary anchor, forward, yaw left/right, four queued forward chunks with a second viewer disconnecting, then reset. This is an integration-test route, not the separate 528-pose example replay.
- [Eight-H200 report](h200-validation.json) and [single-H20 report](h20-validation.json): all 297 expected frames received and decoded in each run.
- [Additional active-viewer test](active-viewer.json): 165/165 frames, with another viewer disconnecting during a four-action rollout and no resume command.
- [Artifact checks](artifact-checks.json): eight participating GPUs, compiled VAE numerical checks, exact raw anchor RGB after reset, checkpoint and source file integrity.
- [Native camera/preprocessing equivalence](native-equivalence.json): 168 steps, four input image modes/orientations.
- [4000×4000 JPEG responsiveness](upload-responsiveness.json): original encoded bytes retained while the event-loop heartbeat continues.
- [Tested adapter file hashes](source-files.json): 22 files, matching the built image.

35 tests, unchanged schema rendering, `reactor validate` and `reactor build` passed.
The native Reactor image was exercised through a platform wrapper adding only AWS CLI, with no adapter or inference-environment changes.
Eight-GPU validation used a 1 TiB host memory limit; the manifest's 512 GiB limit was not separately stress-tested.
Reports omit local video output paths; no credentials, private platform configuration or raw infrastructure logs are published here.

## Transport conditions

Complete-frame acceptance used a controlled loopback WebRTC link, the default codec and 16 FPS. These server environment variables were all set to `20000` (Kbps):

```text
WEBRTC_BWE_MIN_KBPS
WEBRTC_BWE_INITIAL_KBPS
WEBRTC_BWE_MAX_KBPS
WEBRTC_SENDER_MIN_KBPS
WEBRTC_SENDER_MAX_KBPS
```

Production defaults are unchanged. Earlier adaptive-transport runs dropped individual frames; the strict client correctly failed them. These results validate exact frame accounting and inference integration, not lossless delivery on arbitrary public networks or Reactor-managed hosting.

After starting the server with the above test conditions:

```bash
python validate_client.py --url http://127.0.0.1:8080 \
  --image /absolute/path/test/I2V/03_waterfall/image.png \
  --prompt /absolute/path/test/I2V/03_waterfall/prompt.txt \
  --gpus 8 --full --multi-viewer --output /absolute/path/validation
```

Video SHA-256: `759d8c068e6e19d2a0ad1395a2aec03fbfa9d5426fdf933cf8168aa9f58d6351`.
