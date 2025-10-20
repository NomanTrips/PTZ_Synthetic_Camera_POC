# PTZ Synthetic Camera POC — TODO.md

**Goal:** Minimal Python app that plays a video, lets a user control a *synthetic camera* (pan/tilt/zoom) with WASD + mouse wheel, and logs frames + state/action in **LeRobot/pusht‑like** format so we can fine‑tune/evaluate SmolVLA.

---

## 0) Minimal scope (v0)

* Input: one video file (e.g., .mp4)
* Controls: **W/S** tilt ±, **A/D** pan ±, **mouse wheel** zoom ±, **Space** start/stop episode, **Esc** quit
* Output per episode:

  * `videos/.../observation.image/episode_******.mp4` — rendered viewport (e.g., 256×256)
  * `data/.../episode_******.parquet` — rows per frame: state (pan,tilt,zoom), action (dpan,dtilt,dzoom), indices, timestamp
  * `meta/info.json` — LeRobot-friendly metadata

> **Keep it simple:** no autofocus, no tracker, no multiple tasks. FPS fixed (e.g., 24). One camera key: `observation.image`.

---

## 1) Repo scaffold

* [x] `requirements.txt`: `pygame-ce`, `av`, `numpy`, `opencv-python`, `pyarrow`, `imageio`, `imageio-ffmpeg`
* [x] `src/ptz_poc/` package with modules below
* [x] `main.py` CLI: `--video PATH --out_dir PATH --size 256 --fps 24`

**Acceptance:** `python main.py --video sample.mp4 --out_dir out/` opens a window.

---

## 2) VideoReader (PyAV)

* [x] Implement `VideoReader` → iterator yielding `(frame_rgb_np, pts_sec)`
* [x] Normalize to RGB uint8; store source `W,H,FPS`

---

## 3) Synthetic Rig (pan/tilt/zoom)

* [x] `Rig` class with state: `pan_deg`, `tilt_deg`, `zoom_norm ∈ [0,1]`
* [x] Params: `fov_x_deg`, `fov_y_deg`, output size `(H_out,W_out)`, rate/angle limits
* [x] `render(full_frame)` → crop a viewport based on current state, resize to `(H_out,W_out)`
* [x] `apply(dpan, dtilt, dzoom)` → clamp deltas, update state
* [x] Mapping: viewport width = `base_width * exp(-alpha * zoom_norm)`; center shifts with pan/tilt via intrinsics derived from FOV

---

## 4) Input handler (pygame)

* [x] Key repeat for **WASD** → continuous deltas per tick
* [x] Mouse wheel → zoom step per notch
* [x] `Space` toggles recording; `Esc` quits app

---

## 5) HUD renderer

* [ ] Draw viewport to window (blit)
* [ ] Overlay center crosshair
* [ ] Tiny text with `pan, tilt, zoom`

---

## 6) Episode logger (LeRobot‑friendly)

* [ ] `EpisodeWriter` manages one episode:

  * MP4 muxer for `observation.image` (via `imageio-ffmpeg` @ target FPS)
  * In-memory lists for parquet columns: `episode_index, frame_index, timestamp, observation.state[3], action[3]`
* [ ] On `end_episode()`: write Parquet, close MP4

---

## 7) Dataset layout + meta

* [ ] Create folder structure:

  * `meta/info.json`
  * `data/chunk-000/episode_{idx:06d}.parquet`
  * `videos/chunk-000/observation.image/episode_{idx:06d}.mp4`
* [ ] Write `meta/info.json` template (below) at start; update counts after capture

---

## 8) Main loop wiring

* [x] Orchestrate: read next video frame → render viewport → draw HUD → poll input → **log (state_before, action)** → `apply` deltas → advance
* [x] `Space` starts/ends an episode (can start multiple episodes within one run)

---

## LeRobot‑friendly schema (v0)

**`meta/info.json` template** (fill counts later):

```json
{
  "codebase_version": "v2.0",
  "robot_type": "unknown",
  "total_episodes": 0,
  "total_frames": 0,
  "total_tasks": 1,
  "total_videos": 0,
  "total_chunks": 1,
  "chunks_size": 1000,
  "fps": 24,
  "splits": { "train": "0:1000000" },
  "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
  "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
  "features": {
    "observation.image": {
      "dtype": "video",
      "shape": [256, 256, 3],
      "names": ["height","width","channel"],
      "video_info": {
        "video.fps": 24.0,
        "video.codec": "h264",
        "video.pix_fmt": "yuv420p",
        "video.is_depth_map": false,
        "has_audio": false
      }
    },
    "observation.state": { "dtype": "float32", "shape": [3],
      "names": { "motors": ["pan","tilt","zoom"] } },
    "action": { "dtype": "float32", "shape": [3],
      "names": { "motors": ["dpan","dtilt","dzoom"] } },
    "episode_index": { "dtype": "int64", "shape": [1] },
    "frame_index": { "dtype": "int64", "shape": [1] },
    "timestamp": { "dtype": "float32", "shape": [1] },
    "next.reward": { "dtype": "float32", "shape": [1] },
    "next.done": { "dtype": "bool", "shape": [1] },
    "next.success": { "dtype": "bool", "shape": [1] },
    "index": { "dtype": "int64", "shape": [1] },
    "task_index": { "dtype": "int64", "shape": [1] }
  }
}
```

**Parquet columns per frame** (v0):

* `episode_index:int64`, `frame_index:int64`, `timestamp:float32`
* `observation.state: float32[3]` (pan, tilt, zoom)
* `action: float32[3]` (dpan, dtilt, dzoom)
* `task_index:int64` (0)
* `next.reward:float32` (0.0), `next.done:bool` (false), `next.success:bool` (false)

---

## Runbook (v0)

1. `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
2. `python main.py --video path/to/input.mp4 --out_dir dataset_out --size 256 --fps 24`
3. Drive with WASD + wheel for ~30s, `Space` to end episode, `Esc` to quit
4. Inspect `dataset_out/meta/info.json`, `videos/...mp4`, `data/...parquet`
