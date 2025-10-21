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

* [x] Draw viewport to window (blit)
* [x] Overlay center crosshair
* [x] Tiny text with `pan, tilt, zoom`

---

## 6) Episode logger (LeRobot‑friendly)

* [x] `EpisodeWriter` manages one episode:

  * MP4 muxer for `observation.image` (via `imageio-ffmpeg` @ target FPS)
  * In-memory lists for parquet columns: `episode_index, frame_index, timestamp, observation.state[3], action[3]`
* [x] On `end_episode()`: write Parquet, close MP4

---

## 7) Dataset layout + meta

* [x] Create folder structure:

  * `meta/info.json`
  * `data/chunk-000/episode_{idx:06d}.parquet`
  * `videos/chunk-000/observation.image/episode_{idx:06d}.mp4`
* [x] Write `meta/info.json` template (below) at start; update counts after capture

---

## 8) Main loop wiring

* [x] Orchestrate: read next video frame → render viewport → draw HUD → poll input → **log (state_before, action)** → `apply` deltas → advance
* [x] `Space` starts/ends an episode (can start multiple episodes within one run)

---

## 9) Upgrade output to **LeRobot v3.0** (bare minimum)

> Produce v3.0 layout without fancy chunking: write **one** data file and **one** video file with indices `{chunk_index=0,file_index=0}`; create the required `meta/*` parquet files.

* [ ] **info.json → v3.0**

  * Set `codebase_version = "v3.0"`.
  * Remove: `total_videos`, `total_chunks`.
  * Keep/compute: `total_episodes`, `total_frames`, `total_tasks`.
  * Add: `data_files_size_in_mb` (e.g., 100), `video_files_size_in_mb` (e.g., 500).
  * Paths:

    * `data_path = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"`
    * `video_path = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"`
  * Ensure top-level `fps` is **int** (e.g., 24).
  * For every **non-video** feature (state/action/indices/timestamp/next.*): add `"fps": <int>`.

* [ ] **Writer paths (v3)**

  * Write **data** to: `data/chunk-000/file-000.parquet` (merge all frames of the episode).
  * Write **video** to: `videos/observation.image/chunk-000/file-000.mp4`.
  * Keep column names/dtypes same as v2 for rows (state/action/indices/timestamp/next.*), but file names follow v3.

* [ ] **meta/tasks (required)**

  * File: `meta/tasks/chunk-000/file-000.parquet`.
  * Schema: **index = task string**, column `task_index:int64`.
  * For single-task POC:

    * DataFrame: index `["Center and zoom on the target."]`, column `task_index=[0]`.

* [ ] **meta/episodes (required)**

  * File: `meta/episodes/chunk-000/episodes_000.parquet`.
  * Minimal columns (one row per episode):

    * `episode_index:int64` (0-based)
    * `data/chunk_index:int64` (=0)
    * `data/file_index:int64` (=0)
    * `dataset_from_index:int64` (=0)
    * `dataset_to_index:int64` (=num_frames)
    * For each video key (here `observation.image`):

      * `videos/observation.image/chunk_index:int64` (=0)
      * `videos/observation.image/file_index:int64` (=0)
      * `videos/observation.image/from_timestamp:float32` (=0.0)
      * `videos/observation.image/to_timestamp:float32` (=duration_s)
    * `tasks:list[str]` (e.g., `[
      "Center and zoom on the target."]`)
    * `length:int64` (=num_frames)

* [ ] **meta/episodes_stats (required by v3 tools)**

  * File: `meta/episodes_stats/chunk-000/file-000.parquet`.
  * One row per episode with simple stats. Minimal viable set:

    * `episode_index:int64`
    * `stats/action.mean:float32`, `stats/action.std:float32`, `stats/action.min:float32`, `stats/action.max:float32` (aggregate across action dims)
    * `stats/state.mean:float32`, `stats/state.std:float32`, `stats/state.min:float32`, `stats/state.max:float32`
  * Compute over the episode arrays; cast to float32.

**Acceptance:**

* Hub root shows:

```
meta/
  info.json
  tasks/chunk-000/file-000.parquet
  episodes/chunk-000/episodes_000.parquet
  episodes_stats/chunk-000/file-000.parquet
data/
  chunk-000/file-000.parquet
videos/
  observation.image/chunk-000/file-000.mp4
```

* `meta/info.json` has `codebase_version:"v3.0"`, paths with `{chunk_index,file_index}`, and `fps` fields for non-video features.
* `lerobot_dataset_viz.py --repo-id <repo> --revision v3.0 --episode-index 0` opens the viewer.


## Runbook (v0)

1. `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
2. `python main.py --video path/to/input.mp4 --out_dir dataset_out --size 256 --fps 24`
3. Drive with WASD + wheel for ~30s, `Space` to end episode, `Esc` to quit
4. Inspect `dataset_out/meta/info.json`, `videos/...mp4`, `data/...parquet`
