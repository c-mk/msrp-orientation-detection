# ChArUco Panel Tilt Detection

Corner-detection + PnP pose-estimation pipeline for measuring ChArUco
board (panel) tilt from photos, per your abstract.

## What was figured out from your sample photos

I didn't have your board's spec sheet, so I reverse-engineered it directly
from the 14 usable photos you uploaded:

- **Dictionary:** `DICT_4X4_100` (found by sweeping every OpenCV ArUco
  dictionary and counting marker detections — marker IDs ran 0-69, i.e.
  70 markers, and 4x4_100 was the smallest dictionary that fit).
- **Grid size:** 10 columns x 14 rows (found by testing every
  `(squaresX, squaresY)` pair whose marker count matches 70, and checking
  which one lets OpenCV successfully interpolate chessboard corners).
- **Legacy pattern:** `True`. Corner interpolation returned 0 corners until
  I set `board.setLegacyPattern(True)` — your board was generated with the
  older marker/corner numbering convention (common with tools like
  calib.io), which OpenCV >= 4.6 needs to be told about explicitly.

These are baked into `src/board_config.py`.

## What you still need to fill in

Open `src/board_config.py` and set the real, measured (with a ruler or
calipers) values:

```python
SQUARE_LENGTH_M = 0.020   # <- measure one square's edge, meters
MARKER_LENGTH_M = 0.015   # <- measure one marker's edge, meters
```

Note: the **rotation angles (tilt) don't actually depend on getting this
exactly right** — rotation is scale-free. But camera calibration quality
and any real-world *distance* numbers will be off until these are correct.

## Validation against your inclinometer

I ran the full pipeline (calibration + pose estimation) on your uploaded
photos as a sanity check. The frame `u6uu.jpg` came back with a
**pitch of ~0.1-0.2 deg**, matching your inclinometer's 0.2 deg reading.
That's a good sign the pipeline's `pitch` angle is the correct "panel
tilt" quantity for your camera/panel geometry — but confirm this against
a few more inclinometer readings at different known angles before trusting
it fully, since only one data point was checked here.

⚠️ Also: the calibration computed from *your* uploaded images had a
low reprojection error (~0.47px) but poorly-conditioned intrinsics
(very large focal length + distortion values). That's because those
photos were all shot from one fixed camera position with the board at
similar distance — good for *validating the pipeline*, but not a proper
calibration set. For deployment-quality accuracy, calibrate with 15-30
images where the board fills different parts of the frame, at varying
distances and angles (see below).

## Files

```
src/board_config.py     - board/dictionary parameters (edit square/marker size)
src/calibrate_camera.py - computes camera intrinsics from a folder of ChArUco photos
src/detect_tilt.py      - detects the board and reports tilt angles
```

## Setup

```bash
pip install opencv-contrib-python numpy
```

## Usage

### 1. Calibrate your camera

Capture 15-30 photos of the board from your camera, varying distance,
position in frame, and angle. Put them in a folder, then:

```bash
python src/calibrate_camera.py --images_dir path/to/calib_photos \
                                --out camera_calibration.npz
```

Aim for RMS reprojection error under ~1.0 px.

### 2. Detect tilt

Single image:
```bash
python src/detect_tilt.py --image board.jpg --calib camera_calibration.npz
```

Batch folder:
```bash
python src/detect_tilt.py --images_dir photos/ --calib camera_calibration.npz
```

With a known-angle reference frame (recommended — e.g. the photo you took
when the inclinometer read 0.2 deg):
```bash
python src/detect_tilt.py --images_dir photos/ --calib camera_calibration.npz \
                           --baseline reference_0.2deg.jpg
```

### Output columns

- **roll / pitch / yaw** — Euler angles (degrees) of the board relative to
  the camera. **Start by comparing `pitch` across frames** — it matched
  your inclinometer reading in testing.
- **vs_camera_axis** — angle between the board's face-normal and the
  camera's optical axis. 0° = board perfectly facing the camera.
- **vs_baseline** *(only with `--baseline`)* — full 3D rotation difference
  from the reference frame. ⚠️ This combines pitch, roll, **and yaw**
  (in-plane spin), so it will look large even for a frame where the panel
  only spun in-plane and didn't actually tilt. For pure tilt tracking,
  compare `pitch` (and `roll` if your mount can lean sideways too) directly
  between frames instead of relying on this combined number.

## Troubleshooting

- **"pose failed (0 corners found)"** — image is blurry, poorly lit, or the
  board isn't a ChArUco board at all (this happens automatically if you
  point it at an unrelated image).
- **Different board?** Use `board_config.autodetect_dictionary()` against a
  sharp, well-lit photo to find the right dictionary, then sweep
  `(squaresX, squaresY)` combinations with `CharucoDetector.detectBoard()`
  the same way I did here — try `legacyPattern` both `True` and `False`.
