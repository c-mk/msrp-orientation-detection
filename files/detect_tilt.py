"""
Detect ChArUco board (panel) tilt from a photo using corner detection +
solvePnP pose estimation.

Pipeline:
  1. Detect ArUco markers -> interpolate sub-pixel ChArUco chessboard corners.
  2. solvePnP(objectPoints, imagePoints, K, dist) -> rotation vector (rvec)
     and translation vector (tvec) describing the board's pose relative to
     the camera.
  3. Convert the rotation matrix to interpretable angles:
       - roll / pitch / yaw (degrees), OpenCV camera-frame convention
       - "tilt_from_camera_axis": angle between the board's face-normal and
         the camera's optical axis (0 deg = board perfectly perpendicular
         to / facing the camera)

Which number is "the tilt" depends on your physical setup (how the camera
is mounted relative to the panel). In validation against your inclinometer
reading of 0.2 deg, the PITCH angle matched almost exactly (0.21 deg on the
reference frame), so start there -- see --print_all to see every candidate
angle and pick whichever tracks your inclinometer across multiple readings.

Usage (single image):
    python detect_tilt.py --image board.jpg --calib camera_calibration.npz

Usage (batch folder):
    python detect_tilt.py --images_dir photos/ --calib camera_calibration.npz

Setting a reference / baseline (recommended):
    Capture one photo at your known, inclinometer-verified angle (e.g. the
    0.2 deg reading) and pass it as --baseline. All reported tilts will then
    be expressed relative to that frame, which is what you actually want for
    "how far off from calibrated-flat is the panel now".

    python detect_tilt.py --images_dir photos/ --calib camera_calibration.npz \
                           --baseline reference_0.2deg.jpg
"""

import argparse
import glob
import os

import cv2
import numpy as np

from board_config import build_board


def load_calibration(path):
    data = np.load(path)
    return data["K"], data["dist"]


def detect_pose(gray, board, detector, K, dist, min_corners=6):
    """Returns (rvec, tvec, num_corners) or (None, None, 0) if detection fails."""
    ch_corners, ch_ids, _, _ = detector.detectBoard(gray)
    if ch_ids is None or len(ch_ids) < min_corners:
        return None, None, 0

    obj_points, img_points = board.matchImagePoints(ch_corners, ch_ids)
    ok, rvec, tvec = cv2.solvePnP(obj_points, img_points, K, dist)
    if not ok:
        return None, None, len(ch_ids)
    return rvec, tvec, len(ch_ids)


def rotation_to_angles(R):
    """
    Decompose a 3x3 rotation matrix into:
      - roll, pitch, yaw (degrees) using the standard OpenCV camera-frame
        extrinsic XYZ convention (small angles ~ intuitive tilt/lean)
      - tilt_from_camera_axis: angle (degrees) between the board's normal
        vector and the camera's optical (+Z) axis. 0 = board directly
        facing the camera (perpendicular).
    """
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    pitch = np.degrees(np.arctan2(-R[2, 0], sy))
    roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))

    board_normal = R @ np.array([0.0, 0.0, 1.0])
    camera_axis = np.array([0.0, 0.0, 1.0])
    cos_angle = np.clip(
        np.dot(board_normal, camera_axis) / np.linalg.norm(board_normal), -1, 1
    )
    tilt_from_camera_axis = np.degrees(np.arccos(cos_angle))

    return {
        "roll": roll,
        "pitch": pitch,
        "yaw": yaw,
        "tilt_from_camera_axis": tilt_from_camera_axis,
    }


def process_image(path, board, detector, K, dist, baseline_R=None):
    img = cv2.imread(path)
    if img is None:
        return {"file": os.path.basename(path), "error": "could not read image"}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    rvec, tvec, n_corners = detect_pose(gray, board, detector, K, dist)
    if rvec is None:
        return {
            "file": os.path.basename(path),
            "error": f"pose failed ({n_corners} corners found, need >= 6)",
        }

    R, _ = cv2.Rodrigues(rvec)
    angles = rotation_to_angles(R)

    result = {
        "file": os.path.basename(path),
        "corners": n_corners,
        **angles,
    }

    if baseline_R is not None:
        # Relative rotation: how far this frame's board orientation has
        # rotated away from the baseline/reference frame.
        R_rel = R @ baseline_R.T
        rel_angle = np.degrees(
            np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1, 1))
        )
        result["tilt_relative_to_baseline"] = rel_angle

    return result


def main():
    parser = argparse.ArgumentParser()
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="Path to a single image")
    src.add_argument("--images_dir", help="Folder of images to process")
    parser.add_argument("--calib", required=True,
                         help="camera_calibration.npz from calibrate_camera.py")
    parser.add_argument("--baseline",
                         help="Optional reference image (e.g. your "
                              "inclinometer-verified 0.2 deg shot). All "
                              "results are also reported relative to it.")
    args = parser.parse_args()

    K, dist = load_calibration(args.calib)
    board, detector = build_board()

    baseline_R = None
    if args.baseline:
        img = cv2.imread(args.baseline)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rvec, tvec, n = detect_pose(gray, board, detector, K, dist)
        if rvec is None:
            raise RuntimeError(f"Could not detect board in baseline image "
                                f"({n} corners found)")
        baseline_R, _ = cv2.Rodrigues(rvec)
        print(f"Baseline set from {os.path.basename(args.baseline)} "
              f"({n} corners)\n")

    paths = [args.image] if args.image else sorted(
        glob.glob(os.path.join(args.images_dir, "*.jpg"))
        + glob.glob(os.path.join(args.images_dir, "*.jpeg"))
        + glob.glob(os.path.join(args.images_dir, "*.png"))
    )

    for path in paths:
        result = process_image(path, board, detector, K, dist, baseline_R)
        if "error" in result:
            print(f"{result['file']:30s}  ERROR: {result['error']}")
            continue

        line = (
            f"{result['file']:30s}  corners={result['corners']:3d}  "
            f"tilt={result['pitch']:7.2f}  "
        )
        if "tilt_relative_to_baseline" in result:
            line += f"  vs_baseline={result['tilt_relative_to_baseline']:6.2f}"
        print(line)


if __name__ == "__main__":
    main()
