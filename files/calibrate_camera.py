"""
Calibrate the camera's intrinsic parameters (focal length, principal point,
lens distortion) using a folder of ChArUco board photos.

For a GOOD calibration, capture 15-30 images where the board:
  - fills different parts of the frame (corners, edges, center)
  - is shown at different distances
  - is tilted at a variety of angles (not just near-frontal)
All from the SAME camera/lens/focus/zoom setting you'll use for the
real tilt-monitoring shots.

Usage:
    python calibrate_camera.py --images_dir /path/to/calib_photos \
                                --out camera_calibration.npz
"""

import argparse
import glob
import os

import cv2
import numpy as np

from board_config import build_board


def calibrate(images_dir, out_path, min_corners=8):
    board, detector = build_board()

    image_paths = sorted(
        glob.glob(os.path.join(images_dir, "*.jpg"))
        + glob.glob(os.path.join(images_dir, "*.jpeg"))
        + glob.glob(os.path.join(images_dir, "*.png"))
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in {images_dir}")

    all_corners, all_ids = [], []
    img_size = None
    used, skipped = [], []

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            skipped.append(path)
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_size = gray.shape[::-1]  # (width, height)

        ch_corners, ch_ids, _, _ = detector.detectBoard(gray)
        if ch_ids is not None and len(ch_ids) >= min_corners:
            all_corners.append(ch_corners)
            all_ids.append(ch_ids)
            used.append(path)
        else:
            skipped.append(path)

    print(f"Used {len(used)}/{len(image_paths)} images "
          f"(need >= {min_corners} charuco corners per image).")
    if skipped:
        print("Skipped (too few corners or unreadable):")
        for p in skipped:
            print("   ", os.path.basename(p))

    if len(used) < 4:
        raise RuntimeError(
            "Fewer than 4 usable images. Need clearer / more varied shots "
            "of the board for calibration."
        )

    rms, K, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        all_corners, all_ids, board, img_size, None, None
    )

    print(f"\nRMS reprojection error: {rms:.4f} px "
          "(lower is better; < 1.0 is generally good)")
    print("Camera matrix K:\n", K)
    print("Distortion coefficients:\n", dist.ravel())

    np.savez(out_path, K=K, dist=dist, img_size=img_size, rms=rms)
    print(f"\nSaved calibration to {out_path}")
    return K, dist, img_size, rms


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", required=True,
                         help="Folder of ChArUco calibration photos")
    parser.add_argument("--out", default="camera_calibration.npz",
                         help="Output .npz file for camera intrinsics")
    parser.add_argument("--min_corners", type=int, default=8,
                         help="Minimum charuco corners required to use an image")
    args = parser.parse_args()

    calibrate(args.images_dir, args.out, args.min_corners)
