"""
calibrate.py — GuideSense Camera Focal Length Calibration Tool
==============================================================
One-time interactive calibration script to determine camera focal length in pixels (FOCAL_LENGTH_PX).

Usage:
  python3 calibrate.py [--height 170.0] [--distance 200.0] [--camera 0]

Steps:
  1. Position an object of known height (e.g. standing person: 170cm) at a known distance (e.g. 200cm).
  2. Run the script. A live webcam window will open at 640x480 resolution.
  3. Press SPACE or ENTER to freeze the frame for bounding box selection.
  4. Draw a bounding box around the target object (from top to bottom) and press SPACE or ENTER to confirm.
  5. The script calculates and displays FOCAL_LENGTH_PX using the pinhole camera formula:
        FOCAL_LENGTH_PX = (bbox_height_px * known_distance_cm) / known_height_cm
  6. Copy the resulting FOCAL_LENGTH_PX value into distance.py.
"""

import argparse
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="GuideSense Webcam Focal Length Calibration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--height",
        type=float,
        default=170.0,
        help="Known real-world height of the calibration object in cm (default: 170.0 for person)",
    )
    parser.add_argument(
        "--distance",
        type=float,
        default=200.0,
        help="Known distance from camera to the calibration object in cm (default: 200.0)",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam device index (default: 0)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        import cv2
    except ImportError:
        print("[ERROR] OpenCV (cv2) is not installed.")
        print("Please install it using: pip install opencv-python")
        sys.exit(1)

    print("=" * 60)
    print("GuideSense — Camera Calibration Tool")
    print("=" * 60)
    print(f"Target object height : {args.height:.1f} cm")
    print(f"Target distance      : {args.distance:.1f} cm")
    print(f"Camera index         : {args.camera}")
    print("-" * 60)
    print("Opening webcam...")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] Could not open webcam at index {args.camera}.")
        sys.exit(1)

    # Set resolution to 640x480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    window_name = "GuideSense Calibration - Live Preview"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    print("\n[INSTRUCTIONS]")
    print("1. Align the known-height object in the camera frame.")
    print("2. Press [SPACE] or [ENTER] to capture the frame for measurement.")
    print("3. Press [Q] or [ESC] to quit.\n")

    captured_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame from webcam.")
            break

        # Display helper overlay
        display_frame = frame.copy()
        cv2.putText(
            display_frame,
            "Press SPACE / ENTER to capture",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            display_frame,
            f"Object: {args.height:.0f}cm @ {args.distance:.0f}cm",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )

        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key in (32, 13):  # SPACE or ENTER
            captured_frame = frame.copy()
            break
        elif key in (27, ord('q'), ord('Q')):  # ESC or Q
            print("[INFO] Calibration cancelled by user.")
            cap.release()
            cv2.destroyAllWindows()
            return

    cap.release()
    cv2.destroyWindow(window_name)

    if captured_frame is None:
        print("[ERROR] No frame captured.")
        return

    print("\n[ROI SELECTION]")
    print("Select the bounding box around the known-height object:")
    print(" - Click and drag to draw a box.")
    print(" - Press [SPACE] or [ENTER] to confirm.")
    print(" - Press [C] to cancel selection.\n")

    roi_window = "GuideSense Calibration - Draw Bounding Box"
    roi = cv2.selectROI(roi_window, captured_frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(roi_window)

    x, y, w, h = roi
    bbox_height_px = h

    if bbox_height_px <= 0 or w <= 0:
        print("[WARNING] Invalid bounding box selected (height <= 0). Calibration aborted.")
        return

    # Compute focal length in pixels:
    # focal_length_px = (bbox_height_px * known_distance_cm) / known_height_cm
    focal_length_px = (bbox_height_px * args.distance) / args.height

    print("\n" + "=" * 60)
    print("CALIBRATION SUCCESSFUL")
    print("=" * 60)
    print(f"Known Object Height : {args.height:.1f} cm")
    print(f"Known Distance      : {args.distance:.1f} cm")
    print(f"Bounding Box Height : {bbox_height_px} px (width: {w} px)")
    print("-" * 60)
    print(f"CALIBRATED FOCAL_LENGTH_PX = {focal_length_px:.2f}")
    print("-" * 60)
    print("To apply this calibration, update `distance.py`:")
    print(f"    FOCAL_LENGTH_PX: float = {focal_length_px:.1f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
