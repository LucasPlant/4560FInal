import argparse, time, math, sys, os
import numpy as np
import cv2

def load_calibration(path):
    if path is None: 
        return None, None
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        data = np.load(path)
        return data["camera_matrix"], data["dist_coeffs"]
    # try YAML
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise IOError(f"Could not open calibration file: {path}")
    K = fs.getNode("camera_matrix").mat()
    D = fs.getNode("dist_coeffs").mat()
    fs.release()
    return K, D

def approx_intrinsics(width, height, hfov_deg=60.0):
    # crude intrinsics if no calibration: assume pinhole with given HFOV
    hfov = math.radians(hfov_deg)
    fx = (width / 2.0) / math.tan(hfov / 2.0)
    fy = fx
    cx = width / 2.0
    cy = height / 2.0
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]], dtype=np.float64)
    D = np.zeros((5, 1), dtype=np.float64)
    return K, D

def rvec_to_euler_zyx(rvec):
    # Returns roll (x), pitch (y), yaw (z) in radians using ZYX intrinsic order
    R, _ = cv2.Rodrigues(rvec)
    # ZYX: yaw-pitch-roll
    sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
    singular = sy < 1e-6
    if not singular:
        yaw   = math.atan2(R[1,0], R[0,0])
        pitch = math.atan2(-R[2,0], sy)
        roll  = math.atan2(R[2,1], R[2,2])
    else:
        yaw   = math.atan2(-R[0,1], R[1,1])
        pitch = math.atan2(-R[2,0], sy)
        roll  = 0.0
    return roll, pitch, yaw

def get_aruco_dict(name):
    # Accept friendly names
    name = name.upper()
    mapping = {
        "4X4_50": cv2.aruco.DICT_4X4_50,
        "4X4_100": cv2.aruco.DICT_4X4_100,
        "5X5_50": cv2.aruco.DICT_5X5_50,
        "5X5_100": cv2.aruco.DICT_5X5_100,
        "6X6_50": cv2.aruco.DICT_6X6_50,
        "6X6_100": cv2.aruco.DICT_6X6_100,
        "6X6_250": cv2.aruco.DICT_6X6_250,
        "APRILTAG_36H11": cv2.aruco.DICT_APRILTAG_36h11,
    }
    key = name if name in mapping else "6X6_250"
    return cv2.aruco.getPredefinedDictionary(mapping[key])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    ap.add_argument("--video", type=str, default="", help="Video file or stream URL")
    ap.add_argument("--dict", type=str, default="6X6_250", help="ArUco dictionary, e.g. 4X4_50, 6X6_250, APRILTAG_36H11")
    ap.add_argument("--marker-length", type=float, default=0.05, help="Marker side length in meters")
    ap.add_argument("--calib", type=str, default=None, help="Calibration file (.npz or .yml/.yaml) with camera_matrix and dist_coeffs")
    ap.add_argument("--axis-scale", type=float, default=1.0, help="Axis draw length as a multiple of marker side length")
    ap.add_argument("--print-rate", type=float, default=10.0, help="Console print rate (Hz)")
    args = ap.parse_args()

    # video source
    src = args.video if args.video else args.camera
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print("ERROR: could not open video source")
        sys.exit(1)

    # Read one frame to get dimensions
    ok, frame = cap.read()
    if not ok:
        print("ERROR: could not read from source")
        sys.exit(1)
    H, W = frame.shape[:2]

    # Load/approx intrinsics
    K, D = load_calibration(args.calib)
    if K is None:
        K, D = approx_intrinsics(W, H)  # quick start path
        print("[warn] Using approximate intrinsics (assume ~60° HFOV, zero distortion). Distances will be approximate.")

    # ArUco setup (handle newer OpenCV API if available)
    aruco_dict = get_aruco_dict(args.dict)
    # Newer API: ArucoDetector
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    use_new = True

    print_interval = 1.0 / max(args.print_rate, 1e-6)
    last_print = 0.0
    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if use_new:
            corners, ids, _ = detector.detectMarkers(frame)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

        if ids is not None and len(ids) > 0:
            # draw detected markers
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            # estimate pose for each marker
            # estimatePoseSingleMarkers expects side length in meters
            rvecs, tvecs, _obj = cv2.aruco.estimatePoseSingleMarkers(
                corners, args.marker_length, K, D
            )

            for rvec, tvec, mid in zip(rvecs, tvecs, ids.flatten()):
                # draw axis
                axis_len = args.marker_length * args.axis_scale
                cv2.drawFrameAxes(frame, K, D, rvec, tvec, axis_len)

                # Compute yaw/pitch/roll for display
                roll, pitch, yaw = rvec_to_euler_zyx(rvec.reshape(3))
                tx, ty, tz = tvec.reshape(3)

                # HUD text near the marker
                c = np.mean(corners[0].reshape(-1,2), axis=0).astype(int)
                line1 = f"ID {mid}  z:{tz:.3f}m  x:{tx:.3f}  y:{ty:.3f}"
                line2 = f"rpy(deg): {math.degrees(roll):.1f}, {math.degrees(pitch):.1f}, {math.degrees(yaw):.1f}"
                y0 = max(20, c[1]-20)
                cv2.putText(frame, line1, (10, y0), font, 0.55, (0,0,0), 3, cv2.LINE_AA)
                cv2.putText(frame, line1, (10, y0), font, 0.55, (255,255,255), 1, cv2.LINE_AA)
                cv2.putText(frame, line2, (10, y0+22), font, 0.55, (0,0,0), 3, cv2.LINE_AA)
                cv2.putText(frame, line2, (10, y0+22), font, 0.55, (255,255,255), 1, cv2.LINE_AA)

            # Throttle console prints
            now = time.time()
            if now - last_print >= print_interval:
                # print the first marker's pose (typical)
                rvec, tvec = rvecs[0].reshape(3), tvecs[0].reshape(3)
                roll, pitch, yaw = rvec_to_euler_zyx(rvec)
                print(
                    f"ID {ids[0][0]} | t = [{tvec[0]: .3f}, {tvec[1]: .3f}, {tvec[2]: .3f}] m | "
                    f"rpy(deg) = [{math.degrees(roll): .1f}, {math.degrees(pitch): .1f}, {math.degrees(yaw): .1f}]"
                )
                last_print = now

        # FPS overlay
        cv2.putText(frame, "Press q to quit", (10, H-10), font, 0.6, (255,255,255), 2, cv2.LINE_AA)
        cv2.imshow("ArUco Pose", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
