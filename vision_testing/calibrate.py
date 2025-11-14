import argparse, cv2, numpy as np, os

def make_charuco(squares_x=7, squares_y=5, square=0.030, marker=0.022, dict_name="DICT_6X6_250"):
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    board = cv2.aruco.CharucoBoard((squares_x, squares_y), square, marker, aruco_dict)
    return aruco_dict, board

def save_params(path, K, D):
    np.savez(path, camera_matrix=K, dist_coeffs=D)
    print("Saved:", path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--frames", type=int, default=40, help="Target number of good frames")
    ap.add_argument("--dict", default="DICT_6X6_250")
    ap.add_argument("--squares-x", type=int, default=7)
    ap.add_argument("--squares-y", type=int, default=5)
    ap.add_argument("--square", type=float, default=0.030, help="square size (m)")
    ap.add_argument("--marker", type=float, default=0.022, help="marker size (m)")
    ap.add_argument("--out", default="charuco_cam_calib.npz")
    args = ap.parse_args()

    aruco_dict, board = make_charuco(args.squares_x, args.squares_y, args.square, args.marker, args.dict)

    # Optional: create a displayable board image to print (PNG)
    board_img = board.generateImage((1200, 800))
    cv2.imwrite("charuco_to_print.png", board_img)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    detector_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)

    all_charuco_corners = []
    all_charuco_ids = []
    imsize = None
    good = 0

    print("Show the printed board at different angles/distances. Press 'c' to capture a frame, 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok: break
        if imsize is None:
            imsize = (frame.shape[1], frame.shape[0])

        corners, ids, _ = detector.detectMarkers(frame)
        vis = frame.copy()
        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(vis, corners, ids)

            # (optional) corner refinement
            cv2.aruco.refineDetectedMarkers(vis, board, corners, ids, None)

            # interpolate charuco corners
            retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                markerCorners=corners,
                markerIds=ids,
                image=frame,
                board=board
            )
            if retval is not None and charuco_corners is not None and len(charuco_corners) > 6:
                cv2.aruco.drawDetectedCornersCharuco(vis, charuco_corners, charuco_ids, (0,255,0))

        cv2.putText(vis, f"Good frames: {good}/{args.frames}  [c]=capture [q]=quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.imshow("ChArUco calibration", vis)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            if ids is not None and len(ids) > 0:
                retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                    corners, ids, frame, board
                )
                if retval is not None and charuco_corners is not None and len(charuco_corners) > 8:
                    all_charuco_corners.append(charuco_corners)
                    all_charuco_ids.append(charuco_ids)
                    good += 1
                    print(f"Captured frame {good}")
                else:
                    print("Not enough ChArUco corners—move/tilt/relight and try again.")
            else:
                print("No markers—reposition and try again.")

            if good >= args.frames:
                print("Enough frames collected.")
                break

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if good < 5:
        print("Not enough frames to calibrate.")
        return

    # Calibrate
    rms, K, D, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        charucoCorners=all_charuco_corners,
        charucoIds=all_charuco_ids,
        board=board,
        imageSize=imsize,
        cameraMatrix=None,
        distCoeffs=None
    )

    print(f"RMS reprojection error: {rms:.4f}")
    print("K=\n", K)
    print("D=\n", D.ravel())
    save_params(args.out, K, D)

if __name__ == "__main__":
    main()
