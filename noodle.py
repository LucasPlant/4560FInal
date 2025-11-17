import math, time
import numpy as np
import cv2

# def approx_intrinsics(width=640, height=480, hfov_deg=60.0):
#     # crude intrinsics if no calibration: assume pinhole with given HFOV
#     hfov = math.radians(hfov_deg)
#     fx = (width / 2.0) / math.tan(hfov / 2.0)
#     fy = fx
#     cx = width / 2.0
#     cy = height / 2.0
#     K = np.array([[fx, 0, cx],
#                   [0, fy, cy],
#                   [0,  0,  1]], dtype=np.float64)
#     D = np.zeros((5, 1), dtype=np.float64)
#     return K, D


# Configure detector for faster detection (at start of file)
aruco_params = cv2.aruco.DetectorParameters()
aruco_params.adaptiveThreshWinSizeMin = 3
aruco_params.adaptiveThreshWinSizeMax = 23
aruco_params.adaptiveThreshWinSizeStep = 10

g_e_c = np.array([
    [0, 0, 1, 0.0],
    [0, -1, 0, 0.0],
    [1, 0, 0, 0.0],
    [0, 0, 0, 1]
])

# g_a_t = np.array([
#     [0, 0, 1, 0.0],
#     [0, 1, 0, 0.1],
#     [-1, 0, 0, 0.4],
#     [0, 0, 0, 1]
# ])

cap = cv2.VideoCapture(0)
# Attempt to set the desired FPS (e.g., 30 FPS)
requested_fps = 30
cap.set(cv2.CAP_PROP_FPS, requested_fps)
cap.set(cv2.CAP_PROP_EXPOSURE, -6)  # Negative values = shorter exposure (try -4 to -8)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 150)  # Range typically 0-255
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 0.25 = manual mode

# Camera calibration (approximate)
ret, frame = cap.read()
h, w = frame.shape[:2]
focal_length = w / (2 * np.tan(np.radians(60) / 2))
camera_matrix = np.array([[focal_length, 0, w/2],
                        [0, focal_length, h/2],
                        [0, 0, 1]], dtype=np.float64)
dist_coeffs = np.zeros((5, 1))

# Function that runs OpenCV to get one frame and detect ArUco markers
def getPos_singleFrame():
    ret, frame = cap.read()
    frame = cv2.flip(frame, -1)
    # cv2.imshow("Frame", frame)

    if not ret:
        exit("Failed to capture")

    # Detect markers
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    corners, ids, _ = detector.detectMarkers(frame)

    d_e_t = None
    
    if ids is not None:
        # Draw detected markers on the frame
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        
        marker_length = 0.058  # meters
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, marker_length, camera_matrix, dist_coeffs
        )
        
        # Draw 3D axes on each detected marker
        for i in range(len(ids)):
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, 
                            rvecs[i], tvecs[i], marker_length * 0.5)
        
        t_c_a = tvecs[0].reshape(3, 1)

        d_e_a = (g_e_c @ np.block([[t_c_a], [1]]).reshape(4, 1))[0:3]
        d_e_t = d_e_a - np.array([[0.4], [0.0], [0.0]])

        # print("g_e_t:", g_e_t)
    
    # Display the camera feed with annotations
    cv2.imshow("ArUco Detection", frame)
    cv2.waitKey(1)  # Small delay to refresh window
    
    return d_e_t

def main():
    # Sample loop to illustrate usage
    lastPos = getPos_singleFrame()
    while True:
        pos = getPos_singleFrame()
        if pos is not None:
            # Use the detected position
            # print(f"Marker Position: x={pos[0]:.3f}m, y={pos[1]:.3f}m, z={pos[2]:.3f}m")
            lastPos = pos
        else:
            # Use last known position or handle no detection
            if False:
                print(f"Last known Marker Position: x={lastPos[0]:.3f}m, y={lastPos[1]:.3f}m, z={lastPos[2]:.3f}m")
            else:
                print("No marker detected and no last known position.")

    cap.release()
                
if __name__ == "__main__":
    main()