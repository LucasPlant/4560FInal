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

# Function that runs OpenCV to get one frame and detect ArUco markers
def getPos_singleFrame(K, D):
    
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        exit("Failed to capture")

    # Camera calibration (approximate)
    h, w = frame.shape[:2]
    focal_length = w / (2 * np.tan(np.radians(60) / 2))
    camera_matrix = np.array([[focal_length, 0, w/2],
                            [0, focal_length, h/2],
                            [0, 0, 1]], dtype=np.float64)
    dist_coeffs = np.zeros((5, 1))

    # Detect markers
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(frame)

    if ids is not None:
        marker_length = 0.058  # meters
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, marker_length, camera_matrix, dist_coeffs
        )
        
        # Return position of first detected marker
        x, y, z = tvecs[0].reshape(3)
        # print(f"Marker {ids[0][0]}: x={x:.3f}m, y={y:.3f}m, z={z:.3f}m")
        return np.array([x, y, z])
    
    return None

def main():
    # Sample loop to illustrate usage
    lastPos = getPos_singleFrame()
    while True:
        pos = getPos_singleFrame()
        if pos is not None:
            # Use the detected position
            print(f"Marker Position: x={pos[0]:.3f}m, y={pos[1]:.3f}m, z={pos[2]:.3f}m")
            lastPos = pos
        else:
            # Use last known position or handle no detection
            if lastPos is not None:
                print(f"Last known Marker Position: x={lastPos[0]:.3f}m, y={lastPos[1]:.3f}m, z={lastPos[2]:.3f}m")
            else:
                print("No marker detected and no last known position.")
                
if __name__ == "__main__":
    main()