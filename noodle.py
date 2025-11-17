import math, time
import numpy as np
import cv2
import threading

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

# Detection configuration: scale image down for faster detection (0.5 = half size)
# Set to 1.0 to disable scaling
DETECTION_SCALE = 0.5
# Set to True to draw axes and detected markers (slower)
DRAW_ANNOTATIONS = True

# Reuse dictionary and detector objects instead of recreating them every frame
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# Simple timing for profiling
_frame_count = 0
_accum_time = 0.0

g_e_c = np.array([
    [0, 0, 1, 0.0],
    [0, -1, 0, 0.0],
    [1, 0, 0, 0.0],
    [0, 0, 0, 1]
])

def Rx(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c],
    ])

g_e_c = g_e_c

R_a_t = np.array([
    [0, 0, -1],
    [0, 1, 0],
    [1, 0, 0],
])

class CameraStream:
    def __init__(self, src=0, backend=cv2.CAP_ANY):
        # On macOS you can try cv2.CAP_AVFOUNDATION
        self.cap = cv2.VideoCapture(src, backend)
        self.lock = threading.Lock()
        self.frame = None
        self.stopped = False
        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()

    def _update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                # small sleep to avoid busy loop if camera unavailable
                time.sleep(0.01)
                continue
            # store a copy of the latest frame
            with self.lock:
                self.frame = frame.copy()

    def read(self):
        # returns the latest frame (or None)
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def stop(self):
        self.stopped = True
        self._thread.join(timeout=1.0)
        try:
            self.cap.release()
        except Exception:
            pass
    
    def set(self, prop_id, value):
        self.cap.set(prop_id, value)

cap = CameraStream(0)
# Attempt to set the desired FPS (e.g., 30 FPS)
requested_fps = 30
cap.set(cv2.CAP_PROP_FPS, requested_fps)
cap.set(cv2.CAP_PROP_EXPOSURE, -6)  # Negative values = shorter exposure (try -4 to -8)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 150)  # Range typically 0-255
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 0.25 = manual mode
time.sleep(0.1)
# Camera calibration (approximate)
frame = cap.read()
h, w = frame.shape[:2]
focal_length = w / (2 * np.tan(np.radians(60) / 2))
camera_matrix = np.array([[focal_length, 0, w/2],
                        [0, focal_length, h/2],
                        [0, 0, 1]], dtype=np.float64)
dist_coeffs = np.zeros((5, 1))

# Function that runs OpenCV to get one frame and detect ArUco markers
def getPos_singleFrame():
    global _frame_count, _accum_time
    t0 = time.time()

    frame = cap.read()

    frame = cv2.flip(frame, -1)

    # Convert to grayscale and optionally resize for faster detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if DETECTION_SCALE != 1.0:
        small = cv2.resize(gray, None, fx=DETECTION_SCALE, fy=DETECTION_SCALE, interpolation=cv2.INTER_LINEAR)
    else:
        small = gray

    # Detect markers on the (smaller) grayscale image
    corners_small, ids, _ = detector.detectMarkers(small)

    d_e_t = None
    R_e_a = None

    if ids is not None:
        # Scale corners back to original image coordinates if we used a downscale
        if DETECTION_SCALE != 1.0 and len(corners_small) > 0:
            # produce a list of corner arrays in original image coordinates
            corners = [ (np.array(c).astype(np.float32) / DETECTION_SCALE) for c in corners_small ]
        else:
            # ensure corners is a list of arrays (same shape as detectMarkers output)
            corners = [ np.array(c).astype(np.float32) for c in corners_small ]

        if DRAW_ANNOTATIONS:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        marker_length = 0.058  # meters
        # estimatePoseSingleMarkers expects corners in the original image coordinates
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, marker_length, camera_matrix, dist_coeffs
        )

        # Optionally draw 3D axes on each detected marker (drawing is slower)
        if DRAW_ANNOTATIONS:
            for i in range(len(ids)):
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs,
                                  rvecs[i], tvecs[i], marker_length * 0.5)

        t_c_a = tvecs[0].reshape(3, 1)
        d_e_a = (g_e_c @ np.block([[t_c_a], [1]]).reshape(4, 1))[0:3]
        # d_e_t = d_e_a - np.array([[0.3], [0.0], [0.0]])
        d_e_t = d_e_a

        R_a_c = cv2.Rodrigues(rvecs[0])[0]
        R_c_t = R_a_c.T @ R_a_t
        R_e_a = g_e_c[0:3, 0:3] @ R_c_t

    # Display the camera feed with annotations (non-blocking)
    cv2.imshow("ArUco Detection", frame)
    cv2.waitKey(1)

    # timing / fps printing
    elapsed = time.time() - t0
    _frame_count += 1
    _accum_time += elapsed
    if _frame_count % 30 == 0:
        avg = _accum_time / 30.0
        print(f"avg frame time (last 30): {avg*1000:.1f} ms -> {1.0/avg:.1f} FPS")
        _accum_time = 0.0

    return d_e_t

    return np.block([[R_e_a, d_e_t], [0, 0, 0, 1]]) if d_e_t is not None and R_e_a is not None else None

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