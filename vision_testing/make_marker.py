# save as make_marker.py
import argparse, cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--dict", default="6X6_250")
ap.add_argument("--id", type=int, default=0)
ap.add_argument("--pixels", type=int, default=800)
ap.add_argument("--out", default="aruco_6x6_250_id0.png")
args = ap.parse_args()

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
aruco_dict = cv2.aruco.getPredefinedDictionary(mapping[args.dict.upper()])
img = cv2.aruco.generateImageMarker(aruco_dict, args.id, args.pixels)
# add a white border for screen/printing
border = 50
canvas = 255*np.ones((args.pixels+2*border, args.pixels+2*border), dtype=np.uint8)
canvas[border:-border, border:-border] = img
cv2.imwrite(args.out, canvas)
print("wrote", args.out)
