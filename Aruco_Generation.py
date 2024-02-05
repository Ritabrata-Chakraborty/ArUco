import numpy as np
import cv2
import sys
import os

ARUCO_DICT = {
	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
	"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
	"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
	"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
	"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

aruco_type = sys.argv[2]
marker_id = int(sys.argv[1])

window_size = 650
padding = 1

arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_type])

print("ArUCo type '{}' with ID '{}'".format(aruco_type, marker_id))

# Creates an empty image (tag) to draw the ArUco marker on. The image is initialized with zeros.
tag = np.zeros((window_size, window_size, 1), dtype="uint8")
cv2.aruco.generateImageMarker(arucoDict, marker_id, window_size, tag, padding)

border = 15

tag = cv2.copyMakeBorder(tag, border, border, border, border, cv2.BORDER_CONSTANT, value=255)

tag_name = "arucoMarkers/" + aruco_type + "_" + str(marker_id) + ".png"

if not os.path.exists("arucoMarkers/"):
    os.makedirs("arucoMarkers/")

cv2.imwrite(tag_name, tag)
cv2.imshow("ArUCo Tag: {} {}".format(aruco_type, marker_id), tag)

cv2.waitKey(0)
cv2.destroyAllWindows()