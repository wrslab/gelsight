import cv2
import numpy as np
from calibrate_gelsight import takeimg, fishye_calib, imgborder
import pickle

camname = "cam2"
takeimg("fisheye/"+camname, 0, 0, "1tst")
para = pickle.load(open("cam2/cam2_calib.pkl", "rb"))
img = cv2.imread("fisheye/1tst.jpg")
border = imgborder(img, 1, campara=para)
print(border)
