__author__ = 'lukas.bitter'
# !/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import numpy as np
from matplotlib import pyplot as plt

def decompose(img):
    mask = 0x01
    imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for i in range(1, 9):
        res = cv2.bitwise_and(imgGrey, mask)
        name = 'A) and -' + str(bin(mask))
        imgnew = cv2.normalize(res, None, 0, 255, cv2.NORM_MINMAX)
        cv2.imshow(name, imgnew)
        mask <<= 1

# find Harris corners
def harrisCorner(img, gray):
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,2,3,0.04)
    dst = cv2.dilate(dst,None)
    ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
    dst = np.uint8(dst)
    cv2.imshow('dst',img)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()


def nothing(x):
    pass

if __name__ == '__main__':
    img = cv2.imread("leafs/01/RGB/1. Quercus suber/iPAD2_C01_EX01.JPG")
    cv2.namedWindow('image')

    # create trackbars for color change
    cv2.createTrackbar('R','image',0,255,nothing)

    # create switch for ON/OFF functionality
    switch = '0 : OFF \n1 : ON'
    cv2.createTrackbar(switch, 'image',0,1,nothing)

    #cv2.imshow("Image", img)
    b,g,r = cv2.split(img)
    img2 = cv2.merge([r,g,b])

    plt.show()
    k = cv2.waitKey(0) & 0xFF

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,2,3,0.04)

    #result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None)

    # Threshold for an optimal value, it may vary depending on the image.
    img[dst>0.01*dst.max()]=[0,0,255]

    cv2.imshow('dst',img)

    k = cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()