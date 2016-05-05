__author__ = 'lukas.bitter'
# !/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import numpy as np
from matplotlib import pyplot as plt

# find Harris corners
def harrisCorner(img, gray):
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,2,3,0.04)
    dst = cv2.dilate(dst,None)
    ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
    dst = np.uint8(dst)
    #cv2.imshow('dst',img)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()

def findSift(img):
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT()
    kp = sift.detect(gray,None)
    img=cv2.drawKeypoints(gray,kp)
    cv2.imwrite('sift_keypoints.jpg',img)

def nothing(x):
    pass

def loop(img):
    src = img

    # create switch for ON/OFF harris functionality
    harris = '0 : OFF \n1 : ON'
    cv2.createTrackbar(harris, 'image',0,5,nothing)

    while(1):
        cv2.imshow('image',img)
        k = cv2.waitKey(0) & 0xFF
        if k == 27:
            break

        # get current positions of four trackbars
        corner = cv2.getTrackbarPos(harris,'image')

        #grayscale image
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)

        if corner == 1:
            dst = cv2.cornerHarris(gray,2,3,0.04)

            #result is dilated for marking the corners, not important
            dst = cv2.dilate(dst,None)

            # Threshold for an optimal value, it may vary depending on the image.
            img[dst>0.01*dst.max()]=[0,0,255]

        elif corner == 2:
            #sift = cv2.SIFT_create()
            sift = cv2.SIFT()
            kp = sift.detect(gray,None)

            img=cv2.drawKeypoints(gray,kp)
            msg = 'sift_keypoints.jpg'

            cv2.imwrite('sift_keypoints.jpg',img)
        else:
            img = src;

if __name__ == '__main__':
    img = cv2.imread("leafs/01/RGB/1. Quercus suber/iPAD2_C01_EX01.JPG")
    cv2.namedWindow('image')

#    cv2.imshow("Image", img)
    loop(img)

    cv2.destroyAllWindows()
