__author__ = 'lukas.bitter', 'Nicloas Gonin', 'Nils Ryter'
# !/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import numpy as np

# Definition of feature postition on the trackbar:
numHarrisCorner = 1
numSIFT = 2
numSURF = 3
numORB = 4

def nothing(x):
    pass

def loop(img):
    src = img.copy()

    # create trackbars for feature selection
    selFeature = 'Feature selection' #'0 : OFF \n1 : HARRIS \n2 : SIFT'
    paramFeature = 'Feature parameter'
    cv2.createTrackbar(selFeature, 'Image',0,4,nothing)
    cv2.createTrackbar(paramFeature, 'Image',0,30,nothing)

    #grayscale image
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    param = None

    while(1):
        cv2.imshow('Image',img)
        k = cv2.waitKey(1) & 0xFF

        if k == 27:
            break

        # get current positions of trackbars
        feature = cv2.getTrackbarPos(selFeature,'Image')
        newParam = cv2.getTrackbarPos(paramFeature,'Image')

        # Detection if parameter trackbar has changed or not
        if newParam != param:
            param = newParam
            current = None

        # Position 0 of feature trackbar => original image
        if feature == 0:
            current = None
            img = src.copy()

        # Position 1 of feature trackbar => Harris Corner detection
        elif (feature == numHarrisCorner and not current == numHarrisCorner):
            current = numHarrisCorner
            img = src.copy()

            # Max param value should be 31
            harrisParam = 31 if 2*param-1 > 30 else 2*param-1
            print('para: ', param, ' / harrisParam: ', harrisParam)

            dst = cv2.cornerHarris(gray,2,harrisParam,0.04)

            #result is dilated for marking the corners, not important
            dst = cv2.dilate(dst,None)

            # Threshold for an optimal value, it may vary depending on the image.
            img[dst>0.01*dst.max()]=[0,0,255]


        # Position 2 of feature trackbar => SIFT detection
        elif (feature == numSIFT and not current == numSIFT):
            current = numSIFT
            img = src.copy()

            gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            sift = cv2.SIFT(param*20)
            kp = sift.detect(gray,None)
            print 'kp len: ', len(kp)

            img=cv2.drawKeypoints(gray,kp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # Position 3 of feature trackbar => SURF detection
        elif (feature == numSURF and not current == numSURF):
            current = numSURF
            img = src.copy()

            gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            surf = cv2.SURF(param*20)
            kp = surf.detect(gray,None)
            print 'kp len: ', len(kp)

            img=cv2.drawKeypoints(gray,kp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


        # Position 4 of feature trackbar => ORB detection
        elif (feature == numORB and not current == numORB):
            current = numORB
            img = src.copy()

            # Initiate STAR detector
            orb = cv2.ORB(param*20)

            # find the keypoints with ORB
            kp = orb.detect(img,None)
            print 'kp len: ', len(kp)

            # compute the descriptors with ORB
            kp, des = orb.compute(img, kp)

            # draw only keypoints location,not size and orientation
            img = cv2.drawKeypoints(img,kp,color=(0,255,0), flags=0)


if __name__ == '__main__':
    img = cv2.imread("leafs/01/RGB/1. Quercus suber/iPAD2_C01_EX01.JPG")
    cv2.namedWindow('Image')

    loop(img)

    cv2.destroyAllWindows()
