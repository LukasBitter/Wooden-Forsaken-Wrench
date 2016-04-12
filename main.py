__author__ = 'lukas.bitter'

# !/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import numpy as np

def normalize(img):
    imgnew = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imshow("Image normalized", imgnew)


if __name__ == '__main__':
    img = cv2.imread("Unequalized_Hawkes_Bay_NZ.bmp", cv2.IMREAD_GRAYSCALE  )
    cv2.imshow("Image", img)
    k = cv2.waitKey(0) & 0xFF
    print(k)
    while k != 27:
        if k == 110: # n
            print("n")
            normalize(img)
        #cv2.imshow("Image normalized", img)
        k = cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()