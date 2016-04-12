__author__ = 'lukas.bitter'
# !/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import numpy as np

def decompose(img):
    mask = 0x01
    imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for i in range(1, 9):
        res = cv2.bitwise_and(imgGrey, mask)
        name = 'A) and -' + str(bin(mask))
        imgnew = cv2.normalize(res, None, 0, 255, cv2.NORM_MINMAX)
        cv2.imshow(name, imgnew)
        mask <<= 1


if __name__ == '__main__':
    img = cv2.imread("leafs/01/RGB/1. Quercus suber/iPAD2_C01_EX01.JPG")
    cv2.imshow("Image", img)
    k = cv2.waitKey(0) & 0xFF
    print(k)
    while k != 27:
        if k == 100: # d
            print("d")
            decompose(img)
        k = cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()