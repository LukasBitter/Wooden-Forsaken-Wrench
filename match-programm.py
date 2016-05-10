
# !/usr/bin/env python
# -*- coding: utf-8 -*-

#==========================================================
#   IMPORT
#==========================================================

import cv2
import argparse
import numpy as np
from matplotlib import pyplot as plt
from os import listdir
from os.path import isfile, join, isdir, basename

#==========================================================
#   GLOABLS VARIABLES
#==========================================================

args = None
svm = None
list_categories = None
backup_path = "./.svm_backup"


#==========================================================
#   IMAGE PREPROCESSING
#
#   all methods in this section have to respect the
#   given parameter signature:
#
#   input   : a image file
#   output  : a image file
#
#==========================================================

def to_gray_scale(img):
    return img #TODO

def get_contours(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)

    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    return cnt

#==========================================================
#   ALGORITHM METHOD
#
#   all methods in this section have to respect the
#   given parameter signature:
#
#   input   : a image file
#   output  : a float
#
#==========================================================

def find_moments(cnt):
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    return [cx, cy]

def find_area(cnt):
    #Contour Area
    area = cv2.contourArea(cnt)
    return [area]

def find_permimeter(cnt):
    #Contour Perimeter
    perimeter = cv2.arcLength(cnt,True)
    return [perimeter]

def find_contour(cnt):
    #Contour Approximation
    epsilon = 0.1*cv2.arcLength(cnt,True)
    return [epsilon]

def find_boundingRect(cnt):
    #Bounding Rectangle
        #Straight Bounding Rectangle
    x,y,w,h = cv2.boundingRect(cnt)
    return [x,y,w,h]

def find_mimimumEnclosingCircle(cnt):
    #Minimum Enclosing Circle
    (x,y),radius = cv2.minEnclosingCircle(cnt)
    return [x,y,radius]

def find_aspectRation(cnt):
    x,y,w,h = cv2.boundingRect(cnt)
    aspect_ratio = float(w)/h
    return [aspect_ratio]

def find_extent(cnt):
    area = cv2.contourArea(cnt)
    x,y,w,h = cv2.boundingRect(cnt)
    rect_area = w*h
    extent = float(area)/rect_area
    return [extent]

def find_solidity(cnt):
    area = cv2.contourArea(cnt)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = float(area)/hull_area
    return [solidity]

#==========================================================
#   CLASSES
#==========================================================

class StatModel(object):
    '''parent class - starting point to add abstraction'''
    def load(self, fn):
        self.model.load(fn)
    def save(self, fn):
        self.model.save(fn)

class SVM(StatModel):
    '''wrapper for OpenCV SimpleVectorMachine algorithm'''
    def __init__(self):
        self.model = cv2.SVM()

    def train(self, samples, responses):
        #setting algorithm parameters
        params = dict( kernel_type = cv2.SVM_LINEAR,
                       svm_type = cv2.SVM_C_SVC,
                       C = 1 )
        self.model.train(samples, responses, params = params)

    def predict(self, samples):
        return np.float32( [self.model.predict(s) for s in samples])

#==========================================================
#   METHODS
#==========================================================

#creation du vecteur de categorisation.
def create_vector(img):
    vect = []

    #Vector based on contours
    cnt = get_contours(img)
    #vect.extend(find_moments(cnt))
    vect.extend(find_area(cnt))
    vect.extend(find_boundingRect(cnt))
    vect.extend(find_contour(cnt))
    vect.extend(find_mimimumEnclosingCircle(cnt))
    vect.extend(find_permimeter(cnt))
    vect.extend(find_aspectRation(cnt))
    #vect.extend(find_solidity(cnt))
    vect.extend(find_extent(cnt))

    return vect

def train_svm():
    #create the vector array of picture and the vector of their categoriy
    tab_vect = []
    tab_cat = []

    for category in list_categories:
        cat_index = list_categories.index(category)
        # List all picture in category
        p = args.categories + "/"+ category
        list_pictures = [f for f in listdir(p) if isfile(join(p, f))]

        #create the vector array of categoy
        for picture in list_pictures:
            img = cv2.imread(p+"/"+picture)
            v = create_vector(img)
            tab_vect.append(v)
            tab_cat.append(cat_index)

    training_set = np.array(tab_vect, dtype = np.float32)
    training_categories = np.array(tab_cat, dtype = np.int)
    svm.train(training_set, training_categories)

def find_single_image(img):
    return find_images([img])

def find_images(img_array):
    tab_vect = [create_vector(i) for i in img_array]
    input_set = np.array(tab_vect, dtype = np.float32)
    result_set = svm.predict(input_set)
    return [list_categories[int(i)] for i in result_set]

#==========================================================
#   MAIN
#==========================================================

if __name__=='__main__':
    parser = argparse.ArgumentParser(prog= 'categorizer', description='Try to categorize an image')
    parser.add_argument('categories', type=str,
                       help='Path to the folder which contains categories folder which contains images which will train the algo (is it clear now?)')
    parser.add_argument('picture', type=str,
                       help='Path of the image to categorize')
    parser.add_argument('--clean-svm', type=bool,
                      help='Errase the SVM data and train it again')
    args = parser.parse_args()

    # List all categories
    p = args.categories
    list_categories = [f for f in listdir(p) if isdir(join(p, f))]

    # Initalization of SVM
    svm = SVM()
    train_svm()
    svm.save(backup_path)
    if(args.clean_svm or not isfile(backup_path)):
        print ("regenerate the SVM...")
        train_svm()
        svm.save(backup_path)
    elif(isfile(backup_path)):
        svm.load(backup_path)

    # Load the sources pictures
    p = args.picture
    if(isfile(p)):
        img = cv2.imread(args.picture)
        result = find_single_image(img)
        print (zip([basename(p)], result))

    elif(isdir(p)):
        list_pictures_name = [f for f in listdir(p) if isfile(join(p, f))]
        list_pictures_data = [cv2.imread(p+"/"+f) for f in list_pictures_name]
        result = find_images(list_pictures_data)
        print(zip(list_pictures_name, result))

    else:
        print("Error in source pictures path/file")
