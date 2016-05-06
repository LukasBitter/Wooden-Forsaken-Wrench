
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
    return 0.0

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

def example_algo_return_float(img):
    return 0.0

def example_algo_return__fixed_size_array_float(img):
    return [0.0, 0.0, 0.0] # MUST ALWAYS RETURN THE SAME LIST SIZE

def algo_number_2(img):
    return 0.0

def algo_number_3(img):
    return 0.0

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

def create_vector(img):
    vect = []
    # exemple
    vect.extend(example_algo_return_float(img))
    vect.append(example_algo_return__fixed_size_array_float(img))


    vect.append(algo_number_3(img))
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
    return find_images([find_images])

def find_images(img_array):
    tab_vect = [create_vector(i) for i in img_array]
    input_set = np.array(tab_vect, dtype = np.float32)
    result_set = svm.predict(input_set)
    return[list_categories[int(i)] for i in result_set]

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
    if(args.clean_svm or not isfile(backup_path)):
        print "regenerate the SVM..."
        train_svm()
        svm.save(backup_path)
    elif(isfile(backup_path)):
        svm.load(backup_path)

    # Load the sources pictures
    p = args.picture
    if(isfile(p)):
        img = cv2.imread(args.picture)
        result = find_single_image(img)
        print zip([basename(p)], result)

    elif(isdir(p)):
        list_pictures_name = [f for f in listdir(p) if isfile(join(p, f))]
        list_pictures_data = [cv2.imread(p+"/"+f) for f in list_pictures_name]
        result = find_images(list_pictures_data)
        print zip(list_pictures_name, result)

    else:
        print "Error in source pictures path/file"
