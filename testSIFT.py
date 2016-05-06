import numpy as np
import cv2, sys, os
from matplotlib import pyplot as plt
from os import listdir
from os.path import isfile, join, isdir

# Initiate SIFT detector
orb = cv2.SIFT()
bf = cv2.BFMatcher()

#==========================================================
#   CALCULE DES SIFT POUR L'INPUT
#==========================================================

# SIFT sur le fichier source
source = cv2.imread("./leafs/01/RGB/3. Populus nigra/iPAD2_C03_EX03.jpg")
source = cv2.cvtColor(source,cv2.COLOR_BGR2GRAY)
source = np.float32(source)
kpSource, desSource = orb.detectAndCompute(source,None)

#==========================================================
#   GROSSE BOUCLE MOCHE
#==========================================================

model_path = "./test/"
# Get all sample of category
file_in_model_dir = [f for f in listdir(model_path) if isfile(join(model_path, f))]

count = 0
list_sift = []
for file_name in file_in_model_dir:
    print "image no ", count+1
    print "analysis the file : ",model_path+file_name

    # SIFT sur le fichier X
    model = cv2.imread(model_path+file_name,0)
    model = cv2.cvtColor(model ,cv2.COLOR_BGR2GRAY)
    model = np.float32(model)
    kpX, desX = orb.detectAndCompute(model,None)

    # Comparaison entre la source et X
    matches = bf.knnMatch(desSource,desX, k=2)
    good = [] # liste des point cle en commun
    for list_m in matches:
        previous_distance = None
        for distance in list_m:
            if previous_distance != None:
                if previous_distance.distance < 0.75*distance.distance:
                    good.append([previous_distance])

            previous_distance = distance

    count = count+1
    print "matching level : ", len(good) # plus y a de match positif, mieux c'est
