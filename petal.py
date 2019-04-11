#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 18:20:13 2019

@author: pankhuri
"""
import cv2
import numpy as np
import glob

#def bfmatch(des1, des2):
#    '''Brute force matcher for ORB features'''
#    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#    
#    n=0
#    matches = 0
#    
#    # Do matching
#    try:
#        matches = bf.knnmatch(des1,des2,k=3)
#        matches = sorted(matches, key=lambda val: val.distance)
#        n = len(matches)
#        print('Matches =', n)
#    except:
#        print('No features in one image')
#
#    return matches            
     
def pre_pro(image):
    '''Perform image processing for enhancing the features of the images '''
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #improving the contrast of image
    gray = cv2.equalizeHist(gray)
    #sharpening the image
    org_fil = cv2.bilateralFilter(gray, 7, 75, 75)
    k = np.array([[-1, -1, -1],[-1, 9,-1], [-1, -1, -1]])
    org_fil = cv2.filter2D(org_fil, -1, k) 
    return org_fil    
    
        
def detect(image):
    '''
        Implementation of ORB feature detector and descriptor 
        The program takes colored image and return keypoint and descriptor matrix        
    '''    
    
    #preprossing image for final detection
    gray = pre_pro(image)
    orb = cv2.ORB_create(1000,1.2)    
    # Determining key points
    keypoints = orb.detect(gray, None)   
    # Obtaining the descriptors
    keypoints, descriptors = orb.compute(gray, keypoints)
    #print("Number of keypoints Detected: ", len(keypoints))    
    return keypoints, descriptors

if __name__ == "__main__": 
    #reading template image 
    template = cv2.imread('crop1.JPG')
    #doubling the size and detecting features using ORB method
    template = cv2.pyrUp(template)
    key1, des1 = detect(template) 
    
    filenum = 0
    #reading other images in folder, after feature extraction matching with template features
    for filename in glob.glob('*.JPG'):
        #count of all files
        filenum += 1    
            
        original = cv2.imread(filename)
        #original = cv2.imread('DSC01453.JPG')     
        key2, des2 = detect(original)
    
        #matching using descriptor with brute force method        
        #print(filename,", Features matched =", n)
        bf = cv2.BFMatcher(2)
        matches = bf.knnMatch(des1,des2, k=1)
        #print(filename,", Features matched =", n)
        #-- Filter matches using the Lowe's ratio test
        ratio_thresh = 0.75
        good_matches = []
        for m,n in matches:
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)
                print(filename)
                print(good_matches)
        # get the corners from the first image (the object to be "detected")
#        [h,w,~] = size(template)
#        corners = [0 0; w 0; w h; 0 h]
#        display(corners)
#
#        #% apply the homography to the corner points of the box
#        p = cv.perspectiveTransform(corners, H);
#        display(p)
        
#'''Uncomment the following lines for reviewing matched features'''        
#        #stacking images side by side and drawing matched features
#        img3 = cv2.drawMatches(template, key1, original, key2, matches_pt, None, flags=2)
#        img3 = cv2.resize(img3, (960, 640))
#        
#        cv2.imshow('matches', img3)
#        cv2.waitKey()
#        cv2.destroyAllWindows()