#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 23:03:38 2019

@author: pankhuri
"""

import imutils
import cv2
import numpy as np
import glob


#Open template and get canny
template = cv2.imread('crop2.JPG')
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
template = cv2.Canny(template, 10, 25)
(height, width) = template.shape[:2]

filenum = 0
out = []
out1= []
for filename in glob.glob('*.JPG'):
    filenum += 1 
    #open the main image and convert it to gray scale image
    main_image = cv2.imread(filename)
    gray_image = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)
    temp_found = None
    #print(filename)
    for scale in np.linspace(0.2, 1.0, 20)[::-1]:
       #resize the image and store the ratio
       resized_img = imutils.resize(gray_image, width = int(gray_image.shape[1] * scale))
       ratio = gray_image.shape[1] / float(resized_img.shape[1])
       if resized_img.shape[0] < height or resized_img.shape[1] < width:
          break
       #Convert to edged image for checking
       e = cv2.Canny(resized_img, 10, 25)
       match = cv2.matchTemplate(e, template, cv2.TM_CCOEFF)
       threshold = 0.8
       position = np.where(match >= threshold)
       #print(position)
       (_, val_max, _, loc_max) = cv2.minMaxLoc(match)
       if temp_found is None or val_max>temp_found[0]:
          temp_found = (val_max, loc_max, ratio)
    
    #Get information from temp_found to compute x,y coordinate
    (_, loc_max, r) = temp_found
    (x_start, y_start) = (int(loc_max[0]), int(loc_max[1]))
    (x_end, y_end) = (int((loc_max[0] + width)), int((loc_max[1] + height)))
    a = np.asarray([ x_end, y_end ])
    #out1.append(filename)  
    filnum_array = np.array(filename)
    out1.append(filnum_array)
    out.append(a)
    #print(a)
    
arr_num = np.array(out)
arr_str = np.array(out1)

combined=np.column_stack([arr_str,arr_num])
np.savetxt('test_file_name.csv', combined, fmt='{0: ^{1}}'.format("%s", 12), delimiter='\t',newline='\n',header ="Filename, GCPLocation")


#np.savetxt("foo2.csv",ab  ,fmt='[%s, %e]', delimiter='\t',newline='\n',header ="Filename, GCPLocation")
   
    #Draw rectangle around the template
    #cv2.rectangle(main_image, (x_start, y_start), (x_end, y_end), (153, 22, 0), 5)
    #cv2.imshow('Template Found', main_image)
    #cv2.waitKey(0)
