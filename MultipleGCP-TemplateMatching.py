#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 22:29:26 2019

@author: pankhuri
"""

import cv2
import numpy as np
import glob
import imutils
import csv

#open the template as gray scale image
template = cv2.imread('crop2.JPG', 0)
width, height = template.shape[:2] #get the width and height
filenum = 0

out = [] #array for storing location points
out1= [] #array for storing filename

for filename in glob.glob('*.JPG'):
    filenum += 1
    #open the main image and convert it to gray scale image
    main_image = cv2.imread(filename)
    gray_image = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)

    for scale in np.linspace(0.2, 1.0, 20)[::-1]:
       #resize the image and store the ratio
       resized_img = imutils.resize(gray_image, width = int(gray_image.shape[1] * scale))
       ratio = gray_image.shape[1] / float(resized_img.shape[1])
       if resized_img.shape[0] < height or resized_img.shape[1] < width:
          break
       #Convert to edged image for checking
       #e = cv2.Canny(resized_img, 10, 25)
    
    #match the template using cv2.matchTemplate
       match = cv2.matchTemplate(resized_img, template, cv2.TM_CCOEFF_NORMED)
       threshold = 0.8
       position = np.where(match >= threshold) #get the location of template in the image
       for point in zip(*position[::-1]):
       #draw the rectangle around the matched template
       #cv2.rectangle(main_image, point, (point[0] + width, point[1] + height), (0, 204, 153), 0)
       #print(point[0],point[1])
          a = np.asarray([ point[0], point[1] ])/10 #divided by 10 to avoid closer points for later
          out.append(a)          
          filnum_array = np.array(filename)
          out1.append(filnum_array)
          
    #unique = np.unique(out)*10
    

 
arr_num = np.array(out)
arr_str = np.array(out1)
combined=np.column_stack([arr_str,arr_num])
np.savetxt('gcploc.csv', combined, fmt='{0: ^{1}}'.format("%s", 12), delimiter=',',newline='\n',header ="Filename, GCPLocation (multiply by 10)")    

## making data frame from csv file 
#data = pd.read_csv("gcploc.csv")     
## dropping ALL duplicte values 
#data.drop_duplicates(subset ="First Name",keep = False, inplace = True) 
#  
## displaying data 
with open('gcploc.csv') as f:
  data = list(csv.reader(f))
  new_data = [a for i, a in enumerate(data) if a not in data[:i]]
  with open('finalGCPLOCS.csv', 'w') as t:
     write = csv.writer(t)
     write.writerows(new_data)
