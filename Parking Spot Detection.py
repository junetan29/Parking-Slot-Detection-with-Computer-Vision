#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install cvzone


# In[ ]:


import cv2
import pickle

try:
    with open('Parking Position text file path/CarParkPos.txt','rb') as fp:
        position=pickle.load(fp)
except:
    position=[]

# The widthand height of the parking slot
img_width,img_height=35,70

def mouseClick(events,x,y,flags,params):
    # Left click and save the coordinates (x,y) into the Position list
    if (events == cv2.EVENT_LBUTTONDOWN):
        position.append((x,y))

    # Right click to delete the selected frame
    if (events == cv2.EVENT_RBUTTONDOWN):
        for i,pos in enumerate(position):
            (x1,y1)=pos
            if (x1<x<x1+img_width and y1<y<y1+img_height):
                position.pop(i)

    with open('Parking Position text file path/CarParkPos.txt','wb') as fp:
        pickle.dump(position,fp)

while True:
    img = cv2.imread('image path/Challenge4_parking.jpg')
    img = cv2.resize(img, (1127, 400)) 
    for pos in position:
        cv2.rectangle(img=img,pt1=(pos[0],pos[1]),
                      pt2=(pos[0]+img_width,pos[1]+img_height),
                      color=(0,255,0),thickness=2)

    cv2.imshow('Parking',img)

    cv2.setMouseCallback('Parking',mouseClick)
    key=cv2.waitKey(1)
    if key==27:
        break

cv2.destroyAllWindows()


# In[ ]:


import cv2
import pickle
import numpy as np

img = cv2.imread('image path/Challenge4_parking.jpg')
img = cv2.resize(img, (1127, 400)) 

filename = 'Parking Position text file path/CarParkPos.txt'  
with open(filename, 'rb') as f:
    posList = pickle.load(f)

while True:

    spacePark = 0
    imgCopy = img.copy()

    img_w, img_h = img.shape[:2]  
    
    # Convert the grayscale image and detect whether there is a car in the parking space through morphological processing
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Gaussian filter, convolution kernel 3*3, the standard deviation of the convolution kernel along the x and y directions is 1
    imgGray = cv2.GaussianBlur(imgGray, (3,3), 1)
    
    imgThresh = cv2.adaptiveThreshold(imgGray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY_INV, 101, 20)
    
    # Remove scattered white spots, 
    # If there are cars in the parking space, there will be a lot of pixels (white dots) in the parking space. 
    # If there are no cars, there will be basically no white dots in the parking space frame.
    imgMedian = cv2.medianBlur(imgThresh, 5)
    
    # Expand the white part, swell 
    kernel = np.ones((3,3), np.uint8)  
    imgDilate = cv2.dilate(imgMedian, kernel, iterations=1) # The number of iterations is 1

    w, h = 35, 70  

    for pos in posList:
        
        # Crop the straightened rectangular frame, first specify the height h, and then specify the width w
        imgCrop = imgDilate[pos[1]:pos[1]+h, pos[0]:pos[0]+w]
        
        # Calculate how many pixels there are in each cropped single parking space
        count = cv2.countNonZero(imgCrop)
        
        # Determine whether there is a car in the parking space
        if count < 730: # If the number of pixels is less than 730, there is no car.
            color = (0,255,0) # If there is no car, the parking slot frame will be green
            spacePark += 1  # Every time an empty parking space is detected, the number increase one.
            cv2.rectangle(imgCopy, (pos[0],pos[1]), (pos[0]+w,pos[1]+h), color, 2)
    
    # Plot how many empty parking slot are currently left
    cv2.putText(imgCopy, 'FREE: '+str(spacePark), (31,91), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 3)
    
    #cv2.imshow('imgGray', imgGray)
    #cv2.imshow('imgThresh', imgThresh)  
    #cv2.imshow('imgMedian', imgMedian)  
    #cv2.imshow('imgDilate', imgDilate)  
    cv2.imshow('img', imgCopy) 

    if cv2.waitKey(1) & 0xFF==27:  
        break

cv2.destroyAllWindows()

