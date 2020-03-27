import cv2 
import numpy as np 

cap = cv2.VideoCapture(0) 
kernel = np.ones([4, 4])
#kernel[:2, :] = -1  
kernel[:2, :2] = -1 
kernel[2:, 2:] = -1
while True: 
    ret, frame = cap.read() 
    frame = cv2.filter2D(frame, -1, kernel)
    cv2.imshow("name", frame) 
    cv2.waitKey(1000//30)