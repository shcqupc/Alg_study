import cv2 
import numpy as np 
# conda instal opencv 
# pip install opencv-python
img = cv2.imread("img/jin.jpg") 
print(np.shape(img))
cv2.imshow("w", img)
cv2.waitKey(0)