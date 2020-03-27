import matplotlib.image as mpimg # mpimg 用于读取图片
import matplotlib.pyplot as plt
import numpy as np 
img = mpimg.imread('jin.jpg')
img = img[:, :, :]/255
print(np.shape(img))

a1, a2, a3 = 0.2989, 0.5870, 0.1140
img_gray = img[:,:,0]*a1+img[:,:,1]*a2+img[:,:,2]*a3

plt.matshow(img_gray, cmap=plt.get_cmap("gray"))
plt.show()