import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from PIL import Image
print('\n---------------img loading---------------------')
img = plt.imread('Resources/jin.jpg')
img = img[:, :, :]

imga = Image.open('Resources/jin.jpg')
imga = imga.convert('RGBA')
print(np.shape(img))
print('imga',np.shape(imga))
a1, a2, a3 = 0.2989, 0.5870, 0.1140
img_gray = img[:, :, 0] * a1 + img[:, :, 1] * a2 + img[:, :, 2] * a3

U, A, V = LA.svd(img_gray, full_matrices=True)
print(U.shape, A.shape, V.shape)
Lambda = np.diag(A)
print(U.shape, Lambda.shape, V.shape)
re_img = np.dot(np.dot(U[:, :20], Lambda[:20, :20]), V[:20, :])
print('re_img', re_img.shape)

# plt.matshow(img)
# plt.matshow(img_gray, cmap=plt.get_cmap("gray"))
plt.matshow(re_img, cmap=plt.get_cmap("gray"))
plt.imsave('Resources/jin_s.jpg',re_img)
plt.show()
