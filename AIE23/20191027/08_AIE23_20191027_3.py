from numpy import linalg as LA
import numpy as np
import scipy
import sympy as sym
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

sym.init_printing()

print('\n---------------仿射变换---------------------')
x = np.random.random([1000, 2])
print(x)
A = np.array([[5, 1], [-0.5, 1]])
# print(A.shape)
y = np.dot(x, A)
# print(y)
# plt.scatter(x[:, 0], x[:, 1], c='r')
# plt.scatter(y[:, 0], y[:, 1], c='b')
# plt.axis("equal")
# plt.show()
sym.Matrix
print(sym.Matrix([[25, 15, -5], [15, 18, 0], [-5, 0, 11]]))
print('\n----------------SVD--------------------')

print('\n---------------img loading---------------------')
img = plt.imread('Resources/jin.jpg')
# img = mpimg.imread('Resources/jin.jpg')
img = img[:, :, :]
print(np.shape(img))
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
