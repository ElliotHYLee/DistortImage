import cv2
import numpy as np

K = np.array([[  2000,     0.  , 360],
              [    0.  ,  2000,  180],
              [    0.  ,     0.  ,     1.  ]])

# zero distortion coefficients work well for this image
D_pincus = np.array([-3, 0, 0, 0], dtype=np.float)
D_barrel = np.array([3, 0, 0, 0], dtype=np.float)

# use Knew to scale the output
K_pincus, K_barrel = K.copy(), K.copy()
K_pincus[(0,1), (0,1)] = 0.8* K_pincus[(0,1), (0,1)]
K_barrel[(0,1), (0,1)] = 1.2* K_pincus[(0,1), (0,1)]

img = cv2.imread('img_2.jpg')
img = cv2.resize(img, (720, 360), )
img_pincus = cv2.fisheye.undistortImage(img, K, D=D_pincus, Knew=K_pincus)
img_barrel = cv2.fisheye.undistortImage(img, K, D=D_barrel, Knew=K_barrel)


cv2.imwrite('barrel.png', img_barrel)
cv2.imwrite('pincus.png', img_pincus)
