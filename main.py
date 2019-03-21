from ReadData import *
import os

K = np.array([[  2000,     0.  , 360],
              [    0.  ,  2000,  180],
              [    0.  ,     0.  ,     1.  ]])

# zero distortion coefficients work well for this image
D_pincus = np.array([-4, 0, 0, 0], dtype=np.float)
D_barrel = np.array([100, 0, 0, 0], dtype=np.float)

# use Knew to scale the output
K_pincus, K_barrel = K.copy(), K.copy()
K_pincus[(0,1), (0,1)] = 0.7* K_pincus[(0,1), (0,1)]
K_barrel[(0,1), (0,1)] = 2* K_pincus[(0,1), (0,1)]


def main():
    seq=1
    data = ReadData(dsName='airsim', subType='mr', seq=seq)
    barNames = data.getNewImgNames(subtype='bar')
    pinNames = data.getNewImgNames(subtype='pin')
    dirBar = data.path + '/images_bar'
    dirPin = data.path + '/images_pin'

    if not os.path.exists(dirBar):
        os.makedirs(dirBar)
    if not os.path.exists(dirPin):
        os.makedirs(dirPin)

    N = data.imgs.shape[0]

    for i in range(0, N):
        img = data.imgs[i]
        img = np.reshape(img, (360, 720, 3))

        pin = cv2.fisheye.undistortImage(img, K, D=D_pincus, Knew=K_pincus)
        bar = cv2.fisheye.undistortImage(img, K, D=D_barrel, Knew=K_barrel)

        # cv2.imshow('input', img)
        # cv2.imshow('pin', pin)
        # cv2.imshow('bar', bar)
        # cv2.waitKey(1)
        cv2.imwrite(barNames[i], bar*255.0)
        cv2.imwrite(pinNames[i], pin*255.0)
        print(i/N)




if __name__ == '__main__':
    main()