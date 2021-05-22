##Affine Transformation

import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

I1=cv2.imread('img1.jpg')
I1b=I1[0:600,0:800,0]
I1g=I1[0:600,0:800,1]
I1r=I1[0:600,0:800,2]

I2=cv2.imread('img2.jpg')
I1 = cv2.cvtColor(I1, cv2.COLOR_BGR2RGB)
I2 = cv2.cvtColor(I2, cv2.COLOR_BGR2RGB)
rows,cols,ch=I1.shape
pts1=np.float32([[643,260],[547,317],[448,270]])
pts2=np.float32([[415,321],[332,372],[227,321]])
M=cv2.getAffineTransform(pts1,pts2)
dst1=cv2.warpAffine(I1,M,(cols,rows))
dst2=cv2.warpAffine(I2,M,(cols,rows))
plt.subplot(221),plt.imshow(I1),plt.title('Input1')
plt.subplot(222),plt.imshow(dst1),plt.title('Output1')

plt.subplot(223),plt.imshow(I2),plt.title('Input2')
plt.subplot(224),plt.imshow(dst2),plt.title('Output2')
plt.savefig("C:\\Users\\chand\\Documents\\harsha\\CV_LAB\\lab3\\affin2d_function2.png")


###without function
a=np.array([[643,260,1,0,0,0],
            [0,0,0,643,260,1],
            [547,317,1,0,0,0],
            [0,0,0,547,317,1],
            [448,270,1,0,0,0],
            [0,0,0,448,270,1]])
b=np.array([[415],[321],[332],[372],[227],[321]])
x=np.linalg.solve(a,b)
x=np.reshape(x,(2,3))
out_image=np.zeros((600,800,3))
dst_y, dst_x = np.indices((rows, cols))
dst_lin_homg_pts = np.stack((dst_x.ravel(), dst_y.ravel(), np.ones(dst_y.size)))
src_lin_pts = np.round(x.dot(dst_lin_homg_pts)).astype(int)
min_x, min_y = np.amin(src_lin_pts, axis=1)
src_lin_pts -= np.array([[min_x], [min_y]])
trans_max_x, trans_max_y = np.amax(src_lin_pts, axis=1)
srcb = np.ones((trans_max_y+1, trans_max_x+1), dtype=np.uint8)*127
srcb[src_lin_pts[1], src_lin_pts[0]] = I1b.ravel()
#cv2.imshow('src', srcb)
#channel1
srcg = np.ones((trans_max_y+1, trans_max_x+1), dtype=np.uint8)*127
srcg[src_lin_pts[1], src_lin_pts[0]] = I1g.ravel()

#channel2
srcr= np.ones((trans_max_y+1, trans_max_x+1), dtype=np.uint8)*127
srcr[src_lin_pts[1], src_lin_pts[0]] = I1r.ravel()
#cv2.imshow('src', srcb)
#cv2.imshow('src', srcr)
#cv2.imshow('src', srcg)
#stacking of r,g,b
src=np.zeros((628,889,3),'uint8')
src[0:629,0:890,0]=srcr
src[0:629,0:890,1]=srcg
src[0:629,0:890,2]=srcb

img_copy = cv2.cvtColor(src,cv2.COLOR_BGR2RGB)
#plot image
cv2.imshow('src', I1)
cv2.imshow('src', img_copy)
img_copy.save('img21.jpeg')


