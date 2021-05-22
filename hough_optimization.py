# Hough Transform & Disparity
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

image="hough_image_noise.png"
#image=cv2.imread(image)
img=cv2.imread(image,0)
#img = cv2.resize(img, (50,50), interpolation = cv2.INTER_AREA) 
plt.subplot(131)
plt.title("hough img noise")
plt.imshow(img,cmap=plt.cm.gray)
img = cv2.GaussianBlur(img,(9,9),0)
plt.subplot(132)
plt.title("gaussian applied")
plt.imshow(img,cmap=plt.cm.gray)
ret,img=cv2.threshold(img,80,255,cv2.THRESH_BINARY)
img = cv2.Canny(img,127,200)
plt.subplot(133)
plt.title("canny")
plt.imshow(img,cmap=plt.cm.gray)
img = cv2.resize(img, (50,50), interpolation = cv2.INTER_AREA) 
img=np.where(img==255,1,img)
#cv2.imwrite("blur.png",blur)

#adjusted threshold levels to detect the border line
angles=(np.arange(-90,181))*(np.pi)/180
#calculating the r values


cos_theta=np.cos(angles)
sin_theta=np.sin(angles)
cos_theta=cos_theta.reshape(1,271)
sin_theta=sin_theta.reshape(1,271)

x1,y1=np.nonzero(img)
len_x1=len(x1)
len_y1=len(y1)
x1=x1.reshape(len_x1,1)
y1=y1.reshape(len_y1,1)
A1=(x1+1)*cos_theta
B1=(y1+1)*sin_theta
R=A1+B1
plt.figure()
plt.title("Hough transform")
plt.xlabel("r")
plt.ylabel("theta")
plt.savefig('plot_guassian.png')   
for i in range (len_x1):
    plt.plot(angles,R[i][0:272])
  


R=np.floor(R)
R=R.astype(int)
Rmax=np.amax(R)
loc=np.where(Rmax)
loc=np.argmax(R)
i,j = np.unravel_index(R.argmax(), R.shape)
Rmax=int(Rmax+1)
acc=np.zeros((Rmax,271))
for i in range(len_x1):
    for j in range(270):
       if(R[i][j]>=0):
                acc[(R[i][j])][j]=acc[R[i][j]][j]+1

res=acc
Resmax=np.amax(res)
rho,theta = np.unravel_index(res.argmax(), res.shape)
angle_pr=angles[theta]
distance=R[rho][theta]
y3=int(distance*5/np.sin(angle_pr))
x3=int(distance*5/np.cos(angle_pr))
img = cv2.line(img,(0,y3),(x3,0),(255,0,0),1) 
plt.figure()
plt.imshow(img)
#    

