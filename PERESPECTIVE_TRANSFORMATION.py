#Perspective Transformation
import numpy as np
import matplotlib.pyplot as plt
import cv2


I1=cv2.imread('img1.jpg')
I2=cv2.imread('img2.jpg')
rows,cols,ch = I1.shape

pts1=np.float32([[643,260],[547,317],[448,270],[528,203]])
pts2=np.float32([[415,321],[332,372],[227,321],[309,261]])

M=cv2.getPerspectiveTransform(pts1,pts2)
dst=cv2.warpPerspective(I1,M,(rows,cols))
plt.subplot(121)
plt.imshow(I1)
plt.title('Input')
plt.subplot(122)
plt.imshow(dst)
plt.title('Output')
plt.show()


plt.savefig("C:\\Users\\chand\\Documents\\harsha\\CV_LAB\\lab3\\perspective2d_function2.png")
rows1,cols1,ch1=I2.shape
M1=cv2.getPerspectiveTransform(pts1,pts2)
dst1=cv2.warpPerspective(I2,M1,(rows1,cols1))
plt.subplot(121)
plt.imshow(I2)
plt.title('Input2')
plt.subplot(122)
plt.imshow(dst)
plt.title('Output2')
plt.show()

plt.savefig("C:\\Users\\chand\\Documents\\harsha\\CV_LAB\\lab3\\perspective2d_image2_function2.png")
