###Discrete Fourier Transform (DFT) of images.

import matplotlib.pyplot as plt
import numpy as np

image = plt.imread('C://Users//chand//Documents//harsha//CV_LAB//cameraman.bmp')
#
a=np.fft.fft2(image)
shifted_fft=np.fft.fftshift(a)
absolute_fft=abs(shifted_fft)
plt.imshow(absolute_fft)
log_fft=np.log10(absolute_fft)
log_fft_min=np.nanmin(log_fft[np.isfinite(log_fft)])
log_fft_max=np.nanmax(log_fft[np.isfinite(log_fft)])
log_fft1=log_fft/log_fft_max
log_fft2=log_fft1*255
log_fft3=log_fft2.astype(int)
log_fft_max=np.nanmax(log_fft3[np.isfinite(log_fft3)])
plt.imshow(log_fft3)
a = np.mgrid[:5, :5][0]
np.fft.fft2(a)

fig=plt.figure()
ax1=fig.add_subplot(121)
ax2=fig.add_subplot(122)

ax1.imshow(image,cmap='gray')
ax2.imshow(log_fft3,cmap='gray')
ax1.title.set_text("Original Image")
ax2.title.set_text("Fourier Image")
plt.show()
image1 = plt.imread('C://Users//chand//Documents//harsha//CV_LAB//cameraman.bmp')
image2=np.rot90(image1, 3)
a=np.fft.fft2(image2)
shifted_fft=np.fft.fftshift(a)
absolute_fft=abs(shifted_fft)
plt.imshow(absolute_fft)
log_fft=np.log10(absolute_fft)
log_fft_min=np.nanmin(log_fft[np.isfinite(log_fft)])
log_fft_max=np.nanmax(log_fft[np.isfinite(log_fft)])
log_fft1=log_fft/log_fft_max
log_fft2=log_fft1*255
log_fft3=log_fft2.astype(int)
log_fft_max=np.nanmax(log_fft3[np.isfinite(log_fft3)])
plt.imshow(log_fft3)
a = np.mgrid[:5, :5][0]
np.fft.fft2(a)

fig=plt.figure()
ax1=fig.add_subplot(121)
ax2=fig.add_subplot(122)

ax1.imshow(image2,cmap='gray')
ax2.imshow(log_fft3,cmap='gray')
ax1.title.set_text("Original Image")
ax2.title.set_text("Fourier Image")
plt.show()



