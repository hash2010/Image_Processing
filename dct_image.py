###Discrete Cosine Transform (DCT) of images.

import numpy as np
import matplotlib.pyplot as plt
import scipy
from numpy import pi
from numpy import r_

##reading the image file
image=image=plt.imread('C://Users//chand//Documents//harsha//CV_LAB//cameraman.bmp')
#plt.imshow(image, cmap=plt.cm.gray)

def dct2(a):
    return scipy.fftpack.dct( scipy.fftpack.dct( a, axis=0, norm='ortho' ) )

def idct2(a):
  return scipy.fftpack.idct( scipy.fftpack.idct( a, axis=0 , norm='ortho'))
imsize = image.shape
dct = np.zeros(imsize)
for i in r_[:imsize[0]:8]:
    for j in r_[:imsize[1]:8]:
        dct[i:(i+8),j:(j+8)] = dct2( image[i:(i+8),j:(j+8)] )


pos=130
# Display the dct of that block
plt.figure()
plt.imshow(dct[pos:pos+8,pos:pos+8],cmap='gray',vmax= np.max(dct)*0.01,vmin = 0, extent=[0,pi,pi,0])
plt.title( "An 8x8 DCT block")
# Display entire DCT
plt.figure()
plt.imshow(dct,cmap='gray',vmax = np.max(dct)*0.01,vmin = 0)
plt.title( "8x8 DCTs of the image")


# Threshold
thresh = 0.012
dct_thresh = dct * (abs(dct) > (thresh*np.max(dct)))


im_dct = np.zeros(imsize)

for i in r_[:imsize[0]:8]:
    for j in r_[:imsize[1]:8]:
        im_dct[i:(i+8),j:(j+8)] = idct2( dct_thresh[i:(i+8),j:(j+8)] )
        
        


plt.figure(figsize=(16, 5))
plt.subplot(141)
plt.imshow(image, cmap=plt.cm.gray)
plt.title('Original Image', fontsize=15)
plt.subplot(142)
plt.imshow( im_dct, cmap=plt.cm.gray)
plt.title('DCT image', fontsize=15)




