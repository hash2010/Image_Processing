# Scale Invarient Feature transform (SIFT)




import numpy as np
import matplotlib.pyplot as plt
import cv2


I1=cv2.imread('img_left_01.png')
I2=cv2.imread('img_right_01.png')

h=750
w=500
dim=(w,h)
I1= cv2.resize(I1, dim, interpolation = cv2.INTER_AREA) 
I2= cv2.resize(I2, dim, interpolation = cv2.INTER_AREA) 

I1=cv2.cvtColor(I1,cv2.COLOR_RGB2BGR)
I2=cv2.cvtColor(I2,cv2.COLOR_RGB2BGR)
plt.imshow(I1)
Img1=I1
Img2=I2
#plt.imshow(Img1)
I1_gray = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
I2_gray=cv2.cvtColor(I2,cv2.COLOR_BGR2GRAY)

kaze = cv2.KAZE_create()
(kps1, descs1) = kaze.detectAndCompute(I1_gray, None)
img1=cv2.drawKeypoints(I1,kps1,I1)
(kps2, descs2) = kaze.detectAndCompute(I2_gray, None)
img2=cv2.drawKeypoints(I2,kps2,I2)
out_img=np.zeros([750,1000,3])
out_img[0:750,0:500]=img2
out_img[0:750,500:1001]=img1
out_img=out_img.astype(np.uint8)
#plt.imshow(out_img)


#cv.imwrite('sift_keypoints.jpg',img)
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

matches = bf.match(descs1,descs2)
matches = sorted(matches, key = lambda x:x.distance)

out_img2 = cv2.drawMatches(I1, kps1, I2, kps2, matches[:100], I2, flags=2)
plt.imshow(out_img2)
		# computing a homography requires at least 4 matches
if len(matches) > 4:
    ptsA = np.float32([[299,249],[329,452],[364,393],[296,472]])
    ptsB = np.float32([[156,238],[185,441],[218,379],[152,463]])
			# compute the homography between the two sets of points
    (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC)
			# return the matches along with the homograpy matrix
			# and status of each matched point
			# construct the two sets of points
result=np.zeros([750,1000,3])
result=result.astype(np.uint8)			
result2 = cv2.warpPerspective(Img2, H,(I1.shape[1] + I2.shape[1], I1.shape[0]))
result[0:I1.shape[0], 0:I2.shape[1]] = Img1
result[32:750,280:780]=result2[0:718,0:500]
plt.imshow(result)
cropped_img=result[40:750,0:500]
cropped_img=cv2.cvtColor(cropped_img,cv2.COLOR_BGR2RGB)
plt.imshow(cropped_img)
#resul
cv2.imwrite('stitched_img.png',cropped_img)
plt.imshow(Img1)
