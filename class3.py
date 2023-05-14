# -*- coding: utf-8 -*-
"""
Created on Tue May  2 10:45:42 2023

@author: User
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread("bd.jpeg")
cv2.imshow("input",img)

#cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]])
# hist =np.histogram(img[:,:,0],256,[0,256])
# histr1 = cv2.calcHist([img],[0],None,[256],[0,255])
# histr2 = cv2.calcHist([img],[0],None,[256],[0,255])
# histr3 = cv2.calcHist([img],[0],None,[256],[0,255])

# plt.plot(histr)

def equalize_img(normalhist1):
    equalized_img=np.zeros_like(img)
    sum=0.0
    cdf1=normalhist1
    s1=normalhist1
    for i in range(256):
        sum+=cdf1[i]
        cdf1[i]=sum
        s1[i]=round(255*cdf1[i])

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            x=img[i][j]
            equalized_img[i][j]=s1[x]
    cv2.imshow("equalize",equalized_img)
    plt.show()


totalsize=img.shape[0]*img.shape[1]

histr, _ = np.histogram(img[:,:,0],256,[0,256])
plt.plot(histr,color='r')
normalhist1=histr
normalhist1=histr/totalsize
histr, _ = np.histogram(img[:,:,1],256,[0,256])
plt.plot(histr,color='g')
normalhist2=histr
normalhist2=histr/totalsize
histr, _ = np.histogram(img[:,:,2],256,[0,256])
plt.plot(histr,color='b')
normalhist3=histr
normalhist3=histr/totalsize

equalize_img(normalhist1)
equalize_img(normalhist2)
equalize_img(normalhist3)


    




plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.title("Input Image Histogram")
# plt.hist(img.ravel(),256,[0,255])
plt.show()

# img2 = cv2.equalizeHist(img)


# plt.subplot(1, 2, 2)
# plt.title("output Image Histogram")
# plt.hist(img2.ravel(),256,[0,255])


# cv2.imshow("output",img2)

plt.show()

# ## Convert image from RGB to HSV
img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

plt.subplot(2, 2, 3)
histr, _ = np.histogram(img_hsv[:,:,2],256,[0,256])
normalhist=histr
normalhist=histr/totalsize
plt.plot(histr,color = 'b')

# Histogram equalisation on the V-channel
img_hs=equalize_img(normalhist)


# Convert image back from HSV to RGB
image = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
histr, _ = np.histogram(image[:,:,0],256,[0,256])
plt.plot(histr,color='r')
# normalhist1=histr
# normalhist1=histr/totalsize
histr, _ = np.histogram(image[:,:,1],256,[0,256])
plt.plot(histr,color='g')
# normalhist2=histr
# normalhist2=histr/totalsize
histr, _ = np.histogram(image[:,:,2],256,[0,256])
plt.plot(histr,color='b')
# normalhist3=histr



cv2.waitKey(0)
cv2.destroyAllWindows()