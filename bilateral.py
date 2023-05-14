# -*- coding: utf-8 -*-
"""
Created on Fri May 12 16:18:04 2023

@author: User
"""

import cv2 as cv
import numpy as np
import math

img=cv.imread('Lena.jpg',cv.IMREAD_GRAYSCALE)
cv.imshow('Input image',img)

def spatial_kernel(kernelsize,sigma):
    sp_kernel=np.zeros((kernelsize,kernelsize),dtype='float32')
    g=2*sigma**2
    half_ksize=kernelsize//2
    for i in range(-half_ksize,half_ksize+1):
        for j in range(-half_ksize,half_ksize+1):
            sp_kernel[i+half_ksize][j+half_ksize]=(math.exp(-(i*i+j*j)/(2*g)))/(2*g*math.pi)
            
    return sp_kernel


def range_kernel(kernelsize,sigma):
    k=1
    
    

kernelsize=int(input('Enter the size of the kernel'))
kernelsize=kernelsize
sigma=int(input('Enter the value of sigma'))
sigma=sigma
