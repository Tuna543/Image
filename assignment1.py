# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 19:23:05 2023

@author: User
"""

import cv2 as cv
import numpy as np
import math
#import matplotlib.pyplot as plt

img=cv.imread('Lena.jpg',cv.IMREAD_GRAYSCALE)
img1=cv.imread('Lena2.jpg',cv.IMREAD_GRAYSCALE)
cv.imshow('input image', img)
# cv.imshow('input image', img1)


def inverse_log_transform(img):
    out=np.zeros((img.shape[0],img.shape[1]),dtype=np.uint8)
    c=255/np.log2(255+1)
    for i in range(0,img.shape[0]):
        for j in range(0,img.shape[1]):
            s=img.item(i,j)
            val=math.pow(2,(s/c))-1
            out.itemset((i,j),val)
    cv.normalize(out,out,0,255,cv.NORM_MINMAX)
    out=np.round(out).astype(np.uint8)
    cv.imshow('Inverse Log transformation output image',out)
    
def gamma(img):
    out=np.zeros((img.shape[0],img.shape[1]),dtype=np.uint8)
    gm=.23
    c=255/8
    for i in range(0,img.shape[0]):
        for j in range(0,img.shape[1]):
            s=img.item(i,j)
            val=c*pow(s,gm)
            out.itemset((i,j),val)
    cv.normalize(out,out,0,255,cv.NORM_MINMAX)
    out=np.round(out).astype(np.uint8)
    cv.imshow('Gamma Correction image',out)
            
def contrast_streching(img):
    out=np.zeros((img.shape[0],img.shape[1]),dtype=np.uint8)
    x_min=np.min(img)
    x_max=np.max(img)
    for i in range(0,img.shape[0]):
        for j in range(0,img.shape[1]):
            x_input=img.item(i,j)
            val=((x_input-x_min)*255)/(x_max-x_min)
            out.itemset((i,j),val)
    # out=np.round(out).astype(np.uint8)
    cv.imshow('Contrast Stretching output',out)


def gaussian_filtering(kernelsize,img,sigma):
    out=np.zeros((img.shape[0],img.shape[1]),dtype='float32')
    kernel=np.zeros((kernelsize,kernelsize),dtype='float32')
    
    height=kernelsize//2
    output_with_border=cv.copyMakeBorder(src=img,top=height,bottom=height,left=height,right=height,borderType=cv.BORDER_CONSTANT)
    const=2*math.pi*sigma*sigma
    for i in range(-height,height+1):
        for j in range(-height,height+1):
            val=np.exp(-((i*i+j*j)/(2*sigma*sigma)))
            val=val/const
            kernel[i+height][j+height]=val
           
    for i in range(0,kernelsize):
       for j in range(0,kernelsize):
           print(kernel[i][j],end=" ")
       print()
       
    for i in range(0,img.shape[0]):
        for j in range(0,img.shape[1]):
            for m in range(0,height):
                for n in range(0,height):
                    out[i][j]+=output_with_border[i+m-height][j+n-height]*kernel[kernelsize-m-1][kernelsize-n-1]
                    
    cv.normalize(out,out,0,255,cv.NORM_MINMAX)
    out=np.round(out).astype(np.uint8)
    cv.imshow('Gaussian output',out)

def median(img,kernelsize):
    elements=kernelsize*kernelsize
    index1=elements//2
    if(kernelsize%2==0):
        index2=index1-1
    out=np.zeros((img.shape[0]-kernelsize+1,img.shape[1]-kernelsize+1),dtype='float32')
    for i in range(img.shape[0]-kernelsize+1):
        for j in range(img.shape[1]-kernelsize+1):
            arr=[]
            for m in range(kernelsize):
                for n in range(kernelsize):
                    arr.append(img[i+m][j+n])
            arr.sort()
            if(kernelsize%2==1):
                out[i][j]=arr[index1]
            else:
                out[i][j]=(arr[index1]+arr[index2])/2
    cv.normalize(out,out,0,255,cv.NORM_MINMAX)
    out=np.round(out).astype(np.uint8)
    cv.imshow('Median Output',out)
    

def mean(img,kernelsize):
    kernel=np.ones((kernelsize,kernelsize),dtype=np.uint8)
    out=np.zeros((img.shape[0]-kernelsize+1,img.shape[1]-kernelsize+1),dtype='float32')
    s=np.sum(kernel)
    for i in range(img.shape[0]-kernelsize+1):
        for j in range(img.shape[1]-kernelsize+1):
            total=0
            for m in range(kernelsize):
                for n in range(kernelsize):
                    total+=kernel[m][n]*img[i+m][j+n]
            out[i][j]=total/s
    
    cv.normalize(out,out,0,255,cv.NORM_MINMAX)
    out=np.round(out).astype(np.uint8)
    cv.imshow('Mean Output',out)
    
def laplacian(img):
    out=np.zeros((img.shape[0],img.shape[1]),dtype='float32')
    finalout=np.zeros((img.shape[0],img.shape[1]),dtype='float32')
    # ([[0, -1, 0],[-1, 4, -1],[ 0, -1, 0]])
    kernel=np.array([[0, 0, -1, 0, 0],
                      [0, -1, -2, -1, 0],
                      [-1, -2, 16, -2, -1],
                      [0, -1, -2, -1, 0],
                      [0, 0, -1, 0, 0]])
    bordered_img=cv.copyMakeBorder(src=img,top=2,bottom=2,left=2,right=2,borderType=cv.BORDER_CONSTANT)
    
    for i in range(0,img.shape[0]):
        for j in range(0,img.shape[1]):
            for m in range(0,kernel.shape[0]//2):
                for n in range(0,kernel.shape[1]//2):
                    out[i][j]+=bordered_img[i+m-(kernel.shape[0]//2)][j+n-(kernel.shape[1]//2)]*kernel[kernel.shape[0]-m-1][kernel.shape[1]-n-1]
    
    
    for i in range(0,img.shape[0]):
        for j in range(0,img.shape[1]):
            finalout[i][j]=img[i][j]+out[i][j]
            
    
    cv.normalize(out,out,0,255,cv.NORM_MINMAX)
    out=np.round(out).astype(np.uint8)
    cv.imshow('Laplacian output',out)
    
    cv.normalize(finalout,finalout,0,255,cv.NORM_MINMAX)
    finalout=np.round(finalout).astype(np.uint8)
    cv.imshow('Laplacian final output',finalout)
    
def sobel(img):
    out = np.zeros((img.shape[0],img.shape[1]),dtype ='float32')
    out_horizontal = np.zeros((img.shape[0],img.shape[1]),dtype ='float32')
    out_vertical = np.zeros((img.shape[0],img.shape[1]),dtype ='float32')
    kernel_horizontal = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    kernel_vertical = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    
    for i in range(kernel_horizontal.shape[0]//2,img.shape[0]-kernel_horizontal.shape[0]//2):
        for j in range(kernel_horizontal.shape[1]//2,img.shape[1]-kernel_horizontal.shape[1]//2):
            for m in range(kernel_horizontal.shape[0]):
                for n in range(kernel_horizontal.shape[1]):
                    out_horizontal[i][j] += img[i+m-kernel_horizontal.shape[0]//2][j+n-kernel_horizontal.shape[1]//2]*kernel_horizontal[kernel_horizontal.shape[0]-m-1][kernel_horizontal.shape[1]-n-1]
                    out_vertical[i][j] +=img[i+m-kernel_horizontal.shape[0]//2][j+n-kernel_horizontal.shape[1]//2]*kernel_vertical[kernel_horizontal.shape[0]-m-1][kernel_horizontal.shape[1]-n-1]
            out[i][j] = np.sqrt((out_horizontal[i][j])*(out_horizontal[i][j])+(out_vertical[i][j])*(out_vertical[i][j]))

    cv.normalize(out,out,0,255,cv.NORM_MINMAX)
    out=np.round(out).astype(np.uint8)    
    cv.imshow("total sobel output",out)
    
    cv.normalize( out_horizontal, out_horizontal,0,255,cv.NORM_MINMAX)
    out_horizontal=np.round( out_horizontal).astype(np.uint8)    
    cv.imshow("horizontal output", out_horizontal)
    
    cv.normalize(out_vertical,out_vertical,0,255,cv.NORM_MINMAX)
    out_vertical=np.round(out_vertical).astype(np.uint8)    
    cv.imshow("vertical output",out_vertical)

#function calls
# inverse_log_transform(img)
# gamma(img)
# contrast_streching(img)
kernelsize=int(input('Enter the size of kernel: '))
kernelsize=kernelsize
sigma=float(input('Enter the value of sigma: '))
sigma=sigma
# gaussian_filtering(kernelsize, img, sigma)
# median(img1,kernelsize)
# mean(img1,kernelsize)
laplacian(img)
# sobel(img)
cv.waitKey(0)
cv.destroyAllWindows()