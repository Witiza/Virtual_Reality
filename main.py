import numpy as np # Imports numpy under alias np
import cv2

def filterBoxPixel(array,i,j)
    array[i,j] = array[i,j]*1/9+array[i-1,j-1]*1/9+array[i,j-1]*1/9+array[i-1,j]*1/9+array[i+1,j+1]*1/9+array[i,j+1]*1/9

def filterBox(image):
    height,width,_ = image.shape
    ret =
    for i in range(0,height):
        for j in range(0,width):
            image[i,j] =   image[i,j




if __name__=='__main__':
    img = cv2.imread('img/sonic.jpg', cv2.IMREAD_ANYCOLOR)
    filterBox(img)
