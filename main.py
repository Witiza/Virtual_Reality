#encoding: utf-8
import numpy as np # Imports numpy under alias np
import cv2

def convolve(dest, src, i, j, kernel):
    krows, kcols = kernel.shape
    srctmp = src[i:i + krows, j:j + kcols]
    dest[i, j] = (srctmp * kernel[:, :, np.newaxis]).sum(axis=(0, 1))

def convolveGrayscale(dest, src, i, j, kernel):
    krows, kcols = kernel.shape
    srctmp = src[i:i + krows, j:j + kcols]
    dest[i, j] = (srctmp * kernel[:, :]).sum(axis=(0, 1))


def sobelFilter(img):
    rows, cols = img.shape

    kernel_horizontal = np.array([
                [-1.,0.,1.],
                [-2.,0.,2.],
                [-1.,0.,1.],
    ])

    kernel_vertical = np.array([
                [-1.,-2.,-1.],
                [0.,0.,0.],
                [1.,2.,1.],
    ])
    imgpaddingX = np.zeros((rows + 2, cols + 2))
    imgpaddingX[1:-1, 1:-1] = img

    imgpaddingY = np.zeros((rows + 2, cols + 2))
    imgpaddingY[1:-1, 1:-1] = img

    filteredX = np.zeros(img.shape)
    filteredY = np.zeros(img.shape)

    result = np.zeros(img.shape)
    for i in range(0, rows):
        for j in range(0, cols):
            convolveGrayscale(filteredX, imgpaddingX, i, j, kernel_horizontal)


    for i in range(0, rows):
        for j in range(0, cols):
            convolveGrayscale(filteredY, imgpaddingY, i, j, kernel_vertical)


    for i in range(0,rows):
        for j in range(0,cols):
            result[i,j] = np.sqrt(np.power(filteredX[i,j],2) + np.power(filteredY[i,j],2))


    defxfilter = cv2.Sobel(img,cv2.CV_64F,1,0)
    defyfilter = cv2.Sobel(img,cv2.CV_64F,0,1)

    defSobeledImg = np.sqrt(defxfilter * defxfilter + defyfilter * defyfilter)
    cv2.imshow("OwnSobel", np.uint8(result))
    cv2.imshow("Sobel", np.uint8(defSobeledImg))
    return result,filteredX,filteredY

def boxFilter(img):
    rows,cols,_ = img.shape

    kernel = np.ones((5,5))
    kernel = kernel/kernel.sum()

    imgpadding = np.zeros((rows + 4,cols+4,3))
    imgpadding[2:-2,2:-2] = img

    result = np.zeros(img.shape)
    for i in range(0, rows):
        for j in range(0, cols):
            convolve(result, imgpadding, i, j, kernel)



    cv2.imshow("Original", img)

    cv2.imshow("Filtered", np.uint8(result))
    cv2.waitKey(0)

def gaussFilter(img):
    rows, cols = img.shape

    kernel = np.array([
        [2.,  4.,  5.,  4., 2.],
        [4., 9., 12., 9., 4.],
        [5., 12., 15., 12., 5.],
        [4., 9., 12., 9., 4.],
        [2.,  4.,  5.,  4., 2.]
    ])
    kernel = kernel / kernel.sum()


    imgpadding = np.zeros((rows + 4, cols + 4))
    imgpadding[2:-2, 2:-2] = img

    result = np.zeros(img.shape)
    for i in range(0, rows):
        for j in range(0, cols):
            convolveGrayscale(result, imgpadding, i, j, kernel)



    cv2.imshow("OwnGauss", np.uint8(result))
    meh = 5/4
    calla= cv2.GaussianBlur(img,(5,5),meh)
    cv2.imshow("Gauss", np.uint8(calla))
    return result


def cannyEdge(img):
    gauss = gaussFilter(img)

    sobel,sobelx,sobely = sobelFilter(gauss)

    rows, cols = img.shape
    directions = np.uint8(np.zeros(img.shape))



    maxSupression = np.zeros(img.shape)


    for i in range(0, rows):
        for j in range(0, cols):
            directions[i,j] = getAtan(sobelx[i, j], sobely[i, j])


    for i in range(1, rows-1):
        for j in range(1, cols-1):
            maxSupression[i,j] = maximumSupression(sobel,directions,i,j)


    final = np.zeros(img.shape)

    for i in range(1, rows-1):
        for j in range(1, cols-1):
            final[i,j] = hysteresisThresholding(maxSupression,i,j)




    calla = cv2.Canny(img,50,100)


    cv2.imshow("CV2", np.uint8(calla))
    cv2.imshow("Original", img)
    cv2.imshow("Filtered", np.uint8(final))
    cv2.waitKey(0)


def getAtan(x,y):
    angle = np.degrees(np.arctan2(y,x))
    if angle < 0:
        angle += 180

    if angle <= 22.5:
        angle = 0
    if angle > 22.5 and angle <= 67.5:
        angle = 45
    if angle > 67.5 and angle <= 112.5:
        angle = 90
    if angle > 112.5 and angle <= 157.5:
        angle = 135
    if angle > 157.5:
        angle = 0

    return angle

def maximumSupression(sobel,direction,i,j):


    if direction[i,j] == 0:
        if sobel[i,j+1] > sobel[i,j] or sobel[i,j-1] > sobel[i,j]:
            return 0.0
        else:
            return sobel[i,j]

    if direction[i,j] == 45:
        if sobel[i+1,j+1] > sobel[i,j] or sobel[i-1,j-1] > sobel[i,j]:
            return 0.0
        else:
            return sobel[i,j]

    if direction[i,j] == 90:
        if sobel[i+1,j] > sobel[i,j] or sobel[i-1,j] > sobel[i,j]:
            return 0.0
        else:
            return sobel[i,j]

    if direction[i,j] == 135:
        if sobel[i-1,j+1] > sobel[i,j] or sobel[i+1,j-1] > sobel[i,j]:
            return 0.0
        else:
            return sobel[i,j]

def hysteresisThresholding(supressed,i,j):

    max_treshold = 40
    if supressed[i,j] > max_treshold:
        return  255

    elif supressed[i,j] < 10:
        return 0

    else:
       return checkNeighbourPixels(supressed,i,j,max_treshold)

def checkNeighbourPixels(supressed,i,j,max_treshold):
    if supressed[i+1,j] > max_treshold or supressed[i+1,j+1] > max_treshold or supressed[i,j+1] > max_treshold or supressed[i-1,j] > max_treshold or supressed[i-1,j-1] > max_treshold or supressed[i,j-1] > max_treshold:
        return 255
    else:
        return 0







if __name__=='__main__':
    img = cv2.imread('sonic.jpg', cv2.IMREAD_GRAYSCALE)
    cannyEdge(img)
