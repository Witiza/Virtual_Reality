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

    return result,filteredX,filteredY
    cv2.imshow("Original", img)
    cv2.imshow("Filtered", np.uint8(result))
    cv2.waitKey(0)

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
        [1.,  4.,  7.,  4., 1.],
        [4., 16., 26., 16., 4.],
        [7., 26., 41., 26., 7.],
        [4., 16., 26., 16., 4.],
        [1.,  4.,  7.,  4., 1.]
    ])
    kernel = kernel / kernel.sum()


    imgpadding = np.zeros((rows + 4, cols + 4))
    imgpadding[2:-2, 2:-2] = img

    result = np.zeros(img.shape)
    for i in range(0, rows):
        for j in range(0, cols):
            convolveGrayscale(result, imgpadding, i, j, kernel)

    return result
    cv2.imshow("Original", img)
    cv2.imshow("Filtered", np.uint8(result))
    cv2.waitKey(0)


def cannyEdge(img):
    gauss = gaussFilter(img)

    sobel,sobelx,sobely = sobelFilter(gauss)

    rows, cols = img.shape
    directions = np.zeros(img.shape)

    sobelpadding = np.zeros((rows + 2, cols + 2))
    sobelpadding[1:-1, 1:-1] = sobel

    maxSupression = np.zeros(img.shape)

    for i in range(0, rows):
        for j in range(0, cols):
            directions[i,j] = getAtan(sobelx[i, j], sobely[i, j])


    for i in range(0, rows):
        for j in range(0, cols):
            maxSupression[i,j] = maximumSupression(sobelpadding,directions,i,j)

    print(maxSupression)




    cv2.imshow("Original", img)
    cv2.imshow("Filtered", np.uint8(maxSupression))
    cv2.waitKey(0)


def getAtan(x,y):
    angle = np.degrees(np.absolute(np.arctan2(x,y)))
    if angle <= 22.5:
        angle = 0.0
    if angle > 22.5 and angle <= 67.5:
        angle = 45.0
    if angle > 67.5 and angle <= 112.5:
        angle = 90.0
    if angle > 112.5 and angle <= 157.5:
        angle = 135.0
    if angle > 157.5:
        angle = 0.0

    return angle

def maximumSupression(sobel,direction,_i,_j):
    i = _i+1
    j = _j+1
    print(direction[_i,_j])
    if direction[_i,_j] == 0.0:
        if sobel[i,j+1] > sobel[i,j] or sobel[i,j-1] > sobel[i,j]:
            return 0.0
        else:
            return sobel[i,j]

    if direction[_i,_j] == 45.0:
        if sobel[i+1,j+1] > sobel[i,j] or sobel[i-1,j-1] > sobel[i,j]:
            return 0.0
        else:
            return sobel[i,j]

    if direction[_i,_j] == 90.0:
        if sobel[i+1,j] > sobel[i,j] or sobel[i-1,j] > sobel[i,j]:
            return 0.0
        else:
            return sobel[i,j]

    if direction[_i,_j] == 135.0:
        if sobel[i+1,j-1] > sobel[i,j] or sobel[i-1,j+1] > sobel[i,j]:
            return 0.0
        else:
            return sobel[i,j]










if __name__=='__main__':
    img = cv2.imread('droids.jpg', cv2.IMREAD_GRAYSCALE)
  #  filterBox(img)
    cannyEdge(img)
