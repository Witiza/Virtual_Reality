#encoding: utf-8
import numpy as np # Imports numpy under alias np
import cv2

def convolve(dest, src, i, j, kernel):
    krows, kcols = kernel.shape
    srctmp = src[i:i + krows, j:j + kcols]
    dest[i, j] = (srctmp * kernel[:, :, np.newaxis]).sum(axis=(0, 1))


def sobelFilter(img):
    rows, cols,_ = img.shape

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
    imgpaddingX = np.zeros((rows + 2, cols + 2, 3))
    imgpaddingX[1:-1, 1:-1] = img

    imgpaddingY = np.zeros((rows + 2, cols + 2, 3))
    imgpaddingY[1:-1, 1:-1] = img

    filteredX = np.zeros(img.shape)
    filteredY = np.zeros(img.shape)

    result = np.zeros(img.shape)
    for i in range(0, rows):
        for j in range(0, cols):
            convolve(filteredX, imgpaddingX, i, j, kernel_horizontal)


    for i in range(0, rows):
        for j in range(0, cols):
            convolve(filteredY, imgpaddingY, i, j, kernel_vertical)


    for i in range(0,rows):
        for j in range(0,cols):
            result[i,j] = np.sqrt(np.power(filteredX[i,j],2) + np.power(filteredY[i,j],2))

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
    rows, cols, _ = img.shape

    kernel = np.array([
        [1.,  4.,  7.,  4., 1.],
        [4., 16., 26., 16., 4.],
        [7., 26., 41., 26., 7.],
        [4., 16., 26., 16., 4.],
        [1.,  4.,  7.,  4., 1.]
    ])
    kernel = kernel / kernel.sum()


    imgpadding = np.zeros((rows + 4, cols + 4, 3))
    imgpadding[2:-2, 2:-2] = img

    result = np.zeros(img.shape)
    for i in range(0, rows):
        for j in range(0, cols):
            convolve(result, imgpadding, i, j, kernel)

    cv2.imshow("Original", img)
    cv2.imshow("Filtered", np.uint8(result))
    cv2.waitKey(0)


if __name__=='__main__':
    img = cv2.imread('sofi.jpg', cv2.IMREAD_ANYCOLOR)
  #  filterBox(img)
    sobelFilter(img)
