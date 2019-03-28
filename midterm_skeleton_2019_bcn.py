########################################################################
#                         TYPE YOUR NAME HERE
########################################################################

import cv2
import numpy as np

def convolve(dest, src, i, j ,kernel):
    krows, kcols = kernel.shape
    srctmp = src[i:i + krows, j:j + kcols]
    dest[i, j] = (srctmp * kernel[:,:,np.newaxis]).sum(axis=(0,1))

def exercise1():
    img = np.float64(cv2.imread("noise.jpg", cv2.IMREAD_COLOR))
    rows, cols, comp = img.shape

    # TODO: Write your code here
    kernel_horizontally = np.array([[1., 10., 45., 120., 210., 252., 210., 120., 45., 10., 1.]])
    kernel_vertically = kernel_horizontally.reshape(11,1)

    imgpadding = np.zeros((rows,cols+10,3))
    imgpadding[:,5:-5] = img

    result = np.zeros(img.shape)

    for i in range(0,rows):
        for j in range(0, cols):
            convolve(result,imgpadding,i,j,kernel_horizontally)
    result /= kernel_horizontally.sum()

    resultpadding = np.zeros((rows+10,cols,3))
    resultpadding[5:-5,:] = result
    result2 = np.zeros(img.shape)
    for i in range(0,rows):
        for j in range(0, cols):
            convolve(result2,resultpadding,i,j,kernel_vertically)
    result2 /= kernel_vertically.sum()






    cv2.imshow("Original", np.uint8(img))
    cv2.imshow("Filtered", np.uint8(result))
    cv2.imshow("Filtered2", np.uint8(result2))

def convolveMorphology(dest, src, i, j):
    srctmp = src[i:i+3,j:j+3]

    if srctmp.sum() == 2295:
        dest[i,j] = 255
    else:
        dest[i,j] = 0


def exercise2():
    img = cv2.imread("morphology.png", cv2.IMREAD_GRAYSCALE)
    rows, cols = img.shape

    imgpadding = np.zeros((rows+2,cols+2))
    imgpadding[1:-1,1:-1] = img

    result = np.zeros(img.shape)

    for i in range(0,rows):
        for j in range(0,cols):
            convolveMorphology(result,imgpadding,i,j)
    

    # TODO: Write your code here


    cv2.imshow("Input", img)
    cv2.imshow("Result", np.uint8(result))
    cv2.waitKey(0)


if __name__ == '__main__':

    # Uncomment to execute exercise 1
    exercise1()

    # Uncomment to execute exercise 2
    exercise2()

    pass
