import numpy as np # Imports numpy under alias np
import cv2

def convolve(dest, src, i, j, kernel):
    krows, kcols = kernel.shape
    srctmp = src[i:i + krows, j:j + kcols]
    dest[i, j] = (srctmp * kernel[:, :, np.newaxis]).sum(axis=(0, 1))
# def filterBoxPixel(array,i,j)
#     array[i,j] = array[i,j]*1/9+array[i-1,j-1]*1/9+array[i,j-1]*1/9+array[i-1,j]*1/9+array[i+1,j+1]*1/9+array[i,j+1]*1/9
#
# def filterBox(image):
#     height,width,_ = image.shape
#     ret =
#     for i in range(0,height):
#         for j in range(0,width):
#             image[i,j] =   image[i,j
#
def sobelPixel(dest,src,i,j,kernel)


def sobelFilter(img):
    rows, cols,_ = img.shape

    kernel_horizontal = np.array([-1,0,1
                                            [-2,0,2]
                                            [-1,0,1]
                                                    ])

    kernel_vertical = np.array([-1,-2,-1
                                            [0,0,0]
                                            [1,2,1]
                                                  ])
    imgpaddingX = np.zeros((rows + 2, cols + 2, 3))
    imgpaddingX[1:-1, 1:-1] = img

    filtered = np.zeros(img.shape)
    for i in range(0, rows):
        for j in range(0, cols):
            sobelPixel(filtered, imgpaddingX, i, j, kernel)



if __name__=='__main__':
    img = cv2.imread('img/sonic.jpg', cv2.IMREAD_ANYCOLOR)
  #  filterBox(img)

