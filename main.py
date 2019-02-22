import numpy as np # Imports numpy under alias np
import cv2

arr = np.zeros(10)

print(arr)

# exercice 2

arr = np.zeros(10)
arr[4] = 1

print(arr)

#exercise 3

arr = np.arange(10,50)

print(arr)

#exercise 4

arr = np.arange(1,10)
arr = arr.reshape((3,3))

print(arr)

#exercise 5

arr = np.flip(arr,1)

print(arr)

#exercise 6

arr = np.flip(arr,1)

arr = np.flip(arr,0)

print(arr)

#exercise 7

arr = np.identity(3)

print(arr)

# exercise 8

arr = np.random.random((3, 3))

print(arr)

# exercise 9

arr = np.random.random(10)

print(arr)
print('mean: ',arr.mean())

# exercise 10

arr = np.ones((10,10))
arr[1:-1,1:-1] = 0

print(arr)

#exercise 17
arr = np.zeros((64,64))
gradient = np.arange(0.0,64.0)
gradient = gradient / 63.0
gradient = gradient *255.0
arr+=gradient
arr = np.uint8(arr)
cv2.imshow("Image",arr)


#exercise 18
gradient = np.reshape(gradient,(64,1))
arr = np.zeros((64,64))
arr+=gradient
arr = np.uint8(arr)
cv2.imshow("Image2",arr)
cv2.waitKey(0)
