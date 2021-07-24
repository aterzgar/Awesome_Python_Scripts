import cv2 as cv
import matplotlib.pyplot as plt

# converting to gray scale
img = cv.imread('lady.jpeg', cv.IMREAD_GRAYSCALE)

# remove noise
img= cv.GaussianBlur(img, (3, 3), 0)

# convolute with proper kernels
laplacian = cv.Laplacian(img,cv.CV_64F)
sobelx = cv.Sobel(img,cv.CV_64F,1,0,ksize=5)  # x
sobely = cv.Sobel(img,cv.CV_64F,0,1,ksize=5)  # y

#ploting the images
plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
plt.show()