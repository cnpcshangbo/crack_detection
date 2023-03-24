import cv2 as cv
img = cv.imread("Data/Outputs/Crack_Masks/out0050_image_rgb.jpg",
                cv.IMREAD_GRAYSCALE)
print(img.shape)
img_height, img_width = img.shape

with open('readme.txt', 'w') as f:
    f.write('img_width: ' + str(img_width))

middlerow = img[img_height//2, :]

max_value = max(middlerow)


cv.imshow("image", img)
cv.waitKey(0)

# It is for removing/deleting created GUI window from screen
# and memory
cv.destroyAllWindows()
