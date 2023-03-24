import cv2
# Load the two images
img1 = cv2.imread("out0050_image_rgb.jpg")
img2 = cv2.imread("BW.jpg")

# Calculate the intersection and union between the two images
intersection = cv2.bitwise_and(img1, img2)
union = cv2.bitwise_or(img1, img2)

# Calculate the area of the intersection and union
intersection_area = cv2.countNonZero(
    cv2.cvtColor(intersection, cv2.COLOR_BGR2GRAY))
union_area = cv2.countNonZero(cv2.cvtColor(union, cv2.COLOR_BGR2GRAY))

# Calculate the Intersection over Union (IoU)
iou = intersection_area / union_area

print("Intersection over Union (IoU): {:.2f}".format(iou))
