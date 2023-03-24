import cv2

# Load the two images
img1 = cv2.imread("BW.jpg")
img2 = cv2.imread("out0050_image_rgb.jpg", cv2.IMREAD_UNCHANGED)

# Resize the images to the same dimensions (if needed)
img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))

# Convert the overlay image to a 3-channel image (if needed)
if img2.ndim == 2:
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
elif img2.shape[2] == 1:
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)

# Extract the RGB channels of the overlay image
overlay = img2[:, :, 0:3]

# Apply the alpha blending equation
alpha = 0.5  # adjust as needed
output = cv2.addWeighted(overlay, alpha, img1, 1 - alpha, 0)

# Save the output image
cv2.imwrite('output.png', output)
