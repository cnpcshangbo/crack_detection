import coremltools as coremltools
import numpy as np
import PIL.Image
import PIL.Image as Image
import cv2 as cv
import torch as F

Height = 448  # use the correct input image height
Width = 448  # use the correct input image width

# Assumption: the mlmodel's input is of type MultiArray and of shape (1, 3, Height, Width).
model_expected_input_shape = (1, 3, Height, Width) # depending on the model description, this could be (3, Height, Width)

# Load the model.
model = coremltools.models.MLModel('/Users/boshang/Documents/GitHub/crack_detection/Code/SegmentationModel_with_metadata.mlmodel')

def load_image_as_numpy_array(path, resize_to=None):
    # resize_to: (Width, Height)
    img = PIL.Image.open(path)
    if resize_to is not None:
        img = img.resize(resize_to)
    img_np = np.array(img).astype(np.float32) # shape of this numpy array is (Height, Width, 3)
    return img_np

# Load the image and resize using PIL utilities.
img_as_np_array = load_image_as_numpy_array('/Users/boshang/Documents/GitHub/crack_detection/Data/Inputs/IMG_0520.jpg', resize_to=(Width, Height)) # shape (Height, Width, 3)

# PIL returns an image in the format in which the channel dimension is in the end,
# which is different than Core ML's input format, so that needs to be modified.
img_as_np_array = np.transpose(img_as_np_array, (2,0,1)) # shape (3, Height, Width)

# Add the batch dimension if the model description has it.
img_as_np_array = np.reshape(img_as_np_array, model_expected_input_shape)

# Now call predict.
out_dict = model.predict({'input': img_as_np_array})
mask = out_dict['var_10']
print(type(out_dict['var_10']))
# mask = F.sigmoid(mask[0, 0]).data.cpu().numpy()
# img_height, img_width, img_channels = img.shape
# mask = cv.resize(mask, (img_width, img_height), cv.INTER_AREA)
# cv.imwrite(filename='testML.jpg', img=(out_dict['var_338']).astype(np.uint8))
img = Image.fromarray(mask, "RGB")
print(img.size)
# Display the Numpy array as Image
img.show()

# Save the Numpy array as Image
image_filename = "opengenus_image.jpeg"
img.save(image_filename)