import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

import timeit

from PIL import Image
import os

def pil_test():
    cm_hot = mpl.cm.get_cmap('hot') #()->matplotlib.colors.LinearSegmented
    print(os.getcwd())
    os.chdir('./Code')
    img_src = Image.open('../Data/Outputs/Crack_Masks/IMG_0529.jpg').convert('L') #()->PIL.Image
    img_src.thumbnail((512,512))#PIL.Image->PIL.Image
    im = np.array(img_src)#PIL.Image->array
    im = cm_hot(im)#array(512,384)->array(512, 384, 4) (0~1)
    im = np.uint8(im * 255)#array(512, 384, 4) (0~1)->array(512, 384, 4) (0~255)
    im = Image.fromarray(im)#array(512, 384, 4) (0~255) -> PIL.Images
    im.save('test_hot.png')

def rgb2gray(rgb):
    return np.dot(rgb[:,:,:3], [0.299, 0.587, 0.114])

def plt_test():
    img_src = mpimg.imread('../Data/Outputs/Crack_Masks/IMG_0529.jpg')
    # im = rgb2gray(img_src)
    im = img_src
    f = plt.figure(figsize=(4, 4), dpi=128)
    plt.axis('off')
    plt.imshow(im, cmap='hot')
    plt.savefig('test2_hot.jpg', dpi=f.dpi)
    plt.close()

t = timeit.timeit(pil_test, number=30)
print('PIL: %s' % t)
t = timeit.timeit(plt_test, number=30)
print('PLT: %s' % t)