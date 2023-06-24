#!/usr/bin/env python
import os
import shutil
import sys

import sys
import os
import numpy as np
from pathlib import Path
import cv2 as cv
import torch
import torch as F
import torch.nn as nn
#import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from unet.unet_transfer import UNet16, input_size
import matplotlib.pyplot as plt
import argparse
from os.path import join
from PIL import Image
import gc
from utils import load_unet_vgg16, load_unet_resnet_101, load_unet_resnet_34
from tqdm import tqdm
import os
import coremltools as ct
import json
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# change to script dir
os.chdir("/Users/boshang/Documents/GitHub/crack_detection/Code")

if __name__ == '__main__':
    modle_dir = "../Models/model_unet_vgg_16_best.pt"
    input_dir = "../Data/Inputs"
    output_dir = "../Data/Outputs"
    out_pred_dir = output_dir + "/Crack_Masks"
    out_viz_dir = output_dir + "/Visualizations"

    parser = argparse.ArgumentParser()
    parser.add_argument('-img_dir',type=str, help='input dataset directory')
    parser.add_argument('-model_path', type=str, help='trained model path')
    parser.add_argument('-model_type', type=str, choices=['vgg16', 'resnet101', 'resnet34'])
    parser.add_argument('-out_viz_dir', type=str, default=out_viz_dir, required=False, help='visualization output dir')
    parser.add_argument('-out_pred_dir', type=str, default=out_pred_dir, required=False,  help='prediction output dir')
    parser.add_argument('-threshold', type=float, default=0.2 , help='threshold to cut off crack response')
    # parser.
    args = parser.parse_args()
    
    ''' clean previous output '''
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    if args.out_viz_dir != '':
        os.makedirs(args.out_viz_dir, exist_ok=True)
        for path in Path(args.out_viz_dir).glob('*.*'):
            os.remove(str(path))

    if args.out_pred_dir != '':
        os.makedirs(args.out_pred_dir, exist_ok=True)
        for path in Path(args.out_pred_dir).glob('*.*'):
            os.remove(str(path))






    model = load_unet_vgg16(modle_dir)

    channel_means = [0.485, 0.456, 0.406]
    channel_stds  = [0.229, 0.224, 0.225]
    paths = [path for path in Path(input_dir).glob('*.*')]
    for path in tqdm(paths):

        global train_tfms
        train_tfms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(channel_means, channel_stds)])

        img_0 = Image.open(str(path))
        img_0 = np.asarray(img_0)
        if len(img_0.shape) != 3:
            print(f'incorrect image shape: {path.name}{img_0.shape}')
            exit()

        img_0 = img_0[:,:,:3]

        img_height, img_width, img_channels = img_0.shape

        from inference_unet import evaluate_img_tfms, evaluate_img_patch_tfms

        prob_map_full = evaluate_img_tfms(model, img_0, train_tfms)
        
        if out_pred_dir != '':
            cv.imwrite(filename=join(out_pred_dir, f'{path.stem}.jpg'), img=(prob_map_full * 255).astype(np.uint8))

        if out_viz_dir != '':
                    # plt.subplot(121)
                    # plt.imshow(img_0), plt.title(f'{img_0.shape}')
                    if img_0.shape[0] > 2000 or img_0.shape[1] > 2000:
                        img_1 = cv.resize(img_0, None, fx=0.2, fy=0.2, interpolation=cv.INTER_AREA)
                    else:
                        img_1 = img_0

                    # plt.subplot(122)
                    # plt.imshow(img_0), plt.title(f'{img_0.shape}')
                    # plt.show()

                    prob_map_patch = evaluate_img_patch_tfms(model, img_1, train_tfms)

                    #plt.title(f'name={path.stem}. \n cut-off threshold = {args.threshold}', fontsize=4)
                    prob_map_viz_patch = prob_map_patch.copy()
                    prob_map_viz_patch = prob_map_viz_patch/ prob_map_viz_patch.max()
                    prob_map_viz_patch[prob_map_viz_patch < args.threshold] = 0.0
                    fig = plt.figure()
                    st = fig.suptitle(f'name={path.stem} \n cut-off threshold = {args.threshold}', fontsize="x-large")
                    ax = fig.add_subplot(231)
                    ax.imshow(img_1)
                    ax = fig.add_subplot(232)
                    ax.imshow(prob_map_viz_patch)
                    ax = fig.add_subplot(233)
                    ax.imshow(img_1)
                    ax.imshow(prob_map_viz_patch, alpha=0.4)

                    prob_map_viz_full = prob_map_full.copy()
                    prob_map_viz_full[prob_map_viz_full < args.threshold] = 0.0

                    ax = fig.add_subplot(234)
                    ax.imshow(img_0)
                    ax = fig.add_subplot(235)
                    ax.imshow(prob_map_viz_full)
                    ax = fig.add_subplot(236)
                    ax.imshow(img_0)
                    ax.imshow(prob_map_viz_full, alpha=0.4)

                    plt.savefig(join(out_viz_dir, f'{path.stem}.jpg'), dpi=500)
                    plt.close('all')
        break
    
    def display_segmentation(input_image, output_predictions):
        # Create a color pallette, selecting a color for each class
        # palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
        # colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
        # colors = (colors % 255).numpy().astype("uint8")

        # Plot the semantic segmentation predictions of 21 classes in each color
        # r = Image.fromarray(
        #     output_predictions.byte().cpu().numpy()
        # ).resize(input_image.size)
        # r.putpalette(colors)

        # Overlay the segmentation mask on the original image
        alpha_image = input_image.copy()
        alpha_image.putalpha(255)
        
        # r = r.convert("RGBA")
        # r = output_predictions.convert("RGBA")
        
        
        
        # image = Image.fromarray(output_predictions)
        # Get the color map by name:
        # cm = plt.get_cmap('gist_rainbow')
        prob_map_viz_patch = output_predictions.copy()
        # prob_map_viz_patch = prob_map_viz_patch/ prob_map_viz_patch.max()
        prob_map_viz_patch[prob_map_viz_patch < args.threshold*255] = 0.0
        # Apply the colormap like a function to any array:
        # im = cm(output_predictions)
        # print(type(image))
        # im = np.uint8(im * 255)#array(512, 384, 4) (0~1)->array(512, 384, 4) (0~255)
        # im = Image.fromarray(im)#array(512, 384, 4) (0~255) -> PIL.Images
    
        # 
        prob_map_viz_patch = Image.fromarray(prob_map_viz_patch)
        prob_map_viz_patch = prob_map_viz_patch.convert("RGBA")
        prob_map_viz_patch.show()
        prob_map_viz_patch.putalpha(128)
        seg_image = Image.alpha_composite(alpha_image, prob_map_viz_patch)
        seg_image.show()
        
    img_0 = Image.open(str(path))
    out_img=(prob_map_full * 255).astype(np.uint8)
    display_segmentation(img_0, out_img)
    
class WrappedDeeplabv3Resnet101(nn.Module):

    def __init__(self):
        super(WrappedDeeplabv3Resnet101, self).__init__()
        # self.model = torch.hub.load(
        #     'pytorch/vision:v0.6.0',
        #     'deeplabv3_resnet101',
        #     pretrained=True
        # ).eval()
        
        # device = torch.device('cpu')#torch.device('cuda:0' if len(availble_gpus) > 0 else 'cpu')

        # # Model
        # self.model = models.UperNet(num_classes=3, in_channels=3 ,backbone="resnet50", freeze_bn=False, freeze_backbone=False)
        # checkpoint = torch.load('/Users/boshang/Documents/GitHub/visual-inspection-aws/lambda/code/models/best_model.pth', map_location=device) #.eval()

        # if isinstance(checkpoint, dict) and 'state_dict' in checkpoint.keys():
        #     checkpoint = checkpoint['state_dict']
        # if 'module' in list(checkpoint.keys())[0] and not isinstance(self.model, torch.nn.DataParallel):
        #     self.model = torch.nn.DataParallel(self.model)
        # self.model.load_state_dict(checkpoint)
        # # print(model)
        # self.model.to(device)
        self.model = load_unet_vgg16(modle_dir)

    def forward(self, x):
        # res = self.model(x)
        # Extract the tensor we want from the output dictionary
        # x = res["out"]
        # return x
        mask = self.model(x)

        mask = F.sigmoid(mask[0, 0]).data.cpu()#.numpy()
        # img_height, img_width, img_channels = img.shape
        # mask = cv.resize(mask, (img_width, img_height), cv.INTER_AREA)
        return mask
    # def forward(self, x):
    #     conv1 = self.conv1(x)
    #     conv2 = self.conv2(self.pool(conv1))
    #     conv3 = self.conv3(self.pool(conv2))
    #     conv4 = self.conv4(self.pool(conv3))
    #     conv5 = self.conv5(self.pool(conv4))
    #
    #     center = self.center(self.pool(conv5))
    #
    #     dec5 = self.dec5(torch.cat([center, conv5], 1))
    #
    #     dec4 = self.dec4(torch.cat([dec5, conv4], 1))
    #     dec3 = self.dec3(torch.cat([dec4, conv3], 1))
    #     dec2 = self.dec2(torch.cat([dec3, conv2], 1))
    #     dec1 = self.dec1(torch.cat([dec2, conv1], 1))
    #
    #     if self.num_classes > 1:
    #         x_out = F.log_softmax(self.final(dec1), dim=1)
    #     else:
    #         x_out = self.final(dec1)
    #         #x_out = F.sigmoid(x_out)
    #
    #     return x_out
with torch.no_grad():
    img_0 = Image.open(str(path))
    img_0 = np.asarray(img_0)
    if len(img_0.shape) != 3:
        print(f'incorrect image shape: {path.name}{img_0.shape}')
        exit()

    img_0 = img_0[:,:,:3]
    input_width, input_height = input_size[0], input_size[1]

    img_1 = cv.resize(img_0, (input_width, input_height), cv.INTER_AREA)
    # nonlocal train_tfms
    X = train_tfms(Image.fromarray(img_1))
    # X = Variable(X.unsqueeze(0)).cuda()  # [N, 1, H, W]
    # X = Variable(X.unsqueeze(0)).cpu()
    X = X.unsqueeze(0)

    traceable_model = WrappedDeeplabv3Resnet101().eval()
# X.requires_grad = False
trace = torch.jit.trace(traceable_model, X)

input_batch = X
mlmodel = ct.convert(
    trace,
    inputs=[ct.TensorType(name="input", shape=input_batch.shape)],
)

mlmodel.save("SegmentationModel_no_metadata.mlmodel")

# load the model
mlmodel = ct.models.MLModel("SegmentationModel_no_metadata.mlmodel")

labels_json = {"labels": ["background", "aeroplane", "bicycle", "bird", "board", "bottle", "bus", "car", "cat", "chair", "cow", "diningTable", "dog", "horse", "motorbike", "person", "pottedPlant", "sheep", "sofa", "train", "tvOrMonitor"]}

mlmodel.user_defined_metadata["com.apple.coreml.model.preview.type"] = "imageSegmenter"
mlmodel.user_defined_metadata['com.apple.coreml.model.preview.params'] = json.dumps(labels_json)

mlmodel.save("SegmentationModel_with_metadata.mlmodel")