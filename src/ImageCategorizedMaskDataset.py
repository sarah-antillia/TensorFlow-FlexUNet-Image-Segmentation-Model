# Copyright 2025 antillia.com Toshiyuki Arai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# 2025/06/03

# ImageCategorizedMaskDataset.py

import os
import sys
import glob
import shutil
import cv2
import numpy as np

from ConfigParser import ConfigParser

class ImageCategorizedMaskDataset:

  def __init__(self, config_file):
    self.config       = ConfigParser(config_file)
    self.image_width  = self.config.get(ConfigParser.MODEL, "image_width", dvalue=512)

    self.image_height = self.config.get(ConfigParser.MODEL, "image_height", dvalue=512)
    self.image_channels = self.config.get(ConfigParser.MODEL, "image_channels", dvalue=3)
    self.color_order  = "RGB"
    self.mask_file_format = ".npz"
    self.num_classes  = self.config.get(ConfigParser.MODEL, "num_classes")


  def create(self, images_dir, masks_dir ):
    print("ImageCategorizedMaskDataset images_dir {} masks_dir {}".format(images_dir, masks_dir))

    image_files  = glob.glob(images_dir + "/*.jpg")
    image_files += glob.glob(images_dir + "/*.png")
    image_files += glob.glob(images_dir + "/*.tif")

    #print(image_files)
    num_images = len(image_files)

    npz_mask_files = glob.glob(masks_dir  + "/*" + self.mask_file_format)
    num_masks  = len(npz_mask_files)
    print("--- num_image_files: {}  num_mask_files:{}".format(num_images, num_masks))  
    if num_images != num_masks:
       error = "Unmatched the number of images and masks files."
       raise Exception(error)
  
    self.image_dtype = np.uint8
    print("--- num_classes {} image data_type {}".format(self.num_classes, self.image_dtype))
    print("--- num_images {} {} {}".format(num_images, self.image_height, self.image_width, self.image_channels))

    X = np.zeros((num_images, self.image_height, self.image_width, self.image_channels),
                 dtype=self.image_dtype)
       
    self.mask_dtype = bool
    if self.num_classes >1:
      self.mask_dtype = np.int8
      print("--- num_classes {} mask data_type  {}".format(self.num_classes, self.mask_dtype))
    Y = np.zeros((num_images, self.image_height, self.image_width, self.num_classes, ), 
                 dtype=self.mask_dtype)

    for n, image_file in enumerate(image_files):
        img = cv2.imread(image_file)
        if self.color_order == "RGB":
          img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        img = cv2.resize(img, (self.image_width, self.image_height))
        X[n] = img


    for n, npz_mask_file in enumerate(npz_mask_files):
        data = np.load(npz_mask_file)
        mask = data['mask']
        Y[n] = mask
    return X, Y
