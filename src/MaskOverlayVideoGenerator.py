# Copyright 2026 antillia.com Toshiyuki Arai
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

# 2026/04/27

import os
import sys
import cv2
import glob
import shutil
from ConfigParser import ConfigParser
from PIL import Image
import traceback


class MaskOverlayVideoGenerator:

  def __init__(self, file_format=".png", fps=5, video_fileformat=".mp4"):
    self.file_format = file_format
    self.fps = fps
    self.video_fileformat = video_fileformat
    self.fourcc   = cv2.VideoWriter_fourcc(*'mp4v')
    self.duration = 500
    self.loop     = 0

  def generate(self, image_folder, output_video_path):
    images = sorted(glob.glob(os.path.join(image_folder, "*" + self.file_format)))
    
    if not images:
       print("Not found image files")
       return

    if self.video_fileformat == ".mp4":
      self.create_mp4(images, output_video_path)
    elif self.video_fileformat == ".gif":
      self.create_gif(images, output_video_path)


  def create_mp4(self, images, output_video_path):
    frame = cv2.imread(images[0])
    height, width, layers = frame.shape
    size = (width, height)

    video = cv2.VideoWriter(output_video_path, self.fourcc, self.fps, size)

    for image_path in images:
        img = cv2.imread(image_path)
        # img = cv2.resize(img, size)
        video.write(img)

    video.release()
    print("Saved a file {}".format(output_video_path))


  def create_gif(self, image_files, output_video_path):
    images = []
    for image_file in image_files:
      image = Image.open(image_file)
      images += [image]
    if images:
      images[0].save(
        output_video_path,
        save_all=True,
        append_images = images[1:],
        duration = self.duration,
        loop = self.loop)
      print("Saved a file {}".format(output_video_path))


if __name__ == "__main__":
  try:
    config_file = "./train_eval_infer.config"
    if len(sys.argv) == 2:
      config_file = sys.argv[1]
    if not os.path.exists(config_file):
      raise Exception("Not found " + config_file)
  
    config = ConfigParser(config_file)  
    output_dir       = config.get(ConfigParser.INFER3D, "output_dir")
    output_video_dir = config.get(ConfigParser.INFER3D, "video_dir", dvalue="./video_3d")
    video_fps = config.get(ConfigParser.INFER3D, "video_fps", dvalue=10)
    video_type = config.get(ConfigParser.INFER3D, "video_type", dvalue="overlays")
    video_fileformat = config.get(ConfigParser.INFER3D, "video_fileformat", dvalue=".mp4")
    
    if os.path.exists(output_video_dir):
      shutil.rmtree(output_video_dir)
    os.makedirs(output_video_dir)
    
    subdirs = os.listdir(output_dir)
    
    for subdir in subdirs:
      full_subdir = os.path.join(output_dir, subdir)
      overlays_dir = os.path.join(full_subdir, video_type)
      
      basename = os.path.basename(overlays_dir)
      output_fullpath = os.path.join(output_video_dir, basename + video_fileformat) 
       
      generator = MaskOverlayVideoGenerator(fps = video_fps, video_fileformat=video_fileformat)
      generator.generate(overlays_dir, output_fullpath)
  except:
    traceback.print_exc()
    
