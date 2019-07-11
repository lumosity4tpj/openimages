import os
import json
import pandas as pd
import cv2.cv2 as cv2
import argparse
import numpy as np
from progressbar import *

class csv_to_coco(object):
    def __init__(self):
        print('the process of the function: \
                _image & _categories --> _annotation --> save_coco_json ')

    # 构建COCO的image字段
    def _image(self,image_dir,image_txt_path,use_txt=False):
        """
        image{
            "id" : int, 
            "width" : int, 
            "height" : int, 
            "file_name" : str, 
        }
        """
        if use_txt == False:
            images_list = os.listdir(image_dir)
            init_id = 1
            image = []
            information = []
            # for line in images_list:
            widgets = ['image (please keep smiling ^-^) :', Percentage(),' ', Bar('#'), ' ', Timer(), ' ', ETA(), ' ', FileTransferSpeed()]
            pbar = ProgressBar(widgets=widgets)
            for line in pbar(images_list):
                image_single = {}
                image_single['id'] = init_id
                image_single['file_name'] = os.path.join(image_dir,line).replace('\\','/')
                # print(image_single['file_name'])
                # img = cv2.imread(image_single['file_name'])
                # print(img)
                img = cv2.imdecode(np.fromfile(image_single['file_name'],dtype=np.uint8),-1)
                # print(img.shape)
                image_single['height'] = img.shape[0]
                image_single['width'] = img.shape[1]
                init_id += 1
                image.append(image_single)
                height_width = str((image_single['height'],image_single['width']))
                information.append(image_single['file_name']+' '+height_width+'\n')
            # print(image)
            with open(image_txt_path,'w') as m:
                m.writelines(information)
        elif use_txt == True:
            init_id = 1
            image = []
            with open(image_txt_path, 'r') as f:
                samples = f.readlines()
                for line in samples:
                    image_single = {}
                    image_single['id'] = init_id
                    image_single['file_name'] = line.split(' ')[0]
                    image_single['height'] = int(line.split(',')[0].split('(')[-1])
                    image_single['width'] = int(line.split(' ')[-1].split(')')[0])
                    init_id += 1
                    image.append(image_single)
        else:
            print('Error:the param use_txt should be True or False!')
        print('_image is success!')
        return image

    # 构建COCO的categories字段
    def _categories(self,categories_csv_path):
        """
        categories[{
            "id" : int, 
            "name" : str, 
            "supercategory" : str, 
        }]
        """
        # class_bbox_batch = pd.read_csv('D:/download/class-descriptions-boxable.csv',header=None)
        class_bbox_batch = pd.read_csv(categories_csv_path,header=None)
        # print(class_bbox_batch)
        init_id = 1
        categories = []
        for line in class_bbox_batch.values:
            # print(line)
            categories_single = {}
            categories_single['supercategory'] = line[0]
            categories_single['id'] = init_id
            categories_single['name'] = line[1]
            init_id += 1
            categories.append(categories_single)
        # print(categories)
        print('_categories is success!')
        return categories

    # 保存成json格式
    def save_coco_json(self,image_dir,categories_csv_path,save_path,image_txt_path,use_txt):
        image = self._image(image_dir,image_txt_path,use_txt)
        categories = self._categories(categories_csv_path)
        instance = {}
        instance['images'] = image
        instance['categories'] = categories
        json.dump(instance,open(save_path,'w'),ensure_ascii=False,indent=2)
        print('^-^ success!')

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type = str, \
                        default = '/mnt/Guest/tangpeijun/OpenImages/challenge2018',
                        help = 'the path of image data')
    parser.add_argument('--categories_csv_path', type = str, \
                        default = '/mnt/Guest/tangpeijun/OpenImages/class-descriptions-boxable.csv',
                        help = 'the path of categories csv')
    parser.add_argument('--json_save_path', type = str, \
                        default = '/mnt/Guest/tangpeijun/OpenImages/test.json',
                        help = 'the path of json')
    parser.add_argument('--image_txt_path', type = str, \
                        default = '/mnt/Guest/tangpeijun/OpenImages/test.txt',
                        help = 'the path of _image information txt')
    parser.add_argument('--use_txt', type = bool, \
                        default = False,
                        help = 'use or not use the _image information txt')
    return parser.parse_args()

if __name__ == '__main__':
    opt = args()
    csv2json = csv_to_coco()
    csv2json.save_coco_json(opt.image_dir,opt.categories_csv_path,opt.json_save_path,opt.image_txt_path,opt.use_txt)