import os
import json
import pandas as pd
import cv2.cv2 as cv2
import argparse
import numpy as np
from progressbar import *
import zipfile

class csv_to_coco(object):
    def __init__(self):
        print('the process of the function: \
                _image & _categories --> _annotation --> save_coco_json ')

    # 构建COCO的image字段
    def _image(self,image_dir):
        """
        images{
            "id" : int, 
            "width" : int, 
            "height" : int, 
            "file_name" : str, 
        }
        return: images, image_info: dict{str:list}
        """
        if image_dir.split('.')[-1] == 'zip':
            z = zipfile.ZipFile(image_dir,'r')
            z.extractall(os.path.dirname(image_dir))
            print("extractall finshed!")
            images_list = z.namelist()
            # print(images_list)
            # images_list = os.listdir(image_dir)
            del images_list[0]
            images_list = [line.split('.')[0].split('/')[-1] for line in images_list]
            # print(images_list)
            dir_path = image_dir.split('.')[0]
        else:
            images_list = os.listdir(image_dir)
            images_list = [line.split('.')[0] for line in images_list]
            # print(images_list)
            dir_path = image_dir
        # images_list = os.listdir(image_dir)
        init_id = 1
        image_info = {}
        images = []
        # for line in images_list:
        widgets = ['image (please keep smiling ^-^) :', Percentage(),' ', Bar('#'), ' ', Timer(), ' ', ETA(), ' ', FileTransferSpeed()]
        pbar = ProgressBar(widgets=widgets)
        for line in pbar(images_list):
            image_single = {}
            image_single['id'] = init_id
            image_single['file_name'] = dir_path + '/' + line + '.jpg'
            # print(image_single['file_name'])
            # img = cv2.imread(image_single['file_name'])
            # print(img)
            img = cv2.imdecode(np.fromfile(image_single['file_name'],dtype=np.uint8),-1)
            # print(img.shape)
            image_single['height'] = img.shape[0]
            image_single['width'] = img.shape[1]
            init_id += 1
            images.append(image_single)
            image_info[line] = [image_single['id'],image_single['height'],image_single['width']]
        # print(images)
        print('_image is success!')
        return images,image_info

    def dict_generator(self,indict,pre=None,pre_key=None,pre_value=None):
        """
            This code is modified from https://blog.csdn.net/qq_17550379/article/details/80276477.
            return: list[supercategory(str), subcategory(str)]
        """
        pre = pre[:] if pre else []
        pre_key = pre_key if pre_key else ' '
        pre_value = pre_value[:] if pre_value else ['']
        if isinstance(indict, dict):
            for key, value in indict.items():
                if isinstance(value, dict):
                    if len(value) == 0:
                        yield pre+[key, '{}']
                    else:
                        for d in self.dict_generator(value, pre):
                            yield d
                elif isinstance(value, list):
                    if len(value) == 0:                   
                        yield pre+[key, '[]']
                    else:
                        pre_key = key
                        for v in value:
                            for d in self.dict_generator(v, pre, pre_key, pre_value):
                                yield d
                elif isinstance(value, tuple):
                    if len(value) == 0:
                        yield pre+[key, '()']
                    else:
                        for v in value:
                            for d in self.dict_generator(v, pre):
                                yield d
                else:
                    if pre_key == "Subcategory" or "Part":
                        yield pre + pre_value + [value]
                    else:
                        yield pre + [value]
                    pre_value[0] = value
        else:
            yield indict

    # 构建COCO的categories字段
    def _categories(self,categories_csv_path,categories_json_path):
        """
        categories[{
            "id" : int, 
            "name" : str, 
            "supercategory" : str, 
        }]
        """
        ##################### get json_list from json #####################
        dataset = json.load(open(categories_json_path, 'r'))
        assert type(dataset)==dict, 'json file format {} not supported'.format(type(dataset))
        json_list = []
        for i in self.dict_generator(dataset):
            # print(i)
            json_list.append(i)
        # print(len(json_list))

        ############## get catagories_list from csv & match csv and json ####################### 
        class_bbox_batch = pd.read_csv(categories_csv_path,header=None)
        # print(class_bbox_batch)
        categories_info = {}
        categories = []
        init_id = 1
        for line in class_bbox_batch.values:
            for outer in json_list:
                if line[0] == outer[-1]:
                    categories_single = {}
                    categories_single['supercategory'] = outer[0]
                    categories_single['id'] = init_id
                    categories_single['name'] = outer[-1]
                    init_id += 1
                    categories.append(categories_single)
                    categories_info[line[0]] = categories_single['id']
                    break
        for i in range(len(categories)):
            assert i+1 == categories[i]['id'],'error,check the categories'
            if categories[i]['name'] in categories_info:
                assert categories_info[categories[i]['name']] == categories[i]['id'],'error,check the categories'
        # print(len(categories_info))
        # print(categories)
        print('_categories is success!')
        return categories,categories_info

    # 构建COCO的annotation字段
    def _annotation(self,annotation_csv_path,images_dict,categories_dict):
        """
        annotation[{
            "id": int,    
            "image_id": int,
            "bbox": [x,y,width,height],
            "category_id": int,
            "segmentation": RLE or [polygon],
            "area": float,
            "iscrowd": 0 or 1,
        }]
        """
        annotations_bbox_batch = pd.read_csv(annotation_csv_path,iterator=True,engine='python',\
                                            usecols=['ImageID','LabelName','XMin','XMax','YMin','YMax'])
        chunksize = 10000
        chunks = []
        loop = True
        while loop:
            try:
                chunk = annotations_bbox_batch.get_chunk(chunksize)
                chunks.append(chunk)
            except StopIteration:
                loop = False
                print('Iteration is stopped')
        annotations_bbox = pd.concat(chunks,ignore_index=True)
        # print(annotations_bbox)
        init_id = 1
        annotations = []
        widgets = ['annotation (please keep smiling ^-^) :', Percentage(),' ', Bar('#'), ' ', Timer(), ' ', ETA(), ' ', FileTransferSpeed()]
        pbar = ProgressBar(widgets=widgets)
        for line in pbar(annotations_bbox.values):
            # print(line[0])
            if line[0] in images_dict:
                line[2] *= images_dict[line[0]][2]
                line[3] *= images_dict[line[0]][2]
                line[4] *= images_dict[line[0]][1]
                line[5] *= images_dict[line[0]][1]
                # print(line[2],line[3],line[4],line[5])  
                h = line[5]-line[4]
                w = line[3]-line[2]
                x_y_w_h = [line[2],line[5],w,h] # top-left corner
                a = []
                a.append([line[2],line[4],line[2],line[4]+0.5*h,line[2],line[5],line[2]+0.5*w,line[5], \
                        line[3],line[5],line[3],line[5]-0.5*h,line[3],line[4],line[3]-0.5*w,line[4]])
                annotation_single = {}
                annotation_single['segmentation'] = a
                annotation_single['bbox'] = x_y_w_h
                annotation_single['area'] = h * w
                annotation_single['iscrowd'] = 0
                annotation_single['image_id'] = images_dict[line[0]][0]
                if line[1] in categories_dict:
                    annotation_single['category_id'] = categories_dict[line[1]]
                annotation_single['id'] = init_id
                # print('*'*20)
                # print('id:',init_id)
                init_id += 1
                # print(annotation_single)
                annotations.append(annotation_single)
        # print(annotations)
        print('_annotation is success!')
        return annotations

    # 保存成json格式
    def save_coco_json(self,image_dir,categories_csv_path,annotation_csv_path,json_save_path,categories_json_path):
        images,images_dict = self._image(image_dir)
        categories,categories_dict = self._categories(categories_csv_path,categories_json_path)
        annotations = self._annotation(annotation_csv_path,images_dict,categories_dict)
        instance = {}
        instance['images'] = images
        instance['annotations'] = annotations
        instance['categories'] = categories
        json.dump(instance,open(json_save_path,'w'),ensure_ascii=False,indent=2)
        print('^-^ success!')

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type = str, \
                        default = 'E:/openimages/train_08',
                        help = 'the path of image data(file or zip)')
    parser.add_argument('--categories_csv_path', type = str, \
                        default = 'E:/openimages/class-descriptions-boxable.csv',
                        help = 'the path of categories csv')
    parser.add_argument('--categories_json_path', type = str, \
                        default = 'E:/openimages/bbox_labels_600_hierarchy.json',
                        help = 'the path of categories json')    
    parser.add_argument('--annotation_csv_path', type = str, \
                        default = 'E:/openimages/train-annotations-bbox.csv',
                        help = 'the path of annotation csv')
    parser.add_argument('--json_save_path', type = str, \
                        default = 'E:/openimages/train_08.json',
                        help = 'the path of json')
    return parser.parse_args()

if __name__ == '__main__':
    opt = args()
    csv2json = csv_to_coco()
    csv2json.save_coco_json(opt.image_dir,opt.categories_csv_path,opt.annotation_csv_path,opt.json_save_path,opt.categories_json_path)