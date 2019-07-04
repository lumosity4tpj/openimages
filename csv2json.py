import os
import json
import pandas as pd
import cv2.cv2 as cv2
import argparse

class csv_to_coco(object):
    def __init__(self):
        print('the process of the function: \
                _image & _categories --> _annotation --> save_coco_json ')

    # 构建COCO的image字段
    def _image(self,image_dir,image_csv_path,image_txt_path,use_txt=False):
        """
        image{
            "id" : int, 
            "width" : int, 
            "height" : int, 
            "file_name" : str, 
        }
        """
        if use_txt == False:
            # images_batch = pd.read_csv('D:/download/validation-images.csv',iterator=True,engine='python')
            images_batch = pd.read_csv(image_csv_path,iterator=True,engine='python')
            chunksize = 1000
            chunks = []
            loop = True
            while loop:
                try:
                    chunk = images_batch.get_chunk(chunksize)
                    chunks.append(chunk)
                except StopIteration:
                    loop = False
                    print('Iteration is stopped')
            images_dataframe = pd.concat(chunks,ignore_index=True)
            # print(images_dataframe)
            init_id = 1
            image = []
            information = []
            for line in images_dataframe.values:
                image_single = {}
                image_single['id'] = init_id
                image_single['file_name'] = os.path.join(image_dir,line[0]).replace('\\','/')#################################
                print(image_single['file_name'])
                img = cv2.imread(image_single['file_name'])
                # print(img)
                print(img.shape)
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
            print('_image is success!')
        else:
            print('Error:the param use_txt should be True or False!')
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

    # 构建COCO的annotation字段
    def _annotation(self,annotation_csv_path,image,categories):
        """
        annotation[{
            "id": int,    
            "image_id": int,
            "bbox": [x,y,width,height],
            "category_id": int,
            "segmentation": RLE or [polygon],
            "area": float,
            "iscrowd": 0 or 1,
            # add 'others_info':
        }]
        """
        # annotations_bbox_batch = pd.read_csv('D:/download/validation-annotations-bbox.csv',iterator=True,engine='python',\
        #     usecols=['ImageID','LabelName','XMin','XMax','YMin','YMax','IsOccluded','IsTruncated','IsGroupOf','IsDepiction','IsInside'])
        annotations_bbox_batch = pd.read_csv(annotation_csv_path,iterator=True,engine='python',\
            usecols=['ImageID','LabelName','XMin','XMax','YMin','YMax','IsOccluded','IsTruncated','IsGroupOf','IsDepiction','IsInside'])
        chunksize = 1000
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
        annotation = []
        for line in annotations_bbox.values:
            x_y_w_h = [line[2],line[5],line[3]-line[2],line[5]-line[4]] # top-left corner
            h = line[5]-line[4]
            w = line[3]-line[2]
            a = []
            a.append([line[2],line[4],line[2],line[4]+0.5*h,line[2],line[5],line[2]+0.5*w,line[5], \
                    line[3],line[5],line[3],line[5]-0.5*h,line[3],line[4],line[3]-0.5*w,line[4]])
            # IsOccluded_IsTruncated_IsGroupOf_IsDepiction_IsInside = [line[i] for i in range(6,11)]
            annotation_single = {}
            annotation_single['segmentation'] = a
            annotation_single['bbox'] = x_y_w_h
            annotation_single['id'] = init_id
            annotation_single['area'] = 1.0
            annotation_single['iscrowd'] = 0
            # annotation_single['others_info'] = IsOccluded_IsTruncated_IsGroupOf_IsDepiction_IsInside
            print('*'*20)
            print('id:',init_id)
            for i in range(len(image)):
                # print(image[i])
                if os.path.basename(image[i]['file_name']).split('.')[0] == line[0]:
                    annotation_single['image_id'] = image[i]['id']  # or i+1
                    print('image_id:',annotation_single['image_id'])
                    continue
            for i in range(len(categories)):
                if categories[i]['supercategory'] == line[1]:
                    annotation_single['category_id'] = categories[i]['id']  # or i+1
                    print('category_id',annotation_single['category_id'])
                    continue
            init_id += 1
            # print(annotation_single)
            annotation.append(annotation_single)
        # print(annotation)
        print('_annotation is success!')
        return annotation

    # 保存成json格式
    def save_coco_json(self,image_dir,image_csv_path,categories_csv_path,annotation_csv_path,save_path,image_txt_path,use_txt):
        image = self._image(image_dir,image_csv_path,image_txt_path,use_txt)
        categories = self._categories(categories_csv_path)
        annotation = self._annotation(annotation_csv_path,image,categories)
        instance = {}
        instance['images'] = image
        instance['annotation'] = annotation
        instance['categories'] = categories
        json.dump(instance,open(save_path,'w'),ensure_ascii=False,indent=2)
        print('^-^ success!')

# def make_dirs(des_dir):
#     if not os.path.exists(des_dir):
#         os.mkdir(des_dir)

# def move_files(origin_dir,des_dir,folder):
#     imgFiles = [file for file in os.listdir(origin_dir)]
#     for file in imgFiles:
#         lst = file.split("_")
#         origin_file = os.path.join(origin_dir, file)
#         des_file = os.path.join(des_dir, folder, file)
#         os.rename(origin_file, des_file)

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type = str, \
                        default = 'D:/download/validation',
                        help = 'the path of image data')
    parser.add_argument('--image_csv_path', type = str, \
                        default = 'D:/download/validation-images.csv',
                        help = 'the path of image csv')
    parser.add_argument('--categories_csv_path', type = str, \
                        default = 'D:/download/class-descriptions-boxable.csv',
                        help = 'the path of categories csv')
    parser.add_argument('--annotation_csv_path', type = str, \
                        default = 'D:/download/validation-annotations-bbox.csv',
                        help = 'the path of annotation csv')
    parser.add_argument('--json_save_path', type = str, \
                        default = 'D:/download/val.json',
                        help = 'the path of json')
    parser.add_argument('--image_txt_path', type = str, \
                        default = 'D:/download/val_image.txt',
                        help = 'the path of _image information txt')
    parser.add_argument('--use_txt', type = bool, \
                        default = False,
                        help = 'use or not use the _image information txt')
    return parser.parse_args()

if __name__ == '__main__':
    opt = args()
    # make_dirs(opt.image_txt_path)
    csv2json = csv_to_coco()
    csv2json.save_coco_json(opt.image_dir,opt.image_csv_path,opt.categories_csv_path,opt.annotation_csv_path,opt.json_save_path,opt.image_txt_path,opt.use_txt)