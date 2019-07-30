import os
import time
import json
import argparse


class json_to_csv(object):

    def __init__(self,result_json_path,test_json_path,csv_save_path):
        super(json_to_csv,self).__init__()
        self.result_json_path = result_json_path
        self.test_json_path = test_json_path
        self.csv_save_path = csv_save_path

    def get_result_dict(self,indict,thred_value):
        """
        args:
            indict: json      
            thred_value: Threshold
        return: 
            result_dict
                eg: 435:[
                        [211, 0.977134644985199, 49.68760299682617, 680.3424072265625, 180.63580703735352, 1.0],
                        [456, 0.9689508676528931, 106.6458511352539, 332.49053955078125, 170.1101303100586, 182.79473876953125]
                        ] 
        """
        result_dict = {}
        lines = []
        pre_image_id = 0
        if isinstance(indict, list):
            for value in indict:
                if isinstance(value, dict):
                    if value['score'] >= thred_value:
                        if value['image_id'] != pre_image_id and pre_image_id != 0:
                            result_dict[pre_image_id] = lines
                            lines = []
                            lines.append([value['category_id'],value['score'],value['bbox'][0],value['bbox'][1],value['bbox'][2],value['bbox'][3]])
                            pre_image_id = value['image_id']
                        else:
                            lines.append([value['category_id'],value['score'],value['bbox'][0],value['bbox'][1],value['bbox'][2],value['bbox'][3]])
                            pre_image_id = value['image_id']
        else:
            print('error,check the get_result_dict_code & result_json_file')
        return result_dict

    def get_info_dict(self,indict):
        """
        args:
            indict: json      
        return: 
            images_info_dict
                eg: 1: ['00000b4dcff7f799', 683, 1024], 2: ['00001a21632de752', 681, 1024]
            categories_info_dict
                eg: 1: '/m/011k07', 2: '/m/011q46kg', 3: '/m/012074'
        """
        images_info_dict = {}
        categories_info_dict = {}
        categories_info_dict['ImageId'] = 'PredictionString'
        if isinstance(indict, dict):
            for keys,values in indict.items():
                if keys == 'images':
                    for value in values:
                        images_info_dict[value['id']] = [os.path.basename(value['file_name']).split('.')[0],value['height'],value['width']]
                elif keys == 'categories':
                    for value in values:
                        categories_info_dict[value['id']] = value['name']
                else:
                    print('error,check the get_info_dict_code & test_json_file')
        else:
            print('error,check the get_info_dict_code & test_json_file')
        return images_info_dict,categories_info_dict

    # according to the result_json,but length of result < 99999
    def get_result_transfer(self,result_dict,images_info_dict,categories_info_dict):
        csv_dict = {}
        for key in result_dict:
            PredictionString = ''
            for v in result_dict[key]:
                try:
                    v[0] = categories_info_dict[v[0]]
                except:
                    print('error,check the categories')
                else:
                    v[1] = round(v[1],2)
                    v[2] = round(v[2]/images_info_dict[key][2],3)
                    v[3] = round(v[3]/images_info_dict[key][1],3)
                    v[4] = round(v[4]/images_info_dict[key][2],3)
                    v[5] = round(v[5]/images_info_dict[key][1],3)
                    PredictionString += str(v[0])+' '+str(v[1])+' '+str(v[2])+' '+str(v[3])+' '+str(v[4])+' '+str(v[5])+' '
            csv_dict[images_info_dict[key][0]] = PredictionString
        return csv_dict
    
    # according to the test_json, make sure the length is 99999(let the images(not in result) is null)
    def get_result_transfer1(self,result_dict,images_info_dict,categories_info_dict):
        csv_dict = {}
        # bad_list = []
        # good_list = []
        for key in images_info_dict:
            PredictionString = ''
            if key in result_dict:
                for v in result_dict[key]:
                    try:
                        v[0] = categories_info_dict[v[0]]
                    except:
                        print('error,check the categories')
                    else:
                        v[1] = round(v[1],2)
                        v[2] = round(v[2]/images_info_dict[key][2],3)
                        v[3] = round(v[3]/images_info_dict[key][1],3)
                        v[4] = round(v[4]/images_info_dict[key][2],3)
                        v[5] = round(v[5]/images_info_dict[key][1],3)
                        PredictionString += str(v[0])+' '+str(v[1])+' '+str(v[2])+' '+str(v[3])+' '+str(v[4])+' '+str(v[5])+' '
                csv_dict[images_info_dict[key][0]] = PredictionString
                # good_list.append(images_info_dict[key][0])
            else:
                # if images not in result,let the info is ''
                csv_dict[images_info_dict[key][0]] = ''
                # bad_list.append(images_info_dict[key][0])
        # print(len(bad_list)) # 92
        # print(len(good_list))
        return csv_dict

    def save_csv(self):
        with open(self.result_json_path) as f:
            info = json.load(f)
            # print(info)
            result_dict = self.get_result_dict(info,0)
            # print(result_dict)

        with open(self.test_json_path) as f:
            info = json.load(f)
            # print(info)
            images_info_dict,categories_info_dict = self.get_info_dict(info)
            # print(images_info_dict)
            # print(categories_info_dict)

        # csv_dict = self.get_result_transfer(result_dict,images_info_dict,categories_info_dict)
        csv_dict = self.get_result_transfer1(result_dict,images_info_dict,categories_info_dict)

        with open(self.csv_save_path, 'w') as f:
            [f.write('ImageId,PredictionString\n')]
            [f.write('{0},{1}\n'.format(key, value)) for key, value in csv_dict.items()]

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_json_path', type = str, \
                        default = 'C:/Users/lumosity/Desktop/result_40.pkl.bbox.json',
                        help = 'the path of result json')
    parser.add_argument('--test_json_path', type = str, \
                        default = 'C:/Users/lumosity/Desktop/test.json',
                        help = 'the path of test json')
    parser.add_argument('--csv_save_path', type = str, \
                        default = './result.csv',
                        help = 'the path of csv')
    return parser.parse_args()

if __name__ == '__main__':
    s_t = time.time()
    opt = args()
    json2csv = json_to_csv(opt.result_json_path,opt.test_json_path,opt.csv_save_path)
    json2csv.save_csv()
    t_t = time.time() - s_t
    print("the transfer finshed in %.0f h %.0f m %.0f s"%(t_t//3600,(t_t%3600)//60,t_t%60))
