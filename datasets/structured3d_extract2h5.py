import json
import os
from collections import defaultdict
import h5py
import cv2
import numpy as np
import torchvision.transforms as tf
from shapely.geometry import Polygon
from torch.utils import data
import torch

class Structured3D(data.Dataset):
    def __init__(self, phase='training'):
        self.phase = phase
        self.max_wall = 18 #20是什么
        self.max_floor = 1
        self.max_ceiling = 1

        #self.adr = os.path.join('data', 'Structured3D', 'line_anno_5000_500_500.json')
        # self.adr = os.path.join('data', 'Structured3D', '5000_500_500.json')
        self.adr = os.path.join('data', 'Structured3D', 'full_line.json')
        with open(self.adr, 'r') as f:
            files = json.load(f)
        self.data_set = defaultdict(list)
        for _, i in enumerate(files):
            img_name = i[0]
            scene = int(img_name.split('_')[1])
            if scene <= 2999:
                self.data_set['training'].append(i)
            elif scene <= 3249:
                self.data_set['validation'].append(i)
            else:
                self.data_set['test'].append(i)
        print('init over')

    def __getitem__(self, item):
        sample = self.data_set[self.phase][item]
        s0, s1, r, p = sample[0].split('_')[0:4]
        s = s0 + '_' + s1
        p = p.rstrip('.png')
        dirs = os.path.join('/home/Datasets/Structured3D', s, '2D_rendering', r, 'perspective/full', p)
        key_h5 = sample[0].rstrip('.png')

        layout_name = os.path.join(dirs, 'layout.json')

        pparams, labels, segs, endpoints = self.dataload(
            layout_name, sample[1],sample[3],sample[4],ratio_h=2.0,inh=384,inw=640)
        flag_floor =False
        flag_ceiling = False
        boxes_wall = np.empty((0,4),dtype=np.float32)
        boxes_floor = np.empty((0,4),dtype=np.float32)
        boxes_ceiling = np.empty((0,4),dtype=np.float32)
        param_wall = np.empty((0,4),dtype=np.float32)
        param_floor = np.empty((0,4),dtype=np.float32)
        param_ceiling = np.empty((0,4),dtype=np.float32)

        for i, (label, param) in enumerate(zip(labels, pparams)):
            yx = np.where(segs == i) # yx的意思就是矩形在图像中的位置吧
            box = np.array([np.min(yx[1]), np.min(yx[0]), np.max(
                yx[1]), np.max(yx[0])], dtype=np.float32) # 矩形的两个对角点
            if label == 0:
                boxes_wall = np.concatenate([boxes_wall,box.reshape(1,-1)],axis=0)
                param_wall = np.concatenate([param_wall,np.array(param,dtype=np.float32).reshape(1,-1)],axis=0)
            elif label == 1:
                boxes_floor = np.concatenate([boxes_floor,box.reshape(1,-1)],axis=0)
                param_floor = np.concatenate([param_floor,np.array(param,dtype=np.float32).reshape(1,-1)],axis=0)
                flag_floor = True
            elif label == 2:
                boxes_ceiling = np.concatenate([boxes_ceiling,box.reshape(1,-1)],axis=0)
                param_ceiling = np.concatenate([param_ceiling,np.array(param,dtype=np.float32).reshape(1,-1)],axis=0)
                flag_ceiling = True
        sort_id = np.argsort(boxes_wall[:,0]) # 按照坐标从左到右排序
        boxes_wall = boxes_wall[sort_id]
        param_wall = param_wall[sort_id]
        flag = np.zeros(2,dtype=np.float32)
        if flag_floor:
            flag[0] = 1
        if flag_ceiling:
            flag[1] = 1
        with h5py.File('/home/zhangjx/All_model/Non_cubiod/room_layout_transformer/data/structured3d_plane_feature_roi.h5','a') as h5_data:
            try:
                h5_data.create_dataset(key_h5+'/'+'flag',data=flag)
                h5_data.create_dataset(key_h5+'/'+'boxes_floor',data=boxes_floor)
                h5_data.create_dataset(key_h5+'/'+'boxes_ceiling',data=boxes_ceiling)
                h5_data.create_dataset(key_h5+'/'+'boxes_wall',data=boxes_wall)
                h5_data.create_dataset(key_h5+'/'+'param_floor',data=param_floor)
                h5_data.create_dataset(key_h5+'/'+'param_ceiling',data=param_ceiling)
                h5_data.create_dataset(key_h5+'/'+'param_wall',data=param_wall)
                h5_data.create_dataset(key_h5+'/'+'ratio',data=np.array(1/4,dtype=np.float32))
            except:
                pass
            
        ret = {
            'flag_floor'    :   flag_floor,
            'flag_ceiling'  :   flag_ceiling,
            'boxes_floor'   :   boxes_floor,
            'boxes_ceiling' :   boxes_ceiling,
            'boxes_wall'    :   boxes_wall,
            'param_floor'   :   param_floor,
            'param_ceiling' :   param_ceiling,
            'param_wall'    :   param_wall,
        }

        return ret
    def dataload(self, layout_name, lines,lines_floor,lines_ceiling, ratio_h, inh, inw):
        # planes
        with open(layout_name, 'r') as f:
            try:
                anno_layout = json.load(f)
            except:
                pass
            junctions = anno_layout['junctions']
            planes = anno_layout['planes']

            coordinates = []
            for k in junctions:
                coordinates.append(k['coordinate'])
            coordinates = np.array(coordinates) / ratio_h

            pparams = []
            labels = []
            segs = -1 * np.ones([inh, inw])
            i = 0
            for pp in planes:
                if len(pp['visible_mask']) != 0:
                    if pp['type'] == 'wall':
                        cout = coordinates[pp['visible_mask'][0]]
                        polygon = Polygon(cout)
                        if polygon.area >= 1000:
                            cout = cout.astype(np.int32)
                            cv2.fillPoly(segs, [cout], color=i)
                            pparams.append([*pp['normal'], pp['offset'] / 1000.])
                            labels.append(0)
                            i = i + 1
                    else:
                        for v in pp['visible_mask']:
                            cout = coordinates[v]
                            polygon = Polygon(cout)
                            if polygon.area > 1000:
                                cout = cout.astype(np.int32)
                                cv2.fillPoly(segs, [cout], color=i)
                                pparams.append([*pp['normal'], pp['offset'] / 1000.])
                                if pp['type'] == 'floor':
                                    labels.append(1)
                                else:
                                    labels.append(2)
                                i = i + 1
            # seg_tmp = np.array(segs * (255/segs.max()),dtype=np.uint8)
            # img = cv2.imread(layout_name[:-11]+ "rgb_rawlight.png")
            # cv2.imshow('img',img)
            # cv2.imshow('seg',seg_tmp)
            # cv2.waitKey()
        # lines
        endpoints = []
        for line in lines:
            if line[-1] == 2:  # occlusion line
                points = np.array([*line[4], *line[5]]).reshape(2, -1) / ratio_h
                ymin = np.min(points[1])
                ymax = np.max(points[1])
                x0 = line[2] * ymin + line[3] / ratio_h
                x1 = line[2] * ymax + line[3] / ratio_h
                endpoints.append([x0, ymin, x1, ymax])  # start/end point
            elif line[-1] == 1:  # wall/wall line
                wall_id, endpoint = line[0:2], line[2:4]
                xy = coordinates[endpoint]
                if (xy[0, 1] - xy[1, 1]) == 0:
                    continue
                endpoints.append([xy[0, 0], xy[0, 1], xy[1, 0], xy[1, 1]])
            elif line[-1] == 3:  # floor/wall line
                pass
            else:  # ceiling/wall line
                pass
        return pparams, labels, segs, endpoints
    # def dataload(self, layout_name, lines,lines_floor,lines_ceiling):
    #     # planes
    #     with open(layout_name, 'r') as f:
    #         anno_layout = json.load(f)
    #         junctions = anno_layout['junctions']
    #         planes = anno_layout['planes']

    #         coordinates = []
    #         for k in junctions:
    #             coordinates.append(k['coordinate'])
    #         # coordinates = np.array(coordinates) / 2.0
    #         coordinates = np.array(coordinates)

    #         pparams = []
    #         labels = []
    #         # segs = -1 * np.ones([360, 640])
    #         segs = -1 * np.ones([720, 1280])
    #         i = 0
    #         for pp in planes:
    #             if len(pp['visible_mask']) != 0:
    #                 if pp['type'] == 'wall':
    #                     cout = coordinates[pp['visible_mask'][0]]
    #                     polygon = Polygon(cout)
    #                     if polygon.area/2 >= 1000:
    #                         cout = cout.astype(np.int32)
    #                         cv2.fillPoly(segs, [cout], color=i)
    #                         pparams.append([*pp['normal'], pp['offset'] / 1000.])
    #                         labels.append(0)
    #                         i = i + 1
    #                 else:
    #                     for v in pp['visible_mask']:
    #                         cout = coordinates[v]
    #                         polygon = Polygon(cout)
    #                         if polygon.area/2 > 1000:
    #                             cout = cout.astype(np.int32)
    #                             cv2.fillPoly(segs, [cout], color=i)
    #                             pparams.append([*pp['normal'], pp['offset'] / 1000.])
    #                             if pp['type'] == 'floor':
    #                                 labels.append(1)
    #                             else:
    #                                 labels.append(2)
    #                             i = i + 1
    #         # wall: 0        floor:  1     ceiling:   2
    #         # seg_tmp = np.array(segs * (255/segs.max()),dtype=np.uint8)
    #         # img = cv2.imread(layout_name[:-11]+ "rgb_rawlight.png")
    #         # cv2.imshow('img',img)
    #         # cv2.imshow('seg',seg_tmp)
    #         # cv2.waitKey()
    #     # lines
    #     endpoints = []
    #     for line in lines:
    #         if line[-1] == 2:  # occlusion line
    #             points = np.array([*line[4], *line[5]]).reshape(2, -1) / 2
    #             ymin = np.min(points[1])
    #             ymax = np.max(points[1])
    #             x0 = line[2] * ymin + line[3] / 2
    #             x1 = line[2] * ymax + line[3] / 2
    #             endpoints.append([x0, ymin, x1, ymax])  # start/end point
    #         elif line[-1] == 1:  # wall/wall line
    #             wall_id, endpoint = line[0:2], line[2:4]
    #             xy = coordinates[endpoint]
    #             if (xy[0, 1] - xy[1, 1]) == 0:
    #                 continue
    #             endpoints.append([xy[0, 0], xy[0, 1], xy[1, 0], xy[1, 1]])
    #         elif line[-1] == 3:  # floor/wall line
    #             pass
    #         else:  # ceiling/wall line
    #             pass
    #     return pparams, labels, segs, endpoints

    def __len__(self):
        return len(self.data_set[self.phase])

if __name__ == '__main__':
    train_strcutured3d = Structured3D(phase='training')
    val_strcutured3d = Structured3D(phase='validation')
    test_strcutured3d = Structured3D(phase='test')
    train_dataloader = torch.utils.data.DataLoader(train_strcutured3d,batch_size=1)
    val_dataloader = torch.utils.data.DataLoader(val_strcutured3d,batch_size=1)
    test_dataloader = torch.utils.data.DataLoader(test_strcutured3d,batch_size=1)
    for i,data_one in enumerate(train_dataloader):
        if i % 1000 == 0:
            print(i)
    for i,data_one in enumerate(val_dataloader):
        if i % 1000 == 0:
            print(i)
    for i, data_one in enumerate(test_dataloader):
        if i % 1000 == 0:
            print(i)
    pass