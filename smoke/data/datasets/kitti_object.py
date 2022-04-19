import os
import csv
import logging
import random
import numpy as np
from PIL import Image

from torch.utils.data import Dataset
import torch

from smoke.modeling.heatmap_coder import (
    get_transfrom_matrix,
    affine_transform,
    gaussian_radius,
    draw_umich_gaussian,
)
from smoke.modeling.smoke_coder import encode_label
from smoke.structures.params_3d import ParamsList
import cv2

TYPE_ID_CONVERSION = {
    'Car': 0,
    'Cyclist': 1,
    'Pedestrian': 2,
}


class KITTIDataset(Dataset):
    def __init__(self, cfg, root, is_train=True, transforms=None):
        super(KITTIDataset, self).__init__()
        self.root = root
        self.image_dir = os.path.join(root, "image_2")
        self.label_dir = os.path.join(root, "label_2")
        self.calib_dir = os.path.join(root, "calib")

        self.split = cfg.DATASETS.TRAIN_SPLIT if is_train else cfg.DATASETS.TEST_SPLIT
        self.is_train = is_train
        self.transforms = transforms
        root = '/mnt/lustre/zhouyunsong/pod_ad/pod_data/kitti'

        if self.split == "train":
            imageset_txt = os.path.join(root, "val.txt")
        elif self.split == "val":
            imageset_txt = os.path.join(root, "val.txt")
        elif self.split == "trainval":
            imageset_txt = os.path.join(root, "trainval.txt")
        elif self.split == "test":
            imageset_txt = os.path.join(root, "val.txt")
        else:
            raise ValueError("Invalid split!")
        image_files = []
        for line in open(imageset_txt, "r"):
            base_name = line.replace("\n", "")
            image_name = base_name + ".png"
            image_files.append(image_name)
        sort_file = False
        if sort_file:
            image_files.sort()
        self.image_files = image_files
        self.label_files = [i.replace(".png", ".txt") for i in self.image_files]
        self.num_samples = len(self.image_files)
        self.classes = cfg.DATASETS.DETECT_CLASSES

        self.flip_prob = cfg.INPUT.FLIP_PROB_TRAIN if is_train else 0
        self.aug_prob = cfg.INPUT.SHIFT_SCALE_PROB_TRAIN if is_train else 0
        self.shift_scale = cfg.INPUT.SHIFT_SCALE_TRAIN
        self.num_classes = len(self.classes)

        self.input_width = cfg.INPUT.WIDTH_TRAIN
        self.input_height = cfg.INPUT.HEIGHT_TRAIN
        self.output_width = self.input_width // cfg.MODEL.BACKBONE.DOWN_RATIO
        self.output_height = self.input_height // cfg.MODEL.BACKBONE.DOWN_RATIO
        self.max_objs = cfg.DATASETS.MAX_OBJECTS

        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing KITTI {} set with {} files loaded".format(self.split, self.num_samples))

        self.C_mat = [
            [1, 0, 0, 0],
            [0, 0.996, 0.087, 0],
            [0, 0.087, 0.996, 0],
            [0, 0, 0, 1]
        ]
        self.B_mat = [
            [1, 0, 0, 0],
            [0, 1, 0, 0.5],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]
        self.A_mat2 = [
            [1, 0, 0, 0],
            [0, 0.99863, 0.052336, 0],
            [0, 0.052336, 0.99863, 0],
            [0, 0, 0, 1]
        ]
        self.A_mat = [
            [1, 0, 0, 0],
            [0, 0.999390827, 0.0348994967, 0],
            [0, 0.0348994967, 0.999390827, 0],
            [0, 0, 0, 1]
        ]
        with open("/mnt/lustre/zhouyunsong/pod_ad/3DSSD/3DSSD/angle_2.txt","r") as disturb_file:
            P_change = disturb_file.readlines()
        self.P_change = P_change
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # load default parameter here
        original_idx = self.label_files[idx].replace(".txt", "")
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        img = Image.open(img_path)
        
        im = cv2.imread(img_path) #[H, W, 3]
        h, w = im.shape[:2]
        center = (w // 2, h // 2)
        self.root_dir = 'datasets/kitti/training'
        train_branch_num = np.random.randint(3)
        train_branch_num = 1
        B_mat = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]]
        if train_branch_num == 0:
            A_mat = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]]


        with open(os.path.join(self.root_dir, "calib", self.image_files[idx].replace('.png', '') + ".txt"), "r") as file_calib:
            lines_calib = file_calib.readlines()
        with open(os.path.join(self.root_dir, "label_2", self.image_files[idx].replace('.png', '') + ".txt"), "r") as file_label:
            lines_label = file_label.readlines()
        
        pitch = -float(self.P_change[int(original_idx)//2].strip().split(' ')[0])
        roll = float(self.P_change[int(original_idx)//2].strip().split(' ')[1])
        if train_branch_num == 1:
            A_mat = [
                [1, 0, 0, 0],
                [0, np.cos(pitch*np.pi/180), np.sin(pitch*np.pi/180), 0],
                [0, -np.sin(pitch*np.pi/180), np.cos(pitch*np.pi/180), 0],
                [0, 0, 0, 1]
            ]
            B_mat = [
                [np.cos(roll*np.pi/180), -np.sin(roll*np.pi/180), 0, 0],
                [np.sin(roll*np.pi/180), np.cos(roll*np.pi/180), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ]
        calib_dic = {}
        P0_line = lines_calib[0].strip().split(':')[1].split(' ')[1:]
        R0_rect_line = lines_calib[4].strip().split(':')[1].split(' ')[1:]
        P0 = np.array(P0_line, dtype=float).reshape(3,4)
        line_xyz = []
        for label_line in lines_label:
            line_info = label_line.strip().split(' ')
            if line_info[0] not in['Car', 'Cyclist', 'Pedestrian']:
                continue
            line_xyz.append(line_info[-4:-1])
        line_xyz = np.array(line_xyz, dtype=float).reshape(-1,3)


        P0_expand = np.eye(4, dtype=float)
        P0_expand[:3,:] = P0
        M = np.dot(np.dot(P0_expand, np.dot(np.array(A_mat),np.array(B_mat))), np.linalg.inv(P0_expand))
        pitch_disturb = 5
        if train_branch_num == 1:
            im = cv2.warpPerspective(im,M[:3,:3],(w,h))
        img = Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
        
        
        anns, K = self.load_annotations(idx)

        center = np.array([i / 2 for i in img.size], dtype=np.float32)
        size = np.array([i for i in img.size], dtype=np.float32)

        """
        resize, horizontal flip, and affine augmentation are performed here.
        since it is complicated to compute heatmap w.r.t transform.
        """
        flipped = False
        if (self.is_train) and (random.random() < self.flip_prob):
            flipped = True
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            center[0] = size[0] - center[0] - 1
            K[0, 2] = size[0] - K[0, 2] - 1

        affine = False
        if (self.is_train) and (random.random() < self.aug_prob):
            affine = True
            shift, scale = self.shift_scale[0], self.shift_scale[1]
            shift_ranges = np.arange(-shift, shift + 0.1, 0.1)
            center[0] += size[0] * random.choice(shift_ranges)
            center[1] += size[1] * random.choice(shift_ranges)

            scale_ranges = np.arange(1 - scale, 1 + scale + 0.1, 0.1)
            size *= random.choice(scale_ranges)

        center_size = [center, size]
        trans_affine = get_transfrom_matrix(
            center_size,
            [self.input_width, self.input_height]
        )
        trans_affine_inv = np.linalg.inv(trans_affine)
        img = img.transform(
            (self.input_width, self.input_height),
            method=Image.AFFINE,
            data=trans_affine_inv.flatten()[:6],
            resample=Image.BILINEAR,
        )

        trans_mat = get_transfrom_matrix(
            center_size,
            [self.output_width, self.output_height]
        )


        if not self.is_train:
            # for inference we parametrize with original size
            target = ParamsList(image_size=size,
                                is_train=self.is_train)
            target.add_field("trans_mat", trans_mat)
            target.add_field("K", K)
            if self.transforms is not None:
                img, target = self.transforms(img, target)
            if train_branch_num == 2:
                img_new = torch.zeros(img.shape, dtype = torch.float)
                img_new[:,pitch_disturb:,:] = img[:,:-pitch_disturb,:] # H
                img = img_new

            return img, target, original_idx

        heat_map = np.zeros([self.num_classes, self.output_height, self.output_width], dtype=np.float32)
        regression = np.zeros([self.max_objs, 3, 8], dtype=np.float32)
        cls_ids = np.zeros([self.max_objs], dtype=np.int32)
        proj_points = np.zeros([self.max_objs, 2], dtype=np.int32)
        p_offsets = np.zeros([self.max_objs, 2], dtype=np.float32)
        dimensions = np.zeros([self.max_objs, 3], dtype=np.float32)
        locations = np.zeros([self.max_objs, 3], dtype=np.float32)
        rotys = np.zeros([self.max_objs], dtype=np.float32)
        reg_mask = np.zeros([self.max_objs], dtype=np.uint8)
        flip_mask = np.zeros([self.max_objs], dtype=np.uint8)

        for i, a in enumerate(anns):
            a = a.copy()
            cls = a["label"]

            locs = np.array(a["locations"])           

            xyz = locs #[N, 3]
            xyz = np.array([locs[0], locs[1], locs[2], 1], dtype=float)
            xyz = np.dot(np.dot(A_mat,B_mat), xyz)
            xyz[0], xyz[1], xyz[2] = xyz[0]/xyz[3], xyz[1]/xyz[3], xyz[2]/xyz[3]
            locs = xyz[:3]

            rot_y = np.array(a["rot_y"])
            if flipped:
                locs[0] *= -1
                rot_y *= -1

            point, box2d, box3d = encode_label(
                K, rot_y, a["dimensions"], locs
            )
            point = affine_transform(point, trans_mat)
            box2d[:2] = affine_transform(box2d[:2], trans_mat)
            box2d[2:] = affine_transform(box2d[2:], trans_mat)
            box2d[[0, 2]] = box2d[[0, 2]].clip(0, self.output_width - 1)
            box2d[[1, 3]] = box2d[[1, 3]].clip(0, self.output_height - 1)
            h, w = box2d[3] - box2d[1], box2d[2] - box2d[0]

            if (0 < point[0] < self.output_width) and (0 < point[1] < self.output_height):
                point_int = point.astype(np.int32)
                p_offset = point - point_int
                radius = gaussian_radius(h, w)
                radius = max(0, int(radius))
                heat_map[cls] = draw_umich_gaussian(heat_map[cls], point_int, radius)

                cls_ids[i] = cls
                regression[i] = box3d
                proj_points[i] = point_int
                p_offsets[i] = p_offset
                dimensions[i] = np.array(a["dimensions"])
                locations[i] = locs
                rotys[i] = rot_y
                reg_mask[i] = 1 if not affine else 0
                flip_mask[i] = 1 if not affine and flipped else 0

        target = ParamsList(image_size=img.size,
                            is_train=self.is_train)
        target.add_field("hm", heat_map)
        target.add_field("reg", regression)
        target.add_field("cls_ids", cls_ids)
        target.add_field("proj_p", proj_points)
        target.add_field("dimensions", dimensions)
        target.add_field("locations", locations)
        target.add_field("rotys", rotys)
        target.add_field("trans_mat", trans_mat)
        target.add_field("K", K)
        target.add_field("reg_mask", reg_mask)
        target.add_field("flip_mask", flip_mask)
        target.add_field("pitch_roll", np.array([pitch, roll]))

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        if train_branch_num == 2:
            img_new = torch.zeros(img.shape, dtype = torch.float)
            img_new[:,pitch_disturb:,:] = img[:,:-pitch_disturb,:] # H
            img = img_new
        
        return img, target, original_idx

    def load_annotations(self, idx):
        annotations = []
        file_name = self.label_files[idx]
        fieldnames = ['type', 'truncated', 'occluded', 'alpha', 'xmin', 'ymin', 'xmax', 'ymax', 'dh', 'dw',
                      'dl', 'lx', 'ly', 'lz', 'ry']

        if self.is_train:
            with open(os.path.join(self.label_dir, file_name), 'r') as csv_file:
                reader = csv.DictReader(csv_file, delimiter=' ', fieldnames=fieldnames)

                for line, row in enumerate(reader):
                    if row["type"] in self.classes:
                        annotations.append({
                            "class": row["type"],
                            "label": TYPE_ID_CONVERSION[row["type"]],
                            "truncation": float(row["truncated"]),
                            "occlusion": float(row["occluded"]),
                            "alpha": float(row["alpha"]),
                            "dimensions": [float(row['dl']), float(row['dh']), float(row['dw'])],
                            "locations": [float(row['lx']), float(row['ly']), float(row['lz'])],
                            "rot_y": float(row["ry"])
                        })

        # get camera intrinsic matrix K
        with open(os.path.join(self.calib_dir, file_name), 'r') as csv_file:
            reader = csv.reader(csv_file, delimiter=' ')
            for line, row in enumerate(reader):
                if row[0] == 'P2:':
                    K = row[1:]
                    K = [float(i) for i in K]
                    K = np.array(K, dtype=np.float32).reshape(3, 4)
                    K = K[:3, :3]
                    break

        return annotations, K
