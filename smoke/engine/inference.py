import logging
from tqdm import tqdm

import torch
import numpy as np
import os
import math

from smoke.utils import comm
from smoke.utils.timer import Timer, get_time_str
from smoke.data.datasets.evaluation import evaluate
from PIL import Image
import cv2
import sys
sys.path.append("/media/lion/Seagate_Backup/SenseTimeResearch/pod_ad/pod_ad-zhouyunsong/pod/utils")
from vis_helper import *

def compute_on_dataset(model, data_loader, device, timer=None):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")

    for _, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids = batch["images"], batch["targets"], batch["img_ids"]
        images = images.to(device)
        with torch.no_grad():
            if timer:
                timer.tic()
            output = model(images, targets)
            if timer:
                torch.cuda.synchronize()
                timer.toc()
            output = output.to(cpu_device)

        
        item = image_ids[0]
        [sequence, index] = item.split('/')
        P_change_path = os.path.join('/media/lion/Seagate_Backup/SenseTimeResearch/pod_ad/3DSSD/3DSSD/angle_pose', sequence+'.txt')
        with open(P_change_path, 'r') as P_change_file:
            P_change = P_change_file.readlines()


        pitch = -float(P_change[int(index)].strip().split(' ')[0])
        roll = float(P_change[int(index)].strip().split(' ')[1])
        # pitch, roll = 0, 0
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


        root = '/media/lion/Seagate_Backup/SenseTimeResearch/pod_ad/3DSSD/3DSSD/dataset/KITTI/object/'
        img_path = os.path.join(root, sequence, 'testing', 'image_2_origin', index+'.png')

        calib_path = os.path.join(root, sequence, 'testing', 'calib', index+'.txt')
        label_path = os.path.join(root, sequence, 'testing', 'label_new', index+'.txt')

        with open(calib_path, "r") as file_calib:
            lines_calib = file_calib.readlines()
        with Image.open(img_path) as img_origin:
            img_origin = np.array(img_origin.convert('RGB'))
        with open(label_path, "r") as file_label:
            lines_label = file_label.readlines()
        
        w, h = img_origin.shape[1], img_origin.shape[0]
        calib_dic = {}
        P0_line = lines_calib[0].strip().split(':')[1].split(' ')[1:]
        P0 = np.array(P0_line, dtype=float).reshape(3,4)
        P0 = torch.from_numpy(P0).float()
        A_mat = torch.tensor(A_mat, dtype=torch.float)
        B_mat = torch.tensor(B_mat, dtype=torch.float)
        line_xyz = []
        line_output = []
        for label_line in lines_label:
            line_info = label_line.strip().split(' ')
            line_info = line_info[:-1]
            if line_info[0] not in['Car', 'Cyclist', 'Pedestrian']:
                continue
            line_xyz.append(line_info[-4:-1])
            line_output.append(line_info[1:])
        line_xyz = np.array(line_xyz, dtype=float).reshape(-1,3)
        line_xyz = torch.from_numpy(line_xyz).float()
        linx_origin = line_xyz
        const_num = torch.ones((line_xyz[:,0].shape)).float()
        line_xyz = torch.stack((line_xyz[:,0], line_xyz[:,1], line_xyz[:,2], const_num), dim = 0)
        line_xyz = torch.mm(P0, line_xyz).t()
        line_xyz[:,0], line_xyz[:,1] = line_xyz[:,0]/line_xyz[:,2], line_xyz[:,1]/line_xyz[:,2]

        heigh, width = h, w
        theta = -5
        x_center, y_center = width/2, heigh/2

        pts1 = np.float32([[0,0],[w/2,0],[0,h/2]])
        pts2 = np.float32([[2,2],[w/2,0],[0,h/2]])
        M = cv2.getAffineTransform(pts1,pts2)

        pts = np.float32([[1,1,10,1],[1,2,10,1],[1,1,20,1],[-1,1,20,1]])
        pts1 = np.dot(P0.numpy(),pts.T).T
        pts2 = np.dot(P0.numpy(),np.dot(A_mat.numpy(),pts.T)).T
        pts1[:,0], pts1[:,1] = pts1[:,0]/pts1[:,2], pts1[:,1]/pts1[:,2]
        pts1 = pts1[:,:2]
        pts2[:,0], pts2[:,1] = pts2[:,0]/pts2[:,2], pts2[:,1]/pts2[:,2]
        pts2 = pts2[:,:2]

        P0_expand = np.eye(4, dtype=float)
        P0_expand[:3,:] = P0.numpy()
        M = np.dot(np.dot(P0_expand, np.dot(A_mat.numpy(),B_mat.numpy())), np.linalg.inv(P0_expand))
        pitch_disturb = 0
        trans_matrix = np.eye(3, dtype=float)
        trans_matrix = M[:3,:3]
        trans_matrix = torch.from_numpy(trans_matrix).float()
                        
        xyz = output[:, 9:12] #[N, 3]
        const_num = torch.ones((xyz[:,0].shape)).float()
        xyz = torch.stack((xyz[:,0], xyz[:,1], xyz[:,2], const_num), dim = 0)
        xyz_2 = xyz
        xyz = torch.mm(P0, xyz).t()
        xyz_P_change = torch.mm(torch.mm(A_mat,B_mat).inverse(), xyz_2).t()
        xyz_P_change[:,0], xyz_P_change[:,1], xyz_P_change[:,2] = xyz_P_change[:,0]/xyz_P_change[:,3], xyz_P_change[:,1]/xyz_P_change[:,3], xyz_P_change[:,2]/xyz_P_change[:,3]
        xyz_P_change = xyz_P_change[:,:3]
        xyz[:,0], xyz[:,1] = xyz[:,0]/xyz[:,2], xyz[:,1]/xyz[:,2]
        
        
        xyz[:,0], xyz[:,1] = xyz[:,0]*xyz[:,2], xyz[:,1]*xyz[:,2]
        xyz = torch.mm(P0[:,:3].inverse(), xyz.t()).t()
        output[:, 9:12] = xyz

        results_dict.update(
            {img_id: output for img_id in image_ids}
        )
    return results_dict


def inference(
        model,
        data_loader,
        dataset_name,
        eval_types=("detections",),
        device="cuda",
        output_folder=None,

):
    device = torch.device(device)
    num_devices = comm.get_world_size()
    logger = logging.getLogger(__name__)
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))

    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    predictions = compute_on_dataset(model, data_loader, device, inference_timer)
    comm.synchronize()

    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        "Total run time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / img per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset),
            num_devices,
        )
    )
    if not comm.is_main_process():
        return

    return evaluate(eval_type=eval_types,
                    dataset=dataset,
                    predictions=predictions,
                    output_folder=output_folder, )
