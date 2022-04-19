import torch
from torch import nn

import torch.nn.functional as F

from smoke.structures.image_list import to_image_list

from ..backbone import build_backbone, build_transfer
from ..heads.heads import build_heads
import numpy as np
import cv2
from PIL import Image

class KeypointDetector(nn.Module):
    '''
    Generalized structure for keypoint based object detector.
    main parts:
    - backbone
    - heads
    '''

    def __init__(self, cfg):
        super(KeypointDetector, self).__init__()

        self.backbone = build_backbone(cfg)
        self.transfer = build_transfer(cfg)
        self.heads = build_heads(cfg, self.backbone.out_channels, 1)
        #for p in self.parameters():
        #    p.requires_grad = False
        self.heads_para = build_heads(cfg, self.backbone.out_channels, 0)
        
    def forward(self, images, targets=None):
        """
        Args:
            images:
            targets:

        Returns:

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        features = self.backbone(images.tensors)
        P_result, para_losses = self.heads_para(features, targets)
        images_fixed = self.external_parameters_fix(images, P_result, targets)

        

        features_raw = self.backbone(images.tensors)
        features_fixed = self.backbone(images_fixed)

        features, image_loss = self.image_transfer(features_fixed, features_raw)


        result, detector_losses = self.heads(features, targets)

        if self.training:
            losses = {}
            losses.update(image_loss)
            losses.update(para_losses)
            losses.update(detector_losses)

            return losses
        else:
            result_expand = torch.FloatTensor(result.shape[0], result.shape[1]+2)
            result_expand[:,:-3],  result_expand[:,-1]= result[:,:-1], result[:,-1]
            result_expand[:,-3:-1] = P_result[0,:]

            return result

    def external_parameters_fix(self, images, result, targets):
        images_fixed = images.tensors.clone()
        for i in range(images.tensors.shape[0]):
            img_origin = images.tensors[i,...].cpu().numpy() #[3, h. w]
            img_origin = img_origin.transpose(1, 2, 0)
            h, w = img_origin.shape[:2]
            pitch, roll = result.detach().cpu().numpy()[i,0], result.detach().cpu().numpy()[i,1]
            A_mat = [
                [1, 0, 0],
                [0, np.cos(pitch*np.pi/180), np.sin(pitch*np.pi/180)],
                [0, -np.sin(pitch*np.pi/180), np.cos(pitch*np.pi/180)]
            ]
            B_mat = [
                [np.cos(roll*np.pi/180), -np.sin(roll*np.pi/180), 0],
                [np.sin(roll*np.pi/180), np.cos(roll*np.pi/180), 0],
                [0, 0, 1]
            ]
            A_mat, B_mat = torch.tensor(A_mat), torch.tensor(B_mat)
            K = torch.stack([t.get_field("K") for t in targets]).cpu()
            K = K.to(torch.float64)
            M = torch.mm(K[i,...], torch.mm(A_mat, torch.mm(B_mat, K[i,...].inverse())))   
            img = cv2.warpPerspective(img_origin,M.numpy(),(w,h))
            images_fixed[i,...] = torch.from_numpy(img.transpose(2, 0, 1)).cuda()
           
            
        return images_fixed

    def image_transfer(self, features_fixed, features_raw):


        feature_cat = torch.cat((features_fixed, features_raw), 1)

        feature_final = self.transfer(feature_cat)

        Content_net = ContentLoss(features_fixed)
        Style_net = StyleLoss(features_raw)

        image_loss = dict(content_loss=Content_net(feature_final), style_loss=Style_net(feature_final))

        return feature_final, image_loss

class ContentLoss(nn.Module):

    def __init__(self, target, ):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        loss = F.mse_loss(input, self.target)
        return loss


def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=特征图数量
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # 将FXML调整为\ hat FXML

    G = torch.matmul(features, features.t())

    # 我们将gram矩阵的值“规范化”
    # 除以每个要素图中的元素数量。
    return G.div(a * b * c * d)


class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        loss = F.mse_loss(G, self.target)
        return loss