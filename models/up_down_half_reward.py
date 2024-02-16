
import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

from  scipy.ndimage import gaussian_filter as G
from scipy.signal import argrelextrema
import numpy as np
# logger = logging.getLogger(__name__)
def calc_db(keypoints_seqs):
    # keypoints_seqs = keypoints_seqs.data.cpu().numpy()
    beats_np = []
    for keypoints in keypoints_seqs:    
        keypoints = np.array(keypoints).reshape(-1, 24, 3)
        kinetic_vel = np.mean(np.sqrt(np.sum((keypoints[1:] - keypoints[:-1]) ** 2, axis=2)), axis=1)    #변화량 측정
        kinetic_vel = G(kinetic_vel, 5)    #노이즈 제거 필터링
        # print(len(kinetic_vel))
        motion_beats = argrelextrema(kinetic_vel, np.less)    #신호에서 최소최대 찾음
        beat_np = np.zeros(len(keypoints))    #최소최대 필터링, 표시
        beat_np[motion_beats] = 1
        beats_np.append(beat_np)

    motion_beats = torch.from_numpy(np.stack(beats_np)).float().cuda()
    b, t = motion_beats.size()
    return motion_beats.view(b, t//8, 8).max(2)[0]


def ba_reward(keypoints, beats):
    keypoints = keypoints.data.cpu().numpy()
    dance_beats = calc_db(keypoints)
    b, t = beats.size()
    beats = beats.view(b, t//8, 8).float().max(2)[0]

    # To keep same as paper, uncomment the following two lines
    # dance_beats[beats == 0] = 1
    # beats[:, :] = 1
    return (beats*dance_beats - 0.5) * 10


class UpDownReward(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.mrate = config.rate

    def forward(self, pose, music, ds_rate):
        with torch.no_grad():
        # up: 0, 13, 14 normal vector, which should be 15-12, pose(t=0) = [0,0,0]
            n, t, c = pose.size()
            ds_rate = 8
            pose = pose.view(n, t//ds_rate, ds_rate, c//3, 3)

            up_norm = torch.cross(pose[:, :, :, 14, :], pose[:, :, :, 13, :])
            up_norm /= up_norm.norm(dim=-1)[:, :, :, None]
            #print(up_norm.size())    #(128, 28, 8, 3)
            up_side = (pose[:, :, :, 15, :] - pose[:, :, :, 12, :])
            #print(up_side.size())    #(128, 28, 8, 3)
            up_side[0, 0, 0, [0, 2]]=0

            up_direct = torch.sum(up_norm * up_side, dim=-1)
            up_direct /= up_direct.abs() + 1e-5
            up_norm *= up_direct[:, :, :,  None]
            up_norm[:, :, :, 1] = 0

            down_norm = torch.cross(pose[:, :, :, 4, :], pose[:, :, :, 5, :])
            down_norm /= down_norm.norm(dim=-1)[:, :, :, None]
            
            down_side = (pose[:, :, :, 4, :] - pose[:, :, :, 1, :] + pose[:, :, :, 5, :] - pose[:, :, :, 2, :] + pose[:, :, :, 4, :] - pose[:, :, :, 7, :] + pose[:, :, :, 5, :] - pose[:, :, :, 8, :])
            down_side[:, :, :, [0,2]]=0

            down_direct = torch.sum(down_norm * down_side, dim=-1)
            down_direct /= down_direct.abs() + 1e-5
            down_norm *= down_direct[:, :, :, None]
            down_norm[:, :, :, 1] = 0

        #상/하체가 골반 기준 같은방향인지 판단, y좌표 방향만 고려. N값이 양수-같은 방향, 음수-다른 방향
            N = torch.sum(up_side * down_side, dim=-1)

        #몸의 앞/뒷면 방향 같은지 판단, xz에 정사영, M값이 양수-같은 방향, 음수-다른 방향
            M = torch.sum(up_norm * down_norm, dim=-1)
            
            if (N*M).min(dim=-1)[0].all() < 0:   #N*M의 min값이 0보다 작은 경우 몸이 정상적, 양의 reward
                reward = abs((up_norm * down_norm).sum(dim=-1).min(dim=-1)[0])    
            else:    
                reward = -abs((up_norm * down_norm).sum(dim=-1).min(dim=-1)[0])

            #사실 기존 코드랑 다르게 N, M 값을 모두 계산했으니 reward 아래처럼 바꿔도? 이건 나중에 또 해봐야지
            #reward = -((N*M).min(dim=-1)[0])
            reward[reward >= 0] = 1.0

            reward *= self.mrate

            reward += ba_reward(pose, music)

        return reward.clone().detach() 

# reward = UpDownReward(None)
# aa = reward(torch.rand(8, 16, 72), None, 8)
# print(tuple([aa.view(-1).cpu().data.numpy(), ]))
