import torch
import torch.nn as nn
import torch.optim as optim

import constants as c
from d_scale_model import DScaleModel
from loss_functions import adv_loss
from utils import resize_dataset


class DiscriminatorModel(nn.Module):
    def __init__(self, device, height, width, scale_conv_layer_fms, scale_kernel_sizes, scale_fc_layer_sizes):
        super(DiscriminatorModel, self).__init__()
        self.device = device
        self.global_step = 0
        self.height = height
        self.width = width
        self.scale_conv_layer_fms = scale_conv_layer_fms
        self.scale_kernel_sizes = scale_kernel_sizes
        self.scale_fc_layer_sizes = scale_fc_layer_sizes
        self.num_scale_nets = len(scale_conv_layer_fms)

        self.define_model()

    def define_model(self):
        self.scale_nets = nn.ModuleList()
        for scale_num in range(self.num_scale_nets):
            scale_factor = 1. / 2 ** ((self.num_scale_nets - 1) - scale_num)
            self.scale_nets.append(DScaleModel(self.device,
                                               scale_num,
                                               int(self.height * scale_factor),
                                               int(self.width * scale_factor),
                                               self.scale_conv_layer_fms[scale_num],
                                               self.scale_kernel_sizes[scale_num],
                                               self.scale_fc_layer_sizes[scale_num]))

            self.optimizer = optim.SGD(self.parameters(), lr=c.LRATE_D)

    
    def forward(self, x, scale_num):
        return self.scale_nets[scale_num](x)


    def train_step(self, input_frames, gt_output_frames, generator):
        self.train()
        self.zero_grad()

        scale_preds = []
        g_scale_preds = []
        for scale_num in range(self.num_scale_nets):
            scale_factor = 1. / 2 ** ((self.num_scale_nets - 1) - scale_num)
            scale_height = int(self.height * scale_factor)
            scale_width = int(self.width * scale_factor)

            input_frames_scaled = resize_dataset(input_frames, scale_height, scale_width)#.to(self.device, dtype=torch.float)
            gt_output_frames_scaled = resize_dataset(gt_output_frames, scale_height, scale_width)#.to(self.device, dtype=torch.float)
            if scale_num > 0:
                last_gen_frames = g_scale_preds[scale_num - 1]
                last_gen_frames_scaled = resize_dataset(last_gen_frames, scale_height, scale_width)#.to(self.device, dtype=torch.float)
                input_frames_scaled = torch.cat([input_frames_scaled, last_gen_frames_scaled], dim=1).to(dtype=torch.float)

            g_scale_pred = generator.forward(input_frames_scaled, scale_num)
            g_scale_preds.append(g_scale_pred)

            d_scale_input = torch.cat([g_scale_pred, gt_output_frames_scaled], dim=0).to(dtype=torch.float)
            preds = self.forward(d_scale_input, scale_num)

            scale_preds.append(preds)

        labels_ts = torch.cat([torch.zeros((input_frames.shape[0], 1)),
                                      torch.ones((gt_output_frames.shape[0], 1))], dim=0).to(self.device)

        loss = adv_loss(scale_preds, labels_ts)

        loss.backward()
        self.optimizer.step()
        self.global_step += 1

        return loss
