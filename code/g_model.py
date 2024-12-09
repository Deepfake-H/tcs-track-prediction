from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
from torchvision.transforms import Resize
from PIL import Image
import numpy as np
import cv2
import os

import constants as c
from loss_functions import combined_loss
from utils import psnr_error, sharp_diff_error, resize_dataset, generate_datetime_filename, adjust_order_for_img, calculate_center_of_red_point


class GeneratorModel(nn.Module):
    def __init__(self, device, height_train, width_train, height_test, width_test, scale_layer_fms, scale_kernel_sizes):
        super(GeneratorModel, self).__init__()
        self.device = device
        self.global_step = 0
        self.scale_preds = None
        self.scale_gtss = None
        self.d_preds = None
        self.height_train = height_train
        self.width_train = width_train
        self.height_test = height_test
        self.width_test = width_test
        self.scale_layer_fms = scale_layer_fms
        self.scale_kernel_sizes = scale_kernel_sizes
        self.num_scale_nets = len(scale_layer_fms)

        self.define_model()

    def define_model(self):
        self.scale_nets = nn.ModuleList()
        for scale_num in range(self.num_scale_nets):
            layers = []
            for i in range(len(self.scale_kernel_sizes[scale_num])):
                layers.append(nn.Conv2d(self.scale_layer_fms[scale_num][i],
                                        self.scale_layer_fms[scale_num][i + 1],
                                        self.scale_kernel_sizes[scale_num][i],
                                        padding=c.PADDING_G))
                if i == len(self.scale_kernel_sizes[scale_num]) - 1:
                    layers.append(nn.Tanh())
                else:
                    layers.append(nn.ReLU())
            self.scale_nets.append(nn.Sequential(*layers))

        self.optimizer = optim.Adam(self.parameters(), lr=c.LRATE_G)

    def forward(self, x, scale_num):
        return self.scale_nets[scale_num](x)

    def predict_all(self, input_frames):
        scale_preds = []
        for scale_num in range(self.num_scale_nets):
            scale_factor = 1. / 2 ** ((self.num_scale_nets - 1) - scale_num)
            scale_height = int(input_frames.shape[2] * scale_factor)
            scale_width = int(input_frames.shape[3] * scale_factor)

            input_frames_scaled = resize_dataset(input_frames, scale_height, scale_width)#.to(self.device, dtype=torch.float)

            if scale_num > 0:
                last_gen_frames = scale_preds[scale_num - 1]
                last_gen_frames_scaled = resize_dataset(last_gen_frames, scale_height, scale_width)#.to(self.device, dtype=torch.float)
                input_frames_scaled = torch.cat((input_frames_scaled, last_gen_frames_scaled), dim=1)#.to(self.device, dtype=torch.float)

            scale_preds.append(self.forward(input_frames_scaled, scale_num))

        return scale_preds


    def train_step(self, input_frames, gt_frames, discriminator=None):
        self.train()
        self.zero_grad()

        scale_preds = []
        scale_gtss = []
        d_preds = []
        for scale_num in range(self.num_scale_nets):
            scale_factor = 1. / 2 ** ((self.num_scale_nets - 1) - scale_num)
            scale_height = int(self.height_train * scale_factor)
            scale_width = int(self.width_train * scale_factor)

            input_frames_scaled = resize_dataset(input_frames, scale_height, scale_width).to(dtype=torch.float)
            gts_scaled = resize_dataset(gt_frames, scale_height, scale_width).to(dtype=torch.float)

            if scale_num > 0:
                last_gen_frames = scale_preds[scale_num - 1]
                last_gen_frames_scaled = resize_dataset(last_gen_frames, scale_height, scale_width).to(dtype=torch.float)
                input_frames_scaled = torch.cat((input_frames_scaled, last_gen_frames_scaled), dim=1).to(dtype=torch.float)

            preds = self.forward(input_frames_scaled, scale_num)


            scale_preds.append(preds)
            scale_gtss.append(gts_scaled)
            if c.ADVERSARIAL:
                d_pred = discriminator.forward(preds, scale_num)
                d_preds.append(d_pred)

        # scale_gtss_ts = torch.tensor(scale_gtss).to(self.device)
        # d_preds_ts = torch.tensor(d_preds).to(self.device)
        if c.ADVERSARIAL:
            loss = combined_loss(scale_preds, scale_gtss, d_preds, device=self.device)
        else:
            loss = combined_loss(scale_preds, scale_gtss, device=self.device)

        psnr = psnr_error(scale_preds, scale_gtss)
        sharpdiff = sharp_diff_error(scale_preds, scale_gtss)

        loss.backward()
        self.optimizer.step()
        self.global_step += 1

        return loss, psnr, sharpdiff

    def test_batch(self, logger, input_frames, gt_frames, num_rec_out=1, save_imgs=True):
        self.eval()

        if num_rec_out < 1:
            raise ValueError('num_rec_out must be >= 1')


        working_input_frames = deepcopy(input_frames)  # input frames that will shift w/ recursion
        rec_preds = []
        rec_psnr = []
        rec_sharpdiff = []
        for rec_num in range(num_rec_out):
            logger.info('Calculating recursion %d ...' % (rec_num + 1))
            working_gt_frames = gt_frames[:, rec_num * 3:(rec_num + 1) * 3, :, :]

            working_input_frames = resize_dataset(working_input_frames, self.height_test, self.width_test)
            working_gt_frames = resize_dataset(working_gt_frames, self.height_test, self.width_test)

            with torch.no_grad():
                scale_preds = self.predict_all(working_input_frames)
                preds = scale_preds[-1]

                psnr = psnr_error([preds], [working_gt_frames])
                sharpdiff = sharp_diff_error([preds], [working_gt_frames])

                rec_preds.append(preds)
                rec_psnr.append(psnr)
                rec_sharpdiff.append(sharpdiff)

                # remove oldest frames, add newest predictions
                working_input_frames = torch.cat((working_input_frames[:, 3:, :, :], preds), 1)

                torch.cuda.empty_cache()
                logger.info('Recursion %d: | PSNR Error : %.2f Sharpdiff Error : %.2f' % ((rec_num + 1), psnr, sharpdiff))

        file_name_prefix = generate_datetime_filename()
        if save_imgs:
            input_frames_image = adjust_order_for_img(input_frames)
            gt_frames_image = adjust_order_for_img(gt_frames)

            input_images_frames = np.empty([len(input_frames_image),
                                            (input_frames_image[0].shape[2] // (3 * c.NUM_OF_SUBSET)),
                                            c.FULL_HEIGHT,
                                            c.FULL_WIDTH,
                                            3])
            gt_images_frames = np.empty([len(gt_frames_image),
                                         (gt_frames_image[0].shape[2] // 3),
                                         c.FULL_HEIGHT,
                                         c.FULL_WIDTH,
                                         3])

            preds_images_frames = np.copy(gt_images_frames)

            for i in range(len(input_frames_image)):
                for j in range(input_frames_image[i].shape[2] // (3 * c.NUM_OF_SUBSET)):
                    img = Image.fromarray(
                        np.uint8((input_frames_image[i][:, :, j * 3 * c.NUM_OF_SUBSET:j * 3 * c.NUM_OF_SUBSET + 3].detach().cpu().numpy() + 1) * 127.5))
                    img.save(
                        c.IMG_SAVE_DIR + file_name_prefix + "_file" + str(i + 1) + "_input_frame" + str(j + 1) + ".png")
                    input_images_frames[i, j] = img

            for i in range(len(gt_frames_image)):
                for j in range(gt_frames_image[i].shape[2] // 3):
                    img = Image.fromarray(
                        np.uint8((gt_frames_image[i][:, :, j * 3:j * 3 + 3].detach().cpu().numpy() + 1) * 127.5))
                    img.save(
                        c.IMG_SAVE_DIR + file_name_prefix + "_file" + str(i + 1) + "_gt_frame" + str(j + 1) + ".png")
                    gt_images_frames[i, j] = img

            for i in range(len(rec_preds)):
                preds_image = adjust_order_for_img(rec_preds[i])
                for j in range(len(preds_image)):
                    img = Image.fromarray(np.uint8((preds_image[j].detach().cpu().numpy() + 1) * 127.5))
                    img.save(c.IMG_SAVE_DIR + file_name_prefix + "_file" + str(j+1) + "_pred_frame" + str(i+1) + ".png")
                    preds_images_frames[j, i] = img

            # find last input frame and draw red point for ground truth and yellow point for prediction
            for i in range(input_images_frames.shape[0]):
                last_input_frame = input_images_frames[i, -1]
                gt_Xs, gt_Ys, gt_Labels = [], [], []
                pred_Xs, pred_Ys, pred_Labels = [], [], []
                logger.info('File %d' % (i + 1))
                logger.info('Input\tX\tY')
                for j in range(input_images_frames.shape[1]):
                    gt_x, gt_y = calculate_center_of_red_point(input_images_frames[i, j])
                    gt_Xs.append(gt_x)
                    gt_Ys.append(gt_y)
                    gt_Labels.append(j + 1)
                    logger.info('%d\t\t%s\t%s' % (j + 1, gt_x, gt_y))

                logger.info('GT\tX\tY\t|\tPredict\tX\tY')
                before_pred_frame = input_images_frames.shape[1]
                for j in range(gt_images_frames.shape[1]):
                    gt_x, gt_y = calculate_center_of_red_point(gt_images_frames[i, j])
                    gt_Xs.append(gt_x)
                    gt_Ys.append(gt_y)
                    gt_Labels.append(before_pred_frame + j + 1)

                    pred_x, pred_y = calculate_center_of_red_point(preds_images_frames[i, j])
                    pred_Xs.append(pred_x)
                    pred_Ys.append(pred_y)
                    pred_Labels.append(before_pred_frame + j + 1)
                    logger.info('%d\t\t%s\t%s\t|\t\t%s\t%s' % ((j + 1), gt_x, gt_y, pred_x, pred_y))

                for j in range(len(gt_Xs)):
                    if gt_Xs[j] is None or gt_Ys[j] is None:
                        continue
                    cv2.circle(last_input_frame, (gt_Xs[j], gt_Ys[j]), 3, (255 - 40 * j, 20 * j, 0), -1)
                    #cv2.putText(last_input_frame, str(gt_Labels[j]), (gt_Xs[j]-5, gt_Ys[j] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.1, (255, 255, 0), 1)

                for j in range(len(pred_Xs)):
                    if pred_Xs[j] is None or pred_Ys[j] is None:
                        continue
                    cv2.circle(last_input_frame, (pred_Xs[j], pred_Ys[j]), 3, (0, 0, 255), -1)
                    # cv2.putText(last_input_frame, str(pred_Labels[j]), (pred_Xs[j] - 5, pred_Ys[j] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 3)

                overall_image = Image.fromarray(np.uint8(last_input_frame))
                overall_image.save(
                    c.IMG_SAVE_DIR + file_name_prefix + "_file" + str(i + 1) + "_overall.png")


            logger.info('Saved input / ground true / predicted images! File name prefix: %s' % file_name_prefix)
        logger.info('-' * 30)

        return rec_psnr, rec_sharpdiff

    def test_batch(self, logger, input_frames, gt_frames, num_rec_out=1, save_imgs=True):
        self.eval()