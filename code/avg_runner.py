from datetime import datetime

import torch
import argparse
import os
import logging
from pathlib import Path
from tqdm import tqdm
import numpy as np
from glob import glob
from PIL import Image
import cv2
import imageio.v3 as iio
import pandas as pd


from torch import nn

from utils import get_train_batch, get_test_batch, weights_init, generate_datetime_filename, get_clip, \
    get_one_test_batch, adjust_order_for_img, calculate_center_of_red_point
import constants as c
from g_model import GeneratorModel
from d_model import DiscriminatorModel


class AVGRunner:
    def __init__(self, num_gpu, num_steps, model_load_path, num_test_rec):
        """
        Initializes the Adversarial Video Generation Runner.
        :param num_steps:  The number of training steps to run.
        :param model_load_path: The path from which to load a previously-saved model. Default = None.
        :param num_test_rec: The number of recursive generations to produce when testing.
                            Recursive generations use previous generations as input to predict further into the future.
        """
        self.num_gpu = num_gpu
        self.num_steps = num_steps
        self.num_test_rec = num_test_rec

        # create name of log file via datetime
        log_file_name = generate_datetime_filename()
        log_file_path = c.LOG_SAVE_DIR + log_file_name + ".log"
        self.logger = logging.getLogger('avg_runner')
        self.logger.setLevel(logging.INFO)
        file_output_handler = logging.FileHandler(log_file_path)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_output_handler.setFormatter(formatter)
        self.logger.addHandler(file_output_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # Decide which device we want to run on
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and num_gpu > 0) else "cpu")
        self.logger.info('Running on device: %s' % self.device)



        if c.ADVERSARIAL:
            self.logger.info('Init Discriminator...')
            self.d_model = DiscriminatorModel(self.device,
                                              c.TRAIN_HEIGHT,
                                              c.TRAIN_WIDTH,
                                              c.SCALE_CONV_FMS_D,
                                              c.SCALE_KERNEL_SIZES_D,
                                              c.SCALE_FC_LAYER_SIZES_D).to(self.device)

            # Handle multi-GPU if desired
            if (self.device.type == 'cuda') and (num_gpu > 1):
                self.d_model = nn.DataParallel(self.d_model, list(range(num_gpu)))

            # Apply the ``weights_init`` function to randomly initialize all weights
            #  to ``mean=0``, ``stdev=0.02``.
            for i in range(len(self.d_model.scale_nets)):
                self.d_model.scale_nets[i].apply(weights_init)

        self.logger.info('Init Generator...')
        self.g_model = GeneratorModel(self.device,
                                      c.TRAIN_HEIGHT,
                                      c.TRAIN_WIDTH,
                                      c.FULL_HEIGHT,
                                      c.FULL_WIDTH,
                                      c.SCALE_FMS_G,
                                      c.SCALE_KERNEL_SIZES_G).to(self.device)

        # Handle multi-GPU if desired
        if (self.device.type == 'cuda') and (num_gpu > 1):
            self.g_model = nn.DataParallel(self.g_model, list(range(num_gpu)))

        # Apply the ``weights_init`` function to randomly initialize all weights
        #  to ``mean=0``, ``stdev=0.02``.
        for i in range(len(self.g_model.scale_nets)):
            self.g_model.scale_nets[i].apply(weights_init)

        if model_load_path is None and os.path.isfile(c.MODEL_SAVE_DIR + 'model.ckpt'):
            model_load_path = c.MODEL_SAVE_DIR + 'model.ckpt'
        if model_load_path is not None and os.path.isfile(model_load_path):
            self.logger.info('-' * 30)
            self.logger.info('Load checkpoint from %s ...' % model_load_path)
            checkpoint = torch.load(model_load_path)
            self.g_model.load_state_dict(checkpoint['g_model'])
            self.d_model.load_state_dict(checkpoint['d_model'])
            self.logger.info(f'Model restored from {model_load_path}')
            self.logger.info('-' * 30)

    def train(self):
        self.logger.info('-' * 30)
        self.logger.info('Start Training ... (name:%s, steps: %d)' % (c.SAVE_NAME, self.num_steps))
        pbar = tqdm(range(1, self.num_steps + 1))
        for step in pbar:
            d_loss = 0
            if c.ADVERSARIAL:
                batch = get_train_batch()
                input_frames = torch.from_numpy(batch[:, :-3, :, :]).to(self.device)
                gt_output_frames = torch.from_numpy(batch[:, -3:, :, :]).to(self.device)
                d_loss = self.d_model.train_step(input_frames, gt_output_frames, self.g_model)

            batch = get_train_batch()
            input_frames = torch.from_numpy(batch[:, :-3, :, :]).to(self.device)
            gt_output_frames = torch.from_numpy(batch[:, -3:, :, :]).to(self.device)
            g_loss, psnr, sharpdiff = self.g_model.train_step(
                input_frames, gt_output_frames, discriminator=(self.d_model if c.ADVERSARIAL else None))

            if step % c.STATS_FREQ == 0:
                if c.ADVERSARIAL:
                    self.logger.info(
                        'Step %d    | Generator (Loss : %f  PSNR Error : %f  SharpDiff Error : %f)  | Discriminator (Loss : %f)' % (
                        step, g_loss, psnr, sharpdiff, d_loss))
                else:
                    self.logger.info('Step %d    | Generator (Loss : %f  PSNR Error : %f  SharpDiff Error : %f)' % (step, g_loss, psnr, sharpdiff))

            if step % c.MODEL_SAVE_FREQ == 0 or step == self.num_steps:
                self.logger.info('-' * 30)
                self.logger.info('AVG Runner : Step %d' % (step + 1))
                self.logger.info('Saving models...')
                save_model_file_name = c.MODEL_SAVE_DIR + 'model.ckpt'
                save_model_file_hist_name = c.MODEL_SAVE_DIR + 'model-' + str(step) + '.ckpt'
                torch.save({
                    'g_model': self.g_model.state_dict(),
                    'd_model': self.d_model.state_dict(),
                },
                    save_model_file_name)
                self.logger.info('Saved models to: %s' % save_model_file_name)
                torch.save({
                    'g_model': self.g_model.state_dict(),
                    'd_model': self.d_model.state_dict(),
                },
                    save_model_file_hist_name)
                self.logger.info('Saved models to: %s' % save_model_file_hist_name)
                self.logger.info('-' * 30)

            if step % c.TEST_FREQ == 0:
                self.test()

        self.logger.info('All Finished !')

    def test(self):
        self.logger.info('-' * 30)
        self.logger.info('Start Testing ... (recursions: %d)' % self.num_test_rec)
        # pbar = tqdm(range(1, self.num_steps + 1))
        # for step in pbar:
        batch = get_test_batch(c.BATCH_SIZE, num_rec_out=self.num_test_rec)
        input_frames = torch.from_numpy(batch[:, :-3 * self.num_test_rec, :, :]).to(self.device, dtype=torch.float)
        gt_output_frames = torch.from_numpy(batch[:, -3 * self.num_test_rec:, :, :]).to(self.device, dtype=torch.float)
        self.g_model.test_batch(self.logger, input_frames, gt_output_frames, num_rec_out=self.num_test_rec)

        # self.logger.info('Finished Testing !')

    def batch_test(self):
        self.logger.info('Start Batch Testing ... (recursions: %d)' % self.num_test_rec)

        ep_frame_paths = sorted(glob(os.path.join(c.TEST_DIR_LIST[0], c.DATASET_NAME, '*')))
        end_index = len(ep_frame_paths) - (c.HIST_LEN + self.num_test_rec - 1)
        file_name_prefix = c.DATASET_NAME + '-' + generate_datetime_filename()

        predict_img_list = []
        with torch.no_grad():
            pbar = tqdm(range(0, end_index))
            for start_index in pbar:
                clip_frame_intput_paths = ep_frame_paths[start_index:start_index + c.HIST_LEN]
                clip_frame_output_paths = ep_frame_paths[start_index + c.HIST_LEN:start_index + c.HIST_LEN + self.num_test_rec]

                batch = get_one_test_batch(clip_frame_intput_paths, clip_frame_output_paths)
                input_frames = torch.from_numpy(batch[:, :-3 * self.num_test_rec, :, :]).to(self.device, dtype=torch.float)

                scale_preds = self.g_model.predict_all(input_frames)
                preds = scale_preds[-1]
                preds_image = adjust_order_for_img(preds)

                img = Image.fromarray(np.uint8((preds_image[0].detach().cpu().numpy() + 1) * 127.5))
                predict_img_list.append(img)

                (gt_path, gt_image_name) = os.path.split(clip_frame_output_paths[0])
                predict_output_name = gt_image_name.replace("fig.png", "fig-predict.png", 1)
                img.save(c.IMG_SAVE_DIR + file_name_prefix + "-" + predict_output_name)

        # start drawing the images
        draw_img = iio.imread(c.BACKGROUND_IMAGE, mode='RGB')
        draw_img_with_label = iio.imread(c.BACKGROUND_IMAGE, mode='RGB')

        # add history images & ground truth images
        gt_Xs, gt_Ys, gt_Labels = [], [], []
        last_center = None
        for i in range(len(ep_frame_paths)):
            img = iio.imread(ep_frame_paths[i], mode='RGB')
            hist_x, hist_y = calculate_center_of_red_point(img, last_center)
            if hist_x is not None and hist_y is not None:
                last_center = (hist_x, hist_y)
            gt_Xs.append(hist_x)
            gt_Ys.append(hist_y)
            gt_Labels.append(i + 1)

        # add predicted images
        pred_Xs, pred_Ys, pred_Labels = [], [], []
        last_center = None
        for i in range(len(predict_img_list)):
            if gt_Xs[i + c.HIST_LEN - 1] is not None and gt_Ys[i + c.HIST_LEN - 1] is not None:
                last_center = (gt_Xs[i + c.HIST_LEN - 1], gt_Ys[i + c.HIST_LEN - 1])
            pred_x, pred_y = calculate_center_of_red_point(predict_img_list[i], last_center)
            if pred_x is not None or pred_y is not None:
                last_center = (pred_x, pred_y)
            pred_Xs.append(pred_x)
            pred_Ys.append(pred_y)
            pred_Labels.append(c.HIST_LEN + i + 1)

        # draw point on image
        gt_Xs_with_start = [ x + c.BACKGROUND_IMAGE_USE_START_X if x is not None else None for x in gt_Xs]
        gt_Ys_with_start = [ y + c.BACKGROUND_IMAGE_USE_START_Y if y is not None else None for y in gt_Ys]
        for j in range(len(gt_Xs_with_start)):
            if gt_Xs_with_start[j] is None or gt_Ys_with_start[j] is None:
                continue
            cv2.circle(draw_img, (gt_Xs_with_start[j], gt_Ys_with_start[j]), 1, (255, 0, 0), -1)
            cv2.circle(draw_img_with_label, (gt_Xs_with_start[j], gt_Ys_with_start[j]), 1, (255, 0, 0), -1)


            A = -30
            B = -50
            offset = A if j % 2 == 0 else B
            text_x, text_y = gt_Xs_with_start[j] + offset, gt_Ys_with_start[j] + offset
            cv2.arrowedLine(draw_img_with_label, (gt_Xs_with_start[j], gt_Ys_with_start[j]), (text_x, text_y), (255, 0, 0), 1)
            cv2.putText(draw_img_with_label, str(gt_Labels[j]), (text_x - 5, text_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 0, 0), 1)

        pred_Xs_with_start = [ x + c.BACKGROUND_IMAGE_USE_START_X if x is not None else None for x in pred_Xs]
        pred_Ys_with_start = [ y + c.BACKGROUND_IMAGE_USE_START_Y if y is not None else None for y in pred_Ys]
        for j in range(len(pred_Xs_with_start)):
            if pred_Xs_with_start[j] is None or pred_Ys_with_start[j] is None:
                continue
            cv2.circle(draw_img, (pred_Xs_with_start[j], pred_Ys_with_start[j]), 1, (0, 255, 255), -1)
            cv2.circle(draw_img_with_label, (pred_Xs_with_start[j], pred_Ys_with_start[j]), 1, (0, 255, 255), -1)

            A = 30
            B = 50
            offset = A if j % 2 else B
            text_x, text_y = pred_Xs_with_start[j] + offset, pred_Ys_with_start[j] + offset
            cv2.arrowedLine(draw_img_with_label, (pred_Xs_with_start[j], pred_Ys_with_start[j]), (text_x, text_y), (0, 255, 255), 1)
            cv2.putText(draw_img_with_label, str(pred_Labels[j]), (text_x + 5, text_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 255, 255), 1)

        overall_image = Image.fromarray(np.uint8(draw_img))
        overall_image.save(c.IMG_SAVE_DIR + "overall/" + file_name_prefix + "_overall.png")
        overall_image_with_label = Image.fromarray(np.uint8(draw_img_with_label))
        overall_image_with_label.save(c.IMG_SAVE_DIR + "overall/" + file_name_prefix + "_overall_with_label.png")

        self.logger.info('Saved output image to Folder: %s. File name prefix: %s' % (c.IMG_SAVE_DIR, file_name_prefix))

        d = {'gt_Xs': pd.Series(gt_Xs, index=gt_Labels),
             'gt_Ys': pd.Series(gt_Ys, index=gt_Labels),
             'pred_Xs': pd.Series(pred_Xs, index=pred_Labels),
             'pred_Ys': pd.Series(pred_Ys, index=pred_Labels)}
        df = pd.DataFrame(d)

        df['Delta'] = np.sqrt((df['gt_Xs'] - df['pred_Xs']) ** 2 + (df['gt_Ys'] - df['pred_Ys']) ** 2)

        print(df)

        csv_file_path = os.path.join(c.IMG_SAVE_DIR + "overall/", file_name_prefix + "_overall_list.csv")
        df.to_csv(csv_file_path, index=True, index_label="Point")

        self.logger.info('Saved output image to Folder: %s. File name prefix: %s' % (c.IMG_SAVE_DIR, file_name_prefix))
        self.logger.info('Saved data to CSV file: %s' % csv_file_path)


def main():
    parser = argparse.ArgumentParser(description='Adversarial Video Generation Runner')
    parser.add_argument('-l', '--load_path', help='Relative/path/to/saved/model')
    parser.add_argument('-td', '--train_clips_dir', type=str, help='Directory of train clips')
    parser.add_argument('-t', '--test_dir_list', type=str, nargs='+', help='Directory of test images')
    parser.add_argument('-ns', '--num_of_subset', type=int, default=1, help='number of images used for training')
    parser.add_argument('-r', '--recursions', type=int, default=1, help='# recursive predictions to make on test')
    parser.add_argument('-a', '--adversarial', type=bool, default=True, help='Whether to use adversarial training')
    parser.add_argument('-n', '--name', type=str, default='default', help='Subdirectory of ../Data/Save/*/ in which to save output of this run')
    parser.add_argument('-s', '--steps', type=int, default=1000001, help='Number of training steps to run')
    parser.add_argument('-b', '--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('-O', '--overwrite', action='store_true', help='Overwrites all previous data for the model with this save name')
    parser.add_argument('-T', '--test_only', action='store_true', help='Only runs a test step -- no training')
    parser.add_argument('-BT', '--batch_test_only', action='store_true', help='Only runs a batch test -- no training')
    parser.add_argument('-d', '--dataset', type=str, help='Dataset name')
    parser.add_argument('-g', '--gpu', type=int, default=1, help='number of GPUs available. If this is 0, code will run in CPU mode.')
    parser.add_argument('--stats_freq', type=int, help='How often to print(loss/train error stats, in # steps')
    parser.add_argument('--summary_freq', type=int, help='How often to save loss/error summaries, in # steps')
    parser.add_argument('--img_save_freq', type=int, help='How often to save generated images, in # steps')
    parser.add_argument('--test_freq', type=int, help='How often to test the model on test data, in # steps')
    parser.add_argument('--model_save_freq', type=int, help='How often to save the model, in # steps')

    args = parser.parse_args()

    load_path = None

    if args.load_path:
        load_path = args.load_path
    if args.test_dir_list:
        c.set_test_dirs(args.test_dir_list)
    if args.train_clips_dir:
        c.set_train_clips_dir(args.train_clips_dir)
    if args.num_of_subset:
        c.set_number_of_subset(args.num_of_subset)
    if args.recursions:
        num_test_rec = args.recursions
    if args.adversarial is not None:
        c.ADVERSARIAL = args.adversarial
    if args.name:
        c.set_save_name(args.name)
    if args.batch_size:
        c.BATCH_SIZE = args.batch_size
    if args.steps:
        num_steps = args.steps
    if args.overwrite:
        c.clear_save_name()
    if args.gpu:
        num_gpu = args.gpu
    if args.stats_freq:
        c.STATS_FREQ = args.stats_freq
    if args.summary_freq:
        c.SUMMARY_FREQ = args.summary_freq
    if args.img_save_freq:
        c.IMG_SAVE_FREQ = args.img_save_freq
    if args.test_freq:
        c.TEST_FREQ = args.test_freq
    if args.model_save_freq:
        c.MODEL_SAVE_FREQ = args.model_save_freq
    if args.dataset:
        c.DATASET_NAME = args.dataset

    for dir in c.TEST_DIR_LIST:
        assert Path(dir).exists()
    c.FULL_HEIGHT, c.FULL_WIDTH = c.get_test_frame_dims()

    runner = AVGRunner(num_gpu, num_steps, load_path, num_test_rec)
    if args.test_only:
        runner.test()
    elif args.batch_test_only:
        runner.batch_test()
    else:
        runner.train()


if __name__ == '__main__':
    main()