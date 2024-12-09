import os
from datetime import datetime
from glob import glob

import cv2
import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import constants as c


def generate_datetime_filename():
    file_name = str(datetime.now())[:-7]
    file_name = file_name.replace("-", "")
    file_name = file_name.replace(" ", "")
    file_name = file_name.replace(":", "")

    return file_name


def adjust_order_for_torch(input_frames):
    # convert to pytorch Variable: [batch, h, w, channel] -> [batch, channel, h, w]
    return input_frames.permute(0, 3, 1, 2)


def adjust_numpy_order_for_torch(input_numpy_frames):
    # convert to numpy.ndanrray Variable: [batch, h, w, channel] -> [batch, channel, h, w]
    return input_numpy_frames.transpose((0, 3, 1, 2))


def adjust_order_for_img(input_frames):
    # convert to pytorch Variable: [batch, channel, h, w] -> [batch, h, w, channel]
    return input_frames.permute(0, 2, 3, 1)


def resize_dataset(input_frames, height, width):
    """
    Resize the input_frames to the given height and width.
    :param input_frames: [batch, channel, h, w]
    :param height:
    :param width:
    :return: [batch, channel, height, width]
    """
    if input_frames.shape[1] == height and input_frames.shape[2] == width:
        return input_frames

    return F.interpolate(input_frames, size=(height, width), mode='bilinear', align_corners=False)


def get_subdir_list(path):
    return [item for item in os.listdir(path) if os.path.isdir(os.path.join(path, item)) and item[0] != "."]


def conv_out_size(i, p, k, s):
    """
    Gets the output size for a 2D convolution. (Assumes square input and kernel).

    @param i: The side length of the input.
    @param p: The padding type (either 'SAME' or 'VALID').
    @param k: The side length of the kernel.
    @param s: The stride.

    @type i: int
    @type p: string
    @type k: int
    @type s: int

    @return The side length of the output.
    """
    # convert p to a number
    if p == 'same':
        p = k // 2
    elif p == 'valid':
        p = 0
    else:
        raise ValueError('p must be "same" or "valid".')

    return int(((i + (2 * p) - k) / s) + 1)


##
# Data
##

# custom weights initialization called on ``netG`` and ``netD``
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def normalize_frames(frames):
    """
    Convert frames from int8 [0, 255] to float32 [-1, 1].

    @param frames: A numpy array. The frames to be converted.

    @return: The normalized frames.
    """
    new_frames = frames.astype(np.float32)
    new_frames /= (255 / 2)
    new_frames -= 1

    return new_frames


def denormalize_frames(frames):
    """
    Performs the inverse operation of normalize_frames.

    @param frames: A numpy array. The frames to be converted.

    @return: The denormalized frames.
    """
    new_frames = frames + 1
    new_frames *= (255 / 2)
    # noinspection PyUnresolvedReferences
    new_frames = new_frames.astype(np.uint8)

    return new_frames


def clip_l2_diff(clip, hist_len, num_of_subset=1):
    """
    @param clip: A numpy array of shape [train_width=32, train_height=32, (3 * (hist_len * number_of_subset + 1))].
    @param hist_len: The number of historical image used as input
    @param num_of_subset: The number of subset of the dataset used as input
    @return: The sum of l2 differences between the frame pixels of each sequential pair of frames.
    """
    diff = 0
    for i in range(hist_len):
        for j in range(num_of_subset):
            if 3 * ((i + 1) * num_of_subset + j) >= clip.shape[0]:
                break
            frame = clip[3 * (i * num_of_subset + j):3 * (i * num_of_subset + j + 1), :, :]
            next_frame = clip[3 * ((i + 1) * num_of_subset + j):3 * ((i + 1) * num_of_subset + j + 1), :, :]
            # noinspection PyTypeChecker
            diff += np.sum(np.square(next_frame - frame))

    return diff


def get_full_clips(data_dir_list, num_clips, num_rec_out=1):
    """
    Loads a batch of random clips from the unprocessed train or test data.

    @param data_dir_list: List of the directory of the data to read.
    @param num_clips: The number of clips to read.
    @param num_rec_out: The number of outputs to predict. Outputs > 1 are computed recursively,
                        using the previously-generated frames as input. Default = 1.

    @return: An array of shape
             [num_clips,  (3 * (hist_len * num_of_subset + num_rec_out))， c.TRAIN_HEIGHT, c.FULL_WIDTH].
             A batch of frame sequences with values normalized in range [-1, 1].
    """
    num_of_subset = len(data_dir_list)
    assert num_of_subset == c.NUM_OF_SUBSET

    clips = np.empty([num_clips,
                      (3 * (c.HIST_LEN * num_of_subset + num_rec_out)),
                      c.FULL_HEIGHT,
                      c.FULL_WIDTH])

    number = 0

    while number < num_clips:
        # get a random dataset
        dataset = np.random.choice(get_subdir_list(data_dir_list[0]))
        ep_frame_paths = sorted(glob(os.path.join(data_dir_list[0], dataset, '*')))
        start_index = np.random.choice(len(ep_frame_paths) - (c.HIST_LEN + num_rec_out - 1))
        clip_frame_intput_paths = ep_frame_paths[start_index:start_index + c.HIST_LEN]
        clip_frame_output_paths = ep_frame_paths[start_index + c.HIST_LEN:start_index + c.HIST_LEN + num_rec_out]

        clip = get_clip(clip_frame_intput_paths, clip_frame_output_paths, data_dir_list)

        clips[number, :, :, :] = clip

        number += 1

    return clips


def get_clip(input_paths, output_paths, data_dir_list):
    """
    Loads a clip from the unprocessed train or test data.

    @param input_paths: The paths of the input frames.
    @param output_paths: The paths of the output frames.
    @param data_dir_list: List of the directory of the data to read.

    @return: An array of shape
             [(3 * (hist_len * num_of_subset + num_rec_out)), c.TRAIN_HEIGHT, c.TRAIN_WIDTH].
             A frame sequence with values normalized in range [-1, 1].
    """
    num_of_subset = len(data_dir_list)
    clip = np.empty([(3 * (c.HIST_LEN * num_of_subset + 1)), c.FULL_HEIGHT, c.FULL_WIDTH])

    clip_frame_paths = []
    for clip_frame_path in input_paths:
        clip_frame_paths.append(clip_frame_path)
        for i in range(1, num_of_subset):
            new_frame_path = clip_frame_path.replace(data_dir_list[0] + "-", data_dir_list[i] + "-", 1)
            if os.path.isfile(new_frame_path):
                clip_frame_paths.append(new_frame_path)

    if len(clip_frame_paths) != c.HIST_LEN * num_of_subset:
        return None

    clip_frame_paths.extend(output_paths)

    # read in frames
    for frame_num, frame_path in enumerate(clip_frame_paths):
        frame = iio.imread(frame_path, mode='RGB')
        norm_frame = normalize_frames(frame).transpose(2, 0, 1)

        clip[3 * frame_num:(frame_num + 1) * 3, :, :] = norm_frame

    return clip


def process_clip(data_dir_list, num_rec_out=1):
    """
    Gets a clip from the train dataset, cropped randomly to c.TRAIN_HEIGHT x c.TRAIN_WIDTH.

    @return: An array of shape [c.TRAIN_HEIGHT, c.TRAIN_WIDTH, (3 * (c.HIST_LEN + 1))].
             A frame sequence with values normalized in range [-1, 1].
    """
    num_of_subset = len(data_dir_list)
    clip = get_full_clips(data_dir_list, 1, num_rec_out)[0]

    cropped_clips = []
    for k in range(100):
        # Randomly crop the clip. With 0.05 probability, take the first crop offered, otherwise,
        # repeat until we have a clip with movement in it.
        take_first = np.random.choice(2, p=[0.95, 0.05])
        cropped_clip = np.empty([3 * (c.HIST_LEN * num_of_subset + num_rec_out), c.TRAIN_HEIGHT, c.TRAIN_WIDTH])
        for i in range(100):  # cap at 100 trials in case the clip has no movement anywhere
            crop_x = np.random.choice(c.FULL_WIDTH - c.TRAIN_WIDTH + 1)
            crop_y = np.random.choice(c.FULL_HEIGHT - c.TRAIN_HEIGHT + 1)
            cropped_clip = clip[:, crop_y:crop_y + c.TRAIN_HEIGHT, crop_x:crop_x + c.TRAIN_WIDTH]

            if take_first or clip_l2_diff(cropped_clip, c.HIST_LEN, num_of_subset) > c.MOVEMENT_THRESHOLD:
                break

        cropped_clips.append(cropped_clip)

    return cropped_clips


def get_train_batch():
    """
    Loads c.BATCH_SIZE clips from the database of preprocessed training clips.

    @return: An array of shape
            [c.BATCH_SIZE, (3 * (c.HIST_LEN * c.NUM_OF_SUBSET + 1)), c.TRAIN_HEIGHT, c.TRAIN_WIDTH].
    """
    clips = np.empty([c.BATCH_SIZE, (3 * (c.HIST_LEN * c.NUM_OF_SUBSET + 1)), c.TRAIN_HEIGHT, c.TRAIN_WIDTH],
                     dtype=np.float32)
    for i in range(c.BATCH_SIZE):
        path = c.TRAIN_CLIPS_DIR + str(np.random.choice(c.NUM_CLIPS)) + '.npz'
        clip = np.load(path)['arr_0']

        clips[i] = clip

    return clips


def get_test_batch(test_batch_size, num_rec_out=1):
    """
    Gets a clip from the test dataset.

    @param test_batch_size: The number of clips.
    @param num_rec_out: The number of outputs to predict. Outputs > 1 are computed recursively,
                        using the previously-generated frames as input. Default = 1.

    @return: An array of shape:
             [test_batch_size, (3 * (c.HIST_LEN * c.NUM_OF_SUBSET + num_rec_out)), c.TEST_HEIGHT, c.TEST_WIDTH].
             A batch of frame sequences with values normalized in range [-1, 1].
    """
    return get_full_clips(c.TEST_DIR_LIST, test_batch_size, num_rec_out=num_rec_out)


def get_one_test_batch(input_paths, output_paths):
    """
    Gets a clip from the test dataset.

    @param input_paths: The paths of the input frames.
    @param output_paths: The paths of the output frames.

    @return: An array of shape:
             [test_batch_size=1, (3 * (c.HIST_LEN * c.NUM_OF_SUBSET + num_rec_out)), c.TEST_HEIGHT, c.TEST_WIDTH].
             A batch of frame sequences with values normalized in range [-1, 1].
    """
    num_of_subset = len(c.TEST_DIR_LIST)
    clips = np.empty([1, (3 * (c.HIST_LEN * num_of_subset + 1)), c.FULL_HEIGHT, c.FULL_WIDTH])
    clips[0, :, :, :] = get_clip(input_paths, output_paths, c.TEST_DIR_LIST)
    return clips


##
# Error calculation
##

# Computes the Peak Signal to Noise Ratio error between the generated images and the ground truth images using Pytorch
def psnr_error(gen_frames, gt_frames, max_val=1.0):
    """
    Calculate the PSNR error.

    Parameters:
    - y_true: The tensor of real images, List of (batch_size, channels, height, width)[]
    - y_pred: The tensor of predicted images, shape the same as y_true
    - max_val: The maximum possible value of the signal, should be 1.0 for images normalized to [0, 1]

    Returns:
    - psnr: The calculated PSNR value
    """
    scale_errors = []
    for i in range(len(gen_frames)):
        # Calculate MSE (Mean Squared Error)
        mse = torch.mean((gen_frames[i] - gt_frames[i]) ** 2)

        # Avoid log(0) situation
        mse = torch.clamp(mse, min=1e-10)

        # Calculate PSNR value according to the definition of PSNR
        psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
        scale_errors.append(torch.mean(psnr))

    return torch.mean(torch.stack(scale_errors))


def calculate_gradients(img):
    """
    Calculate the gradients of the given image tensor.

    Parameters:
    - img: Image tensor, shape (batch_size, channels, height, width)

    Returns:
    - Gradient tensor in x and y directions.
    """
    a = img[:, :, :-1, :-1] - img[:, :, :-1, 1:]
    b = img[:, :, :-1, :-1] - img[:, :, 1:, :-1]
    grad = torch.sqrt(torch.pow(a, 2) + torch.pow(b, 2))
    return grad


# Computes the Sharpness Difference error between the generated images and the ground images using Pytorch
def sharp_diff_error(gen_frames, gt_frames):
    """
    Computes the Sharpness Difference error between the generated images and the ground truth images.

    @param gen_frames: A tensor of shape [batch_size, height, width, 3]. The frames generated by the generator model.
    @param gt_frames: A tensor of shape [batch_size, height, width, 3]. The ground-truth frames for each frame in gen_frames.

    @return: A scalar tensor. The Sharpness Difference error over each frame in the batch.
    """
    scale_errors = []
    for i in range(len(gen_frames)):
        # Calculate gradients
        true_grad = calculate_gradients(gt_frames[i])
        pred_grad = calculate_gradients(gen_frames[i])

        # Compute the sharpness difference
        sharp_diff = torch.abs(true_grad - pred_grad)
        scale_errors.append(torch.mean(sharp_diff))

    # Return the mean sharpness difference error
    return torch.mean(torch.stack(scale_errors))


def calculate_center_of_red_point(image, last_center=None):
    img_int8 = np.uint8(image)
    image_hsv = cv2.cvtColor(img_int8, cv2.COLOR_RGB2HSV)

    # lower mask (0-10)
    lower_red = np.array([0, 30, 30])
    upper_red = np.array([15, 255, 255])
    mask0 = cv2.inRange(image_hsv, lower_red, upper_red)

    # upper mask (170-180)
    lower_red_alt = np.array([160, 30, 30])
    upper_red_alt = np.array([180, 255, 255])
    mask1 = cv2.inRange(image_hsv, lower_red_alt, upper_red_alt)

    # join masks
    mask = mask0 + mask1

    # Morphological operations to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None

    # Find the contour with the highest mean red value
    max_mean_val = 0
    reddest_contour = None
    closest_contour = None
    min_distance = float('inf')

    for contour in contours:
        mask_contour = np.zeros_like(mask)
        cv2.drawContours(mask_contour, [contour], -1, 255, thickness=cv2.FILLED)
        mean_val = cv2.mean(image_hsv, mask=mask_contour)[1]  # Mean saturation value (H, S, V)

        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            if last_center is not None:
                distance = np.sqrt((cX - last_center[0]) ** 2 + (cY - last_center[1]) ** 2)
                if distance < min_distance:
                    min_distance = distance
                    closest_contour = contour

            if mean_val > max_mean_val:
                max_mean_val = mean_val
                reddest_contour = contour

    if last_center is not None and closest_contour is not None:
        if min_distance > 100:
            return None, None
        chosen_contour = closest_contour
    else:
        chosen_contour = reddest_contour

    if chosen_contour is not None:
        M = cv2.moments(chosen_contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            return cX, cY
        elif chosen_contour.shape[0] >= 1 and chosen_contour.shape[1] >= 1 and chosen_contour.shape[2] >= 2:
            return chosen_contour[0, 0, 0], chosen_contour[0, 0, 1]
    return None, None
