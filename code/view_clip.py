import argparse
import numpy as np
import tkinter
import matplotlib
# matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import utils
from torchvision.transforms import Resize
from utils import resize_dataset



parser = argparse.ArgumentParser()
####################################################################
# Parse command line
####################################################################
parser.add_argument('--folder', type=str, default='../data/clips/vor-wv/', help='File to view')
parser.add_argument('--file', type=str, default='0', nargs='+', help='File to view')

args = parser.parse_args()


def main():
    file_num = len(args.file)

    images = []
    # resized = []
    for file in args.file:
        file_path = args.folder + file + ".npz"
        clips = np.load(file_path)['arr_0']
        clips = utils.denormalize_frames(clips)
        # clips_resized = resize_dataset(clips.reshape(1, 32, 32, 12), 8, 8)[0]

        for i in range(int(clips.shape[2] / 3)):
            clip = clips[:, :, 3 * i:3 * (i + 1)]
            images.append(clip)
            # clip_resized = clips_resized[:, :, 3 * i:3 * (i + 1)]
            # resized.append(clip_resized)

    plt.figure()
    for i in range(1, 4 * file_num + 1):
        plt.subplot(file_num, 4, i)
        plt.imshow(images[i-1])
        print("images[i-1]:", images[i-1].shape)
        plt.xticks([])
        plt.yticks([])

        # plt.subplot(file_num * 2, 4, i+4)
        # plt.imshow(resized[i - 1])
        # print("resized[i-1]:", resized[i - 1].shape)
        # plt.xticks([])
        # plt.yticks([])

    plt.show()


if __name__ == '__main__':
    main()