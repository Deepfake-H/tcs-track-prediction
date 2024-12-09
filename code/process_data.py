import argparse
import utils
import glob
import os
from glob import glob
import numpy as np
import gc
import constants as c

from tqdm import tqdm

parser = argparse.ArgumentParser()
####################################################################
# Parse command line
####################################################################
parser.add_argument('--train_dir', type=str, default='../data/train', help='Directory of train datasets')
parser.add_argument('--output_clips_dir', type=str, default='../data/clips', help='Directory of clips')
parser.add_argument('--subsets', type=str, nargs='+', default='wv',
                    help='Prefix of target image in dataset folder.')
parser.add_argument('--num_clips', type=int, default=500000, help='Clips to process for training')
parser.add_argument('--predict_len', type=int, default=1, help='The number of outputs to predict')

args = parser.parse_args()


def main():
    # train_dir = os.path.join(args.train_dir, args.subset)
    dataset_path_list = []
    c.set_number_of_subset(len(args.subsets))
    for ds in args.subsets:
        dataset_path = '{}/{}'.format(args.train_dir, ds)
        assert os.path.exists(dataset_path)
        dataset_path_list.append(dataset_path)

    output_dataset_name = "-".join(args.subsets)
    output_clips_dataset_dir = os.path.join(args.output_clips_dir, output_dataset_name)
    if not os.path.exists(output_clips_dataset_dir):
        os.makedirs(output_clips_dataset_dir)

    num_prev_clips = len(glob(os.path.join(output_clips_dataset_dir, '*')))
    print('Start generate {} clips from train DIR: {} on subsets: {}'.format(args.num_clips, args.train_dir, args.subsets))
    print('Output Dir: {}'.format(output_clips_dataset_dir))
    pbar = tqdm(range(num_prev_clips, args.num_clips + num_prev_clips, 100))
    for clip_num in pbar:
        clips = utils.process_clip(dataset_path_list, args.predict_len)
        for count, clip in enumerate(clips):
            np.savez_compressed(os.path.join(output_clips_dataset_dir, str(clip_num + count)), clip)

        if (clip_num + 1) % 10000 == 0:
            gc.collect()

    print('Finished!')


if __name__ == '__main__':
    main()
