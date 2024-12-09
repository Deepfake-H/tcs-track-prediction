import argparse
import glob
import os
from glob import glob
import utils

import cv2
from tqdm import tqdm

parser = argparse.ArgumentParser()
####################################################################
# Parse command line
####################################################################
parser.add_argument('--raw_dir', type=str, default='../data/raw', help='Directory of raw datasets')
parser.add_argument('--output_dir', type=str, default='../data/train', help='Directory for output')
parser.add_argument('--datasets', type=str, default='ALL_DATASETS', nargs='+',
                    help='Raw datasets. ALL_DATASETS means all under --raw_dir. e.g.: 2001039S11139-WINSOME, 2011-TEST')
parser.add_argument('--prefixes', type=str, default='wv', nargs='+',
                    help='Prefix of target image in dataset folder. eg. wv, vor, wind, z')
parser.add_argument('--full_width', type=int, default=1187, help='Full Width of pre-processed image')
parser.add_argument('--full_height', type=int, default=989, help='Full Height of pre-processed image')
parser.add_argument('--resize_width', type=int, default=1187, help='Resize Width of pre-processed image')
parser.add_argument('--resize_height', type=int, default=989, help='Resize Height of pre-processed image')
parser.add_argument('--start_x', type=int, default=242, help='X value of start pixel')
parser.add_argument('--start_y', type=int, default=233, help='Y value of start pixel')
parser.add_argument('--single_file_name', type=str, default='', help='preprocess one single image')

args = parser.parse_args()


def pre_process_one_dataset(args, dataset, prefix):
    pre_processed_dataset_path = '{}/{}/{}'.format(args.output_dir, prefix, dataset)
    if not os.path.exists(pre_processed_dataset_path):
        os.makedirs(pre_processed_dataset_path)

    search_str = '{}/{}/{}*.png'.format(args.raw_dir, dataset, prefix)
    pbar = tqdm(glob(search_str))
    for filename in pbar:
        (path, v_name) = os.path.split(filename)
        save_file_name = '{}/{}'.format(pre_processed_dataset_path, v_name)
        pre_process_one_single_file(args, filename, save_file_name)

def pre_process_one_single_file(args, file_name, save_file_name):
    cut_x1 = args.start_x
    cut_y1 = args.start_y
    cut_x2 = args.start_x + args.full_width
    cut_y2 = args.start_y + args.full_height

    image = cv2.imread(file_name)
    cropped = image[cut_y1:cut_y2, cut_x1:cut_x2]
    width = int(args.resize_width)
    height = int(args.resize_height)
    dim = (width, height)
    resized = cv2.resize(cropped, dim, interpolation=cv2.INTER_AREA)
    cv2.imwrite(save_file_name, resized)

def main():
    if args.single_file_name != '':
        (path, v_name) = os.path.split(args.single_file_name)
        save_file_name = '{}/{}'.format(path, 'Preprocessed_' + v_name)
        pre_process_one_single_file(args, args.single_file_name, save_file_name)
        return

    assert os.path.exists('{}'.format(args.raw_dir))
    dataset_list = []
    if args.datasets == 'ALL_DATASETS':
        dataset_list = utils.get_subdir_list('{}'.format(args.raw_dir))
    else:
        for ds in args.datasets:
            dataset_path = '{}/{}'.format(args.raw_dir, ds)
            assert os.path.exists(dataset_path)
            dataset_list.append(ds)

    print('Will pre-process following datasets: {}'.format(dataset_list))

    dataset_count = len(dataset_list)
    prefix_count = len(args.prefixes)
    for dataset_num, dataset in enumerate(dataset_list):
        print('({}/{}) Pre-processing dataset: {}'.format(dataset_num + 1, dataset_count, dataset))
        for prefix_num, prefix in enumerate(args.prefixes):
            print('({}/{}) prefix: {}'.format(prefix_num + 1, prefix_count, prefix))
            pre_process_one_dataset(args, dataset, prefix)


if __name__ == '__main__':
    main()
