import argparse
import numpy as np
from glob import glob
from tqdm import tqdm
import os


parser = argparse.ArgumentParser()
####################################################################
# Parse command line
####################################################################
parser.add_argument('--folder', type=str, default='../data/clips/wv/', help='Folder to process')
parser.add_argument('--max_num', type=int, default=500000, help='maximum number of clips to keep')

args = parser.parse_args()


def main():

    search_str = '{}*.npz'.format(args.folder)
    pbar = tqdm(glob(search_str))
    for filename in pbar:
        (path, v_name) = os.path.split(filename)
        v_name_int = int(v_name.split('.')[0])
        if v_name_int > args.max_num:
            os.remove(filename)

if __name__ == '__main__':
    main()