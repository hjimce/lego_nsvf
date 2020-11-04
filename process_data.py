from poses.pose_utils import gen_poses, preprocess_imgs
import sys
import argparse

# Example usage:
# python process_data.py
parser = argparse.ArgumentParser()
parser.add_argument('--scenedir', type=str, default='lego_data_dir',
                    help='Data directory. e.g. lego_data_dir')
parser.add_argument('--colmapdir', type=str, default='sparse/0', 
                    help='The name of folder to load colmap .bin data from. COLMAP defualt is \'sparse/0\'.')
parser.add_argument('--downsample_factor', type=int, default=1)
args = parser.parse_args()


if __name__=='__main__':
    gen_poses(args.scenedir, args.colmapdir, args.downsample_factor)
