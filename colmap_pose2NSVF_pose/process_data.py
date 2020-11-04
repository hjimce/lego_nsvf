from poses.pose_utils import gen_poses, preprocess_imgs
import sys

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--scenedir', type=str,
                    help='Data directory. e.g. video-1234')
parser.add_argument('--colmapdir', type=str,
					default='sparse/undist', 
                    help='The name of folder to load colmap .bin data from. COLMAP defualt is \'sparse/0\'.')
parser.add_argument('--imgdir', type=str,
					default='images_orig_res', 
                    help='The name of folder to load original resolution rgb images from.')
parser.add_argument('--imgoutdir', type=str,
					default='rgb', 
                    help='The name of folder to save resized images to.')
parser.add_argument('--downsample_factor', type=int, default=2)
parser.add_argument('--mask', type=int, default=0, 
                    help='Boolean indicating whether to mask input images.')
args = parser.parse_args()


if __name__=='__main__':
    gen_poses(args.scenedir, args.colmapdir, args.downsample_factor)
    # if not (args.imgdir == args.imgoutdir):
    #     preprocess_imgs(args.scenedir, args.imgdir, args.imgoutdir, args.downsample_factor, 
    #         tomask=args.mask, maskdir="npy")
