import argparse, os, glob
import numpy as np

def load_matrix(path):
    return np.array([[float(w) for w in line.strip().split()] for line in open(path)]).astype(np.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenedir', type=str,
                        help='Data directory. e.g. video-1234')
    parser.add_argument('--indir', type=str, default="pose",
                        help='Data directory. e.g. video-1234')
    parser.add_argument('--outdir', type=str, required=True, 
                        help='Data directory. e.g. video-1234')
    parser.add_argument('--noise_std', type=float,
                        help='Data directory. e.g. video-1234')
    args = parser.parse_args()

    if not os.path.exists(os.path.join(args.scenedir, args.outdir)):
        os.mkdir(os.path.join(args.scenedir, args.outdir))

    pose_fn = sorted(glob.glob(os.path.join(args.scenedir, args.indir, "*")))

    for fn in pose_fn:
        pose = load_matrix(fn)
        pose[:3,:4] = pose[:3,:4] + np.random.randn(3,4) * args.noise_std
        lines = [
            "{} {} {} {}\n".format(pose[0,0], pose[0,1], pose[0,2], pose[0,3]),
            "{} {} {} {}\n".format(pose[1,0], pose[1,1], pose[1,2], pose[1,3]),
            "{} {} {} {}\n".format(pose[2,0], pose[2,1], pose[2,2], pose[2,3]),
            "0.0 0.0 0.0 1.0\n",
        ]
        with open(os.path.join(args.scenedir, args.outdir, os.path.basename(fn)), "w+") as f:
            f.writelines(lines)
