import os
import subprocess
import argparse


# $ DATASET_PATH=/path/to/dataset

# $ colmap feature_extractor \
#    --database_path $DATASET_PATH/database.db \
#    --image_path $DATASET_PATH/images

# $ colmap exhaustive_matcher \
#    --database_path $DATASET_PATH/database.db

# $ mkdir $DATASET_PATH/sparse

# $ colmap mapper \
#     --database_path $DATASET_PATH/database.db \
#     --image_path $DATASET_PATH/images \
#     --output_path $DATASET_PATH/sparse

# $ mkdir $DATASET_PATH/dense
if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', type=str,
                        help='Data directory. e.g. video-1234')
    parser.add_argument('--images_folder', type=str, default="images_orig_res",
                        help="Folder under basedir in which images are stored.")
    args = parser.parse_args()


    logfile_name = os.path.join(args.basedir, 'colmap_output.txt')
    logfile = open(logfile_name, 'w')
    
    # Command line calls
    feature_extractor_args = [
        'colmap', 'feature_extractor', 
            '--database_path', os.path.join(args.basedir, 'database.db'), 
            '--image_path', os.path.join(args.basedir, args.images_folder),
            '--ImageReader.single_camera', '1',     # all images have the same intrinsic parameter
            '--SiftExtraction.use_gpu', '1',
    ]
    feat_output = ( subprocess.check_output(feature_extractor_args, universal_newlines=True) )
    logfile.write(feat_output)
    print('Features extracted')

    exhaustive_matcher_args = [
        'colmap', 'sequential_matcher', 
            '--database_path', os.path.join(args.basedir, 'database.db'), 
    ]

    match_output = ( subprocess.check_output(exhaustive_matcher_args, universal_newlines=True) )
    logfile.write(match_output)
    print('Features matched')

    # Run again on exhaustive
    exhaustive_matcher_args = [
        'colmap', 'exhaustive_matcher', 
            '--database_path', os.path.join(args.basedir, 'database.db'), 
    ]

    match_output = ( subprocess.check_output(exhaustive_matcher_args, universal_newlines=True) )
    logfile.write(match_output)
    print('Features matched')

    p = os.path.join(args.basedir, 'sparse')
    if not os.path.exists(p):
        os.makedirs(p)

    mapper_args = [
        'colmap', 'mapper',
            '--database_path', os.path.join(args.basedir, 'database.db'),
            '--image_path', os.path.join(args.basedir, args.images_folder),
            '--output_path', os.path.join(args.basedir, 'sparse'), # --export_path changed to --output_path in colmap 3.6
            '--Mapper.num_threads', '16',
            '--Mapper.init_min_tri_angle', '4',
            '--Mapper.multiple_models', '0',
            '--Mapper.extract_colors', '0',     # @yxie20 We considered extract color for the pre-training routine (perf-opt Task 2)
    ]

    map_output = ( subprocess.check_output(mapper_args, universal_newlines=True) )
    logfile.write(map_output)
    logfile.close()
    print('Sparse map created')
    
    print( 'Finished running COLMAP, see {} for logs'.format(logfile_name) )


