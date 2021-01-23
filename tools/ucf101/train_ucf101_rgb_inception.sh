cd ../..

./tools/dist_train_recognizer.sh configs/TSN/ucf101/tsn_rgb_bninception.py 1 --validate --work_dir ./workdir/ucf101_rgb_inception
