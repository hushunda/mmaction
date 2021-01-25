# test all
cd ../../

./tools/dist_test_recognizer.sh configs/TSN/ucf101/tsn_rgb_sknet.py 6
./tools/dist_test_recognizer.sh configs/TSN/ucf101/tsn_rgb_bninception.py 6
./tools/dist_test_recognizer.sh configs/TSN/ucf101/tsn_rgb_resnet101.py 6

./tools/dist_test_recognizer.sh configs/TSN/ucf101/tsn_flow_sknet.py 6
./tools/dist_test_recognizer.sh configs/TSN/ucf101/tsn_flow_bninception.py 6
./tools/dist_test_recognizer.sh configs/TSN/ucf101/tsn_flow_resnet101.py 6
