# test all
cd ../../

./tools/dist_test_recognizer.sh configs/TSN/ucf101/tsn_rgb_sknet.py
./tools/dist_test_recognizer.sh configs/TSN/ucf101/tsn_rgb_bninception.py
./tools/dist_test_recognizer.sh configs/TSN/ucf101/tsn_rgb_resnet101.py

./tools/dist_test_recognizer.sh configs/TSN/ucf101/tsn_flow_sknet.py
./tools/dist_test_recognizer.sh configs/TSN/ucf101/tsn_flow_bninception.py
./tools/dist_test_recognizer.sh configs/TSN/ucf101/tsn_flow_resnet101.py
