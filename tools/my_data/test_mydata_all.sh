# test all
cd ../../

sh ./tools/dist_test_recognizer.sh configs/TSN/my_data/tsn_rgb_sknet.py 2
sh ./tools/dist_test_recognizer.sh configs/TSN/my_data/tsn_rgb_bninception.py 2
sh ./tools/dist_test_recognizer.sh configs/TSN/my_data/tsn_rgb_resnet101.py 2

#sh ./tools/dist_test_recognizer.sh configs/TSN/my_data/tsn_flow_sknet.py 2
sh ./tools/dist_test_recognizer.sh configs/TSN/my_data/tsn_flow_bninception.py 2
sh ./tools/dist_test_recognizer.sh configs/TSN/my_data/tsn_flow_resnet101.py 2
