#! /usr/bin/bash env

cd ../../
PYTHONPATH=. python data_tools/build_file_list.py hmdb51 data/hmdb51/rawframes/ --level 2 --format rawframes
echo "Filelist for rawframes generated."

PYTHONPATH=. python data_tools/build_file_list.py hmdb51 data/hmdb51/videos/ --level 2 --format videos
echo "Filelist for videos generated."

cd data_tools/hmdb51/