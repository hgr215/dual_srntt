#!/usr/bin/env bash
echo
echo -- Start Testing --
echo

#hgr

# test on default SRNTT model
#python main.py \
#    --is_train False \
#    --input_dir data/test/input \
#    --ref_dir data/test/ref \
#    --use_init_model_only False \
#    --result_dir results
##    --x2


## test on default SRNTT-l2 model
#python main.py \
#    --is_train False \
#    --input_dir data/test/CUFED5/001_0.png \
#    --ref_dir data/test/CUFED5/001_2.png \
#    --use_init_model_only True \
#    --result_dir demo_testing_srntt-l2

## test on your own model
#python main.py \
#    --is_train False \
#    --use_init_model_only False \
#    --result_dir results\
#    --save_dir demo_training_srntt\
#    --load_step 0

## test x2 model
python main.py \
    --input_dir data/test/input \
    --ref_dir data/test/ref \
    --x2_train\
    --is_train False \
    --use_init_model_only False \
    --result_dir results\
    --save_dir demo_training_srntt\
    --load_step 26\
    --is_original_image True\
    --patch_size 10\
    --stride 5