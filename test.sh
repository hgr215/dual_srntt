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

## test x2 model p10s5
#python main.py \
#    --input_dir data/test/input \
#    --ref_dir data/test/ref \
#    --x2_train \
#    --is_train False \
#    --use_init_model_only False \
#    --result_dir results\
#    --save_dir demo_training_srntt\
#    --load_step 35 \
#    --is_original_image True \
#    --patch_size 10 \
#    --stride 5

#python main.py \
#    --input_dir data/test/input \
#    --ref_dir data/test/ref \
#    --x2_train \
#    --is_train False \
#    --use_init_model_only False \
#    --result_dir results \
#    --save_dir model_x2 \
#    --load_step 26 \
#    --is_original_image True

## test x2 model p10s2
#python main.py \
#    --input_dir data/test/input_sti \
#    --ref_dir data/test/ref_sti \
#    --x2_train \
#    --is_train False \
#    --use_init_model_only False \
#    --result_dir results/p10s2_sti \
#    --save_dir demo_training_srntt_p10s2\
#    --load_step 35 \
#    --is_original_image True \
#    --patch_size 10 \
#    --stride 2 \
#    --cuda 3

## test hw transfer
#python main.py \
#    --input_dir data/test/input_h \
#    --ref_dir data/test/ref_h \
#    --x2_train \
#    --is_train False \
#    --use_init_model_only False \
#    --result_dir results/huawei_transfer_model2x_p3s1 \
#    --save_dir model_x2 \
#    --load_step 52 \
#    --is_original_image True

## test hw liqi
#python main.py \
#    --input_dir data/test/input_liqi \
#    --ref_dir data/test/ref_liqi \
#    --x2_train \
#    --is_train False \
#    --use_init_model_only False \
#    --result_dir results/2x_liqi \
#    --save_dir model_x2 \
#    --load_step 26 \
#    --is_original_image True

## test init hw
#python main.py \
#    --input_dir data/test/input_hwtrain \
#    --ref_dir data/test/ref_hwtrain \
#    --x2_train \
#    --is_train False \
#    --use_init_model_only False \
#    --result_dir results/hw_init_trainim2 \
#    --save_dir model_hw_init \
#    --load_step 28 \
#    --is_original_image True \
#    --cuda 0

## test init real_hw
#python main.py \
#    --input_dir data/test/input_huawei \
#    --ref_dir data/test/ref_huawei \
#    --x2_train \
#    --is_train False \
#    --use_init_model_only False \
#    --result_dir results/hw_init_hwTest \
#    --save_dir my_models/modelWithCE_hwReal_init \
#    --load_step 19 \
#    --is_original_image True \
#    --cuda 3

## test init real_hw preCE
#python main.py \
#    --input_dir data/test/input_hwtrain \
#    --ref_dir data/test/ref_hwtrain \
#    --x2_train \
#    --is_train False \
#    --use_init_model_only False \
#    --result_dir results/hw_init_preCE \
#    --save_dir my_models/modelWithCE_hwReal_init_preCE \
#    --load_step 24 \
#    --is_original_image True \
#    --cuda 3
#

# test all transfer real_hw preCE
python main.py \
    --input_dir data/test/input_huawei \
    --ref_dir data/test/ref_huawei \
    --x2_train \
    --is_train False \
    --use_init_model_only False \
    --result_dir results/hw_all_transfer_from26 \
    --save_dir my_models/model_x2 \
    --load_step 46 \
    --is_original_image True \
    --cuda 1

python train2.py --cuda 1
