#!/usr/bin/env bash
echo
echo -- Start Training --
echo

#training_set=${1-CUFED}
#training_set=${1-dual}
training_set=${1-dual_hw}

# # download training set
#  python download_dataset.py --dataset_name ${training_set}

# # calculate the swapped feature map in the offline manner
#python offline_patchMatch_textureSwap.py --data_folder data/train/${training_set}


 # train a new model
#python main.py \
#    --is_train True \
#    --input_dir data/train/${training_set}/input \
#    --ref_dir data/train/${training_set}/ref \
#    --map_dir data/train/${training_set}/map_321 \
#    --use_pretrained_model False \
#    --num_init_epochs 2 \
#    --num_epochs 100 \
#    --save_dir demo_training_srntt
##    --is_gan True

 # train a 2x new model
#python main.py \
#    --is_train True \
#    --input_dir data/train/${training_set}/input \
#    --ref_dir data/train/${training_set}/ref \
#    --map_dir data/train/${training_set}/map_321_2x \
#    --use_pretrained_model False \
#    --num_init_epochs 2 \
#    --num_epochs 35 \
#    --save_dir demo_training_srntt \
#    --x2_train
##    --is_gan True

# train p10 model
#python main.py \
#    --is_train True \
#    --input_dir data/train/${training_set}/input \
#    --ref_dir data/train/${training_set}/ref \
#    --map_dir data/train/${training_set}/map_321_2x_p10s5 \
#    --use_pretrained_model False \
#    --num_init_epochs 2 \
#    --num_epochs 35 \
#    --save_dir demo_training_srntt \
#    --x2_train

## train p10s2 model hot
#python main.py \
#    --is_train True \
#    --input_dir data/train/${training_set}/input \
#    --ref_dir data/train/${training_set}/ref \
#    --map_dir data/train/${training_set}/map_321_2x_p10s5 \
#    --use_pretrained_model False \
#    --num_init_epochs 2 \
#    --num_epochs 35 \
#    --save_dir demo_training_srntt_p10s2 \
#    --x2_train \
#    --hot_start True \
#    --patch_size 10 \
#    --stride 2

python main.py \
    --is_train True \
    --input_dir data/train/${training_set}/input \
    --ref_dir data/train/${training_set}/ref \
    --map_dir data/train/${training_set}/map_321_2x \
    --use_pretrained_model False \
    --num_init_epochs 2 \
    --num_epochs 35 \
    --save_dir demo_training_srntt \
    --x2_train
#    --is_gan True


## train based on a pre-trained model
# load_step: step to load; num_epochs: epochs remaining
#python main.py \
#    --is_train True \
#    --input_dir data/train/${training_set}/input \
#    --ref_dir data/train/${training_set}/ref \
#    --map_dir data/train/${training_set}/map_321 \
#    --use_pretrained_model True \
#    --num_init_epochs 0 \
#    --num_epochs 70 \
#    --save_dir demo_training_srntt\
#    --load_step 29\
#    --is_gan False

## train p3s1 model hot dual_hw (transfer)
#python main.py \
#    --is_train True \
#    --input_dir data/train/${training_set}/input \
#    --ref_dir data/train/${training_set}/ref \
#    --map_dir data/train/${training_set}/map_321_2x \
#    --use_pretrained_model True \
#    --num_init_epochs 0 \
#    --num_epochs 20 \
#    --load_step 32 \
#    --save_dir model_x2 \
#    --x2_train \
#    --hot_start True \
##    --patch_size 10 \
##    --stride 2

# train from hw dataset
python main.py \
    --is_train True \
    --input_dir data/train/dual_hw/input \
    --ref_dir data/train/dual_hw/ref \
    --map_dir data/train/dual_hw/map_321_2x \
    --use_pretrained_model False \
    --num_init_epochs 2 \
    --num_epochs 35 \
    --save_dir model_hw_init \
    --x2_train

