#!/usr/bin/env bash
echo
echo -- Start Training --
echo

training_set=${1-CUFED}

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

