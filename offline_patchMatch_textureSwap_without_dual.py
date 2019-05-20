import tensorflow as tf
import numpy as np
from glob import glob
from os.path import exists, join, split
from os import makedirs, environ
from SRNTT.model import *
from SRNTT.vgg19 import *
from SRNTT.swap import *
from scipy.misc import imread, imresize
import argparse

environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

parser = argparse.ArgumentParser('offline_patchMatch_textureSwap')
parser.add_argument('--data_folder', type=str, default='data/train/CUFED', help='The dir of dataset: CUFED or DIV2K')
parser.add_argument('--scale', type=int, default=2)
args = parser.parse_args()

data_folder = args.data_folder
scale = args.scale
print('scale:%d' % scale)
if 'CUFED' in data_folder:
    input_size = 160 // scale
elif 'DIV2K' in data_folder:
    input_size = 320 // scale
else:
    raise Exception('Unrecognized dataset!')

input_path = join(data_folder, 'input')
ref_path = join(data_folder, 'ref')
matching_layer = ['relu3_1', 'relu2_1', 'relu1_1']
save_path = join(data_folder, 'map_321_2x')
if not exists(save_path):
    makedirs(save_path)

input_files = sorted(glob(join(input_path, '*.png')))
ref_files = sorted(glob(join(ref_path, '*.png')))
n_files = len(input_files)
assert n_files == len(ref_files)

vgg19_model_path = 'SRNTT/models/VGG19/imagenet-vgg-verydeep-19.mat'
tf_input = tf.placeholder(dtype=tf.float32, shape=[1, input_size, input_size, 3])
srntt = SRNTT(vgg19_model_path=vgg19_model_path)
net_upscale, _ = srntt.model(tf_input / 127.5 - 1, is_train=False)
net_vgg19 = VGG19(model_path=vgg19_model_path)
swaper = Swap()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    tf.global_variables_initializer().run()
    print_format = '%%0%dd/%%0%dd' % (len(str(n_files)), len(str(n_files)))
    for i in range(n_files):
        file_name = join(save_path, split(input_files[i])[-1].replace('.png', '.npz'))
        if exists(file_name):
            continue
        print(print_format % (i + 1, n_files))
        img_in_lr = imresize(imread(input_files[i], mode='RGB'), (input_size, input_size), interp='bicubic')
        img_ref = imresize(imread(ref_files[i], mode='RGB'), (input_size * scale, input_size * scale),
                           interp='bicubic')  # $
        img_ref_lr = imresize(img_ref, (input_size, input_size), interp='bicubic')
        # img_in_sr = (net_upscale.outputs.eval({tf_input: [img_in_lr]})[0] + 1) * 127.5
        # img_ref_sr = (net_upscale.outputs.eval({tf_input: [img_ref_lr]})[0] + 1) * 127.5
        img_in_sr = imresize(img_in_lr, [scale * input_size, scale * input_size], interp='bicubic')
        img_ref_sr = imresize(img_ref_lr, [scale * input_size, scale * input_size], interp='bicubic')

        # get feature maps via VGG19
        map_in_sr = net_vgg19.get_layer_output(sess=sess, feed_image=img_in_sr, layer_name=matching_layer[0])
        map_ref = net_vgg19.get_layer_output(sess=sess, feed_image=img_ref, layer_name=matching_layer)
        map_ref_sr = net_vgg19.get_layer_output(sess=sess, feed_image=img_ref_sr, layer_name=matching_layer[0])

        # patch matching and swapping
        other_style = []
        for m in map_ref[1:]:
            other_style.append([m])

        maps, weights, correspondence = swaper.conditional_swap_multi_layer(
            content=map_in_sr,
            style=[map_ref[0]],
            condition=[map_ref_sr],
            other_styles=other_style
        )

        # save maps
        np.savez(file_name, target_map=maps, weights=weights, correspondence=correspondence)
