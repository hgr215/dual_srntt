import tensorflow as tf
import cv2
import os
import numpy as np
from os.path import join

# '''dual from DIV2K sti'''
# data_dir = './data/train/dual'
# # crop_size = 320  # crop size of label input
# crop_size = 160
# epoches = 12  # num of patches per image
#
# filenames = os.listdir(join(data_dir, 'train'))
# num = len(filenames)
# if not os.path.exists(join(data_dir, 'input')):
#     os.mkdir(join(data_dir, 'input'))
# if not os.path.exists(join(data_dir, 'ref')):
#     os.mkdir(join(data_dir, 'ref'))
#
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# with tf.Session(config=config) as sess:
#     in_tf = tf.placeholder(tf.uint8, [None, None, None])
#     gt_tf = tf.random_crop(in_tf, [crop_size, crop_size, 3])
#     l_tf = tf.image.resize_image_with_crop_or_pad(gt_tf, crop_size // 2, crop_size // 2)
#     for e in range(epoches):
#         for i in range(num):
#             im = cv2.imread(join(data_dir, 'train', filenames[i]), cv2.IMREAD_COLOR)
#             assert im is not None
#             im_gt, im_l = sess.run(fetches=[gt_tf, l_tf], feed_dict={in_tf: im})
#             assert cv2.imwrite(join(data_dir, 'input', filenames[i][:-4] + '_%d.png' % e), im_gt)
#             assert cv2.imwrite(join(data_dir, 'ref', filenames[i][:-4] + '_%d.png' % e), im_l)
#             print('%d %s processed.' % (i, filenames[i]))

'''huawei_files'''
# use is_ori = True, down and up mode, where input is SU actually:

data_dir = './data/train/dual_hw'
# crop_size = 320  # crop size of label input
crop_size = 160
epoches_same = 920  # num of patches per image
epoches_diff = 115  # num of patches per image

filenames = os.listdir(join(data_dir, 'train'))
num = len(filenames) // 2
if not os.path.exists(join(data_dir, 'input')):
    os.mkdir(join(data_dir, 'input'))
if not os.path.exists(join(data_dir, 'ref')):
    os.mkdir(join(data_dir, 'ref'))

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# building the graph:
in_4_tf = tf.placeholder(tf.float32, [1, None, None, 3])
in_5_tf = tf.placeholder(tf.float32, [1, None, None, 3])
var_4_tf = tf.get_variable("in_4", validate_shape=False, initializer=in_4_tf)
var_5_tf = tf.get_variable("in_5", validate_shape=False, initializer=in_5_tf)
# var_4_tf = tf.get_variable('in_4', shape=[1, None, None, 3], dtype=tf.float32, validate_shape=False)
# var_5_tf = tf.get_variable('in_5', shape=[1, None, None, 3], dtype=tf.float32, validate_shape=False)
# ass_4 = tf.assign(var_4_tf, in_4_tf, name='load_in4', validate_shape=False)
# ass_5 = tf.assign(var_5_tf, in_5_tf, name='load_in5', validate_shape=False)
# size_4 = tf.Tensor(var_4_tf.shape[1:3], dtype=tf.int32)
# size_5 = tf.Tensor(var_5_tf.shape[1:3], dtype=tf.int32)
size_4 = tf.shape(var_4_tf)[1:3]
size_5 = tf.shape(var_5_tf)[1:3]
print(type(size_4))
h_4 = size_4[0]
w_4 = size_4[1]
h_5 = size_5[0]
w_5 = size_5[1]

im_4u_tf = tf.image.resize_bicubic(var_4_tf, size_4 * 5 // 4)

# crop same patch
im_4uc_tf = tf.image.resize_image_with_crop_or_pad(im_4u_tf, size_5[0], size_5[1])
im_45_tf = tf.concat([im_4uc_tf, var_5_tf], axis=-1)
im_45_crop = tf.random_crop(im_45_tf, [1, crop_size, crop_size, 6])
im_c_input = im_45_crop[:, :, :, :3]
im_c_ref = im_45_crop[:, :, :, 3:]

# crop diff patch
h_of_patch = tf.random.uniform(shape=(), minval=0, maxval=size_5[0] - crop_size, dtype=tf.int32)
w_of_patch = tf.random.uniform(shape=(), minval=0, maxval=size_5[1] - crop_size, dtype=tf.int32)
border = (size_5 - size_4) // 2
# im_border_patch_input1 = im_4u_tf[border[0] - crop_size // 2:border[0] + crop_size // 2,
#                          border[1] + w_of_patch - crop_size // 2:border[1] + w_of_patch + crop_size // 2]  # up
# im_border_patch_input2 = im_4u_tf[h_4 - border[0] - crop_size // 2:h_4 - border[0] + crop_size // 2,
#                          border[1] + w_of_patch - crop_size // 2:border[1] + w_of_patch + crop_size // 2]  # down
#
# im_border_patch_input3 = im_4u_tf[h_of_patch - crop_size // 2:h_of_patch + crop_size // 2,
#                          border[1] - crop_size // 2:border[1 + crop_size // 2]]  # left
#
# im_border_patch_input4 = im_4u_tf[h_of_patch - crop_size // 2:h_of_patch + crop_size // 2,
#                          w_4 - border[1] - crop_size // 2:w_4 - border[1 + crop_size // 2]]  # right
#
# im_border_patch_ref1 = var_5_tf[border[0] - crop_size // 2:border[0] + crop_size // 2,
#                        w_of_patch - crop_size // 2:w_of_patch + crop_size // 2]  # up

border_patch_input1 = tf.image.crop_to_bounding_box(im_4u_tf, border[0] - crop_size // 2, border[1] + w_of_patch,
                                                    crop_size, crop_size)  # up
border_patch_input2 = tf.image.crop_to_bounding_box(im_4u_tf, h_4 - border[0] - crop_size // 2, border[1] + w_of_patch,
                                                    crop_size, crop_size)  # down
border_patch_input3 = tf.image.crop_to_bounding_box(im_4u_tf, border[0] + h_of_patch, border[1] - crop_size // 2,
                                                    crop_size, crop_size)  # left
border_patch_input4 = tf.image.crop_to_bounding_box(im_4u_tf, border[0] + h_of_patch, w_4 - border[1] - crop_size // 2,
                                                    crop_size, crop_size)  # right

border_patch_ref1 = tf.image.crop_to_bounding_box(var_5_tf, 0, w_of_patch, crop_size, crop_size)  # up
border_patch_ref2 = tf.image.crop_to_bounding_box(var_5_tf, h_5 - crop_size, w_of_patch, crop_size, crop_size)  # down
border_patch_ref3 = tf.image.crop_to_bounding_box(var_5_tf, h_of_patch, 0, crop_size, crop_size)  # left
border_patch_ref4 = tf.image.crop_to_bounding_box(var_5_tf, h_of_patch, w_5 - crop_size, crop_size, crop_size)  # right

with tf.Session(config=config) as sess:
    for i in range(1, num + 1):
        print(join(data_dir, 'train', '%d_4.jpg' % i))
        im_4 = cv2.imread(join(data_dir, 'train', '%d_4.jpg' % i), cv2.IMREAD_COLOR)
        im_5 = cv2.imread(join(data_dir, 'train', '%d_5.jpg' % i), cv2.IMREAD_COLOR)

        sess.run(tf.global_variables_initializer(),feed_dict={
            in_4_tf: im_4[np.newaxis, ...],
            in_5_tf: im_5[np.newaxis, ...]
        })
        # _, _ = sess.run([var_4_tf, var_5_tf],
        #                 feed_dict={in_4_tf: im_4[np.newaxis, ...],
        #                            in_5_tf: im_5[np.newaxis, ...]})  # feed the im to GPU

        for e in range(epoches_same):
            im_c_input_np, im_c_ref_np = sess.run([im_c_input, im_c_ref])
            assert cv2.imwrite(join(data_dir, 'input', '%d_same_patch%d.png' % (i, e)), im_c_input_np.squeeze())
            assert cv2.imwrite(join(data_dir, 'ref', '%d_same_patch%d.png' % (i, e)), im_c_ref_np.squeeze())
            print('%d same_%d processed.' % (i, e))

        # for e in range(epoches_diff):
        #     border_patch_input1_np, border_patch_ref1_np = sess.run([border_patch_input1, border_patch_ref1])
        #     assert cv2.imwrite(join(data_dir, 'input', '%d_diffU_patch%d.png' % (i, e)), border_patch_input1_np.squeeze())
        #     assert cv2.imwrite(join(data_dir, 'ref', '%d_diffU_patch%d.png' % (i, e)), border_patch_ref1_np.squeeze())
        #     print('%d diff1_%d processed.' % (i, e))
        #
        #     border_patch_input2_np, border_patch_ref2_np = sess.run([border_patch_input2, border_patch_ref2])
        #     assert cv2.imwrite(join(data_dir, 'input', '%d_diffD_patch%d.png' % (i, e)), border_patch_input2_np.squeeze())
        #     assert cv2.imwrite(join(data_dir, 'ref', '%d_diffD_patch%d.png' % (i, e)), border_patch_ref2_np.squeeze())
        #     print('%d diff2_%d processed.' % (i, e))
        #
        #     border_patch_input3_np, border_patch_ref3_np = sess.run([border_patch_input3, border_patch_ref3])
        #     assert cv2.imwrite(join(data_dir, 'input', '%d_diffL_patch%d.png' % (i, e)), border_patch_input3_np.squeeze())
        #     assert cv2.imwrite(join(data_dir, 'ref', '%d_diffL_patch%d.png' % (i, e)), border_patch_ref3_np.squeeze())
        #     print('%d diff3_%d processed.' % (i, e))
        #
        #     border_patch_input4_np, border_patch_ref4_np = sess.run([border_patch_input4, border_patch_ref4])
        #     assert cv2.imwrite(join(data_dir, 'input', '%d_diffR_patch%d.png' % (i, e)), border_patch_input4_np.squeeze())
        #     assert cv2.imwrite(join(data_dir, 'ref', '%d_diffR_patch%d.png' % (i, e)), border_patch_ref4_np.squeeze())
        #     print('%d diff4_%d processed.' % (i, e))
