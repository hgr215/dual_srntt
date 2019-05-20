import tensorflow as tf
import cv2
import os
from os.path import join

data_dir = './data/train/dual'
# crop_size = 320  # crop size of label input
crop_size = 160
epoches = 12  # num of patches per image

filenames = os.listdir(join(data_dir, 'train'))
num = len(filenames)
if not os.path.exists(join(data_dir, 'input')):
    os.mkdir(join(data_dir, 'input'))
if not os.path.exists(join(data_dir, 'ref')):
    os.mkdir(join(data_dir, 'ref'))

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    in_tf = tf.placeholder(tf.uint8, [None, None, None])
    gt_tf = tf.random_crop(in_tf, [crop_size, crop_size, 3])
    l_tf = tf.image.resize_image_with_crop_or_pad(gt_tf, crop_size // 2, crop_size // 2)
    for e in range(epoches):
        for i in range(num):
            im = cv2.imread(join(data_dir, 'train', filenames[i]), cv2.IMREAD_COLOR)
            assert im is not None
            im_gt, im_l = sess.run(fetches=[gt_tf, l_tf], feed_dict={in_tf: im})
            assert cv2.imwrite(join(data_dir, 'input', filenames[i][:-4] + '_%d.png' % e), im_gt)
            assert cv2.imwrite(join(data_dir, 'ref', filenames[i][:-4] + '_%d.png' % e), im_l)
            print('%d %s processed.' % (i, filenames[i]))
