import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES']='3'

tf.set_random_seed(1)

x = tf.random_normal(shape=[1, 3, 3, 1])  # 正向卷积的结果，要作为反向卷积的输出
kernel = tf.random_normal(shape=[2, 2, 3, 1])  # 正向卷积的kernel的模样

# strides 和padding也是假想中 正向卷积的模样。
y = tf.nn.conv2d_transpose(x, kernel, output_shape=[1, 6, 6, 1],
                           strides=[1, 2, 2, 1], padding="SAME")
# 在这里，output_shape=[1,6,6,3]也可以，考虑正向过程，[1,6,6,3]时，然后通过
# kernel_shape:[2,2,3,1],strides:[1,2,2,1]也可以
# 获得x_shape:[1,3,3,1]。
# output_shape 也可以是一个 tensor
sess = tf.Session()
tf.global_variables_initializer().run(session=sess)

print(y.eval(session=sess))
