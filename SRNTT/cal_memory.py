short_size = 320
patch_size = 3  # patch size defined at feature level
stride = 1
long_size = short_size // 2
# long_size=500
cell_size = 4  # 4 byte float32
feature_scale = 4  # scale factor of relu3_1 feature
print('Memory use of long %d, SU %d, Patch size %d, stride %d:' % (long_size, short_size, patch_size, stride))

short_size //= feature_scale
long_size //= feature_scale
print('feature size: %d, %d' % (long_size, short_size))
num_iter = ((short_size - patch_size) // stride + 1) ** 2
memory = num_iter * ((long_size - patch_size) // stride + 1) ** 2
memory = memory * cell_size / 1024.0 / 1024.0  # MB
print('%0.2f MB' % (memory))
print('Num of iter when transfer the patches: %d' % num_iter)
