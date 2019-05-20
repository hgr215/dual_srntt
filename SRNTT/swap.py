import tensorflow as tf
import numpy as np
import os
import time


class Swap(object):
    def __init__(self, input_size, patch_size=3, stride=1, sess=None, matching_layer=['relu3_1', 'relu2_1', 'relu1_1']):
        '''
        require same input size, once Swap built.
        :param sess:
        input_size: (1,height,width,channel) of content feature.
        '''
        self.patch_size = patch_size
        self.input_size = np.array(input_size, dtype=np.int32)  # used for fast-swap scheme.scheme
        self.stride = stride
        self.sess = sess
        self.content = None
        self.style = None
        self.condition = None
        self.conv_input = tf.placeholder(dtype=tf.float32, shape=[1, None, None, None], name='swap_input')
        self.conv_filter = tf.placeholder(dtype=tf.float32, shape=[self.patch_size, self.patch_size, None, None],
                                          name='swap_filter')

        self.tconv_input = tf.placeholder(dtype=tf.float32, shape=[1, None, None, None], name='swap_t_input')
        self.t_conv_filter = tf.placeholder(dtype=tf.float32, shape=[None, None, None, None],
                                            name='tconv_filter')  # --may have problem?
        # self.t_conv_filter = tf.placeholder(dtype=tf.float32, shape=[self.patch_size, self.patch_size, None, None])
        self.conv = tf.nn.conv2d(
            input=self.conv_input,
            filter=self.conv_filter,
            strides=(1, self.stride, self.stride, 1),
            padding='VALID',  # --no padding
            name='feature_patchmatch'
        )
        # self.output_hw = 1, patch_size + (hi - 1) * stride, patch_size + (wi - 1) * stride, ci  # --cal out size
        self.t_conv = tf.nn.conv2d_transpose(
            value=self.tconv_input,
            output_shape=self.input_size,
            filter=self.t_conv_filter,
            strides=(1, self.stride, self.stride, 1),
            padding='VALID',
            name='swap_transpose_conv'
        )
        # cal overlap:
        _, h_c, w_c, channels = self.input_size
        over_in = np.ones((1, (h_c - patch_size) // stride + 1, (w_c - patch_size) // stride + 1, 1))
        # over_in_tf = tf.placeholder(dtype=tf.float32, shape=over_in.shape)
        # over_filter_tf = tf.placeholder(dtype=tf.float32, shape=(patch_size, patch_size, 1, 1))
        over_map = tf.nn.conv2d_transpose(
            value=self.tconv_input,
            output_shape=[1, h_c, w_c, 1],
            filter=self.t_conv_filter,
            strides=(1, stride, stride, 1),
            padding='VALID',
            name='swap_transpose_conv_over'
        )
        self.over_map = over_map.eval(
            {self.tconv_input: over_in, self.t_conv_filter: np.ones((patch_size, patch_size, 1, 1))}, session=self.sess)
        print('overmap.shape: ', self.over_map.shape)

        # extraction op:
        self.patch_extr_ops = {}  # --{patch_size:op}
        patches_tf = tf.image.extract_image_patches(
            images=self.tconv_input,
            ksizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.stride, self.stride, 1],
            rates=[1, 1, 1, 1],
            padding='VALID',
            name='patches_extraction%d' % self.patch_size
        )
        patches_tf = tf.reshape(patches_tf, (-1, self.patch_size, self.patch_size, channels))
        patches_tf = tf.transpose(patches_tf, (1, 2, 3, 0))
        self.patch_extr_ops[self.patch_size] = patches_tf

        if matching_layer == ['relu3_1', 'relu2_1', 'relu1_1']:
            pass
            ratio = 1
            self.other_tconv = []
            self.other_osize = []
            self.over_map_other = []
            for ii in range(2):
                ratio *= 2
                self.other_osize.append((self.input_size * [1, ratio, ratio, 1 / ratio]).astype(np.int32))
                self.other_tconv.append(
                    tf.nn.conv2d_transpose(
                        value=self.tconv_input,
                        output_shape=self.other_osize[ii],
                        filter=self.t_conv_filter,
                        strides=(1, self.stride * ratio, self.stride * ratio, 1),
                        padding='VALID',
                        name='swap_transpose_conv_%d' % (ii + 1)
                    )
                )
                # other layer overmap:
                over_map_t = tf.nn.conv2d_transpose(
                    value=self.tconv_input,
                    output_shape=[1, h_c * ratio, w_c * ratio, 1],
                    filter=self.t_conv_filter,
                    strides=(1, stride * ratio, stride * ratio, 1),
                    padding='VALID',
                    name='swap_transpose_conv_over_%d' % ii
                )
                over_map_t_np = over_map_t.eval({self.tconv_input: over_in,
                                                 self.t_conv_filter: np.ones(
                                                     (patch_size * ratio, patch_size * ratio, 1, 1))},
                                                session=self.sess)
                print('overmap.shape', over_map_t_np.shape)
                self.over_map_other.append(over_map_t_np)

                # other extraction ops:
                patches_tf = tf.image.extract_image_patches(
                    images=self.tconv_input,
                    ksizes=[1, self.patch_size * ratio, self.patch_size * ratio, 1],
                    strides=[1, self.stride * ratio, self.stride * ratio, 1],
                    rates=[1, 1, 1, 1],
                    padding='VALID',
                    name='patches_extraction%d' % (self.patch_size * ratio)
                )
                patches_tf = tf.reshape(patches_tf,
                                        (-1, self.patch_size * ratio, self.patch_size * ratio, channels // ratio))
                patches_tf = tf.transpose(patches_tf, (1, 2, 3, 0))
                self.patch_extr_ops[self.patch_size * ratio] = patches_tf
        else:
            print('cannot cal input size with %s, for the t_conv' % matching_layer)
            exit(0)

        # self.t_conv=tf.layers.conv2d_transpose(
        #     inputs=self.tconv_input,
        #     filters=self.t_conv_filter,
        #     kernel_size=self.patch_size,
        #     strides=(self.stride,self.stride),
        #     padding='valid',
        #     use_bias=False,
        #
        #
        # )

    def style2patches(self, feature_map=None, ):
        """
        sample patches from the style (reference) map
        :param feature_map: array, [H, W, C]
        :return: array (conv kernel), [H, W, C_in, C_out]  # --C_in = C
        """
        if feature_map is None:
            feature_map = self.style
        h, w, c = feature_map.shape
        t0 = time.time()
        # patches_tf = tf.image.extract_image_patches(
        #     images=feature_map[np.newaxis, ...],
        #     ksizes=[1, self.patch_size, self.patch_size, 1],
        #     strides=[1, self.stride, self.stride, 1],
        #     rates=[1, 1, 1, 1],
        #     padding='VALID',
        #     name='patches_extraction'
        # )
        if self.patch_size not in self.patch_extr_ops:
            print('no extr op for %d' % self.patch_size)
            exit(0)

        patches_tf = self.patch_extr_ops[self.patch_size]
        # print('op building time: %.4f' % (time.time() - t0))
        patches = patches_tf.eval({self.tconv_input: feature_map[np.newaxis, ...]}, session=self.sess)
        print('op building + eval time: %.4f' % (time.time() - t0))
        return patches

    def style2patches_slow(self, feature_map=None):
        """
        sample patches from the style (reference) map
        :param feature_map: array, [H, W, C]
        :return: array (conv kernel), [H, W, C_in, C_out]
        """
        if feature_map is None:
            feature_map = self.style
        h, w, c = feature_map.shape
        patches = []
        for ind_row in range(0, h - self.patch_size + 1, self.stride):
            for ind_col in range(0, w - self.patch_size + 1, self.stride):
                patches.append(feature_map[ind_row:ind_row + self.patch_size, ind_col:ind_col + self.patch_size, :])
        return np.stack(patches, axis=-1)

    def conditional_swap_multi_layer(self, content, style, condition, other_styles=None,
                                     is_weight=False):
        """
        feature swapping with multiple references on multiple feature layers
        :param content: array (h, w, c), feature map of content
        :param style: list of array [(h, w, c)], feature map of each reference
        :param condition: list of array [(h, w, c)], augmented feature map of each reference for matching with content map
        :param patch_size: int, size of matching patch
        :param stride: int, stride of sliding the patch
        :param other_styles: list (different layers) of lists (different references) of array (feature map),
                [[(h_, w_, c_)]], feature map of each reference from other layers
        :param is_weight, bool, whether compute weights
        :return: swapped feature maps - [3D array, ...], matching weights - 2D array, matching idx - 2D array
        """
        print('content.shape:', content.shape)

        assert isinstance(content, np.ndarray)
        self.content = np.squeeze(content)
        assert len(self.content.shape) == 3

        assert isinstance(style, list)
        self.style = [np.squeeze(s) for s in style]
        assert all([len(self.style[i].shape) == 3 for i in range(len(self.style))])

        assert isinstance(condition, list)
        self.condition = [np.squeeze(c) for c in condition]  # --condition is blurred style map
        assert all([len(self.condition[i].shape) == 3 for i in range(len(self.condition))])
        assert len(self.condition) == len(self.style)

        # --style and the content size can be different, but size of condition and style must be same
        num_channels = self.content.shape[-1]
        assert all([self.style[i].shape[-1] == num_channels for i in range(len(self.style))])
        # assert all([self.condition[i].shape[-1] == num_channels for i in range(len(self.condition))])
        assert all([self.style[i].shape == self.condition[i].shape for i in range(len(self.style))])

        if other_styles is not None:
            assert isinstance(other_styles, list)
            assert all([isinstance(s, list) for s in other_styles])
            other_styles = [[np.squeeze(s) for s in styles] for styles in other_styles]
            assert all([all([len(s.shape) == 3 for s in styles]) for styles in other_styles])

        patch_size = self.patch_size
        stride =self.stride

        # split content, style, and condition into patches
        t_e = time.time()
        patches_content = self.style2patches(self.content)
        patches_style = np.concatenate(list(map(self.style2patches, self.style)), axis=-1)  # --shape of (p,p,c,n_p*1)
        patches = np.concatenate(list(map(self.style2patches, self.condition)), axis=-1)  # --patches of condition
        print('patch extract time: %0.2f s.' % (time.time() - t_e))
        # normalize content and condition patches
        norm = np.sqrt(np.sum(np.square(patches), axis=(0, 1, 2)))
        patches_style_normed = patches / norm
        norm = np.sqrt(np.sum(np.square(patches_content), axis=(0, 1, 2)))
        # if is_weight:   patches_content_normed = patches_content / norm  # no use if not is_weight
        del norm, patches, patches_content

        # match content and condition patches (batch-wise matching because of memory limitation)
        # the size of a batch is 512MB
        t_m = time.time()
        batch_size = int(1024. ** 2 * 512 / (self.content.shape[0] * self.content.shape[1]))
        num_out_channels = patches_style_normed.shape[-1]  # --num_p
        print('\tMatching ...')
        max_idx, max_val = None, None
        for idx in range(0, num_out_channels, batch_size):
            print('\t  Batch %02d/%02d' % (idx / batch_size + 1, np.ceil(1. * num_out_channels / batch_size)))
            batch = patches_style_normed[..., idx:idx + batch_size]
            if self.sess:  # --sess is not None
                # --here content patches is not normed, which is not necessary
                # for thres is not needed be in 0~1.(th is not used here)
                corr = self.conv.eval({self.conv_input: [self.content], self.conv_filter: batch}, session=self.sess)
            else:
                corr = self.conv.eval({self.conv_input: [self.content], self.conv_filter: batch})

            _, h_co, w_co, _ = corr.shape  # (1,hc,wc,num_p)
            corr = np.squeeze(corr)
            max_idx_tmp = np.argmax(corr, axis=-1) + idx
            max_val_tmp = np.max(corr, axis=-1)
            del corr, batch
            # del batch
            if max_idx is None:
                max_idx, max_val = max_idx_tmp, max_val_tmp
            else:
                indices = max_val_tmp > max_val
                max_val[indices] = max_val_tmp[indices]
                max_idx[indices] = max_idx_tmp[indices]
        print('patch match time: %0.2f s.' % (time.time() - t_m))

        '''my swap'''
        del max_val
        # max_idx = max_idx[np.newaxis, ...]  # shape of (1,h,w)
        map_t = np.zeros_like(self.content)[np.newaxis, ...]
        maps = []
        if other_styles:
            map_t_other = []
            patches_style_other = []
            self.t_convs = []
            for style in other_styles:  # for in different layer
                ratio = float(style[0].shape[0]) / self.style[0].shape[0]
                assert int(ratio) == ratio
                ratio = int(ratio)
                self.patch_size = patch_size * ratio
                self.stride = stride * ratio
                patches_style_other.append(np.concatenate(list(map(self.style2patches, style)), axis=-1))
                map_t_other.append(
                    np.zeros((1, self.content.shape[0] * ratio, self.content.shape[1] * ratio, style[0].shape[2])))

        for idx in range(0, num_out_channels, batch_size):
            filter_b = patches_style[:, :, :, idx * batch_size:(idx + 1) * batch_size]
            print('batch_size: %d' % batch_size)
            num_patches_b = filter_b.shape[-1]  # --real batch_size
            scores_b = np.zeros((h_co, w_co, num_patches_b + 1))  # last channel is for indexing convenience
            max_idx_b = max_idx.copy()
            coor_out = (max_idx < idx) + (max_idx >= (idx + batch_size))  # (x,y) whose patch_idx not in batch
            max_idx_b[~coor_out] -= idx * batch_size
            max_idx_b[coor_out] = -1
            max_idx_b = max_idx_b.reshape((-1,))
            scores_b = scores_b.reshape((-1, num_patches_b + 1))
            scores_b[range(h_co * w_co), max_idx_b[range(h_co * w_co)]] = 1
            scores_b = scores_b.reshape((h_co, w_co, num_patches_b + 1))
            scores_b = scores_b[:, :, :-1]  # remove last channel

            # print('patch_style.shape: ' , patches_style.shape)
            # condition layer swap
            print('filter_b.shape: ', filter_b.shape, scores_b[np.newaxis, ...].shape)
            map_t += self.t_conv.eval({self.tconv_input: scores_b[np.newaxis, ...], self.t_conv_filter: filter_b},
                                      session=self.sess)
            # other layer swap:
            if other_styles:
                for ii in range(len(map_t_other)):
                    filter_b = patches_style_other[ii][:, :, :, idx * batch_size:(idx + 1) * batch_size]
                    map_t_other[ii] += self.other_tconv[ii].eval(
                        {self.tconv_input: scores_b[np.newaxis, ...], self.t_conv_filter: filter_b}, session=self.sess)

        # cal overlap (flatten t_conv, like in pytorch swap project)
        # over_input = np.ones((1, h_co, w_co, 1))
        # over_filter = np.ones((patches_style.shape[0], patches_style.shape[1], 1, 1))
        # over_map = self.t_conv.eval({self.tconv_input: over_input, self.t_conv_filter: over_filter},
        #                             session=self.sess)
        map_t /= self.over_map
        maps.append(map_t.squeeze())
        for ii in range(len(map_t_other)):
            # over_filter = np.ones((patches_style_other[ii].shape[0], patches_style_other[ii].shape[1], 1, 1))
            # over_map = self.other_tconv[ii].eval({self.tconv_input: over_input, self.t_conv_filter: over_filter},
            #                                      session=self.sess)
            map_t_other[ii] /= self.over_map_other[ii]
            maps.append(map_t_other[ii].squeeze())

        weights = None
        '''end'''

        # compute matching similarity (inner product)
        # if is_weight:
        #     print('\tWeighting ...')
        #     corr2 = np.matmul(
        #         np.transpose(np.reshape(patches_content_normed, (-1, patches_content_normed.shape[-1]))),
        #         np.reshape(patches_style_normed, (-1, patches_style_normed.shape[-1]))
        #     )
        #     weights = np.reshape(np.max(corr2, axis=-1), max_idx.shape)
        #     del patches_content_normed, patches_style_normed, corr2
        # else:
        #     weights = None
        #     del patches_content_normed, patches_style_normed

        return maps, weights, max_idx
