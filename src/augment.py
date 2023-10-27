import tensorflow as tf


class MixStripes(tf.keras.layers.Layer):
    def __init__(
            self,
            dim,
            mix_width,
            stripes_num=None,
            mask_ratio=None,
            random_noise_mask=False,
            fixed_stripes_num=False,
            mini_batch_mix_mask=False,
            **kwargs):
        super(MixStripes, self).__init__(**kwargs)
        self.dim = dim
        self.mix_width = mix_width
        self.stripes_num = stripes_num
        self.random_noise_mask = random_noise_mask
        self.fixed_stripes_num = fixed_stripes_num
        self.mask_ratio = mask_ratio
        self.supports_masking = True
        self.mini_batch_mix_mask = mini_batch_mix_mask

    def call(self, inputs, mask=None, training=None, add_loss=False):
        if not add_loss:
            if training is False:
                return inputs, None
        if self.mix_width == 0 or self.stripes_num == 0:
            return inputs, None

        masks = self.compute_batch_mask(inputs)
        if self.random_noise_mask:
            mask_emb = tf.random.uniform(
                [], minval=0, maxval=1, dtype=inputs.dtype)
            inputs = tf.where(masks, mask_emb, inputs)
        elif self.mini_batch_mix_mask:
            batch_size = tf.shape(inputs)[0]
            all_indices = tf.range(batch_size)
            for i in range(batch_size):
                random_index = tf.random.shuffle(
                    tf.where(all_indices != i)[:, 0])[0]

                mixed_batch = tf.where(
                    masks[i], (inputs[i] + inputs[random_index]) * 0.5, inputs[i])
                inputs = tf.tensor_scatter_nd_update(
                    inputs, [[i]], [mixed_batch])

        else:
            inputs = tf.where(
                masks, tf.constant(
                    0.0, dtype=tf.float16), inputs)

        if mask is not None:
            inputs *= tf.expand_dims(tf.cast(mask, dtype=tf.float32), -1)
        return inputs, masks

    @tf.function
    def compute_batch_mask(self, batch):
        time_steps, freqs = tf.shape(batch)[1], tf.shape(batch)[2]
        batch_size = tf.shape(batch)[0]

        min_rando_stripes_num = 1
        if self.fixed_stripes_num:
            min_rando_stripes_num = self.stripes_num
        random_stripes_num = tf.random.uniform(
            [1],
            minval=min_rando_stripes_num,
            maxval=self.stripes_num + 1,
            dtype=tf.int32)[0]

        if self.mask_ratio:
            random_stripes_num = tf.reduce_sum(
                tf.where(
                    tf.random.uniform(
                        [time_steps],
                        minval=0,
                        maxval=1) < self.mask_ratio,
                    1,
                    0))

        if self.dim == 1:
            x_range = tf.range(time_steps)[None, None, :]
            bgn = tf.random.uniform([batch_size, random_stripes_num, 1],
                                    minval=0,
                                    maxval=time_steps - self.mix_width,
                                    dtype=tf.int32)
            mask = tf.reduce_any(
                (bgn <= x_range) & (x_range < (bgn + self.mix_width)),
                axis=1, keepdims=True)
            mask = tf.broadcast_to(tf.expand_dims(tf.squeeze(
                mask, axis=1), axis=-1), [batch_size, time_steps, freqs])
        elif self.dim == 2:
            y_range = tf.range(freqs)[None, None, :]
            bgn = tf.random.uniform([batch_size, random_stripes_num, 1],
                                    minval=0,
                                    maxval=freqs - self.mix_width,
                                    dtype=tf.int32)
            mask = tf.reduce_any(
                (bgn <= y_range) & (y_range < (bgn + self.mix_width)),
                axis=1, keepdims=True)
            mask = tf.broadcast_to(mask, [batch_size, time_steps, freqs])
        return mask

    def get_config(self):
        config = super(MixStripes, self).get_config()
        return dict(list(config.items()))


class SpecMixAugmentation(tf.keras.layers.Layer):
    def __init__(self,
                 time_mix_width,
                 freq_mix_width,
                 time_mask_ratio=None,
                 freq_mask_ratio=None,
                 random_noise_mask=False,
                 mini_batch_mix_mask=False,
                 **kwargs):
        super(SpecMixAugmentation, self).__init__(**kwargs)
        self.time_mixer = MixStripes(
            dim=1,
            mix_width=time_mix_width,
            stripes_num=1,
            mask_ratio=time_mask_ratio,
            random_noise_mask=random_noise_mask,
            mini_batch_mix_mask=mini_batch_mix_mask)
        self.freq_mixer = MixStripes(
            dim=2,
            mix_width=freq_mix_width,
            stripes_num=1,
            mask_ratio=freq_mask_ratio,
            random_noise_mask=random_noise_mask,
            mini_batch_mix_mask=mini_batch_mix_mask)
        self.supports_masking = True

    def call(self, inputs, mask=None, training=None):
        # inputs: (batch_size, time_steps, freqs)

        x, _ = self.time_mixer(inputs, mask=mask, training=training)
        x, _ = self.freq_mixer(x, mask=mask, training=training)
        return x

    def get_config(self):
        config = super(SpecMixAugmentation, self).get_config()
        return dict(list(config.items()))
