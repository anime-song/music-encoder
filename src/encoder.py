import tensorflow as tf

from dilated_transformer import DilatedTransformer
from augment import MixStripes
from model_utils import cosine_similarity_matmul, cosine_similarity


class MaskedEncoder(tf.keras.layers.Layer):
    def __init__(
        self,
        hidden_size,
        num_heads,
        num_layers,
        intermediate_size,
        batch_size,
        patch_length,
        embedding_dim,
        dropout=0.1,
        layer_norm_eps=1e-5,
        is_gelu_approx=False,
        norm_first=True,
        temperature=1.0,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.layer_norm_eps = layer_norm_eps
        self.is_gelu_approx = is_gelu_approx
        self.norm_first = norm_first
        self.patch_length = patch_length
        self.batch_size = batch_size
        self.temperature = temperature

        self.transformer = DilatedTransformer(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            intermediate_size=self.intermediate_size,
            num_layers=self.num_layers,
            layer_norm_eps=self.layer_norm_eps,
            is_gelu_approx=self.is_gelu_approx,
            dropout=self.dropout,
            norm_first=self.norm_first,
            attention_window_size=5,
            batch_size=batch_size,
            seq_len=patch_length
        )

        self.dense = tf.keras.layers.Dense(
            embedding_dim,
            dtype=tf.float32)

        self.final_projection = tf.keras.layers.Dense(
            embedding_dim, dtype=tf.float32)

        self.num_negative_samples = 100
        self.spec_augment = MixStripes(
            dim=1,
            mix_width=10,
            stripes_num=200,
            mask_ratio=0.065,
            random_noise_mask=True,
            fixed_stripes_num=False)

    def call(
            self,
            inputs,
            attention_mask=None,
            training=False,
            add_loss=True):
        quantized = self.dense(inputs)

        mask = None
        if add_loss:
            inputs, mask = self.spec_augment(
                inputs, training=training, add_loss=add_loss)

            mask = mask[:, :, 0]

        inputs = self.transformer(inputs, training=training)
        inputs = self.final_projection(inputs)

        if add_loss:
            contrastive_loss, pos_sim, neg_sim = self.contrastive_loss(
                mask, inputs, quantized)
            self.add_loss(contrastive_loss)
            self.add_metric(contrastive_loss, name="context")
            self.add_metric(pos_sim, name="pos_sim")
            self.add_metric(neg_sim, name="neg_sim")

        return inputs, quantized, mask

    def contrastive_loss(self, mask, context, quantized):
        loss = []
        pos_sims = []
        neg_sims = []
        for batch_index in range(self.batch_size):
            mask_indices = tf.where(mask[batch_index])
            c_t = tf.gather_nd(context[batch_index], mask_indices)
            q_t = tf.gather_nd(quantized[batch_index], mask_indices)

            pos_similarity = cosine_similarity(c_t, q_t)
            pos_sims.append(tf.reduce_mean(pos_similarity))

            neg_similarity = cosine_similarity_matmul(c_t, q_t)
            neg_sims.append(tf.reduce_mean(neg_similarity))

            neg_similarity_mask = tf.linalg.diag(
                tf.ones(tf.shape(neg_similarity)[-1]))
            neg_similarity = tf.exp(
                neg_similarity / self.temperature) * (1 - neg_similarity_mask)

            random_indices = tf.random.shuffle(
                tf.range(tf.shape(neg_similarity)[-1]))[:self.num_negative_samples]
            neg_similarity = tf.gather(neg_similarity, random_indices, axis=1)

            neg_similarity = tf.concat(
                [neg_similarity, pos_similarity[:, tf.newaxis]], axis=-1)

            numerator = tf.exp(pos_similarity / self.temperature)
            denominator = tf.reduce_sum(neg_similarity, axis=-1)
            loss_m = -tf.math.log(numerator / denominator)
            loss.append(tf.reduce_mean(loss_m))

        return tf.reduce_mean(loss), tf.reduce_mean(
            pos_sims), tf.reduce_mean(neg_sims)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_heads": self.num_heads,
                "num_layers": self.num_layers,
                "intermediate_size": self.intermediate_size,
                "dropout": self.dropout,
                "layer_norm_eps": self.layer_norm_eps,
                "is_gelu_approx": self.is_gelu_approx,
                "norm_first": self.norm_first,
                "temperature": self.temperature
            }
        )
        return config
