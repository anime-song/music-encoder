import tensorflow as tf

from config import MusicEncoderConfig
from encoder import MaskedEncoder
from feature_extractor import FeatureExtractorLayer
from model_utils import cosine_similarity_matmul
from vector_quantize_tf import ResidualVQ


class QuantizeModel(tf.keras.Model):
    def __init__(
            self,
            config: MusicEncoderConfig,
            batch_size,
            seq_len,
            **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.batch_size = batch_size
        self.seq_len = seq_len

        self.residual_vq = ResidualVQ(
            input_dim=252 * 3,
            codebook_size=config.codebook_size,
            embedding_dim=config.quantizer_embedding_dim,
            num_quantizers=config.num_quantizers,
            batch_size=batch_size,
            ema_decay=config.ema_decay,
            threshold_ema_dead_code=config.threshold_ema_dead_code,
            commitment_cost=config.commitment_cost,
            sample_codebook_temperature=config.sample_codebook_temperature,
            kmeans_init=config.kmeans_init,
            dtype=tf.float32
        )

    def call(
            self,
            inputs,
            training=False):
        inputs = tf.transpose(inputs, (0, 1, 3, 2))
        inputs = tf.reshape(inputs, (self.batch_size, -1, 252 * 3))

        quantized_inputs, all_quantized, encoding_indices = self.residual_vq(
            inputs, training=training)
        quantized_inputs = tf.keras.activations.relu(quantized_inputs)

        quantized_loss = tf.reduce_mean(
            tf.square(
                tf.cast(
                    inputs,
                    dtype=tf.float32) - quantized_inputs))
        self.add_loss(quantized_loss)
        self.add_metric(quantized_loss, name="quantized")

        return quantized_inputs, all_quantized, encoding_indices


class MusicEncoder(tf.keras.Model):
    def __init__(
            self,
            config: MusicEncoderConfig,
            batch_size,
            seq_len,
            **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.batch_size = batch_size
        self.seq_len = seq_len

        self.feature_extract_layers = [
            FeatureExtractorLayer(
                filter_sizes=config.filter_sizes,
                kernel_sizes=config.kernel_sizes,
                strides=config.strides,
                is_gelu_approx=config.is_gelu_approx,
                layer_id=i
            )
            for i in range(len(config.filter_sizes))
        ]

        encoded_seq_len = seq_len

        self.context_encoder = MaskedEncoder(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            intermediate_size=config.intermediate_size,
            batch_size=batch_size,
            patch_length=encoded_seq_len,
            embedding_dim=config.embedding_dim,
            dropout=config.dropout,
            layer_norm_eps=config.layer_norm_eps,
            is_gelu_approx=config.is_gelu_approx,
            norm_first=config.norm_first,
            temperature=config.temperature,
        )

        model_input = tf.keras.layers.Input(shape=(None, 12 * 3 * 7, 3))
        self.quantize_layer = QuantizeModel(
            config, batch_size=batch_size, seq_len=seq_len)
        model_out = self.quantize_layer(model_input)
        self.quantize_model = tf.keras.Model(
            inputs=[model_input], outputs=model_out)

        self.projection_layers = [
            tf.keras.layers.Dense(
                config.quantizer_embedding_dim,
                dtype=tf.float32) for i in range(
                config.num_mlm_target_quantizers)]

        self.coarse_adapter = tf.keras.layers.Dense(
            config.hidden_size, name="coarse_adapter")
        self.num_mlm_target_quantizers = config.num_mlm_target_quantizers

    def load_quantize_model(self, path):
        self.quantize_model.load_weights(path)
        self.quantize_model.trainable = False

    def call(
            self,
            inputs,
            training=False,
            return_context=False,
            add_loss=True,
            return_encoder=True):
        original_inputs = inputs
        inputs = tf.transpose(inputs, (0, 1, 3, 2))
        inputs = tf.reshape(inputs, (self.batch_size, -1, 252 * 3))

        _, all_quantized, encoding_indices = self.quantize_model(
            original_inputs, training=training)

        for feature_extractor_layer in self.feature_extract_layers:
            inputs = feature_extractor_layer(inputs, training=training)
        feature = inputs

        coarse_tokens = tf.concat(
            all_quantized[:self.num_mlm_target_quantizers], axis=-1)
        inputs = tf.concat(
            [tf.cast(coarse_tokens, dtype=inputs.dtype), inputs], axis=-1)
        inputs = self.coarse_adapter(inputs)

        inputs, quantized_context, mask = self.context_encoder(
            inputs, training=training, add_loss=add_loss)

        if return_encoder:
            return inputs

        embeddings = self.quantize_layer.residual_vq.get_embeddings()
        mlm_losses = []
        mlm_accuracies = []
        for i in range(len(self.projection_layers)):
            proj = self.projection_layers[i](inputs)[..., tf.newaxis, :]
            encoded = tf.transpose(embeddings[i])[tf.newaxis, tf.newaxis, ...]

            similarity = cosine_similarity_matmul(proj, encoded) / 0.1
            similarity = tf.squeeze(similarity, axis=-2)

            if add_loss:
                masked_similarity = tf.gather_nd(similarity, tf.where(mask))
                masked_labels = tf.gather_nd(
                    encoding_indices[i], tf.where(mask))
                loss = tf.keras.losses.SparseCategoricalCrossentropy(
                    from_logits=True)(masked_labels, masked_similarity)
                mlm_losses.append(loss)
                acc = tf.keras.metrics.sparse_categorical_accuracy(
                    masked_labels, masked_similarity)
                mlm_accuracies.append(acc)

        self.add_loss(tf.reduce_mean(mlm_losses))
        self.add_metric(
            tf.reduce_mean(mlm_losses),
            name="mlm_loss")
        self.add_metric(
            tf.reduce_mean(mlm_accuracies),
            name="mlm_acc"
        )

        if return_context:
            return inputs, quantized_context, feature

        return inputs

    def freeze_feature_extractor(self):
        for i in range(len(self.feature_extract_layers)):
            self.feature_extract_layers[i].trainable = False
