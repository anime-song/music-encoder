import tensorflow as tf
from model_utils import ReGLU


def masked_softmax(logits, mask):
    # マスクを適用する前に非常に大きな負の値を追加します
    mask = tf.cast(mask, dtype=logits.dtype)
    logits += (mask * -1e4)
    return tf.nn.softmax(logits, axis=-1)


class DilatedMultiheadSelfAttentionWithRelativePositionalEmbedding(
        tf.keras.layers.Layer):
    def __init__(
            self,
            dmodel,
            num_heads,
            batch_size,
            seq_len,
            dropout=0,
            attn_len=5,
            layer_index=0,
            **kwargs):
        super(
            DilatedMultiheadSelfAttentionWithRelativePositionalEmbedding,
            self).__init__(
            **kwargs)
        self.attn_len = attn_len
        self.dmodel = dmodel
        self.num_heads = num_heads
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.head_dim = dmodel // num_heads
        assert self.head_dim * num_heads == dmodel, "embed_dim must be divisible by num_heads"

        self.query = tf.keras.layers.Dense(
            dmodel)
        self.key = tf.keras.layers.Dense(
            dmodel)
        self.value = tf.keras.layers.Dense(
            dmodel)
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.Er = self.add_weight(
            shape=(
                num_heads,
                self.head_dim,
                attn_len),
            initializer='random_normal',
            trainable=True,
            name="Er_{}".format(layer_index))

    def call(self, inputs, layer=0, training=False):
        query = inputs
        key = inputs
        value = inputs

        if training:
            batch = self.batch_size
            time = self.seq_len
        else:
            batch = tf.shape(query)[0]
            time = tf.shape(query)[1]
        d_model = tf.shape(query)[2]
        dtype = query.dtype

        q = tf.reshape(
            self.query(query), [
                batch, time, self.num_heads, 1, self.head_dim])
        k = tf.reshape(
            self.key(key), [
                batch, time, self.num_heads, 1, self.head_dim])
        v = tf.reshape(
            self.value(value), [
                batch, time, self.num_heads, 1, self.head_dim])

        q = tf.transpose(q, [0, 2, 1, 3, 4])
        k = tf.transpose(k, [0, 2, 1, 3, 4])
        v = tf.transpose(v, [0, 2, 1, 3, 4])

        k = tf.concat(
            (
                self.kv_roll(k[:, 0: 4], layer, padding_value=0, shift=0),
                self.kv_roll(k[:, 4: 5], layer, padding_value=0, shift=-2),
                self.kv_roll(k[:, 5: 6], layer, padding_value=0, shift=-1),
                self.kv_roll(k[:, 6: 7], layer, padding_value=0, shift=1),
                self.kv_roll(k[:, 7: 8], layer, padding_value=0, shift=2)
            ),
            axis=1
        )
        v = tf.concat(
            (
                self.kv_roll(v[:, 0: 4], layer, padding_value=0, shift=0),
                self.kv_roll(v[:, 4: 5], layer, padding_value=0, shift=-2),
                self.kv_roll(v[:, 5: 6], layer, padding_value=0, shift=-1),
                self.kv_roll(v[:, 6: 7], layer, padding_value=0, shift=1),
                self.kv_roll(v[:, 7: 8], layer, padding_value=0, shift=2)
            ),
            axis=1
        )

        Er_t = tf.expand_dims(self.Er, axis=1)
        Er_t = tf.expand_dims(Er_t, axis=0)

        qk = tf.matmul(q, tf.transpose(k, [0, 1, 2, 4, 3]))
        attn = (qk + tf.matmul(q, Er_t)) / \
            tf.cast(tf.math.sqrt(float(self.head_dim)), dtype=dtype)
        attn = masked_softmax(attn, qk == 0)

        out = tf.matmul(attn, v)
        out = tf.transpose(tf.squeeze(out, axis=-2), [0, 2, 1, 3])
        out = tf.reshape(out, [batch, time, d_model])
        out = self.dropout(out, training=training)

        return out, attn

    def kv_roll(self, tensor, layer, padding_value=0, shift=1):
        time = tf.shape(tensor)[2]
        paddings = tf.constant([[0, 0], [0, 0], [(
            2**layer) * (self.attn_len // 2), (2**layer) * (self.attn_len // 2)], [0, 0], [0, 0]])
        tensor = tf.pad(
            tensor,
            paddings,
            "CONSTANT",
            constant_values=padding_value)
        tensor_list = [tf.roll(tensor, shift=-i * (2**layer), axis=2)
                       for i in range(shift, self.attn_len + shift)]
        tensor = tf.concat(tensor_list, axis=-2)
        return tensor[:, :, :time, :, :]


class DilatedTransformerLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        hidden_size,
        num_heads,
        intermediate_size,
        batch_size,
        seq_len,
        attention_window_size=5,
        layer_norm_eps=1e-5,
        is_gelu_approx=False,
        dropout=0.1,
        norm_first=True,
        layer_index=0,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.layer_norm_eps = layer_norm_eps
        self.is_gelu_approx = is_gelu_approx
        self.dropout = dropout
        self.norm_first = norm_first
        self.attention_window_size = attention_window_size

        self.attention_layer = DilatedMultiheadSelfAttentionWithRelativePositionalEmbedding(
            self.hidden_size,
            num_heads=self.num_heads,
            batch_size=self.batch_size,
            seq_len=self.seq_len,
            dropout=self.dropout,
            attn_len=attention_window_size,
            layer_index=layer_index)

        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)

        self.layer_norm = tf.keras.layers.LayerNormalization(
            epsilon=layer_norm_eps)

        self.intermediate = tf.keras.layers.Dense(
            intermediate_size)
        self.attention_output = tf.keras.layers.Dense(
            hidden_size)

        self.final_layer_norm = tf.keras.layers.LayerNormalization(
            epsilon=layer_norm_eps)

        self.activation = ReGLU()
        # self.activation = tf.keras.activations.relu

    def call(self, inputs, training=False, layer=0):
        # Attention
        residual = inputs
        if self.norm_first:
            inputs = self.layer_norm(inputs)
        inputs, scores = self.attention_layer(
            inputs,
            training=training,
            layer=layer)
        inputs = self.dropout1(inputs, training=training)
        inputs = inputs + residual
        if not self.norm_first:
            inputs = self.layer_norm(inputs)

        # FFN
        residual = inputs
        if self.norm_first:
            inputs = self.final_layer_norm(inputs)
        inputs = self.intermediate(inputs)
        inputs = self.activation(inputs)
        inputs = self.dropout2(
            self.attention_output(inputs),
            training=training)
        inputs = inputs + residual
        if not self.norm_first:
            inputs = self.final_layer_norm(inputs)

        return inputs, scores

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_heads": self.num_heads,
                "intermediate_size": self.intermediate_size,
                "layer_norm_eps": self.layer_norm_eps,
                "is_gelu_approx": self.is_gelu_approx,
                "dropout": self.dropout,
                "norm_first": self.norm_first,
                "attention_window_size": self.attention_window_size,
            }
        )

        return config


class DilatedTransformer(tf.keras.layers.Layer):
    def __init__(
        self,
        hidden_size,
        num_heads,
        intermediate_size,
        num_layers,
        batch_size,
        seq_len,
        layer_norm_eps=1e-5,
        is_gelu_approx=False,
        dropout=0.1,
        norm_first=True,
        attention_window_size=5,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.layer_norm_eps = layer_norm_eps
        self.is_gelu_approx = is_gelu_approx
        self.dropout = dropout
        self.norm_first = norm_first
        self.num_layers = num_layers
        self.attention_window_size = attention_window_size
        self.batch_size = batch_size
        self.seq_len = seq_len

        self.layers = [
            DilatedTransformerLayer(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                intermediate_size=self.intermediate_size,
                layer_norm_eps=self.layer_norm_eps,
                is_gelu_approx=self.is_gelu_approx,
                dropout=self.dropout,
                norm_first=self.norm_first,
                attention_window_size=attention_window_size,
                batch_size=batch_size,
                seq_len=seq_len,
                layer_index=i) for i in range(
                self.num_layers)]

    def call(self, inputs, training=False):
        for i, layer in enumerate(self.layers):
            inputs, _ = layer(inputs, layer=i, training=training)
        return inputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_heads": self.num_heads,
                "intermediate_size": self.intermediate_size,
                "num_layers": self.num_layers,
                "layer_norm_eps": self.layer_norm_eps,
                "is_gelu_approx": self.is_gelu_approx,
                "dropout": self.dropout,
                "norm_first": self.norm_first,
                "attention_window_size": self.attention_window_size
            }
        )

        return config
