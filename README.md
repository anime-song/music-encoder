# MusicEncoder

wav2vec2.0 と HuBERT を組み合わせた音楽のためのモデルです。

# データ作成

訓練データを作成するには Dataset/Source フォルダにオーディオデータを配置します。
オーディオデータはすべて mp3 である必要があります。

以下を実行すると Dataset/Processed フォルダに訓練データが作成されます。

```
python preprocess/convert_spectrogram.py
```

# 訓練

初めに量子化モデルを訓練します。

```
python train_quantize_model.py
```

その後、エンコーダーを訓練します。

```
python train_encoder.py
```

# ファインチューニング
ファインチューニングは以下のようにして行います。
ファインチューニング時はadd_lossをFalseにすることを忘れないようにしてください。

```python
model_input = tf.keras.layers.Input(shape=(None, 12 * 3 * 7, 3))
config = MusicEncoderConfig()
music_encoder = MusicEncoder(
    config=config,
    batch_size=batch_size,
    seq_len=patch_len)
music_encoder_out = music_encoder(model_input, add_loss=False, return_encoder=True)
music_encoder.quantize_model.trainable = False
model = tf.keras.Model(inputs=[model_input], outputs=music_encoder_out)
model.load_weights("./model/music_encoder/music_encoder.ckpt")
music_encoder.freeze_feature_extractor()

inputs = model(model_input)
inputs = tf.keras.layers.LSTM(512, return_sequences=True)(inputs)
inputs = tf.keras.layers.GlobalAveragePooling1D()(inputs)
genre = tf.keras.layers.Dense(10, activation="softmax", dtype=tf.float32)(inputs)
model = tf.keras.Model(inputs=model_input, outputs=genre)
```

# 学習済みモデルの重み

学習モデルの重みは後日公開予定です。
