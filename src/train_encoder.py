import cv2
import tensorflow as tf
import numpy as np

from config import MusicEncoderConfig
from util import DataGeneratorBatch, load_from_npz
from model import MusicEncoder
from gradient_accumulator import GradientAccumulateModel
from preprocess import fft


def allocate_gpu_memory(gpu_number=0):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')

    if len(physical_devices) > 0:
        try:
            print("Found {} GPU(s)".format(len(physical_devices)))
            tf.config.experimental.set_memory_growth(
                physical_devices[gpu_number], True)
            print("#{} GPU memory is allocated".format(gpu_number))
        except RuntimeError as e:
            print(e)
    else:
        print("Not enough GPU hardware devices available")


def lr_warmup_cosine_decay(global_step,
                           warmup_steps,
                           hold=0,
                           total_steps=0,
                           start_lr=0.0,
                           target_lr=1e-3):
    learning_rate = 0.5 * target_lr * \
        (1 + np.cos(np.pi * (global_step - warmup_steps - hold) /
         float(total_steps - warmup_steps - hold)))
    warmup_lr = target_lr * (global_step / warmup_steps)
    if hold > 0:
        learning_rate = np.where(global_step > warmup_steps + hold,
                                 learning_rate, target_lr)
    learning_rate = np.where(
        global_step < warmup_steps,
        warmup_lr,
        learning_rate)
    return learning_rate


class WarmupCosineDecay(tf.keras.callbacks.Callback):
    def __init__(
            self,
            total_steps=0,
            warmup_steps=0,
            start_lr=0.0,
            target_lr=1e-3,
            hold=0,
            global_steps=0):

        super(WarmupCosineDecay, self).__init__()
        self.start_lr = start_lr
        self.hold = hold
        self.total_steps = total_steps
        self.global_step = global_steps
        self.target_lr = target_lr
        self.warmup_steps = warmup_steps
        self.lrs = []

    def on_batch_end(self, batch, logs=None):
        self.global_step = self.global_step + 1
        lr = self.model.optimizer.lr.numpy()
        self.lrs.append(lr)

    def on_batch_begin(self, batch, logs=None):
        lr = lr_warmup_cosine_decay(
            global_step=self.global_step,
            total_steps=self.total_steps,
            warmup_steps=self.warmup_steps,
            start_lr=self.start_lr,
            target_lr=self.target_lr,
            hold=self.hold)
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)


class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, test_data, base_model):
        self.base_model = base_model
        self.x_data, self.y_data = test_data.__getitem__(0)
        self.x_data = self.x_data[0]
        for i in range(min(self.x_data.shape[0], 8)):
            cv2.imwrite("./img/original_l_{}.jpg".format(i),
                        cv2.flip(fft.minmax(self.y_data[i][:, :, 0].transpose(1, 0)) * 255, 0))

    def on_train_batch_begin(self, batch, logs=None):
        if batch % 250 == 0:
            context, _, feature = self.base_model(
                self.x_data, return_context=True)

            for i in range(min(context.shape[0], 8)):
                cv2.imwrite(
                    "./img/context_{}.jpg".format(i),
                    cv2.flip(
                        fft.minmax(
                            context[i, :, :].numpy().transpose(
                                1,
                                0).astype("float32")) *
                        255,
                        0))
            for i in range(min(feature.shape[0], 8)):
                cv2.imwrite(
                    "./img/quantized_{}.jpg".format(i),
                    cv2.flip(
                        fft.minmax(
                            feature[i, :, :].numpy().transpose(
                                1,
                                0).astype("float32")) *
                        255,
                        0))


if __name__ == "__main__":
    # GPUメモリ制限
    allocate_gpu_memory()

    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print('Compute dtype: %s' % policy.compute_dtype)
    print('Variable dtype: %s' % policy.variable_dtype)

    model_name = "music_encoder"

    epochs = 200
    batch_size = 2
    accum_steps = 2
    patch_len = 8192
    cache_size = 250
    initial_epoch = 0
    initial_value_loss = None
    log_dir = "./logs/music_encoder"

    x_train, x_test, dataset = load_from_npz()
    monitor = 'val_loss'
    model_input = tf.keras.layers.Input(shape=(None, 12 * 3 * 7, 3))
    config = MusicEncoderConfig()
    music_encoder = MusicEncoder(
        config=config,
        batch_size=batch_size,
        seq_len=patch_len)
    music_encoder.load_quantize_model(
        "./model/quantize_model/quantize_model.ckpt")
    output = music_encoder(model_input)
    model = tf.keras.Model(inputs=[model_input], outputs=output)

    # ジェネレーター作成
    train_gen = DataGeneratorBatch(
        files=x_train,
        dataset=dataset,
        batch_size=batch_size,
        patch_length=patch_len,
        initialepoch=initial_epoch,
        max_queue=2,
        cache_size=cache_size)

    test_gen = DataGeneratorBatch(
        files=x_test,
        dataset=dataset,
        batch_size=batch_size,
        patch_length=patch_len,
        validate_mode=True,
        cache_size=cache_size)

    plot_gen = DataGeneratorBatch(
        files=x_test[:10],
        dataset=dataset,
        batch_size=batch_size,
        patch_length=patch_len,
        validate_mode=True,
        cache_size=cache_size)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1)

    ckpt_callback_best = tf.keras.callbacks.ModelCheckpoint(
        filepath="./model/music_encoder/{}.ckpt".format(model_name),
        monitor=monitor,
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        initial_value_threshold=initial_value_loss)

    plot_callback = CustomCallback(plot_gen, music_encoder)
    lrs = WarmupCosineDecay(
        len(train_gen) *
        epochs,
        warmup_steps=len(train_gen) *
        epochs *
        0.1,
        target_lr=1e-5,
        global_steps=len(train_gen) *
        initial_epoch)
    callbacks = [
        plot_callback,
        ckpt_callback_best,
        tensorboard_callback,
        lrs
    ]

    model = GradientAccumulateModel(
        accum_steps=accum_steps,
        inputs=model.inputs,
        outputs=model.outputs,
        mixed_precision=True)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=1e-3,
        beta_1=0.9,
        beta_2=0.98)

    model.compile(optimizer=optimizer)
    model.summary()

    history = model.fit(
        x=train_gen,
        validation_data=test_gen,
        initial_epoch=initial_epoch,
        epochs=epochs,
        shuffle=False,
        callbacks=callbacks
    )
