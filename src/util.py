import time
import random
import threading
import copy
import zipfile
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
import h5py


def get_shape_from_npz(npz_filename, array_key):
    with zipfile.ZipFile(npz_filename, 'r') as z:
        with z.open(array_key + '.npy') as f:
            # ファイルの先頭部分から必要な情報を取得
            major, minor = np.lib.format.read_magic(f)
            header = np.lib.format._read_array_header(
                f, version=(major, minor))
            shape = header[0]
    return shape


def load_from_npz(directory="./Dataset/Processed"):
    h5 = h5py.File(directory + "/dataset.hdf5", mode="r")
    group = h5.require_group("musics")

    datasetList = [
        name for name in group if isinstance(
            group[name], h5py.Dataset)]

    dataset = group

    train_files, test_files = train_test_split(
        datasetList, test_size=0.1, random_state=42)

    return train_files, test_files, dataset


def load_data(self):
    while not self.is_epoch_end:
        should_added_queue = len(self.data_cache_queue) < self.max_queue
        while should_added_queue:
            self._load_cache(self.file_index)
            should_added_queue = len(self.data_cache_queue) < self.max_queue

        time.sleep(0.1)


class DataLoader:
    def __init__(
            self,
            files,
            dataset,
            validate_mode,
            seq_len,
            max_queue=1,
            cache_size=100):
        self.dataset = dataset
        self.files_index = files.index
        self.files = sorted(set(copy.deepcopy(files)), key=self.files_index)
        self.file_index = 0
        self.data_cache = {}
        self.data_cache_queue = []

        self.validate_mode = validate_mode
        self.cache_size = cache_size
        self.max_queue = max_queue
        self.is_epoch_end = False
        self.seq_len = seq_len
        self.start()

    def _load_cache(self, start_idx):
        cache = {}
        for i in range(self.cache_size):
            idx = start_idx + i
            if idx >= len(self.files):
                idx = random.randint(0, len(self.files) - 2)

            data = self.dataset[self.files[idx]]

            n_frames = data.shape[1]
            if n_frames <= self.seq_len:
                start = 0
            else:
                start = np.random.randint(0, n_frames - self.seq_len)

            spect = (data[:, start: start + self.seq_len] /
                     np.iinfo(np.uint16).max).astype("float32")

            cache[self.files[idx]] = [spect, 0]

        self.data_cache_queue.append(cache)
        self.file_index += self.cache_size

    def on_epoch_end(self):
        self.is_epoch_end = True
        self.load_segment_next.join()

        self.is_epoch_end = False
        if not self.validate_mode:
            self.files = random.sample(self.files, len(self.files))

        self.file_index = 0
        self.data_cache.clear()
        self.data_cache_queue.clear()

        self.start()

    def start(self):
        self.load_segment_next = threading.Thread(
            target=load_data, args=(self,))
        self.load_segment_next.start()

    def select_data(self):
        while len(self.data_cache) <= 0:
            if len(self.data_cache_queue) >= 1:
                self.data_cache = self.data_cache_queue.pop(0)
                break

            time.sleep(0.1)

        file_name, data = random.choice(list(self.data_cache.items()))
        spectrogram_data = data[0]

        del self.data_cache[file_name]
        return spectrogram_data

    def __len__(self):
        return len(self.files)


class DataGeneratorBatch(keras.utils.Sequence):
    def __init__(self,
                 files,
                 dataset,
                 batch_size=32,
                 patch_length=128,
                 initialepoch=0,
                 validate_mode=False,
                 max_queue=1,
                 cache_size=500):

        print("files size:{}".format(len(files)))
        self.dataloader = DataLoader(
            files,
            dataset,
            validate_mode,
            seq_len=patch_length,
            max_queue=max_queue,
            cache_size=cache_size)

        self.batch_size = batch_size
        self.patch_length = patch_length

        total_seq_length = 0
        for file in files:
            length = dataset[file].shape[1]
            total_seq_length += (length // self.patch_length) * \
                self.patch_length

        self.batch_len = int(
            (total_seq_length // self.patch_length // self.batch_size)) + 1

        # データ読み込み
        self.epoch = initialepoch

    def on_epoch_end(self):
        self.dataloader.on_epoch_end()
        self.epoch += 1

    def __getitem__(self, index):
        X = np.full(
            (self.batch_size,
             self.patch_length,
             252,
             3),
            0,
            dtype="float32")

        for batch in range(self.batch_size):
            data = self.dataloader.select_data()
            X[batch, :data.shape[1], :, 0] = data[0]
            X[batch, :data.shape[1], :, 1] = data[1]
            X[batch, :data.shape[1], :, 2] = data[2]

        return [X[:, :, :, :]], X[:, :, :, :]

    def __len__(self):
        return self.batch_len
