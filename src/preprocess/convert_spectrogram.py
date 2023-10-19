import librosa
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
import os

import fft
import h5py

DATA_SET_PATH = "./Dataset/Processed"

lock = multiprocessing.Lock()


def save_to_h5(filepath, music_name, y):
    with lock:  # ロックを取得
        with h5py.File(filepath, mode="a") as h5:
            group = h5.require_group("musics")
            group.create_dataset(
                name=music_name,
                data=y,
                dtype=y.dtype,
                shape=y.shape)


def exists_dataset(music_name):
    filepath = os.path.join(
        DATA_SET_PATH,
        "dataset") + ".hdf5"
    with lock:  # ロックを取得
        with h5py.File(filepath, mode="r") as h5:
            if f"/musics/{music_name}" in h5:
                return True
            else:
                return False


def create_dataset(files):
    for train_data in files:
        try:
            f = train_data

            music_name = f.split("\\")[-1].split(".mp3")[0]
            print(music_name)

            # 音声読み込み
            y, sr = librosa.load(f, sr=22050, mono=False)
            if len(y.shape) == 1:
                y = np.array([y, y])

            # スペクトル解析
            S = fft.cqt(
                y,
                sr=sr,
                n_bins=12 * 3 * 7,
                bins_per_octave=12 * 3,
                hop_length=256 + 128 + 64,
                Qfactor=25.0)

            S = (S * np.iinfo(np.uint16).max).astype("uint16")

            filepath = os.path.join(
                DATA_SET_PATH,
                "dataset") + ".hdf5"

            save_to_h5(filepath, music_name, S)

        except Exception as e:
            print(music_name, e)


if __name__ == '__main__':
    files = librosa.util.find_files("./Dataset/Source", ext=["mp3"])

    multi = True
    if multi:
        n_proc = 6
        N = int(np.ceil(len(files) / n_proc))
        y_split = [files[idx:idx + N] for idx in range(0, len(files), N)]

        Parallel(
            n_jobs=n_proc,
            backend="multiprocessing",
            verbose=1)([
                delayed(create_dataset)(
                    files=[f]) for f in files])
