import click
import os
import librosa
import numpy as np
from utils import add_padding


def make_spectrogram(file):
    y, sr = librosa.load(file)
    S = np.abs(librosa.stft(y))
    parent_folder = os.path.dirname(file)
    out_dir = os.path.join(parent_folder, "spectrogram")
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    filename = os.path.basename(file).split(".")[0] + ".npy"
    np.save(os.path.join(out_dir, filename), S)


def make_spectrogram_DEAM(file):
    y, sr = librosa.load(file)
    S = np.abs(librosa.stft(y))[:, :691]
    S = librosa.feature.melspectrogram(S=S, sr=sr, n_mels=32)
    parent_folder = os.path.dirname(file)
    out_dir = os.path.join(parent_folder, "DEAM_spectrograms")
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    filename = os.path.basename(file).split(".")[0] + ".npy"
    np.save(os.path.join(out_dir, filename), S)


def make_spectrogram_DEAM_padding(file):
    y, sr = librosa.load(file)
    S = np.abs(librosa.stft(y))
    if S.shape[1] < 1000:
        S = add_padding(S, S.shape[0], 1000)
    elif S.shape[1] > 1000:
        S = S[:, :1000]
    S = librosa.feature.melspectrogram(S=S, sr=sr, n_mels=32)
    parent_folder = os.path.dirname(file)
    out_dir = os.path.join(parent_folder, "DEAM_spectrograms_padding")
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    filename = os.path.basename(file).split(".")[0] + ".npy"
    np.save(os.path.join(out_dir, filename), S)


@click.command()
@click.option('--path', default='D:/magisterka-dane-mp3/00', help='mp3 directory path')
def main(path):
    for file in os.listdir(path):
        if file.endswith(".mp3"):
            make_spectrogram_DEAM(os.path.join(path, file))
            make_spectrogram_DEAM_padding(os.path.join(path, file))


if __name__ == '__main__':
    main()
