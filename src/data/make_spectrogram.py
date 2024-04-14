import click
import os
import librosa
import numpy as np


def make_spectrogram(file):
    y, sr = librosa.load(file)
    S = np.abs(librosa.stft(y))
    parent_folder = os.path.dirname(file)
    out_dir = os.path.join(parent_folder, "spectrogram")
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    filename = os.path.basename(file).split(".")[0] + ".npy"
    np.save(os.path.join(out_dir, filename), S)


@click.command()
@click.option('--path', default='D:/magisterka-dane-mp3/00', help='mp3 directory path')
def main(path):
    for file in os.listdir(path):
        if file.endswith(".mp3"):
            make_spectrogram(os.path.join(path, file))


if __name__ == '__main__':
    main()
