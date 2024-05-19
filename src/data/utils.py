import numpy as np
from typing import Dict
import torch
import gdown
import os
import zipfile
import torchaudio
import pandas as pd
import math
import urllib.request
import librosa

ROOT_DIR = os.path.split(os.environ['VIRTUAL_ENV'])[0]


def load_melspectrogram(path) -> Dict:
    full_path = os.path.join(ROOT_DIR, path)
    S = np.load(full_path)[:, :691]
    S = librosa.feature.melspectrogram(S=S, sr=44100, n_mels=32)
    y = torch.from_numpy(S)
    return {'audio': y}


def load_mtg_melspectrogram(path) -> Dict:
    y = torch.from_numpy(np.load(path)[:, :512])
    return {'audio': y}


def load_audio(audio_path, duration) -> Dict:
    waveform, sample_rate = torchaudio.load(audio_path)
    num_samples = int(duration * sample_rate)
    waveform = waveform[:, :num_samples]
    waveform_mono = torch.mean(waveform, dim=0).unsqueeze(0)
    return {'audio': waveform_mono}


def make_melspectrogram(audio_path) -> Dict:
    y, sr = librosa.load(audio_path)
    S = np.abs(librosa.stft(y))[:, :691]
    S = librosa.feature.melspectrogram(S=S, sr=sr, n_mels=32)
    y = torch.from_numpy(S)
    return {'audio': y}


def collate_list_of_dicts(input_set) -> Dict:
    dictionary = {}
    class_list_set = set()

    for item in input_set:
        # class_list_set.update(item['label'])  # in case of list in 'label
        class_list_set.add(item['label'])

    dictionary['classlist'] = list(class_list_set)
    dictionary['audio'] = torch.stack([item['audio'] for item in input_set])
    dictionary['target'] = torch.stack([item['target'] for item in input_set])
    return dictionary


def export_zip_file(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(ROOT_DIR + "/data/raw")


def download_dataset(url, dataset_name, file_name, export, google):
    full_path = ROOT_DIR + '/data/external/' + dataset_name
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    if google:
        gdown.download(url, full_path + "/" + file_name)
    else:
        urllib.request.urlretrieve(url, full_path + "/" + file_name)
    if export:
        export_zip_file(full_path + "/" + file_name)


def open_csv(file_name):
    with open(file_name) as file:
        content = file.read().splitlines()
        for row in content:
            print(row)
            break


def assign_quarter_label(arousal, valence, pmemo):
    if pmemo:
        middle = 0.5
    else:
        middle = 0
    if arousal > middle and valence > middle:
        return "Q1"
    elif arousal < middle and valence > middle:
        return "Q2"
    elif arousal < middle and valence < middle:
        return "Q3"
    elif arousal > middle and valence < middle:
        return "Q4"


def assign_label(arousal, valence, pmemo):  # arousal-x, valence-y
    if pmemo:
        middle = 0.5
    else:
        middle = 0
    if arousal >= middle and valence >= middle:
        if valence + middle > math.sqrt(3) * arousal:
            return "power"
        elif valence + middle < 1 / math.sqrt(3) * arousal:
            return "joy"
        return "surprise"
    elif arousal < middle and valence >= middle:
        if - valence + middle < math.sqrt(3) * arousal:
            return "tension"
        elif - valence + middle > 1 / math.sqrt(3) * arousal:
            return "fear"
        return "anger"
    elif arousal < middle and valence < middle:
        if abs(valence) + middle < abs(arousal):
            return "bitterness"
        else:
            return "sadness"
    elif arousal >= middle and valence < middle:
        if - valence + middle > math.sqrt(3) * arousal:
            return "peace"
        elif - valence + middle < 1 / math.sqrt(3) * arousal:
            return "transcendence"
        return "tenderness"


def assign_octant_label(arousal, valence):
    octant_labels = [(['O1', 'O2'], ['O4', 'O3']), (['O7', 'O8'], ['O5', 'O6'])]
    octant = octant_labels[valence < 0.5][arousal < 0.5][abs(valence) < abs(arousal)]
    return octant


def batch_device(dictionary, DEVICE):
    for key, value in dictionary.items():
        if torch.is_tensor(value):
            dictionary[key] = value.to(DEVICE)


if __name__ == '__main__':
    data = load_melspectrogram('D:/magisterka-dane/' + '00/7400.npy')
    print(data['audio'].shape)

    # PMEmo label assign
    df = pd.read_csv("../../data/raw/PMEmo2019/annotations/static_annotations.csv")
    for index, row in df.iterrows():
        df.at[index, 'quadrant'] = assign_quarter_label(row['Arousal(mean)'], row['Valence(mean)'], True)
        df.at[index, 'label'] = assign_label(row['Arousal(mean)'], row['Valence(mean)'], True)
    df.to_csv('../../data/processed/pmemo_labels.csv', sep='\t')

    # DEAM dataset label assign
    df = pd.read_csv(
        "../../data/external/DEAM_Annotations/annotations/annotations averaged per song/dynamic (per second annotations)/arousal.csv",
        index_col=0)
    df2 = pd.read_csv(
        "../../data/external/DEAM_Annotations/annotations/annotations averaged per song/dynamic (per second annotations)/valence.csv",
        index_col=0)
    df['arousal(mean)'] = df.mean(axis=1)
    df2['valence(mean)'] = df2.mean(axis=1)
    df['valence(mean)'] = df2['valence(mean)']
    for index, row in df.iterrows():
        df.at[index, 'quadrant'] = assign_quarter_label(row['arousal(mean)'], row['valence(mean)'], False)
        df.at[index, 'label'] = assign_label(row['arousal(mean)'], row['valence(mean)'], False)
    df[['valence(mean)', 'arousal(mean)', 'quadrant', 'label']].to_csv('../../data/processed/deam_mean_quarter.csv',
                                                                       sep='\t')

    # TROMPA-MER label assign
    df = pd.read_csv("../../data/external/TROMPA_MER/summary_anno.csv", index_col=0, sep='\t')
    for index, row in df.iterrows():
        df.at[index, 'label'] = assign_label(row['norm_energy'], row['norm_valence'], False)
    df[['norm_energy', 'norm_valence', 'quadrant', 'label']].to_csv('../../data/processed/trompa_mer_labels.csv',
                                                                    sep='\t')
