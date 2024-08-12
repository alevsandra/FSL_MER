import click
import librosa
import random
import matplotlib.pyplot as plt
from audiomentations import SpecCompose, SpecFrequencyMask
from audiomentations.core.transforms_interface import BaseSpectrogramTransform
from make_dataset import TrompaMer, PMEmo, DEAM
import numpy as np
import os
import csv

ROOT_DIR = os.path.split(os.environ['VIRTUAL_ENV'])[0]


class SpecTimeMask(BaseSpectrogramTransform):
    supports_multichannel = True

    def __init__(
            self,
            min_mask_fraction: float = 0.03,
            max_mask_fraction: float = 0.2,
            fill_mode: str = "constant",
            fill_constant: float = 0.0,
            p: float = 0.5,
    ):
        super().__init__(p)
        self.min_mask_fraction = min_mask_fraction
        self.max_mask_fraction = max_mask_fraction
        assert fill_mode in ("mean", "constant")
        self.fill_mode = fill_mode
        self.fill_constant = fill_constant

    def randomize_parameters(self, magnitude_spectrogram):
        super().randomize_parameters(magnitude_spectrogram)
        if self.parameters["should_apply"]:
            num_time_bins = magnitude_spectrogram.shape[1]
            min_frequencies_to_mask = int(
                round(self.min_mask_fraction * num_time_bins)
            )
            max_frequencies_to_mask = int(
                round(self.max_mask_fraction * num_time_bins)
            )
            num_frequencies_to_mask = random.randint(
                min_frequencies_to_mask, max_frequencies_to_mask
            )
            self.parameters["start_time_index"] = random.randint(
                0, num_time_bins - num_frequencies_to_mask
            )
            self.parameters["end_time_index"] = (
                    self.parameters["start_time_index"] + num_frequencies_to_mask
            )

    def apply(self, magnitude_spectrogram):
        if self.fill_mode == "mean":
            fill_value = np.mean(
                magnitude_spectrogram[:, self.parameters["start_time_index"]: self.parameters["end_time_index"]])
        else:
            # self.fill_mode == "constant"
            fill_value = self.fill_constant
        magnitude_spectrogram = magnitude_spectrogram.copy()
        magnitude_spectrogram[:, self.parameters["start_time_index"]: self.parameters["end_time_index"]] = fill_value
        return magnitude_spectrogram


def plot_spectrogram(spectrogram):
    fig, ax = plt.subplots()
    S_dB = librosa.amplitude_to_db(spectrogram, ref=np.max)
    img = librosa.display.specshow(S_dB, x_axis='time',
                                   y_axis='mel', ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel-frequency spectrogram')
    plt.show()


def get_indices(dataset, classes):
    return [index for c in classes for index in dataset.class_to_indices[c]]


def write_to_csv(file, data):
    csv_file = open(file, 'a', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(data)
    csv_file.close()


@click.command()
@click.option('--out_path', default='data/processed/augmentation', help='out path for augmented data')
@click.option('--out_path_annotations', default='data/processed', help='out path for annotations')
def main(out_path, out_path_annotations):
    # Q2 - 'tension', 'fear', 'anger'
    # Q4 - 'peace', 'transcendence', 'tenderness'
    classes_for_augmentation = ['tension', 'fear', 'anger', 'peace', 'transcendence', 'tenderness']
    classes_for_augmentation_pmemo = ['tension', 'transcendence', 'tenderness']
    trompa = TrompaMer(False, classes_for_augmentation, augmentation=True)
    pmemo = PMEmo(False, classes_for_augmentation_pmemo, augmentation=True)
    deam = DEAM(False, classes_for_augmentation, augmentation=True)

    # create one list of spectrograms
    items = [trompa[i] for i in get_indices(trompa, classes_for_augmentation)]
    items.extend([pmemo[i] for i in get_indices(pmemo, classes_for_augmentation_pmemo)])
    items.extend([deam[i] for i in get_indices(deam, classes_for_augmentation)])

    # create output path if not present
    full_path = os.path.join(ROOT_DIR, out_path)
    if not os.path.isdir(full_path):
        os.makedirs(full_path)

    # create annotations file
    annotation_file = os.path.join(ROOT_DIR, out_path_annotations, "augmentation_annotations.csv")
    write_to_csv(annotation_file, ['id', 'label'])

    # define transformations
    augment = SpecCompose([
        SpecFrequencyMask(p=0.5),
        SpecTimeMask(p=0.5)
    ])

    i = 4000
    for element in items:
        augmented = augment(element['audio'][:, :691])

        filename = os.path.join(full_path, str(i) + ".npy")
        np.save(filename, augmented)

        write_to_csv(annotation_file, [i, element['label']])

        i += 1


if __name__ == '__main__':
    main()
