import librosa
import random
import matplotlib.pyplot as plt
from audiomentations import SpecCompose, SpecFrequencyMask
from audiomentations.core.transforms_interface import BaseSpectrogramTransform
from make_dataset import TrompaMer
import numpy as np


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


if __name__ == '__main__':
    trompa = TrompaMer(False,
                       ['joy', 'power', 'surprise', 'anger', 'tension', 'fear', 'sadness', 'bitterness', 'peace',
                        'tenderness', 'transcendence'], augmentation=True)
    input_spec = trompa[0]['audio']

    augment = SpecCompose([
        SpecFrequencyMask(p=0),
        SpecTimeMask(p=1)
    ])

    augmented = augment(input_spec)

    plot_spectrogram(augmented)
