import os
import tarfile
from tqdm import tqdm
import urllib.request


def main_download(out_path: str = "."):
    for i in tqdm(range(0, 100)):
        if i < 10:
            j = "0" + str(i)
        else:
            j = i
        filename = "autotagging_moodtheme_melspecs-" + str(j) + ".tar"
        url = "https://cdn.freesound.org/mtg-jamendo/autotagging_moodtheme/melspecs/" + filename
        if not os.path.isfile(filename):
            urllib.request.urlretrieve(url, filename)
            tar = tarfile.open(filename)
            tar.extractall(path=out_path)
            tar.close()
        try:
            os.remove(filename)
        except OSError:
            pass


if __name__ == '__main__':
    main_download('data/')
