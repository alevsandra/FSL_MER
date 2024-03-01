Getting started
===============

This is where you describe how to get set up on a clean install, including the
commands necessary to get the raw data (using the `sync_data_from_s3` command,
for example), and then how to make the cleaned, final data sets.

After cloning the repo, you'll need to install the requirements:

    pip install -r requirements.txt

Then you'll need to pull the submodules code:

    git submodule update --init --recursive

Download pytorch version, which is compatible with CUDA according to https://pytorch.org/get-started/locally/:

    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121