#!/bin/bash

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

set -x
set -e

cd $ROOT/transgender

# install anaconda3
if [ ! -e .anaconda3/bin/activate ]; then
  curl https://raw.githubusercontent.com/wkentaro/dotfiles/3c249b5c1a7ceffe369cf63d51b7f64a0c773321/local/bin/install_anaconda3.sh | bash -s .
fi
set +x && source .anaconda3/bin/activate && set -x

if [ ! -e StarGAN ]; then
  git clone https://github.com/wkentaro/StarGAN.git -b real-harem
fi
cd $(pwd)/StarGAN

# install pytorch
conda install cuda80 pytorch torchvision -c soumith -y
conda install dlib opencv -c conda-forge -y
pip install -I numpy  # FIXME: needed after opencv installation
pip install scikit-image

$(pwd)/install_face2016.sh

$(pwd)/install_pretrained_model.sh

set +e  # FIXME: pytorch program crashes on exit
python $(pwd)/transgender_jsk.py

gnome-open $(pwd)/logs/transgender_jsk
