#!/bin/bash

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

set -x
set -e

cd $ROOT/transgender

# install anaconda3
$(pwd)/install_anaconda3.sh .
set +x && source .anaconda3/bin/activate && set -x

if [ ! -e StarGAN ]; then
  git clone https://github.com/wkentaro/StarGAN.git -b real-harem
fi
cd $(pwd)/StarGAN

# install pytorch
conda install cuda80 pytorch torchvision -c soumith -y
conda install dlib opencv -c menpo -y
pip install scikit-image

$(pwd)/install_face2016.sh

$(pwd)/install_pretrained_model.sh

set +e  # FIXME: pytorch program crashes on exit
python $(pwd)/transgender_jsk.py

gnome-open $(pwd)/logs/transgender_jsk
