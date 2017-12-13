#!/bin/bash

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd $ROOT

set -x
set -e

if [ ! -e StarGAN ]; then
  git clone https://github.com/wkentaro/StarGAN.git -b real-harem
fi
cd StarGAN

# install anaconda3
./install_anaconda3.sh .

set +x && source .anaconda3/bin/activate && set -x

# install pytorch
conda install cuda80 pytorch -c soumith

./install_face2016.sh

./install_pretrained_model.sh

set +e  # FIXME: pytorch program crashes on exit
python transgender_jsk.py

gnome-open logs/transgender_jsk
