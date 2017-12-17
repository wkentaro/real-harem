#!/bin/bash

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

set -x

cd $ROOT/face2face

if [ ! -e .anaconda3/bin/activate ]; then
  curl https://raw.githubusercontent.com/wkentaro/dotfiles/3c249b5c1a7ceffe369cf63d51b7f64a0c773321/local/bin/install_anaconda3.sh | bash -s .
fi
set +x && source .anaconda3/bin/activate && set -x
conda install dlib opencv -c conda-forge -y
pip install -I numpy  # FIXME: needed after opencv installation
pip install imutils Pillow matplotlib scikit-image

./install_shape_predictor.sh

python sample_face2face.py
