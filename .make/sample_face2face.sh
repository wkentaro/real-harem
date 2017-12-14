#!/bin/bash

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

set -x

cd $ROOT/face2face

./install_anaconda3.sh .
set +x && source .anaconda3/bin/activate && set -x
conda install dlib opencv -c menpo -y

pip install imutils numpy Pillow matplotlib scikit-image

./install_shape_predictor.sh

python sample_face2face.py
