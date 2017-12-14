#!/bin/bash

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

set -x

cd $HERE

url='http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
file=shape_predictor_68_face_landmarks.dat
if [ ! -e $file ]; then
  wget $url
  bzip2 -dk shape_predictor_68_face_landmarks.dat.bz2
fi
