#!/usr/bin/env sh

CAFFE_BIN=/home/qinshuo/WorkPlace/caffe/build/tools/caffe.bin
#CAFFE_BIN=/home/qinshuo/WorkPlace/caffe/build/tools/caffe.bin
GPU_ID=0

INIT_WEIGHT=/media/D/SpineDataset/spine_seg/weight/a_iter_5000.caffemodel
MODEL=/home/qinshuo/WorkPlace/seg_spine/config_py/solver.prototxt


CMD="${CAFFE_BIN} train \
    --gpu=${GPU_ID} 
    --weights=${INIT_WEIGHT}
    --solver=${MODEL} "


#--gpu=${GPU_ID}"
#    --weights=${INIT_WEIGHT}

echo Running ${CMD} && ${CMD}






