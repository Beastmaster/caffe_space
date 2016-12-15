#!/usr/bin/env sh



CAFFE_BIN=/home/qinshuo/WorkPlace/caffe-dec/build/tools/caffe.bin
GPU_ID=0

CONFIG_DIR=/home/qinshuo/WorkPlace/seg_spine/config
#MODEL=../../exper/voc12_test/model/train_iter_800.caffemodel
#MODEL=../../exper/seg/model/train_iter_6000.caffemodel
MODEL=/media/D/SpineDataset/spine_seg/weight/_iter_1000.caffemodel
TEST_ITER=184

CMD="${CAFFE_BIN} test \
--model=${CONFIG_DIR}/deploy.prototxt \
--weights=${MODEL} \
--gpu=${GPU_ID}  \
--iterations=${TEST_ITER}"


#--gpu=${GPU_ID}  \

echo Running ${CMD} && ${CMD}
