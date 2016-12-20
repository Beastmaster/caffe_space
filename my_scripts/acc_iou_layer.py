import caffe

import numpy as np
from PIL import Image
import os

import random

class AccuracyIOULayer(caffe.Layer):
    """
    Calculate segmentation accuracy
    
    Feed softmax score and label
    """

    def setup(self, bottom, top):
        """
        Setup data layer according to parameters:

        """
        # config
        #params = eval(self.param_str)

        # two tops: data and label
        if len(top) > 1:
            raise Exception("No more than 1 top needed.")
        # data layers have no bottoms
        if len(bottom) != 2:
            raise Exception("Need to define two tops: data and label.")
        random.seed(100)

    def reshape(self, bottom, top):
        # reshape top a single number
        top[0].reshape( 2 )
        #top[0].reshape( 4 )

    def forward(self, bottom, top):
        # assign output
        bottom_data = bottom[0].data[0,0,:]
        seg_score_map = np.zeros(bottom_data.shape)
        seg_score_map[bottom_data>=0.5] = 1
        label = bottom[1].data[0]

        sum_map = label + seg_score_map
        union = sum_map.copy()
        union[union>0] = 1

        interect = sum_map.copy()
        interect[interect <2 ] = 0

        pos1 = float(np.count_nonzero(union))/float(np.count_nonzero(label))
        pos2 = float(np.count_nonzero(interect))/float(np.count_nonzero(sum_map))

        top[0].data[...] = [pos1,pos2]

    def backward(self, top, propagate_down, bottom):
        pass

