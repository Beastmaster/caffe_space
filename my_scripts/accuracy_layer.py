'''
Accuracy layer
'''

import caffe

import numpy as np
from PIL import Image
import os

import random

class AccuracyLayer(caffe.Layer):
    """
    Calculate segmentation accuracy
    
    Feed softmax score and label
    """

    def setup(self, bottom, top):
        """
        Setup data layer according to parameters:

        """
        # config
        self.prob_threshold = 0.5
        self.accuracy_type = "IOU"
        params = eval(self.param_str)
        if "prob_threshold" in params:
            self.prob_threshold = params["prob_threshold"]
        if "accuracy_type" in params:
            self.accuracy_type  = params["accuracy_type"]

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
        score_map = np.greater(bottom_data,self.prob_threshold)
        label = np.greater(bottom[1].data[0],self.prob_threshold)

        union = np.logical_or(score_map,label)
        interact = np.logical_and(score_map,label)

        try:
            pos1 = float(np.count_nonzero(label))/float(np.count_nonzero(union))
            pos2 = float(np.count_nonzero(interect))/float(np.count_nonzero(label))
        except:
            pos1 = 0.0
            pos2 = 0.0

        top[0].data[...] = [pos1,pos2]

    def backward(self, top, propagate_down, bottom):
        pass

