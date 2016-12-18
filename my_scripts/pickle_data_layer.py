import caffe

import numpy as np
from PIL import Image
import os
import cPickle as pickle

import random

class SliceDataLayer(caffe.Layer):
    """
    Load (input image, label image) pairs from a cPickle file
    pickle data with "data_augment.py"
    """

    def setup(self, bottom, top):
        """
        Setup data layer according to parameters:

        """
        # config
        params = eval(self.param_str)
        
        self.tran_data = params['train_data']
        self.mean = params['mean']
        self.batch = params['batch']
        self.phase = params['phase']
        self.crop_size = params['crop']
       
        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define two tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")
        
        # train list existence
        try:
            with open(self.tran_data,'r') as ff:
                data = pickle.load(ff)
        except ValueError: 
            raise Exception('Train pickle file not exists')        
        

        self.iter = 0
        self.random = True
        # randomization: seed and pick
        self.seed = 50
        if self.random:
            random.seed(self.seed)
            #self.idx = random.randint(0, len(self.indices)-1)
            self.indice_arr = np.arange(len(self.indices))
            np.random.shuffle(self.indice_arr)
        else:
            self.indice_arr = np.arange(len(self.indices))


    def reshape(self, bottom, top):
        # pick next input
        if self.random:
            self.idx = self.indice_arr[self.iter]
        else:
            self.idx =self.iter
        
        self.iter = self.iter+1
        if self.iter>=len(self.indices):
            self.iter=0

        # load image + label image pair
        self.data = self.load_image(self.idx)
        self.label = self.load_label(self.idx)
        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape( *self.data.shape)
        top[1].reshape( *self.label.shape)


    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data
        top[1].data[...] = self.label
        

    def backward(self, top, propagate_down, bottom):
        pass


    def load_image(self, idx):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        def load_per_img(idx):
            line = self.indices[idx].split() 
            img_name = self.data_dir+line[0]
            im = Image.open(img_name,'r')
            im = im.convert('L')
            im = im.resize((self.crop_size,self.crop_size),Image.NEAREST)
            in_ = np.array(im, dtype=np.float32)
            #in_ = np.expand_dims(in_,axis = 0)       
            in_ -= self.mean
            return in_
        
        img_batch = np.zeros([self.batch,1,self.crop_size,self.crop_size])
        for ii in range(self.batch):
            ii_temp = ii + idx
            if ii_temp<len(self.indices):
                pass
            else:
                ii_temp = ii_temp-len(self.indices)
            img_batch[ii,:,:,:] = load_per_img(self.indice_arr[ii_temp])

        return img_batch


    def load_label(self, idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        def load_per_lab(idx):
            line = self.indices[idx].split() 
            label_name = self.label_dir+line[1]
            im = Image.open(label_name)
            im = im.convert('L')
            im = im.resize((self.crop_size,self.crop_size),Image.NEAREST)
            label = np.array(im, dtype=np.uint8)
            # label = np.expand_dims(label,axis=2)
            return label

        label_batch = np.zeros([self.batch,1,self.crop_size,self.crop_size])
        for ii in range(self.batch):
            ii_temp = ii + idx
            if ii_temp<len(self.indices):
                pass
            else:
                ii_temp = ii_temp-len(self.indices)
            label_batch[ii,:,:,:] = load_per_lab(self.indice_arr[ii_temp])

        return label_batch
