

import numpy as np
from PIL import Image
import os
import cPickle as pickle














class DataAugmentation:
    '''
    Rotate, resize, crop

    2D image

    base tuple: (data,label)
    '''
    def __init__(self,name):
        self.data = []
        self.pkl_name = name
    
    def add_data(self,image,label):
        _data = self._open_img(image)
        _label = self._open_img(label,dtype = np.uint8)
        for i in range(0,360,20):
            _data_i = self._rotate(_data,i)
            _label_i = self._rotate(_label,i)
            _tuple = (_data_i,_label_i)
            self.data.append(_tuple)
    def get_data(self):
        return self.data


    def flush(self):
        with open(self.pkl_name,'wb') as output:
            pickle.dump(self.data,output)
            print "Pickling to {} done!".format(self.pkl_name)

    def pre_load(self,pkl):
        with open(self.pkl_name,'rb') as pkl:
            self.data = pickle.load(pkl)

    def _open_img(self,name,dtype=np.float32):
        img = Image.open(name,'r')
        img2 = np.array(img.rotate(45),dtype=dtype)
        return img2

    def _rotate(self,image,angle):
        '''
        angle is degree(360)
        '''
        im2 = Image.fromarray(image)
        return np.array(im2.rotate(angle),dtype=image.dtype)

    def _resize(self,image):
        pass


    def _crop(self,image):
        pass




