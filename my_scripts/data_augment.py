
'''
Rotate image 

Write to lmdb 

This function is useless

'''

from PIL import Image
import cPickle as pickle
import lmdb
import numpy as np
import os




class DataAugBase:
    def __init__(self):
        pass

    def _open_img(self,name,dtype=np.float32):
        try:
            img = Image.open(name,'r')
        except:
            print "file: {} does not exist".format(name)
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



class DataAugmentation(DataAugBase):
    '''
    Rotate, resize, crop
    2D image
    base tuple: {number: (data,label)}
                  string   pickled numpy array
    '''
    def __init__(self,name):
        self.bEnableAug = True
        self.db_name = name
        self.env = lmdb.open(self.db_name,writemap=True,map_size=1024*1024*1024*1024)
        self.txn = self.env.begin(write=True)

    def disable_augmentation(self):
        self.bEnableAug = False
    def enable_augmentation(self):
        self.bEnableAug = True

    def add_data(self,image,label):
        _data = self._open_img(image)
        _label = self._open_img(label,dtype = np.uint8)
        if self.bEnableAug:
            for i in range(0,360,20):
                _data_i = self._rotate(_data,i)
                _label_i = self._rotate(_label,i)
                _tuple = (_data_i,_label_i)
                ser = pickle.dumps(_tuple)
                idx = self.txn.stat()['entries']
                self.txn.put(str(idx),ser)
                self.txn.commit()
                self.txn = self.env.begin(write=True)
        else:
            _tuple = (_data_i,_label_i)
            ser = pickle.dumps(_tuple)
            idx = self.txn.stat()['entries']
            self.txn.put(str(idx),ser)
            self.txn.commit()
            self.txn = self.env.begin(write=True)

    def flush(self):  # flush will close the database
        txn = self.env.begin(write=True)
        txn.commit()
        self.env.close()

    def pre_load(self,db_name):
        self.env = lmdb.open(self.db_name,map_size=1024*1024*1024*1024)
        txn = self.env.begin(write=True)
        temp_env = lmdb.open(db_name)
        with temp_env.begin(write=False) as temp_txn:
            for kk in temp_env.stat():
                txn.put(kk,temp_txn.get(kk))
                txn.commit()
                txn = self.env.begin(write=True)
    




class DataAugmentationFile(DataAugBase):
    '''
    Rotate image and save to file
    '''
    def __init__(self,ff,dir=''):
        self.init_file = ff
        idx = ff.rfind('.')
        self.new_file = ff[:idx]+'aug'+ff[idx:]


    def set_new_list(self,ff):
        self.new_file = ff
    
    def set_new_img_dir(self,dd):
        self.img_dir = dd
    def set_new_label_dir(self,dd):
        self.label_dir = 

    def update(self):
        with open(self.init_file,'r') as init_fp:
            for line in init_fp.read().splitlines():
                img = dir+line.split()[0]
                label = dir+line.split()[1]
                




if __name__=='__main__':
    file_list = '/home/qinshuo/WorkPlace/caffe_space/seg_spine/spine_list/train_list.txt'
    ddir='/media/D/SpineDataset/spine_seg'
    #db_file = '/home/qinshuo/WorkPlace/caffe_space/seg_spine/spine_list/train_data'
    db_file = '/media/D/test_train'

    with open(file_list,'r') as ff:
        indices = ff.read().splitlines()

    data_aug = DataAugmentation(db_file)
    for id,ffs in enumerate(indices):  
        image = ddir + ffs.split()[0]
        label = ddir + ffs.split()[1]
        data_aug.add_data(image,label)
        print 'Dumping {} th file ..'.format(id)

    data_aug.flush()
    print "LMDB {} Done!".format(db_file)