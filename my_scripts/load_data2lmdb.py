

import lmdb
import numpy as np
import sys
caffe_root = '/home/qinshuo/WorPlace/caffe/'
sys.path.insert(0,caffe_root+'python')
import caffe



class write_data2lmdb:
    def open_lmdb(self,name):
        db_name = name
        self.env = lmdb.open(db_name)
        self.txn = env.begin(write=True)

    def add(self,data,label):
        pass


    def flush(self):
        pass



