



import glob
import os
import sys

def generate_list(train_dir,label_dir,file_name):
    '''
    Generate train data and label list
    split by space
    '''
    def ext_files(path):
        train_list = glob.glob(os.path.join(path,'*.jpg'))
        train_list.extend(glob.glob(os.path.join(path,'*.png')))
        train_list = sorted(train_list)
        return [os.path.join(path,v) for v in train_list]
    
    train_list = ext_files(train_dir)
    label_list = ext_files(label_dir)
    if len(train_list) != len(label_list):
        print "Warning: train list and label list have different size!"


    with open(file_name,'w') as ff:
        for tt,ll in zip(train_list,label_list):
            line = '{}  {}  \n'.format(tt,ll)
            ff.write(line)


if __name__ == '__main__':
    if len(sys.argv) >3:
        train_dir = sys.argv[1]
        label_dir = sys.argv[2]
        file_name = sys.argv[3]

    else:
        train_dir = '/media/D/GE_qinshuo/train/source'
        label_dir = '/media/D/GE_qinshuo/train/label'
        file_name = '/home/qinshuo/WorkPlace/caffe_space/extract_skin/data_list/head_train.txt'
    
    generate_list(train_dir,label_dir,file_name)
