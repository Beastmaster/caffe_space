


import numpy as np
import matplotlib.pyplot as plt

import sys
# add self-defined python script path
sys.path.append('/home/qinshuo/WorkPlace/caffe_space/my_scripts/')
import numpy as np
caffe_root = '/home/qinshuo/WorkPlace/caffe/'
sys.path.insert(0,caffe_root+'python')
import caffe

# setup GPU/CPU mode and device number
caffe.set_mode_gpu() 
caffe.set_device(0)

net_def = '/home/qinshuo/WorkPlace/caffe_space/seg_spine/config/seg_spine_deploy.prototxt'
weight = '/home/qinshuo/WorkPlace/caffe_space/seg_spine/weight/seg_spine_net1.caffemodel'


# Load the fully convolutional network to transplant the parameters.
net = caffe.Net(net_def, weight,caffe.TEST)
#[(k, v.data.shape) for k, v in net.blobs.items()]
#[(k, v[0].data.shape,v[1].data.shape) for k, v in net.params.items()]


# show image
def show_image(*img_arrays):
    num = len(img_arrays)
    plt.figure()
    for i in range(num):
        plt.subplot(num/4 + 1,4,i+1)
        data = img_arrays[i]
        filt_min, filt_max = data.min(), data.max()
        try:
            plt.title("filter #{} output".format(i))
            plt.imshow(data, vmin=filt_min, vmax=filt_max,cmap="gray")
            plt.tight_layout()
            plt.axis('off')
        except:
            print "index {} cannot display".format(i)
            pass

# read a image, convert to gray image
def load_img(name,size=[512,512]):
    from PIL import Image
    test_img = Image.open(name)
    test_img = test_img.convert('L')
    test_img = test_img.resize(size,Image.NEAREST)
    return np.array(test_img, dtype=np.float32)
    


ff = '/media/D/SpineDataset/spine_seg/test/St11.jpg'

# expand dimension
test_img_array = np.expand_dims(load_img(ff),axis = 0) 
# forward this the net
out = net.forward_all(data=test_img_array)
print 'forward done'
data = net.blobs['data'].data
conv1_1 = net.blobs['deconv4_3'].data
output = out['prob']
output[output>=0.1]=1
show_image(output[0][1])


