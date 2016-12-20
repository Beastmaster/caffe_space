
import numpy as np
import matplotlib.pyplot as plt

import sys
# add self-defined python script path
sys.path.append('/home/qinshuo/WorkPlace/caffe_space/my_scripts/')
import numpy as np
caffe_root = '/home/qinshuo/WorkPlace/caffe/'
sys.path.insert(0,caffe_root+'python')
import caffe


# define layer model
solver_file = '/home/qinshuo/WorkPlace/caffe_space/seg_spine/config/seg_spine_solver.prototxt'

# setup GPU/CPU mode and device number
caffe.set_mode_gpu() 
caffe.set_device(0)

solver = caffe.SGDSolver(solver_file)

# check net layers
print("blobs {}\nparams {}".format(solver.net.blobs.keys(), solver.net.params.keys()))

#solver.net.forward()
solver.step(1)
print "First forward done.."


def net_blobs_overview(net):
    print "Net blobs:"
    for k, v in net.blobs.items():
        print '{: <15} \t weights: {:<20} biases :{} '.format(k, net.blobs[k].data.shape,net.blobs[k].data.shape)
def net_params_overview(net):
    print "Net params:"
    for k, v in net.params.items():
        print '{: <15} \t weights: {:<20} biases :{} '.format(k, v[0].data.shape,v[1].data.shape)
def check_layer_size(net,layer_name):
    if layer_name in net.params.keys():
        layer = net.params[layer_name]
        print "Layer \"{}\" size is: {},{}".format(layer_name,layer[0].data.shape,layer[1].data.shape)
    elif layer_name in net.blobs.keys():
        layer = net.blobs[layer_name]
        print "Layer \"{}\" size is: {}".format(layer_name,layer.data.shape)
    else:
        print "Layer \"{}\" does not exists".format(layer_name)

net_blobs_overview(solver.net)

solver.step(1)

def forward_steps(solver,step):
    for i in range(step):
        solver.step(1)
        p_step = 200
        if (i+1)%p_step == 0:
            train_loss = solver.net.blobs['loss'].data
            train_acc  = solver.net.blobs['accuracy'].data
            print "iteration: {}   loss: {}    accuracy: {}".format(str(i),str(train_loss),train_acc)
    import datetime
    print datetime.datetime.now().time()
    print "Iteration {} done!".format(step)


# In[8]:

forward_steps(solver,5000)
# save pre-trained model
solver.net.save('/home/qinshuo/WorkPlace/caffe_space/seg_spine/weight/seg_spine_net1.caffemodel')

forward_steps(solver,10000)
# save pre-trained model
solver.net.save('/home/qinshuo/WorkPlace/caffe_space/seg_spine/weight/seg_spine_net1.caffemodel')

forward_steps(solver,10000)
# save pre-trained model
solver.net.save('/home/qinshuo/WorkPlace/caffe_space/seg_spine/weight/seg_spine_net1.caffemodel')

forward_steps(solver,10000)
# save pre-trained model
solver.net.save('/home/qinshuo/WorkPlace/caffe_space/seg_spine/weight/seg_spine_net1.caffemodel')

