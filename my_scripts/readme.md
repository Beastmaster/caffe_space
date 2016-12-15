# This folder hold all self-defined layers #



## load_data2lmdb.py ##
Load image data to lmdb, no use by now

## slice_data_layer.py ##
1. Module name: 
slice_data_layer
2. Layer name(on prototxt file): 
SliceDataLayer
3. Function: 
Load trainning data from a image list. 
** List format: ** 
image.jpg(png)  label.jpg(png)
image.jpg(png)  label.jpg(png)
    ...             ...
image.jpg(png)  label.jpg(png)

## acc_iou_layer.py ##
1. Module name:
acc_iou_layer
2. Layer name:
AccuracyIOULayer
3. Function: 
Compute semantic segmentation accuracy.


