# readme

## dependence

- python 3.6
- TensorFlow
- Numpy
- Pillow
- Scipy

## install

1. put  `vgg19.npy` into folder `./vg199`, you can get it [here](https://pan.baidu.com/s/1Nr8YAFS10YbxW9-u-haWlg)
2. get  MSCOCO train set, then put all the images into folder `./MSCOCO_train2k`, you can get it [here](https://pan.baidu.com/s/1aKosAB6L67RvfTbsHaSWVw)

## run

- just type `python main.py` in your command line
- details are in `__main__` of`main.py` 

## docs

- `ckpt` is checkpoint file use by tensorflow
- `model` contains trained packaged model
- `MSCOCO_train2k` stores the training set
- `ouput` contains the transfer result