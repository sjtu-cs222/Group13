from __future__ import division
import os
from os import listdir
from os.path import isfile, join
import time
import tensorflow as tf
import numpy as np
import sys
import functools
from scipy.misc import imresize
from PIL import Image
from tensorflow import graph_util
import os
import shutil
import tensorflow as tf
from StyleTransferModel.model import Model


def load_image(path, shape=None, crop='center'):
    img = Image.open(path).convert("RGB")

    if isinstance(shape, (list, tuple)):
        # crop to obtain identical aspect ratio to shape
        width, height = img.size
        target_width, target_height = shape[0], shape[1]

        aspect_ratio = width / float(height)
        target_aspect = target_width / float(target_height)

        if aspect_ratio > target_aspect:  # if wider than wanted, crop the width
            new_width = int(height * target_aspect)
            if crop == 'right':
                img = img.crop((width - new_width, 0, width, height))
            elif crop == 'left':
                img = img.crop((0, 0, new_width, height))
            else:
                img = img.crop(((width - new_width) / 2, 0,
                                (width + new_width) / 2, height))
        else:  # else crop the height
            new_height = int(width / target_aspect)
            if crop == 'top':
                img = img.crop((0, 0, width, new_height))
            elif crop == 'bottom':
                img = img.crop((0, height - new_height, width, height))
            else:
                img = img.crop((0, (height - new_height) / 2,
                                width, (height + new_height) / 2))

        # resize to target now that we have the correct aspect ratio
        img = img.resize((target_width, target_height))

    elif isinstance(shape, (int, float)):
        width, height = img.size
        large = max(width, height)
        ratio = shape / float(large)
        width_n, height_n = ratio * width, ratio * height
        img = img.resize((int(width_n), int(height_n)))
    return img


def save_image(path, image):
    res = Image.fromarray(np.uint8(np.clip(image, 0, 255.0)))
    res.save(path)


def mkdir_if_not_exists(*args):
    for arg in args:
        if not os.path.exists(arg):
            os.makedirs(arg)


class hyper_param():
    def __init__(self, style_img, train_set_path, checkpoint_dir, output_dir, content_img):
        self.style = style_img
        self.batch_size = 1
        self.max_iter = 2e4
        self.learning_rate = 1e-3
        self.iter_print = 5e2
        self.checkpoint_iterations = 1e3
        self.train_path = train_set_path
        self.content_weight = 80
        self.style_weight = 1e2
        self.tv_weight = 2e2
        self.continue_train = False
        self.sample_path = content_img
        self.checkpoint_dir = checkpoint_dir
        self.serial = output_dir


def transfer( model, output_dir, content_img,interp = -1 ):
    with open(model, 'rb') as f:
        style_graph_def = tf.GraphDef()
        style_graph_def.ParseFromString(f.read())

    style_graph = tf.Graph()
    with style_graph.as_default():
        tf.import_graph_def(style_graph_def, name='')
    style_graph.finalize()

    sess_style = tf.Session(graph = style_graph)
    content = style_graph.get_tensor_by_name('content_input:0')
    shortcut = style_graph.get_tensor_by_name('shortcut:0')
    interp_opt = style_graph.get_tensor_by_name('interpolation_factor:0')
    style_output_tensor = style_graph.get_tensor_by_name('add_39:0')

    # TODO: remove here by deleting training batch size dependencies
    train_batch_size = content.get_shape().as_list()[0]

    img = np.array(load_image(content_img, 1024), dtype=np.float32)
    border = np.ceil(np.shape(img)[0]/20/4).astype(int) * 5
    container = [imresize(img, (np.shape(img)[0] + 2 * border, np.shape(img)[1] + 2 * border, 3))]
    container[0][border : np.shape(img)[0] + border, border : np.shape(img)[1] + border, :] = img
    container = np.repeat(container, train_batch_size, 0)

    mkdir_if_not_exists(output_dir)
    if interp < 0:
        shortcuts = [[False, False], [False, True], [True, False]]
        styles_res = []
        for sc in shortcuts:
            styles_res.append(sess_style.run(style_output_tensor, feed_dict={
                content: container, shortcut: sc, interp_opt: 0}))
        # 保存图片
        count = 1
        for style in styles_res:
            save_image(os.path.join(output_dir, "style_%d.jpg" % count), np.squeeze(
                style[0][border: np.shape(img)[0] + border, border: np.shape(img)[1] + border, :]))
            count+=1
    else :
        for i in range(interp):
            style_img = sess_style.run(
            style_output_tensor,
            feed_dict={
                content: container,
                shortcut: [True, True],
                interp_opt: i / interp * 2
            })
            save_image(
                os.path.join(
                    output_dir, "style_interp_{}_{}.jpg".format(i,interp)),
                np.squeeze(style_img[0][border : np.shape(img)[0] + border, border : np.shape(img)[1] + border, :])
                )

    sess_style.close()

def save_model(checkpoint_dir,output):
    #保存模型
    meta_graph = [meta for meta in os.listdir(
        checkpoint_dir) if '.meta' in meta]
    assert (len(meta_graph) > 0)

    sess = tf.Session()
    saver = tf.train.import_meta_graph(
        os.path.join(checkpoint_dir, meta_graph[0]))
    saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
    graph = tf.get_default_graph()

    input_graph_def = graph.as_graph_def()

    output_node_names = 'add_39'
    output_graph_def = graph_util.convert_variables_to_constants(
        sess, input_graph_def, output_node_names.split(","))

    with tf.gfile.GFile(output, "wb") as f:
        f.write(output_graph_def.SerializeToString())
    sess.close()


def model_train(train_param):
    

    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.4
    sess = tf.Session(config=config)
    train_model = Model(sess, train_param)

    style_image_basename = os.path.basename(train_param.style)
    style_image_basename = style_image_basename[:style_image_basename.find(
        ".")]

    print("[Checkpoint Directory: {}]".format(train_param.checkpoint_dir))
    print("[Transfer Output Directory: {}]".format(train_param.serial))
    mkdir_if_not_exists(train_param.serial, train_param.checkpoint_dir)
    print("[----------train start-----------]")
    if train_param.continue_train:
        train_model.finetune_model(train_param)
    else:
        train_model.train(train_param)


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == "__main__":
    
    
    '''
    模型训练  训练图像风格迁移模型
    保存模型  将ckpt内的临时模型打包并保存
    风格迁移  使用训练好的模型进行风格迁移（不进行install也可运行）
    '''
    
    '''
    ######################################
    # 模型训练
    style_pic = "./model/newstyle.jpg"
    train_set_dir = "./MSCOCO_train2k"
    checkpoint_dir = "./ckpt"
    output_dir = "./output"
    content_pic = "content.jpg"
    train_param = hyper_param(style_pic,train_set_dir,checkpoint_dir,output_dir,content_pic)
    model_train(train_param)
    ##################################################

    ######################################
    # 保存模型 
    checkpoint_dir = "./ckpt"
    model_dir = "./model"
    save_model(checkpoint_dir,model_dir)
    ######################################
    '''

    ######################################
    # 风格迁移
    model = "./model/newstyle.pb" 
    output_dir = "./output"
    content_pic = "content.jpg"
    output_num = -1 #控制输出图像张数， 连续笔画控制，默认输出三张，
    transfer(model,output_dir,content_pic,output_num)
    ######################################
