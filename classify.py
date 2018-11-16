#! /usr/bin/env python
# -*- coding: utf-8 -*-
from nets.inception_v3 import inceptionv3
from misc.utils import *
import tensorflow as tf
from os import listdir
from convert_to_csv import convert
import numpy as np
import argparse
import pandas as pd
import time
import os

def validate_arguments(args):
    nets = ['inceptionv3']
    if not(args.network in nets):
        exit (-1)
    if args.evaluate:
        if args.img_list is None or args.gt_labels is None:
            exit (-1)

def choose_net(network):    
    MAP = {
        'inceptionv3': inceptionv3,
    }
    
    if network == 'caffenet':
        size = 227
    elif network == 'inceptionv3':
        size = 299
    else:
        size = 224
        
    #placeholder to pass image
    input_image = tf.placeholder(shape=[None, size, size, 3],dtype='float32', name='input_image')

    return MAP[network](input_image), input_image

def evaluate(net, im_list, in_im, labels, net_name,batch_size=30):
    top_1 = 0
    top_5 = 0
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    img_list = open(im_list).readlines()
    gt_labels = open(labels, 'utf-8').readlines()
    t_s = time.time()
    isotropic,size = get_params(net_name)
    batch_im = np.zeros((batch_size, size,size,3))
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        img_loader = loader_func(net_name,isotropic,size,sess)
        for i in range(len(img_list)/batch_size):
            lim = min(batch_size,len(img_list)-i*batch_size)
            for j in range(lim):
                im = img_loader(img_list[i*batch_size+j].strip())
                batch_im[j] = np.copy(im)
            gt = np.array([int(gt_labels[i*batch_size+j].strip()) for j in range(lim)])
            softmax_scores = sess.run(net['prob'], feed_dict={in_im: batch_im})
            inds = np.argsort(softmax_scores, axis=1)[:,::-1][:,:5]
            top_1+= np.sum(inds[:,0] == gt)
            top_5 += np.sum([gt[i] in inds[i] for i in range(lim)])
    
def predict(net, im_path, in_im, net_name):
    synset = open('misc/ilsvrc_synsets.txt', encoding='windows-1256').readlines()
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    t_s = time.time()
    isotropic,size = get_params(net_name)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        img_loader = loader_func(net_name,isotropic,size,sess)
        im = img_loader(im_path.strip())
        im = np.reshape(im,[1,size,size,3])
        softmax_scores = sess.run(net['prob'], feed_dict={in_im: im})
        inds = np.argsort(softmax_scores[0])[::-1][:5]
        labels = []
        for i in inds:
            n = synset[i].split(' ')
            for i in range(1, len(n)):
                if n[i][-1] == ',':
                    labels.append(n[i][0:-1])
                elif "\n" in n[i]:
                    labels.append(n[i][0:-2])
                else: labels.append(n[i])
        return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', default='inceptionv3', help='The network eg. googlenet')
    parser.add_argument('--folder', default='misc/sample.jpg',  help='Path to input image')
    parser.add_argument('--evaluate', default=False,  help='Flag to evaluate over full validation set')
    parser.add_argument('--img_list',  help='Path to the validation image list')
    parser.add_argument('--csv',default='dataset.csv', help='Path to csv file')
    parser.add_argument('--gt_labels', help='Path to the ground truth validation labels')
    parser.add_argument('--batch_size', default=50,  help='Batch size for evaluation code')
    args = parser.parse_args()
    validate_arguments(args)
    net, inp_im  = choose_net(args.network)

    df = pd.read_csv(args.csv, encoding='windows-1256')
    header = []
    for i in df.columns:
        header.append(i)
    if "is_label" not in df.columns:
        header.append('is_label')
        df.to_csv(args.csv, columns=header,  index=False)


    #print(df)
    if args.evaluate:
        evaluate(net, args.img_list, inp_im, args.gt_labels, args.network,args.batch_size)
    else:
        for img_path in listdir(args.folder):
            # print (img_path)
            filename, file_extension = os.path.splitext(args.folder+img_path)
            if file_extension == '.jpg':
                full_path = args.folder + img_path
                result = predict(net, full_path, inp_im, args.network)
                # print(len(result))
                convert(result, args.csv, img_path)
            # elif file_extension == '.mp4':
            #     full_path = get_image(args.folder, img_path)
            #     result = predict(net, full_path, inp_im, args.network)
            #     convert(result, args.csv, img_path)


if __name__ == '__main__':
    main()
