# Copyright 2020 UMONS-Numediart-USHERBROOKE-Necotis.
#
# MAFNet of University of Mons and University of Sherbrooke – Mathilde Brousmiche is free software : you can redistribute it 
# and/or modify it under the terms of the Lesser GNU General Public License as published by the Free Software Foundation, 
# either version 3 of the License, or any later version. This program is distributed in the hope that it will be useful, 
# but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  
# See the Lesser GNU General Public License for more details. 

# You should have received a copy of the Lesser GNU General Public License along with this program.  If not, see <http://www.gnu.org/licenses/>.
# Each use of this software must be attributed to University of MONS – Numédiart Institute and to University of SHERBROOKE - Necotis Lab (Mathilde Brousmiche).
# This software was further extended by Abdul Majid.

import tensorflow as tf
import os
import sklearn
import pickle
import numpy as np
import argparse
import model as model

parser = argparse.ArgumentParser(description='AVE')

# Data specifications
parser.add_argument('--data_path', type=str, default="pickle_files",
                    help='data path')

parser.add_argument('--n_epoch', type=int, default=300,
                    help='number of epoch')
parser.add_argument('--batch_size', type=int, default=4,
                    help='number of batch size')
parser.add_argument('--learning_rate', type=float, default=3E-6,
                    help='number of batch size')
parser.add_argument('--patience', type=int, default=50,
                    help='patience for early stopping')

parser.add_argument('--x_image_shape', type=int, default=1920, #remeber change in mafnet img also
                    help='image feature size')
# R50 - 2048
# D121 - 1024
# D201 - 1920
# INCV2 - 1536
# Effnet - 1280
# VGG - 512
parser.add_argument('--x_sound_shape', type=int, default=1024,
                    help='sound feature size')
parser.add_argument('--n_classes', type=int, default=2,
                    help='number of classes')
parser.add_argument('--n_hidden', type=int, default=512,
                    help='number of hidden neurons')

parser.add_argument('--prob_img', type=float, default=0.,
                    help='rate for updating weights from image pathway')
parser.add_argument('--prob_all', type=float, default=.9,
                    help='rate for updating weights from both pathways')

parser.add_argument('--train', action='store_true', default=False,
                    help='train a new model')

parser.add_argument('--pick', default="non_coswara")

args = parser.parse_args()


def load_data(path, dset, shuffle=True):
    print('Load data ' + dset + 'set')
    pkl_file = open(os.path.join(path, dset+'Spex_Set_visual_'+args.pick+'.p'), 'rb')
    visual = pickle.load(pkl_file)
    visual = np.asarray(visual)

    pkl_file = open(os.path.join(path, dset + 'Spex_Set_audio_'+args.pick+'.p'), 'rb')
    audio = pickle.load(pkl_file)
    audio = np.asarray(audio)

    pkl_file = open(os.path.join(path, dset + 'Spex_Set_target_'+args.pick+'.p'), 'rb')
    target = pickle.load(pkl_file)
    target = np.asarray(target)
    if shuffle:
        visual, audio, target = sklearn.utils.shuffle(visual, audio, target)

    return visual, audio, target


def train_model(train_data, val_data):
    print('Training ...')

    # Data Generator
    tf.compat.v1.disable_eager_execution()
    x_image = tf.compat.v1.placeholder(tf.float32, shape=(None, 1, args.x_image_shape), name='x_image')
    x_sound = tf.compat.v1.placeholder(tf.float32, shape=(None, 1, args.x_sound_shape), name='x_sound')

    y = tf.compat.v1.placeholder(tf.float32, shape=(None, args.n_classes), name='target')

    #for comaptibility
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.compat.v1.Session(config=config)

    # setup network
    MAFnet = model.MAFnet_tuned(x_image, x_sound, y, args.x_image_shape, args.x_sound_shape, args.n_hidden, args.n_classes)

    var_list_image = []
    var_list_sound = []
    for var in tf.compat.v1.trainable_variables():
        if 'image' in var.name:
            var_list_image.append(var)
        if 'sound' in var.name:
            var_list_sound.append(var)
        if (not 'image' in var.name) and not ('sound' in var.name):
            var_list_image.append(var)


    train_step = tf.compat.v1.train.AdamOptimizer(args.learning_rate).minimize(MAFnet.loss)
    train_step_image = tf.compat.v1.train.AdamOptimizer(args.learning_rate).minimize(MAFnet.loss, var_list=var_list_image)
    train_step_sound = tf.compat.v1.train.AdamOptimizer(args.learning_rate).minimize(MAFnet.loss, var_list=var_list_sound)

    saver = tf.compat.v1.train.Saver()
    sess.run(tf.compat.v1.global_variables_initializer())

    # Train initialization
    min_delta = 0.0001
    patience_cnt = 0
    best_acc_val = 0
    best_epoch = 0
    
    f=open('both_fusion_final'+args.pick,'w')
    f2=open('both_tpr'+args.pick,'w')
    f3=open('both_fpr'+args.pick,'w')
    
    N = len(train_data[0])
    for n in range(args.n_epoch):

        total_loss = 0
        train_data = sklearn.utils.shuffle(train_data[0], train_data[1], train_data[2])
        for i in range(int(N / args.batch_size)):
            random_choice = np.random.choice(['image', 'sound', 'all'], size=int(N / args.batch_size),
                                             p=[args.prob_img, 1 - args.prob_img - args.prob_all, args.prob_all])

            feed_dict = {x_image: train_data[0][i*args.batch_size:(i+1)*args.batch_size],
                         x_sound: train_data[1][i*args.batch_size:(i+1)*args.batch_size],
                         y: train_data[2][i*args.batch_size:(i+1)*args.batch_size],
                         MAFnet.isTraining: True}

            if random_choice[i] == 'image':
                _, l = sess.run([train_step_image, MAFnet.loss], feed_dict=feed_dict)
            elif random_choice[i] == 'sound':
                _, l = sess.run([train_step_sound, MAFnet.loss], feed_dict=feed_dict)
            elif random_choice[i] == 'all':
                _, l = sess.run([train_step, MAFnet.loss], feed_dict=feed_dict)

            total_loss += l

        acc_val,y_matrix,ypred,yactual = sess.run(MAFnet.acc, feed_dict={ x_image: val_data[0],
                                               x_sound: val_data[1],
                                               y: val_data[2],
                                               MAFnet.isTraining: False})

        # Early stopping
        if n > 0 and (acc_val - best_acc_val) > min_delta:
            patience_cnt = 0
            best_acc_val = acc_val
            best_epoch = n
            saver.save(sess, os.path.join('model', 'MAFnet'))
        else:
            patience_cnt += 1
        acc_val=100*acc_val
        print('>> Epoch [{}/{}] : Accuracy val {:.2f} Best epoch : {}'.format(n + 1, args.n_epoch,
                                                                              acc_val , best_epoch+1))
        s=f'{acc_val},'
        f.write(s)

        tn=y_matrix[0,0]
        fp=y_matrix[0,1]
        fn=y_matrix[1,0]
        tp=y_matrix[1,1]

        tpr=tp/(tp+fn)
        fpr=fp/(fp+tn)

        s=f'{tpr},'
        f2.write(s)

        s=f'{fpr},'
        f3.write(s)
        # print(a1)
        if patience_cnt > args.patience:
            print("Early stopping...")
            print("Best epoch : " + str(best_epoch+1))
            break
    f.close()
    f2.close()
    f3.close()
    return

def test_model(test_data):
    print('Testing ...')
    # Data Generator
    tf.compat.v1.disable_eager_execution()
    x_image = tf.compat.v1.placeholder(tf.float32, shape=(None, 1, args.x_image_shape), name='x_image')
    x_sound = tf.compat.v1.placeholder(tf.float32, shape=(None, 1,  args.x_sound_shape), name='x_sound')#5 was 1
    y = tf.compat.v1.placeholder(tf.float32, shape=(None, args.n_classes), name='target')

    # compatibility
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.compat.v1.Session(config=config)

    # setup network
    MAFnet = model.MAFnet_tuned(x_image, x_sound, y, args.x_image_shape, args.x_sound_shape, args.n_hidden, args.n_classes)

    f=open('ypred_both'+args.pick+'.txt','w')
    f2=open('yactual_both'+args.pick+'.txt','w')

    saver = tf.compat.v1.train.Saver()
    saver.restore(sess, os.path.join('model', 'MAFnet'))
    accuracy,ymat,ypred,yactual= sess.run(MAFnet.acc, feed_dict={x_image: test_data[0],
                                               x_sound: test_data[1],
                                               y: test_data[2],
                                               MAFnet.isTraining: False})
    ac=accuracy*100
    ac=round(ac,2)
    print('\nAccuracy : {}%'.format(ac))

    s=''
    l=[]
    for i in ypred:
        l.append(i[0])
    s=f'{l}'
    f.write(s)

    l=[]
    for i in yactual:
        l.append(i[0])
    s=f'{l}'
    f2.write(s)

    f.close()
    f2.close()
    return


if __name__=='__main__':
    if args.train == True:
        train_data = load_data(args.data_path, dset='train')
        val_data = load_data(args.data_path, dset='validation')
        train_model(train_data, val_data)
    else:
        test_data = load_data(args.data_path, dset='test', shuffle=False)
        test_model(test_data)
