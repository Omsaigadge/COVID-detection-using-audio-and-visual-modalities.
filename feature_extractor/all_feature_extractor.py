# Copyright 2023 Abdul Majid.
#
# This feature extractor of KLE Technological University is free software : you can redistribute it 
# and/or modify it under the terms of the Lesser GNU General Public License as published by the Free Software Foundation, 
# either version 3 of the License, or any later version. This program is distributed in the hope that it will be useful, 
# but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  
# See the Lesser GNU General Public License for more details. 

# You should have received a copy of the Lesser GNU General Public License along with this program.  If not, see <http://www.gnu.org/licenses/>.
# Each use of this software must be attributed to Abdul Majid.

from keras.applications.densenet import DenseNet121,DenseNet201, preprocess_input
import tensorflow as tf
import keras.utils as image
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import os
import argparse
import pickle

# Data specifications
parser.add_argument('--save_path', type=str, default="/pickle_files/",
                    help='save path')
parser.add_argument('--pick', default="non_coswara")

args = parser.parse_args()


covid={
    'negative':0,
    'positive':1
}

def DenseNet_extractor_img(train_files, test_files, validation_files):

    model = DenseNet201(
                    weights='imagenet',
                    include_top=False,
                    input_shape=(224, 224, 3))

    all_f=[train_files,test_files,validation_files]
    all_fs=['train_files','test_files','validation_files']

    split=""
    for f_type in all_fs:
        if f_type=='train_files':
            split='train'
            f_type=all_f[0]
        elif f_type=='test_files':
            split='test'
            f_type=all_f[1]
        elif f_type=='validation_files':
            split='validation'
            f_type=all_f[2]
        else:
            raise NameError
        img_feature = []
        target = []
        N=len(f_type)
        milestones = [1,15, 30, 45, 60, 75, 90, 100]
        for n, file in enumerate(f_type):
            img_name = file.split(',')[0]
            input_x = []
            img_path=img_name
            x = image.load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(x)
            x = preprocess_input(x)
            input_x.append(x)

            input_x = np.asarray(input_x)
            features = model.predict(input_x)
            features = np.mean(features, axis = (1,2))
            tmp = []
            tmp.append(np.mean(features[0:1,:], axis=0))
            img_feature.append(tmp)

            target.append(to_categorical(covid[file.split(',')[1]], num_classes=2))
            
            percentage_complete = (100.0 * (n+1) / N)
            while len(milestones) > 0 and percentage_complete >= milestones[0]:
                print(f"{milestones[0]} completed of "+split+" split")
                milestones = milestones[1:]

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        img_feature = np.asarray(img_feature)
        output = open(os.path.join(save_path, split + 'Spex_Set_visual_'+args.pick+'.p'), 'wb')
        pickle.dump(img_feature, output)

        target = np.asarray(target)
        output = open(os.path.join(save_path, split + 'Spex_Set_target_'+args.pick+'.p'), 'wb')
        pickle.dump(target, output)
        
        print(split+' vcf completed')
    return





def DenseNet_extractor_aud(train_files, test_files, validation_files):

    model = DenseNet121(
                    weights='imagenet',
                    include_top=False,
                    input_shape=(640, 480, 3))

    all_f=[train_files,test_files,validation_files]
    all_fs=['train_files','test_files','validation_files']

    split=""
    for f_type in all_fs:
        if f_type=='train_files':
            split='train'
            f_type=all_f[0]
        elif f_type=='test_files':
            split='test'
            f_type=all_f[1]
        elif f_type=='validation_files':
            split='validation'
            f_type=all_f[2]
        else:
            raise NameError
        img_feature = []
        target = []
        N=len(f_type)
        milestones = [1,15, 30, 45, 60, 75, 90, 100]
        for n, file in enumerate(f_type):
            img_name = file.split(',')[0]
            input_x = []
            img_path=img_name
            x = image.load_img(img_path, target_size=(640, 480))
            x = image.img_to_array(x)
            x = preprocess_input(x)
            input_x.append(x)

            input_x = np.asarray(input_x)
            features = model.predict(input_x)
            features = np.mean(features, axis = (1,2))
            tmp = []
            tmp.append(np.mean(features[0:1,:], axis=0))
            img_feature.append(tmp)

            target.append(to_categorical(covid[file.split(',')[1]], num_classes=2))
            
            percentage_complete = (100.0 * (n+1) / N)
            while len(milestones) > 0 and percentage_complete >= milestones[0]:
                print(f"{milestones[0]} completed of "+split+" split")
                milestones = milestones[1:]

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        img_feature = np.asarray(img_feature)
        output = open(os.path.join(save_path, split + 'Spex_Set_audio_'+args.pick+'.p'), 'wb')
        pickle.dump(img_feature, output)

        target = np.asarray(target)
        output = open(os.path.join(save_path, split + 'Spex_Set_target_'+args.pick+'.p'), 'wb')
        pickle.dump(target, output)
        
        print(split+' vcf completed')
    return




if __name__ == '__main__':
    save_path = args.save_path
    save_path=os.path.abspath(os.pardir)+save_path
    print(save_path)
    add_path=os.path.abspath(os.pardir)+'/dataset'
    print(add_path)

    f = open("audio_xray.txt", 'r')
    lines = f.read().split('\n')
    try:
        lines.remove('')
    except:
        lines=lines
    
    train_files, test_files  = train_test_split(lines, test_size=0.3, random_state=42)
    validation_files, test_files = train_test_split(test_files, test_size=0.3, random_state=42)
    print('train_files: ',len(train_files))
    print('test_files: ',len(test_files))
    print('validation_files: ',len(validation_files))
    # input()
    
    tf_aud=[]
    tf_img=[]
    for i in train_files:
        aud=add_path+i.split(',')[0]
        img=add_path+i.split(',')[2]
        helth=i.split(',')[1]
        aud=aud+','+helth
        img=img+','+helth
        tf_aud.append(aud)
        tf_img.append(img)
        
        
    tsf_aud=[]
    tsf_img=[]
    for i in test_files:
        aud=add_path+i.split(',')[0]
        img=add_path+i.split(',')[2]
        helth=i.split(',')[1]
        aud=aud+','+helth
        img=img+','+helth
        tsf_aud.append(aud)
        tsf_img.append(img)
        
    vf_aud=[]
    vf_img=[]
    for i in validation_files:
        aud=add_path+i.split(',')[0]
        img=add_path+i.split(',')[2]
        helth=i.split(',')[1]
        aud=aud+','+helth
        img=img+','+helth
        vf_aud.append(aud)
        vf_img.append(img)

    DenseNet_extractor_aud(tf_aud,tsf_aud,vf_aud)
    DenseNet_extractor_img(tf_img,tsf_img,vf_img)
