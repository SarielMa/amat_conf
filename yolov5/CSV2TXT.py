


import os, sys, random, shutil
import xml.etree.ElementTree as ET
from glob import glob
import pandas as pd
from shutil import copyfile
import pandas as pd
from sklearn import preprocessing, model_selection
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np

img_width = 640
img_height = 480

def width(df):
  return int(df.xmax - df.xmin)
def height(df):
  return int(df.ymax - df.ymin)
def x_center(df):
  return int(df.xmin + (df.width/2))
def y_center(df):
  return int(df.ymin + (df.height/2))
def w_norm(df):
  return df/img_width
def h_norm(df):
  return df/img_height

df = pd.read_csv('content/blood_cell_detection.csv')

le = preprocessing.LabelEncoder()

le.fit(df['cell_type'])
print(le.classes_)
labels = le.transform(df['cell_type'])
df['labels'] = labels

df['width'] = df.apply(width, axis=1)
df['height'] = df.apply(height, axis=1)

df['x_center'] = df.apply(x_center, axis=1)
df['y_center'] = df.apply(y_center, axis=1)

df['x_center_norm'] = df['x_center'].apply(w_norm)
df['width_norm'] = df['width'].apply(w_norm)

df['y_center_norm'] = df['y_center'].apply(h_norm)
df['height_norm'] = df['height'].apply(h_norm)

#df.head(30)

df_train, df_valid = model_selection.train_test_split(df, test_size=0.1, random_state=13, shuffle=True)
print(df_train.shape, df_valid.shape)

if not os.path.exists('content/bcc'):

    os.mkdir('content/bcc/')
    os.mkdir('content/bcc/images/')
    os.mkdir('content/bcc/images/train/')
    os.mkdir('content/bcc/images/valid/')
    
    os.mkdir('content/bcc/labels/')
    os.mkdir('content/bcc/labels/train/')
    os.mkdir('content/bcc/labels/valid/')

def segregate_data(df, img_path, label_path, train_img_path, train_label_path):
  filenames = []
  for filename in df.filename:
    filenames.append(filename)
  filenames = set(filenames)
  
  for filename in filenames:
    yolo_list = []

    for _,row in df[df.filename == filename].iterrows():
      yolo_list.append([row.labels, row.x_center_norm, row.y_center_norm, row.width_norm, row.height_norm])

    yolo_list = np.array(yolo_list)
    follows = str(row.prev_filename.split('.')[0])+".txt"
    
    txt_filename = os.path.join(train_label_path,follows)
    # Save the .img & .txt files to the corresponding train and validation folders
    np.savetxt(txt_filename, yolo_list, fmt=["%d", "%f", "%f", "%f", "%f"])
    shutil.copyfile(os.path.join(img_path,row.prev_filename), os.path.join(train_img_path,row.prev_filename))
 
## Apply function ## 
src_img_path = "BCCD/JPEGImages/"
src_label_path = "BCCD/Annotations/"

train_img_path = "content/bcc/images/train"
train_label_path = "content/bcc/labels/train"

valid_img_path = "content/bcc/images/valid"
valid_label_path = "content/bcc/labels/valid"

segregate_data(df_train, src_img_path, src_label_path, train_img_path, train_label_path)
segregate_data(df_valid, src_img_path, src_label_path, valid_img_path, valid_label_path)

print("No. of Training images", len(os.listdir('content/bcc/images/train')))
print("No. of Training labels", len(os.listdir('content/bcc/labels/train')))

print("No. of valid images", len(os.listdir('content/bcc/images/valid')))
print("No. of valid labels", len(os.listdir('content/bcc/labels/valid')))