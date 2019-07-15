# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import hashlib
import io
import logging
import os

from lxml import etree
import PIL.Image
import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util


class Object:
    def __init__(self, xmin, ymin, xmax, ymax, label):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.label = label


class Image:
    def __init__(self, filename):
        self.filename = filename
        self.Objects = []
        self.length = 0

    def append(self, Object):
        self.Objects.append(Object)
        self.length += 1


flags = tf.app.flags
flags.DEFINE_string(
    'data_dir', '', 'Root directory to raw PASCAL VOC dataset.')
flags.DEFINE_string('set', 'train', 'Convert training set, validation set or '
                    'merged set.')
flags.DEFINE_string('annotations_file', 'Annotations',
                    '(Relative) path to annotations directory.')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_boolean('val', False, 'Valid or not')
FLAGS = flags.FLAGS

SETS = ['train', 'val', 'trainval', 'test']

VAL = FLAGS.val


def read_anno(filepath):
    Image_list = []
    with open(filepath) as fp:
        line = fp.readline()
        while(line):
            arr = line.split()
            pattern2 = re.compile("\Atrain_images/2")
            pattern20 = re.compile("\Atrain_images/20")
            # print(arr[0], end='')
            if(VAL and not pattern20.match(arr[0])):
                line = fp.readline()
                continue
            elif(not VAL and (not pattern2.match(arr[0]) or pattern20.match(arr[0]))):
                #print(line)
                line = fp.readline()
                continue

            IM = 0
            print(arr[0])
            for x in arr:
                if(IM == 0):
                    IM = Image(x.split('/')[1])
                else:
                    feats = x.split(',')
                    # print(feats)
                    # print(len(feats))
                    Ob = Object(feats[0], feats[1],
                                feats[2], feats[3], feats[4])
                    IM.append(Ob)
            Image_list.append(IM)
            line = fp.readline()
    return Image_list


def dict_to_tf_example(data, dataset_directory):
    img_path = data.filename
    full_path = os.path.join(dataset_directory, img_path)
    with tf.gfile.GFile(full_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()

    width = 400
    height = 600

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    poses = []
    if data.length != 0:
        for obj in data.Objects:
            xmin.append(float(obj.xmin) / width)
            ymin.append(float(obj.ymin) / height)
            xmax.append(float(obj.xmax) / width)
            ymax.append(float(obj.ymax) / height)
            classes.append(int(obj.label)+1)

    #----------------------------wed-----------------------------------#
    print(data.filename+" "+str(data.length))

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(
            data.filename.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(
            data.filename.encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/label': dataset_util.int64_list_feature(classes)}))
    return example


def main(_):
    if FLAGS.set not in SETS:
        raise ValueError('set must be in : {}'.format(SETS))

    data_dir = FLAGS.data_dir

    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

    examples_list = read_anno(FLAGS.annotations_file)
    print('start!')
    for (idx, example) in enumerate(examples_list):
        if idx % 10 == 0:
            logging.info('On image %d of %d', idx, len(examples_list))

        tf_example = dict_to_tf_example(example, FLAGS.data_dir)
        writer.write(tf_example.SerializeToString())
        print('On image %d of %d', idx, len(examples_list))

    writer.close()


if __name__ == '__main__':
    tf.app.run()
