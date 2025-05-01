import os
import json
import tensorflow as tf
from PIL import Image
import numpy as np
from tqdm import tqdm

def create_tf_example(image_path, annotations, category_index):
    with tf.io.gfile.GFile(image_path, 'rb') as fid:
        encoded_jpg = fid.read()
    
    image = Image.open(image_path)
    width, height = image.size
    
    filename = os.path.basename(image_path)
    image_format = b'jpeg'
    
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []
    
    for ann in annotations:
        bbox = ann['bbox']
        category_id = ann['category_id']
        
        # Convert COCO bbox format [x,y,width,height] to [ymin,xmin,ymax,xmax]
        xmin = bbox[0] / width
        ymin = bbox[1] / height
        xmax = (bbox[0] + bbox[2]) / width
        ymax = (bbox[1] + bbox[3]) / height
        
        xmins.append(xmin)
        xmaxs.append(xmax)
        ymins.append(ymin)
        ymaxs.append(ymax)
        classes_text.append(category_index[category_id].encode('utf8'))
        classes.append(category_id)
    
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename.encode('utf8')])),
        'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename.encode('utf8')])),
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_jpg])),
        'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_format])),
        'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmins)),
        'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmaxs)),
        'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymins)),
        'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymaxs)),
        'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_text)),
        'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=classes)),
    }))
    
    return tf_example

def convert_coco_to_tfrecord(coco_ann_file, image_dir, output_path, category_index):
    with open(coco_ann_file, 'r') as f:
        coco_data = json.load(f)
    
    # Create image_id to annotations mapping
    image_id_to_anns = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in image_id_to_anns:
            image_id_to_anns[image_id] = []
        image_id_to_anns[image_id].append(ann)
    
    # Create image_id to filename mapping
    image_id_to_filename = {}
    for img in coco_data['images']:
        image_id_to_filename[img['id']] = img['file_name']
    
    # Create TFRecord writer
    writer = tf.io.TFRecordWriter(output_path)
    
    # Convert each image and its annotations
    for image_id, annotations in tqdm(image_id_to_anns.items()):
        if image_id not in image_id_to_filename:
            continue
            
        image_path = os.path.join(image_dir, image_id_to_filename[image_id])
        if not os.path.exists(image_path):
            continue
            
        tf_example = create_tf_example(image_path, annotations, category_index)
        writer.write(tf_example.SerializeToString())
    
    writer.close()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco_ann_file', type=str, required=True)
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--category_index_file', type=str, required=True)
    args = parser.parse_args()
    
    # Load category index
    with open(args.category_index_file, 'r') as f:
        category_index = json.load(f)
    
    convert_coco_to_tfrecord(
        args.coco_ann_file,
        args.image_dir,
        args.output_path,
        category_index
    )

if __name__ == '__main__':
    main() 