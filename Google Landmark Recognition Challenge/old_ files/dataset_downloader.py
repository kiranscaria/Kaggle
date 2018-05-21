#!/usr/bin/python

# Note to Kagglers: This script will not run directly in Kaggle kernels. You
# need to download it and run it on your local machine.

# Downloads images from the Google Landmarks dataset using multiple threads.
# Images that already exist will not be downloaded again, so the script can
# resume a partially completed download. All images will be saved in the JPG
# format with 90% compression quality.

import sys, os, multiprocessing, urllib2, csv
from PIL import Image
from StringIO import StringIO

data_file = 'train.csv'
out_dir = '/mnt/disks/dataset/train'

def ParseData(data_file):
    csvfile = open(data_file, 'r')
    csvreader = csv.reader(csvfile)
    key_url_list = [line for line in csvreader]
    return key_url_list[1:]  # Chop off header


def DownloadImage(key_url_label):
   
    (key, url, label) = key_url_label
    
    filename = os.path.join(out_dir, '%s.jpg' % key)

    if os.path.exists(filename):
        print('Image %s already exists. Skipping download.' % filename)
        return

    try:
        response = urllib2.urlopen(url)
        image_data = response.read()
    except:
        print('Warning: Could not download image %s from %s' % (key, url))
        return

    try:
        pil_image = Image.open(StringIO(image_data))
    except:
        print('Warning: Failed to parse image %s' % key)
        return 

    try:
        pil_image_rgb = pil_image.convert('RGB')
    except:
        print('Warning: Failed to convert image %s to RGB' % key)
        return 

    try:
        pil_image_rgb.save(filename, format='JPEG', quality=90)
        with open('/mnt/disks/dataset/downloaded_train.csv', 'a') as out_file:
            out_file.write(key + ',' + label + '\n')
    except:
        print('Warning: Failed to save image %s' % filename)
        return 


def Run():

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    
    key_url_label_list = ParseData(data_file)
    print(key_url_label_list[:5])
    
    pool = multiprocessing.Pool(processes=300)
    pool.map(DownloadImage, key_url_label_list)


if __name__ == '__main__':
    Run()
