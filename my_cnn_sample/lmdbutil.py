#encoding=utf-8
#LMDB utility
#reference site
# http://shengshuyang.github.io/hook-up-lmdb-with-caffe-in-python.html
# https://github.com/kostyaev/ml-utils/blob/master/create_multilabel_lmdb.py

import os
import numpy as np
import lmdb
from PIL import Image, ImageOps
from caffe.proto import caffe_pb2
from caffe.io import datum_to_array, array_to_datum

CH_RGB  = 3 # channel count of rgb
CH_GRAY = 1 # channel count of grayscale

def read_lmdb( db ):
  '''
  #read_lmdb: 
  #  db: target lmdb path
  '''
  print ('[start] read lmdb [{}]'.format(db))
  env = lmdb.open( db, readonly=True)
  with env.begin() as txn:
    cursor = txn.cursor()
    count = 0
    shape = None
    for key, value in cursor:
      datum = caffe_pb2.Datum()
      datum.ParseFromString(value)
      x = datum_to_array(datum)
      y = datum.label
      if count == 0: # show shape of first item as sample
        shape = x.shape
      count += 1
    print ('[finished] total count:{} , shape:{}'.format(count,shape))
  env.close()

def lmdb_to_images( db, dest ,length ):
  '''
  #lmdb_to_img: output image-files from lmdb. file path : <dest>/<class_id>/<key>.jpg 
  #  lmdb: target lmdb path
  #  dest: directory path
  #  length: limit of item count to output
  '''
  print ('[start] creating images from lmdb[{}]'.format(db))
  if not os.path.isdir(dest):
    os.mkdir(dest)
    print ('  directory \'{}\' created.'.format(dest))

  env = lmdb.open( db, readonly=True)
  with env.begin() as txn:
    cursor = txn.cursor()
    count = 0
    for key, value in cursor:
      datum = caffe_pb2.Datum()
      datum.ParseFromString(value)
      x = datum_to_array(datum)
      y = datum.label
      #print ( '{0}: x.shape:{1} label:{2}'.format(key,x.shape,y))
      if not os.path.isdir(os.path.join(dest,str(y))):
        os.mkdir(os.path.join(dest,str(y)))

      if datum.channels == CH_GRAY:
        img = Image.fromarray( np.uint8(x[0])) # shape (h,w)
        img.save( os.path.join(dest, str(y), str(key) + '.jpg' ))
      elif datum.channels == CH_RGB:
        img = Image.fromarray( np.uint8(x.transpose((1,2,0)))) # shape (ch,h,w)->(h,w,ch)
        img.save( os.path.join(dest, str(y), str(key) + '.jpg' ))
      else: #error
        print ('  invalid channel')
        break

      count += 1
      if count >= length:
        break
    print ('[finished] total {} images are written in \'{}\''.format(count,dest))
  env.close()

def resize_lmdb( src_db, dest_db, width, height, length ):
  '''
  #resize_lmdb: create lmdb from existing lmdb with resizing
  #  src_db:  source lmdb name
  #  dest_db: new lmdb name
  #  width:   new image width
  #  height:  new image height
  #  length:  limit of item count to process
  '''
  print ('[start] resize lmdb [{}]->[{}]'.format(src_db,dest_db))
  src_env = lmdb.open( src_db, readonly=True)
  map_size = 100000000*length #buffer size
  dest_env = lmdb.Environment(dest_db,map_size)
  dest_txn = dest_env.begin(write=True, buffers=True)
  with src_env.begin() as src_txn:
    cursor = src_txn.cursor()
    count = 0
    before, after = None,None
    for key, value in cursor:
      datum = caffe_pb2.Datum()
      datum.ParseFromString(value)
      x = datum_to_array(datum) # (c,h,w)
      y = datum.label

      if datum.channels == CH_GRAY:
        img_array = x.transpose((1,2,0)) # (c,h,w)->(h,w,c)
        img_array = img_array.reshape(img_array.shape[0],img_array.shape[1]) # (h,w,1)->(h,w)
        img = Image.fromarray(np.uint8(img_array))
        img = img.resize((width,height))
        img_array = np.asarray(img)
        img_array = img_array.reshape(height,width,1) # (h,w)->(h,w,1)
      elif datum.channels == CH_RGB:
        img_array = x.transpose((1,2,0)) # (c,h,w)->(h,w,c)
        img = Image.fromarray(np.uint8(img_array))
        img = img.resize((width,height))
        img_array = np.asarray(img)
      img_array = img_array.transpose((2,0,1)) # (h,w,c) -> (c,h,w)
      out_datum = array_to_datum(img_array,y)
      dest_txn.put( key.encode('ascii'), out_datum.SerializeToString())
      if count == 0:
        before = x.shape
        after = img_array.shape
        #print ( '{0}: x.shape:{1} label:{2} -> x.shape{3} label:{4}'.format(key,x.shape,y,img.shape,y))
      count += 1
      if count >= length:
        break
    print ('[finished] total count {}. shape {} -> {}'.format(count,before,after))
  dest_txn.commit()
  dest_env.close()
  src_env.close()


def create_lmdb_from_array(dest_db, imgs, labels):
  '''
  # multi cahennl lmdb
  # imgs : list of img array
  # labels : list of label id
  # imgs[0].shape = (channel, height, width)
  # len(imgs) = item count
  # len(labels) = item count
  '''
  #print ('[start] creating lmdb. ch:{}, w:{}, h:{}, item count:{}'.format(imgs[0].shape[0],imgs[0].shape[1],imgs[0].shape[2],len(imgs)))
  map_size = 100000000*len(imgs[0]) #buffer size
  env = lmdb.Environment(dest_db,map_size)
  txn = env.begin(write=True, buffers=True)
  count = 0
  for idx, img in enumerate(imgs):
    str_id = '{:08}'.format(idx)
    clsid = labels[idx]
    #print ( '  {0} shape:{1}, class:{2}'.format(str_id, img.shape , clsid))
    datum = array_to_datum(img,clsid)
    txn.put(str_id.encode('ascii'), datum.SerializeToString())
    count += 1
  txn.commit()
  env.close()
  print ('[finished] {} items have written to \'{}\''.format(count,dest_db))

def create_lmdb_from_filelist( db_name, images, labels ):
  '''
  #create_lmdb_from_filelist: create lmdb from list of image file names. 
  #  db_name:   new lmdb name
  #  images: list of image file names
  #  labels: list of labels (index must be syncronized with images)
  '''
  print ('[start] creating lmdb from file list')
  img_array = []
  for idx, name in enumerate(images):
    img    = np.asarray(Image.open(name)) # (h,w,c)
    if len(img.shape) == 2: # gray (h,w)
      img = img.reshape(1, img.shape[0], img.shape[1]) #shape (h,w)->(c,h,w)
    elif len(img.shape) == 3 & img.shape[2] == CH_RGB:
      img = img.transpose((2,0,1)) # shape (h,w,c)->(c,h,w)
    else:
      continue
    img_array.append(img)
  create_lmdb_from_array(db_name,img_array,labels)
  print ('[finished] creating lmdb from file list')


def create_lmdb_from_dir( db_name, src_dir ):
  '''
  #create_lmdb_from_dir: create lmdb from images. input src : <src>/<class_id>/xxxxx.jpg 
  #  db_name: new lmdb name
  #  src_dir : directory includes images
  '''
  print ('[start] creating lmdb from image directory')
  imgs = []
  labels = []
  for root, dirs, files in os.walk(src_dir, topdown = False):
    if root == src_dir:
      continue
    print ('  dir:{0} has {1} items.'.format(root,len(files)))
    for idx, fname in enumerate(files):
      #print ('  name:{0}, label:{1}'.format(fname,os.path.basename(root)))
      imgs.append( os.path.join(root,fname) )
      labels.append( int(os.path.basename(root)) ) # integer
  create_lmdb_from_filelist( db_name, imgs, labels )
  print ('[finished] creating lmdb from image directory')


# sample script creating multi channel lmdb
# rgb(3ch) -> rgb+gray (4ch)
def _test_multi_channel_lmdb( db, src_rgb_dir ):
  array_4ch = []
  labels = []
  print ('[test][start] sample script creating 4ch lmdb')
  for root, dirs, files in os.walk(src_rgb_dir, topdown = False):
    if root == src_rgb_dir:
      continue
    print ('  dir:{0} has {1} items.'.format(root,len(files)))
    for idx, fname in enumerate(files):

      labels.append( int(os.path.basename(root)) ) # integer

      rgb = Image.open(os.path.join(root,fname))
      gray = ImageOps.grayscale(rgb)
      img_rgb  = np.asarray(rgb) # (h,w,c)
      img_gray = np.asarray(gray) # (h,w)

      img_rgb  = img_rgb.transpose((2,0,1)) # (h,w,c) -> (c,h,w)
      img_gray = img_gray.reshape(1,img_gray.shape[0],img_gray.shape[1]) # (h,w)->(1,h,w)
      img_4ch  = np.concatenate( [img_rgb,img_gray])
      array_4ch.append(img_4ch)

  create_lmdb_from_array(db,array_4ch,labels)
  print ('[test][finished] sample script creating 4ch lmdb')


if __name__=='__main__':

  ## test code

  ### read_lmdb
  read_lmdb('cifar10_train_lmdb') #rgb
  read_lmdb('mnist_train_lmdb') #gray

  '''
  ### lmdb_to_images
  lmdb_to_images( 'cifar10_test_lmdb', 'cifar10_test_img' , 100 )
  lmdb_to_images( 'mnist_test_lmdb', 'mnist_test_img' , 100 )
  '''

  '''
  ### resize_lmdb( src_db, dest_db, width, height, length )
  resize_lmdb( 'cifar10_train_lmdb', 'cifar10_train_lmdb_resize', 64, 48, 500) #original size 32x32 x50000
  resize_lmdb( 'cifar10_test_lmdb', 'cifar10_test_lmdb_resize', 48, 64, 100)
  resize_lmdb( 'mnist_train_lmdb', 'mnist_train_lmdb_resize', 56, 42, 600)   #original size 28x28 x60000
  resize_lmdb( 'mnist_test_lmdb', 'mnist_test_lmdb_resize', 42, 56, 100)
  lmdb_to_images( 'cifar10_train_lmdb_resize', 'cifar10_train_resize_img' , 500 )
  lmdb_to_images( 'cifar10_test_lmdb_resize', 'cifar10_test_resize_img' , 100 )
  lmdb_to_images( 'mnist_train_lmdb_resize', 'mnist_train_resize_img' , 600 )
  lmdb_to_images( 'mnist_test_lmdb_resize', 'mnist_test_resize_img' , 100 )
  '''

  '''
  ### create_lmdb_from_dir( db_name, src_dir )
  create_lmdb_from_dir( 'mnist_test_lmdb_resize', 'mnist_test_resize_img' )
  create_lmdb_from_dir( 'cifar10_test_lmdb_resize', 'cifar10_test_resize_img' )
  lmdb_to_images( 'mnist_test_lmdb_resize', 'mnist_test_resize_img' , 100 )
  lmdb_to_images( 'cifar10_test_lmdb_resize', 'cifar10_test_resize_img' , 100 )
  '''

  '''
  ### _test_multi_channel_lmdb( db, src_rgb_dir )
  _test_multi_channel_lmdb( 'cifar10_4ch_lmdb', 'cifar10_train_resize_img')
  read_lmdb('cifar10_4ch_lmdb')
  '''


