import os
import numpy as np
from scipy import io
import lmdb
import caffe
from PIL import Image

NUM_IDX_DIGITS = 10
IDX_FMT = '{:0>%d' % NUM_IDX_DIGITS + 'd}'


def scalars_to_lmdb(scalars, path_dst,
                    lut=None):
    '''
    Generate LMDB file from list of scalars
    '''
    db = lmdb.open(path_dst, map_size=int(1e12))

    with db.begin(write=True) as in_txn:

        if not hasattr(scalars, '__iter__'):
            scalars = np.array([scalars])

        for idx, x in enumerate(scalars):
            print 'scalar:::'+str(idx)
            if not hasattr(x, '__iter__'):
                content_field = np.array([x])
            else:
                content_field = np.array(x)

            # validate these are scalars
            if content_field.size != 1:
                raise AttributeError("Unexpected shape for scalar at i=%d (%s)"
                                     % (idx, str(content_field.shape)))

            # guarantee shape (1,1,1)
            while len(content_field.shape) < 3:
                content_field = np.expand_dims(content_field, axis=0)
            content_field = content_field.astype(int)

            if lut is not None:
                content_field = lut(content_field)

            dat = caffe.io.array_to_datum(content_field)
            in_txn.put(IDX_FMT.format(idx), dat.SerializeToString())

    db.close()

    return 0

def arrays_to_lmdb(arrs, path_dst):
    '''
    Generate LMDB file from list of ndarrays
    '''
    db = lmdb.open(path_dst, map_size=int(1e12))

    with db.begin(write=True) as in_txn:

        for idx, x in enumerate(arrs):
            print 'array:::'+str(idx)
            content_field = x
            while len(content_field.shape) < 3:
                content_field = np.expand_dims(content_field, axis=0)

            dat = caffe.io.array_to_datum(content_field)
            in_txn.put(IDX_FMT.format(idx), dat.SerializeToString())

    db.close()

    return 0

def img_to_lmdb(paths_src,path_dst):
    in_db = lmdb.open(path_dst, map_size=int(1e12))
    with in_db.begin(write=True) as in_txn:
        for in_idx, in_ in enumerate(paths_src):
            print 'img:::'+str(in_)
            # load image:
            # - as np.uint8 {0, ..., 255}
            # - in BGR (switch from RGB)
            # - in Channel x Height x Width order (switch from H x W x C)
            im = np.array(Image.open(in_)) # or load whatever ndarray you need
            im = im[:,:,::-1]
            im = im.transpose((2,0,1))
            im_dat = caffe.io.array_to_datum(im)
            in_txn.put('{:0>10d}'.format(in_idx), im_dat.SerializeToString())
    in_db.close()


def matfiles_to_lmdb(paths_src, path_dst, fieldname,
                     lut=None):
    '''
    Generate LMDB file from set of mat files with integer data
    Source: https://github.com/BVLC/caffe/issues/1698#issuecomment-70211045
    credit: Evan Shelhamer

    '''
    db = lmdb.open(path_dst, map_size=int(1e12))

    with db.begin(write=True) as in_txn:

        for idx, path_ in enumerate(paths_src):
            print 'label:::'+str(idx)
            content_field = io.loadmat(path_)[fieldname]
            # get shape (1,H,W)
            while len(content_field.shape) < 3:
                content_field = np.expand_dims(content_field, axis=0)
            content_field = content_field.astype(int)

            if lut is not None:
                content_field = lut(content_field)

            img_dat = caffe.io.array_to_datum(content_field)
            in_txn.put(IDX_FMT.format(idx), img_dat.SerializeToString())

    db.close()

    return 0