import numpy as np
import tensorflow as tf
import h5py
from tqdm import trange

def rgb_to_yuv(rgb):
    rgb = np.float32(rgb) / 255.0
    y = 0.299 * rgb[:,:,0] + 0.587 * rgb[:,:,1] + 0.114 * rgb[:,:,2]
    u = -0.147 * rgb[:,:,0] - 0.289 * rgb[:,:,1] + 0.436 * rgb[:,:,2]
    v = 0.615 * rgb[:,:,0] - 0.515 * rgb[:,:,1] - 0.100 * rgb[:,:,2]
    yuv = np.stack([y, u, v], axis=-1)
    return yuv

def rgb_to_yuv_gpu(rgb):
    rgb = tf.cast(rgb, tf.float32) / 255.0
    y = 0.299 * rgb[:,:,0] + 0.587 * rgb[:,:,1] + 0.114 * rgb[:,:,2]
    u = -0.147 * rgb[:,:,0] - 0.289 * rgb[:,:,1] + 0.436 * rgb[:,:,2]
    v = 0.615 * rgb[:,:,0] - 0.515 * rgb[:,:,1] - 0.100 * rgb[:,:,2]
    yuv = tf.stack([y, u, v], axis=-1)
    return yuv

def augment(x):
    bs = x.shape[0]
    thetas = np.random.uniform(-1, 1, [bs,1]) * np.pi
    rotated = rotate_z(thetas, x)
    scale = np.random.uniform(size=[bs,1,3]) * 0.45 + 0.8
    return rotated * scale

def augment_gpu(x, y, bs):
    thetas = tf.random.uniform([bs,1], -1, 1) * tf.constant(np.pi)
    rotated = rotate_z_gpu(thetas, x)
    scale = tf.random.uniform([bs,1,3]) * 0.45 + 0.8
    return rotated * scale, y

def standardize_gpu(x, y, num_pts):
    perm = tf.random.shuffle(tf.range(10000))
    x = tf.gather(x, perm[:num_pts], axis=1)
    z = tf.clip_by_value(x, -20, 20)
    mean, var = tf.nn.moments(z, [1,2], keepdims=True)
    std = tf.sqrt(var)
    return (z - mean) / std, y

class DataLoader(object):
    def __init__(self, params, channel, y=0, do_standardize=False, do_augmentation=False, n_obj=5):
        for key, val in params.items():
            setattr(self, key, val)
        self.channel = channel  
        self.y = y if isinstance(y, (list, tuple)) else [y]
        
        # Filter data
        filt = np.full(self.labels.shape, False)
        for yi in self.y:
            lflt = self.labels == yi
            locs = np.where(lflt)[0]
            if len(locs) > n_obj:
                lflt[locs[n_obj]:] = False
                        filt = np.logical_or(filt, lflt)
        self.labels = self.labels[filt]
        self.data = self.data[filt, :, :]
        self.data = rgb_to_yuv(self.data)  # Convert RGB to YUV
        
        # 根据通道选择数据
        if self.channel == 'Y':
            self.data = self.data[:, :, 0:1] 
        elif self.channel == 'U':
            self.data = self.data[:, :, 1:2]  
        elif self.channel == 'V':
            self.data = self.data[:, :, 2:3]  
        
        self.max_n_pt = self.data.shape[1]

        n_repeat = 30000 // sum(filt)
        self.data = np.tile(self.data, (n_repeat, 1, 1))
        self.labels = np.tile(self.labels, (n_repeat,))
        for i in trange(len(self.data)):
            pt_perm = np.random.permutation(self.max_n_pt)
            self.data[i] = self.data[i, pt_perm]
        
        self.len_data = len(self.data)
        self.prep1 = (lambda x, y: standardize_gpu(rgb_to_yuv_gpu(x), y, self.num_points_per_object)) if do_standardize else rgb_to_yuv_gpu
        self.prep2 = (lambda x, y: augment_gpu(self.prep1(x, y)[0], y, self.batch_size)) if do_augmentation else self.prep1

        data_placeholder = tf.placeholder(self.data.dtype, self.data.shape)
        labels_placeholder = tf.placeholder(self.labels.dtype, self.labels.shape)
        dataset = tf.data.Dataset.from_tensor_slices((data_placeholder, labels_placeholder))
        dataset = dataset.shuffle(30000).repeat().batch(self.batch_size).map(self.prep2, num_parallel_calls=2)
        dataset = dataset.prefetch(buffer_size=10000)
        iterator = dataset.make_initializable_iterator()
        self.sess.run(iterator.initializer, feed_dict={data_placeholder: self.data, labels_placeholder: self.labels})
        bdata, blabel = iterator.get_next()
        bdata.set_shape((self.batch_size, self.num_points_per_object, 3))
        self.next_batch = (bdata, blabel)

    def __iter__(self):
        return self.iterator()

    def iterator(self):
        while True:
            yield self.sess.run(self.next_batch)

