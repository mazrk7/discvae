###########################################################################################
# Script to load the Sprites video dataset (stored in .npy format)
# Adapted from Yingzhen Li (https://github.com/YingzhenLi/Sprites)
###########################################################################################

import math
import time

import numpy as np


def sprites_act(path, seed=0, return_labels=False):
    directions = ['front', 'left', 'right']
    actions = ['walk', 'spellcard', 'slash']
    start = time.time()
    path = path + '/npy/'

    X_train = []
    X_test = []
    if return_labels:
        A_train = [];
        A_test = []
        D_train = [];
        D_test = []
    for act in range(len(actions)):
        for i in range(len(directions)):
            label = 3 * act + i
            print(actions[act], directions[i], act, i, label)
            x = np.load(path + '%s_%s_frames_train.npy' % (actions[act], directions[i]))
            X_train.append(x)
            y = np.load(path + '%s_%s_frames_test.npy' % (actions[act], directions[i]))
            X_test.append(y)
            if return_labels:
                a = np.load(path + '%s_%s_attributes_train.npy' % (actions[act], directions[i]))
                A_train.append(a)
                d = np.zeros([a.shape[0], a.shape[1], 9])
                d[:, :, label] = 1;
                D_train.append(d)

                a = np.load(path + '%s_%s_attributes_test.npy' % (actions[act], directions[i]))
                A_test.append(a)
                d = np.zeros([a.shape[0], a.shape[1], 9])
                d[:, :, label] = 1;
                D_test.append(d)

    X_train = np.concatenate(X_train, axis=0) * 256.0
    X_test = np.concatenate(X_test, axis=0) * 256.0
    # Concatenate all images along the time axis
    concatenated = np.concatenate(X_train, axis=0)
    mean = np.mean(concatenated, axis=0)
    print(mean.shape)
    np.random.seed(seed)
    ind = np.random.permutation(X_train.shape[0])
    X_train = X_train[ind]
    if return_labels:
        A_train = np.concatenate(A_train, axis=0)
        D_train = np.concatenate(D_train, axis=0)
        A_train = A_train[ind]
        D_train = D_train[ind]
    ind = np.random.permutation(X_test.shape[0])
    X_test = X_test[ind]
    if return_labels:
        A_test = np.concatenate(A_test, axis=0)
        D_test = np.concatenate(D_test, axis=0)
        A_test = A_test[ind]
        D_test = D_test[ind]
        print(A_test.shape, D_test.shape, X_test.shape, 'shapes')
    print(X_train.shape, X_test.min(), X_test.max())
    end = time.time()
    print('data loaded in %.2f seconds...' % (end - start))

    if return_labels:
        return X_train, X_test, A_train, A_test, D_train, D_test, mean
    else:
        return X_train, X_test, mean


def reshape_and_tile_images(array, shape=(28, 28), n_cols=None):
    if n_cols is None:
        n_cols = int(math.sqrt(array.shape[0]))
    n_rows = int(np.ceil(float(array.shape[0]) / n_cols))
    if len(shape) == 2:
        order = 'C'
    else:
        order = 'F'

    def cell(i, j):
        ind = i * n_cols + j
        if i * n_cols + j < array.shape[0]:
            return array[ind].reshape(*shape, order='C')
        else:
            return np.zeros(shape)

    def row(i):
        return np.concatenate([cell(i, j) for j in range(n_cols)], axis=1)

    return np.concatenate([row(i) for i in range(n_rows)], axis=0)


def plot_gif(x_seq, shape, path, filename):
    n_cols = int(np.sqrt(x_seq.shape[0]))
    x_seq = x_seq[:n_cols ** 2]
    T = x_seq.shape[1]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig = plt.figure()
    x0 = reshape_and_tile_images(x_seq[:, 0], shape, n_cols)
    im = plt.imshow(x0, animated=True, cmap='gray')
    plt.axis('off')

    def update(t):
        x_frame = reshape_and_tile_images(x_seq[:, t], shape, n_cols)
        im.set_array(x_frame)
        return im,

    anim = FuncAnimation(fig, update, frames=np.arange(T), \
                         interval=1000, blit=True)
    anim.save(path + filename + '.gif', writer='imagemagick')
    print('image saved as ' + path + filename + '.gif')
