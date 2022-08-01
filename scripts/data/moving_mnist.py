###########################################################################################
# Script to generate Moving MNIST video dataset (frame by frame) as described in:
# [1] arXiv:1502.04681 - Unsupervised Learning of Video Representations Using LSTMs
#     Srivastava et al
# Adapted version of Tencia Lee (https://gist.github.com/tencia/afb129122a64bde3bd0c)
# Saves in a .npz format
###########################################################################################


import argparse
import math
import os
import sys

import numpy as np
from PIL import Image


def arr_from_img(im, mean=0, std=1):
    """Converts an image into an np.array format.

    Args:
        im: Image of shape (H, W).
        mean: Mean to subtract from image.
        std: Standard deviation for normalisation.

    Returns:
        Image in np.float32 format (W, H, C) and scaled to [0,1].
    """

    w, h = im.size
    arr = im.getdata()
    c = int(np.product(arr.size) / (w * h))
    scaled_arr = np.asarray(im, dtype=np.float32).reshape((w, h, c)).transpose(2, 0, 1) / 255.

    return (scaled_arr - mean) / std


def get_picture_array(X, index, mean=0, std=1):
    """Extract image from np.array.

    Args:
        X: Dataset of shape (N, C, W, H).
        index: Index of image we want to fetch.
        mean: Mean to add if necessary.
        std: Standard deviation to add if necessary.

    Returns:
        Image in shape (H, W, C) or (H, W) if single channel.
    """

    c, w, h = X.shape[1], X.shape[2], X.shape[3]
    ret = (((X[index] + mean) * 255.) * std).reshape(c, w, h).transpose(2, 1, 0).clip(0, 255).astype(np.uint8)

    if c == 1:
        ret = ret.reshape(h, w)

    return ret


# Loads the MNIST data from online source
def load_dataset():
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def _download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, filename)

    import gzip
    def _load_mnist(images_file, labels_file):
        if not os.path.exists(images_file):
            _download(images_file)

        if not os.path.exists(labels_file):
            _download(labels_file)

        with gzip.open(images_file, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 1, 28, 28).transpose(0, 1, 3, 2)

        with gzip.open(labels_file, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)

        return (data / np.float32(255)), labels

    return _load_mnist('train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz')


def generate_moving_mnist(shape=(64, 64), seq_len=30, seqs=10000, digit_sz=28, num_digits=2, speed_max=5):
    """Generates and returns video frames and associated labels as np.uint8 arrays.

    Args:
        shape: Shape of the Moving MNIST patches (W, H).
        seq_len: Frame length of a video sequence.
        seqs: Number of video sequences to generate.
        digit_sz: Size of the digit images within the video patch.
        num_digits: Number of digits within a video sequence.
        speed_max: Maximum velocity a digit can take.

    Returns:
        dataset: Dataset of np.float32 type (seq_len, seqs, 1, W, H).
        labels: Labels associated with dataset in np.uint8 type (seqs, num_digits).
    """

    mnist_data, mnist_y = load_dataset()
    num_examples = mnist_data.shape[0]
    width, height = shape

    # Get limitations on motion within a patch
    lims = (x_lim, y_lim) = width - digit_sz, height - digit_sz

    # Create empty dataset and labels np.arrays
    dataset = np.empty((seqs, seq_len, 1, width, height), dtype=np.uint8)
    labels = np.empty((seqs, num_digits), dtype=np.uint8)
    lab_dist = np.zeros(10, dtype=np.int32)
    for seq_idx in range(seqs):
        # Randomly generate direc/speed/position, calculate velocity vector
        direcs = np.pi * (np.random.rand(num_digits) * 2 - 1)
        speeds = np.random.randint(speed_max, size=num_digits) + 2
        veloc = np.asarray([(v * math.cos(d), v * math.sin(d)) for d, v in zip(direcs, speeds)])

        # Get randomly sampled images and there corresponding labels from MNIST dataset
        rand_indexes = np.random.randint(0, num_examples, size=num_digits)
        mnist_images = [Image.fromarray(get_picture_array(mnist_data, r)).resize((digit_sz, digit_sz), Image.Resampling.LANCZOS) \
                        for r in rand_indexes]
        mnist_labels = [mnist_y[r] for r in rand_indexes]
        lab_dist[mnist_labels[0]] += 1

        # Create tuples of (x,y) initial positions for each digit in the patch
        positions = np.asarray([(np.random.rand() * x_lim, np.random.rand() * y_lim) for _ in range(num_digits)])

        # Create actual video sequence now with random motion
        for frame_idx in range(seq_len):
            canvases = [Image.new('L', (width, height)) for _ in range(num_digits)]
            canvas = np.zeros((1, width, height), dtype=np.float32)

            # Superimpose digit images onto the canvas
            for i, canv in enumerate(canvases):
                canv.paste(mnist_images[i], tuple(map(lambda p: int(round(p)), positions[i])))
                canvas += arr_from_img(canv)

            # Update positions based on velocity
            next_pos = positions + veloc

            # Bounce off walls if we hit one by changing direction
            for i, pos in enumerate(next_pos):
                for j, coord in enumerate(pos):
                    if coord < -2 or coord > (lims[j] + 2):
                        veloc[i] = tuple(list(veloc[i][:j]) + [-1 * veloc[i][j]] + list(veloc[i][j + 1:]))

            # Update positions array
            positions = positions + veloc

            # Copy additive canvas to data array
            dataset[seq_idx, frame_idx] = (canvas * 255).clip(0, 255).astype(np.uint8)
            labels[seq_idx] = mnist_labels

    print("Digit distribution:")
    print(lab_dist)

    # Normalise
    dataset = (dataset / np.float32(255))
    # Binarise
    dataset[dataset >= .5] = 1.
    dataset[dataset < .5] = 0.

    return dataset, labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default='',
                        help="Path to load the dataset from.")
    parser.add_argument('--filename', default='',
                        help="Name of the dataset file.")
    parser.add_argument('--patch_size', default=64, type=int,
                        help="Patch image size that comprises the moving digit.")
    parser.add_argument('--digit_size', default=28, type=int,
                        help="Size of the MNIST digit within the patch/frame.")
    parser.add_argument('--num_digits', default=1, type=int,
                        help="Number of digits in each patch/frame of video.")
    parser.add_argument('--seq_length', default=20, type=int,
                        help="Number of frames in video sequence.")
    parser.add_argument('--seqs_count', default=10000, type=int,
                        help="Number of video sequences to generate.")
    parser.add_argument('--speed_max', default=5, type=int,
                        help="The random velocity of each digit will be capped at one "
                             "speed level higher than this parameter.")
    args = parser.parse_args()

    dataset, labels = generate_moving_mnist(shape=(args.patch_size, args.patch_size), seq_len=args.seq_length, \
                                            seqs=args.seqs_count, digit_sz=args.digit_size,
                                            num_digits=args.num_digits, speed_max=args.speed_max)

    dest = os.path.join(args.dataset_path,
                        '{}_num{}_seq{}'.format(args.filename, args.num_digits, args.seq_length))
    if not os.path.exists(args.dataset_path):
        os.makedirs(args.dataset_path)

    # Fixed upper bounds for splits
    train_ub = args.seqs_count * 8 // 10
    valid_ub = args.seqs_count * 9 // 10

    # Create Moving MNIST dataset as an .npz file or jpg frames
    train_set = dataset[0:train_ub]
    train_y = labels[0:train_ub]
    valid_set = dataset[train_ub:valid_ub]
    valid_y = labels[train_ub:valid_ub]
    test_set = dataset[valid_ub:args.seqs_count]
    test_y = labels[valid_ub:args.seqs_count]

    # Compute training set mean
    dense_images = [seq for seq in train_set]
    # Concatenate all images along the time axis
    concatenated = np.concatenate(dense_images, axis=0)
    mean = np.mean(concatenated, axis=0)

    # Save resulting dataset
    np.savez(dest, train=train_set, valid=valid_set, test=test_set, train_mean=mean, train_y=train_y, valid_y=valid_y,
             test_y=test_y)
