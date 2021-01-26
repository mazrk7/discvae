import matplotlib
import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

# Prevent text from rendering as paths
new_rc_params = {'text.usetex': False,
                 "svg.fonttype": 'none'
                 }
matplotlib.rcParams.update(new_rc_params)

# Causes matplot lib to use Type 42 fonts for PDF and avoid Type 3
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

from sklearn.manifold import TSNE
from scipy.optimize import linear_sum_assignment


def reduce_dimensionality(data, dim=2, perplexity=20):
    """Fit the numpy data array into a t-SNE embedding."""

    if (data.shape[-1] > 2):
        tsne = TSNE(n_components=dim, verbose=1, perplexity=perplexity, n_iter=1000)
        data = tsne.fit_transform(data)

    return data


def entropy(logits, targets):
    """Computes entropy as -sum(targets*log(predicted))."""

    log_q = tf.nn.log_softmax(logits)

    return -tf.reduce_sum(targets * log_q, axis=-1)


def cluster_acc(y_pred, y_true):
    """Computes the clustering accuracy metric."""

    y_true = y_true.astype(np.int64)

    D = max(y_pred.max(), y_true.max()) + 1
    cost_matrix = np.zeros((D, D), dtype=np.int64)

    # Confusion matrix
    for i in range(y_pred.size):
        cost_matrix[y_pred[i], y_true[i]] += 1

    indices = linear_sum_assignment(-cost_matrix)
    indices = np.asarray(indices)
    indices = np.transpose(indices)
    acc = sum([cost_matrix[i, j] for i, j in indices]) * 1.0 / y_pred.size

    return acc


def compute_NMI(cluster_assignments, class_assignments):
    """Computes the Normalized Mutual Information between cluster and class assignments.
    Taken from SOM-VAE (2018): https://github.com/ratschlab/SOM-VAE/blob/master/som_vae/utils.py

    Args:
        cluster_assignments (list): List of cluster assignments for every point.
        class_assignments (list): List of class assignments for every point.
    Returns:
        float: The NMI value.
    """

    assert len(cluster_assignments) == len(class_assignments), "The inputs have to be of the same length."

    clusters = np.unique(cluster_assignments)
    classes = np.unique(class_assignments)

    num_samples = len(cluster_assignments)
    num_classes = len(classes)

    assert num_classes > 1, "There should be more than one class."

    cluster_class_counts = {cluster_: {class_: 0 for class_ in classes} for cluster_ in clusters}

    for cluster_, class_ in zip(cluster_assignments, class_assignments):
        cluster_class_counts[cluster_][class_] += 1

    cluster_sizes = {cluster_: sum(list(class_dict.values())) for cluster_, class_dict in cluster_class_counts.items()}
    class_sizes = {class_: sum([cluster_class_counts[clus][class_] for clus in clusters]) for class_ in classes}

    I_cluster_class = H_cluster = H_class = 0

    for cluster_ in clusters:
        for class_ in classes:
            if cluster_class_counts[cluster_][class_] == 0:
                pass
            else:
                I_cluster_class += (cluster_class_counts[cluster_][class_] / num_samples) * \
                                   (np.log((cluster_class_counts[cluster_][class_] * num_samples) / \
                                           (cluster_sizes[cluster_] * class_sizes[class_])))

    for cluster_ in clusters:
        H_cluster -= (cluster_sizes[cluster_] / num_samples) * np.log(cluster_sizes[cluster_] / num_samples)

    for class_ in classes:
        H_class -= (class_sizes[class_] / num_samples) * np.log(class_sizes[class_] / num_samples)

    NMI = (2 * I_cluster_class) / (H_cluster + H_class)

    return NMI


def image_seq_summary(seqs, name, step, num=None):
    """Visualizes sequences as TensorBoard summaries.

    Args:
        seqs: A tensor of shape [batch_size, time, H, W, C].
        name: String name of this summary.
        step: Step to associate with this summary.
        num: Integer for the number of examples to visualize. Defaults to
          all examples.
    """

    seqs = tf.unstack(seqs[:num])
    joined_seqs = [tf.concat(tf.unstack(seq), 1) for seq in seqs]
    joined_seqs = tf.expand_dims(tf.concat(joined_seqs, 0), 0)
    tf.summary.image(name, joined_seqs, max_outputs=1, step=step)


def pack_images(images, rows, cols):
    """Helper utility to make a field of images."""

    shape = tf.shape(input=images)
    width = shape[-3]
    height = shape[-2]
    depth = shape[-1]

    images = tf.reshape(images, (-1, width, height, depth))
    batch = tf.shape(input=images)[0]

    rows = tf.minimum(rows, batch)
    cols = tf.minimum(batch // rows, cols)

    images = images[:rows * cols]
    images = tf.reshape(images, (rows, cols, width, height, depth))
    images = tf.transpose(a=images, perm=[0, 2, 1, 3, 4])
    images = tf.reshape(images, [1, rows * width, cols * height, depth])

    return images


def image_tile_summary(tensor, name, step, rows=8, cols=8):
    tf.summary.image(name, pack_images(tensor, rows, cols), max_outputs=1, step=step)


def tsne_visualise(path, step, latent_var, labels, num_colours):
    feat_cols = ['index' + str(i) for i in range(latent_var.shape[1])]
    df = pd.DataFrame(latent_var, columns=feat_cols)

    df['labels'] = labels
    df['dim_1'] = latent_var[:, 0]
    df['dim_2'] = latent_var[:, 1]

    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x='dim_1', y='dim_2',
        hue='labels',
        palette=sns.color_palette('hls', num_colours),
        data=df,
        s=50,  # Marker size
        legend='full',
        alpha=0.3)

    # Axes formatting
    ax = plt.gca()
    ax.set_xlabel("dim_1", size=20)
    ax.set_ylabel("dim_2", size=20)
    # Turn off tick labels
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_title("t-SNE Projection of Latent Space", size=24)
    ax.legend(fontsize=16, loc=0)

    plt.savefig('{}/tsne_proj_{}.svg'.format(path, step), dpi=1000, format='svg', bbox_inches='tight')


def plot_density(path, step, samples):
    g = sns.jointplot(
        x=samples[..., 0],
        y=samples[..., 1],
        kind='kde',
        space=0,
        color='b')

    plt.savefig('{}/density_estimate_{}.pdf'.format(path, step))


def plot_batch_sequence(path, step, batch_seq, name):
    seq_len = np.size(batch_seq, 0)
    num_seq = np.size(batch_seq, 1)
    f = plt.figure(figsize=(10, num_seq))
    grid = gs.GridSpec(num_seq, seq_len,
                       wspace=0.0, hspace=0.0, top=0.85, bottom=0.25, left=0.05, right=0.95)

    for j in range(num_seq):
        for i in range(seq_len):
            img = np.clip(batch_seq[i, j], 0, 1)
            ax = plt.subplot(grid[j, i])
            ax.imshow(img, cmap='gray')
            ax.set_yticks([])
            ax.set_xticks([])

    plt.savefig('{}/{}_{}.pdf'.format(path, name, step))


def plot_video_sequence(path, step, sequence, name):
    f = plt.figure(figsize=(10, 2))
    seq_len = np.size(sequence, 0)
    grid = gs.GridSpec(1, seq_len,
                       wspace=0.0, hspace=0.0, top=0.85, bottom=0.25, left=0.05, right=0.95)

    for i in range(seq_len):
        img = np.clip(np.squeeze(sequence[i]), 0, 1)
        ax = plt.subplot(grid[0, i])
        ax.imshow(img, cmap='gray')
        ax.set_yticks([])
        ax.set_xticks([])

    plt.savefig('{}/{}_{}.pdf'.format(path, name, step))


def plot_k_samples(path, step, targets, k_samples, infer_c=None, num_k_display=10, name='inject'):
    f = plt.figure(figsize=(10, num_k_display + 2))
    sample_length = np.size(k_samples, 1)
    grid = gs.GridSpec(num_k_display + 2, sample_length,
                       wspace=0.0, hspace=0.0, top=0.95, bottom=0.05, left=0.05, right=0.95)

    # Just put it as the last index
    if infer_c is None:
        infer_c = num_k_display - 1

    infer_c_sample = k_samples[infer_c]
    k_samples_removed = np.delete(k_samples, infer_c, axis=0)
    k_samples_display = k_samples_removed[:num_k_display]

    k = 0
    for k_sample in k_samples_display:
        for i in range(sample_length):
            k_img = np.clip(np.squeeze(k_sample[i]), 0, 1)
            ax = plt.subplot(grid[k, i])
            ax.imshow(k_img, cmap='gray')
            ax.set_yticks([])
            ax.set_xticks([])

        k += 1

    for i in range(sample_length):
        infer_c_img = np.clip(np.squeeze(infer_c_sample[i]), 0, 1)
        ax = plt.subplot(grid[num_k_display, i])
        ax.imshow(infer_c_img, cmap='gray')
        ax.set_yticks([])
        ax.set_xticks([])

        target_img = np.clip(np.squeeze(targets[i]), 0, 1)
        ax = plt.subplot(grid[num_k_display + 1, i])
        ax.imshow(target_img, cmap='gray')
        ax.set_yticks([])
        ax.set_xticks([])

    plt.savefig('{}/cluster_{}_{}.pdf'.format(path, name, step))
