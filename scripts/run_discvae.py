from absl import app, flags, logging

import runners

FLAGS = flags.FLAGS

# Shared flags
flags.DEFINE_enum('model', 'discvae',
                  ['discvae', 'vrnn', 'gmvae', 'vae'],
                  "The deep generative model to be trained.")
flags.DEFINE_enum('mode', 'train',
                  ['train', 'eval'],
                  "The mode of the binary.")
flags.DEFINE_integer('latent_size', 128,
                     "The size of the latent state of the generative model.")
flags.DEFINE_integer('dynamic_latent_size', 32,
                     "The size of the dynamic latent state of a disentangled sequential model.")
flags.DEFINE_integer('hidden_size', 512,
                     "The size of the intermediate hidden states for dense networks.")
flags.DEFINE_integer('rnn_size', 128,
                     "The size of the dynamic RNN internal states.")
flags.DEFINE_integer('mixture_components', 10,
                     "Number of mixture components to use in prior.")
flags.DEFINE_integer('encoded_data_size', 512,
                     "The dimension of the encoded data sequence.")
flags.DEFINE_integer('encoded_latent_size', 32,
                     "The dimension of the encoded latent state.")
flags.DEFINE_float('init_temp', 1.0,
                   "Initial temperature for the Gumbel-Softmax distribution.")
flags.DEFINE_integer('batch_size', 16,
                     "Batch size.")
flags.DEFINE_string('logdir', '/tmp/smc_vi',
                    "The directory to keep checkpoints and summaries in.")
flags.DEFINE_integer('random_seed', 0,
                     "A random seed for seeding the TensorFlow graph.")

# Training/Evaluation flags
flags.DEFINE_float('learning_rate', 0.0003,
                   "The learning rate for ADAM.")
flags.DEFINE_float('clip_norm', float(1e10),
                   "Threshold for global norm gradient clipping.")
flags.DEFINE_float('beta', 1.0,
                   "Beta parameter for the b-VAE formulation.")
flags.DEFINE_integer('num_epochs', 20,
                     "The number of epochs to train for.")
flags.DEFINE_integer('summarise_every', 50,
                     "The number of steps between qualitative/debugging summaries.")
flags.DEFINE_integer('save_every', 5,
                     "The number of epochs between saving checkpoints.")
flags.DEFINE_string('gpu_num', '0',
                    "Comma-separated list of GPU ids to use.")
flags.DEFINE_enum('split', 'train',
                  ['train', 'test', 'valid'],
                  "Split to evaluate the model on.")

# Sampling flags
flags.DEFINE_integer('num_samples', 1,
                     "Number of samples for use in training and reconstruction/generation.")
flags.DEFINE_integer('sample_length', 10,
                     "The number of timesteps to sample for.")
flags.DEFINE_integer('prefix_length', 10,
                     "The number of timesteps to condition the model on before sampling.")

# Dataset flags
flags.DEFINE_enum('dataset', 'moving_mnist',
                  ['moving_mnist', 'sprites'],
                  "The dataset to be used.")
flags.DEFINE_string('dataset_path', '',
                    "Path to load the dataset from.")
flags.DEFINE_string('filename', '',
                    "Name of the dataset file.")
flags.DEFINE_integer('patch_size', 64,
                     "Patch image size.")
flags.DEFINE_integer('num_channels', 1,
                     "Number of channels in the image.")
flags.DEFINE_integer('seq_length', 20,
                     "Number of frames in video sequence.")


def main(unused_argv):
    del unused_argv

    logging.set_verbosity(logging.INFO)
    logging.info("Arguments: {}".format(FLAGS.flag_values_dict()))

    if FLAGS.mode == 'train':
        runners.run_train(FLAGS)
    elif FLAGS.mode == 'eval':
        runners.run_eval(FLAGS)


if __name__ == '__main__':
    app.run(main)
