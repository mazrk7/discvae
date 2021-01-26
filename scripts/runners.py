import os
import sys

import numpy as np
import tensorflow as tf
import torch
from absl import logging

import utils
from data import lpc_sprites
from models import discvae, gmvae, vae, vrnn


def create_moving_mnist(config, split, shuffle, num_digits=1):
    """Creates the Moving MNIST dataset for a given config.

    Args:
        config: A configuration object with config values accessible as properties.
            Most likely a FLAGS object.
        split: The dataset split to load.
        shuffle: If true, shuffle the dataset randomly.
        num_digits: Number of digits in the frames.

    Returns:
        dataset: A tf.data.Dataset iterable, where each batched example consists of the shifted 
            forward input digit sequences of shape [time, batch_size, patch_size, patch_size, num_channels],
            their sequence targets of the same shape, the lengths of each sequence and the labels for 
            each bouncing digit, both as int Tensors of shape [batch_size].
        mean: A float Tensor of shape [patch_size, patch_size, num_channels] containing the
            mean loaded from the training set.
    """

    path = os.path.join(config.dataset_path,
                        '{}_num{}_seq{}.npz'.format(config.filename, num_digits, config.seq_length))
    with np.load(path) as data:
        sequences = data[split]
        mean = data['train_mean'].reshape(-1)
        num_examples = len(sequences)
        seq_labels = data[split + '_y']

        def _seq_generator():
            """A generator that yields the digit image video sequences."""
            for i in range(num_examples):
                # Flatten and reshape to remove channel
                seq = sequences[i].reshape(-1, config.patch_size * config.patch_size * config.num_channels)
                yield seq, seq.shape[0], seq_labels[i]

        dataset = tf.data.Dataset.from_generator(
            _seq_generator,
            output_types=(tf.float64, tf.int64, tf.int64),
            output_shapes=([None, config.patch_size * config.patch_size], [], [num_digits])
        )

        if shuffle:
            dataset = dataset.shuffle(num_examples, reshuffle_each_iteration=True)

        # Batch sequences together, padding them to a common length in time
        dataset = dataset.padded_batch(config.batch_size,
                                       padded_shapes=([None, config.patch_size * config.patch_size], [], [num_digits]))

        def _process_seq_data(data, lengths, labels):
            """Generates target distribution and ensures that resulting Tensor is time-major."""
            data = tf.cast(tf.transpose(data, perm=[1, 0, 2]), dtype=tf.float32)
            lengths = tf.cast(lengths, dtype=tf.int32)

            # Mean centre the inputs
            images = data - tf.constant(mean, dtype=tf.float32, shape=[1, 1, mean.shape[0]])
            # Shift the inputs one step forward in time
            # Remove the last timestep so targets and inputs are same length
            images = tf.pad(images, [[1, 0], [0, 0], [0, 0]], mode='CONSTANT')[:-1]
            # Mask out unused timesteps
            images *= tf.expand_dims(tf.transpose(tf.sequence_mask(lengths, dtype=images.dtype)), 2)

            # Reshape into image sequences
            bs = tf.shape(lengths)[0]
            images = tf.reshape(images, [-1, bs, config.patch_size, config.patch_size, config.num_channels])
            img_targets = tf.reshape(data, [-1, bs, config.patch_size, config.patch_size, config.num_channels])

            return images, img_targets, lengths, tf.cast(labels, dtype=tf.int32)

        dataset = (dataset.map(_process_seq_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                   .prefetch(tf.data.experimental.AUTOTUNE))

        train_mu = tf.reshape(mean, [config.patch_size, config.patch_size, config.num_channels])

        return dataset, tf.constant(train_mu, dtype=tf.float32)


def create_lpc_sprites(config, shuffle):
    """Creates the LPC-Sprites dataset for a given config.

    Args:
        config: A configuration object with config values accessible as properties.
            Most likely a FLAGS object.
        shuffle: If true, shuffle the dataset randomly.

    Returns:
        train_set: A tf.data.Dataset iterable, where each batched example consists of the sprites
            sequences of shape [time, batch_size, patch_size, patch_size, num_channels], their sequence
            targets of the same shape, the lengths of each sequence and the labels for each
            action behaviour, both as int Tensors of shape [batch_size].
        test_set: Equivalent to the training set iterable, except for the test split.
    """

    # X_train contains the video frames (N_train, T, patch_size, patch_size, num_channels)
    # A_train contains the attribute labels (N_train, T, 4, 6)
    # D_train contains the action labels (N_train, T, 9)
    X_train, X_test, A_train, A_test, D_train, D_test, X_mu = lpc_sprites.sprites_act(config.dataset_path,
                                                                                      return_labels=True)
    num_train = len(X_train)
    num_test = len(X_test)
    mean = X_mu.reshape(-1)

    def _train_seq_generator():
        """A generator that yields the training sprite video sequences."""
        for i in range(num_train):
            # Flatten
            seq = X_train[i].reshape(-1, config.patch_size * config.patch_size * config.num_channels)
            act_label = np.argmax(D_train[i], axis=1)
            attr_label = np.argmax(A_train[i], axis=2)
            cat_label = np.insert(attr_label[0], 0, act_label[0], axis=0)

            # Assume same label for entire sequence
            yield seq, seq.shape[0], cat_label

    def _test_seq_generator():
        """A generator that yields the test sprite video sequences."""
        for i in range(num_test):
            # Flatten
            seq = X_test[i].reshape(-1, config.patch_size * config.patch_size * config.num_channels)
            act_label = np.argmax(D_test[i], axis=1)
            attr_label = np.argmax(A_test[i], axis=2)
            cat_label = np.insert(attr_label[0], 0, act_label[0], axis=0)

            # Assume same label for entire sequence
            yield seq, seq.shape[0], cat_label

    train_set = tf.data.Dataset.from_generator(
        _train_seq_generator,
        output_types=(tf.float64, tf.int64, tf.int64),
        output_shapes=([None, config.patch_size * config.patch_size * config.num_channels], [], [5])
    )

    test_set = tf.data.Dataset.from_generator(
        _test_seq_generator,
        output_types=(tf.float64, tf.int64, tf.int64),
        output_shapes=([None, config.patch_size * config.patch_size * config.num_channels], [], [5])
    )

    if shuffle:
        train_set = train_set.shuffle(num_train, reshuffle_each_iteration=True)

    # Batch sequences together, padding them to a common length in time
    train_set = train_set.padded_batch(config.batch_size,
                                       padded_shapes=(
                                           [None, config.patch_size * config.patch_size * config.num_channels], [],
                                           [5]))
    test_set = test_set.padded_batch(config.batch_size,
                                     padded_shapes=(
                                         [None, config.patch_size * config.patch_size * config.num_channels], [], [5]))

    def _process_seq_data(data, lengths, labels):
        """Generates target distribution and ensures that resulting Tensor is time-major."""
        data = tf.cast(tf.transpose(data, perm=[1, 0, 2]), dtype=tf.float32)
        lengths = tf.cast(lengths, dtype=tf.int32)

        # Shift the inputs one step forward in time
        # Remove the last timestep so targets and inputs are same length
        images = tf.pad(data, [[1, 0], [0, 0], [0, 0]], mode='CONSTANT')[:-1]
        # Mask out unused timesteps
        images *= tf.expand_dims(tf.transpose(tf.sequence_mask(lengths, dtype=images.dtype)), 2)

        # Reshape into image sequences
        bs = tf.shape(lengths)[0]
        images = tf.reshape(images, [-1, bs, config.patch_size, config.patch_size, config.num_channels])
        img_targets = tf.reshape(data, [-1, bs, config.patch_size, config.patch_size, config.num_channels])

        return images, img_targets, lengths, tf.cast(labels, dtype=tf.int32)

    train_set = (train_set.map(_process_seq_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                 .prefetch(tf.data.experimental.AUTOTUNE))
    test_set = (test_set.map(_process_seq_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                .prefetch(tf.data.experimental.AUTOTUNE))

    train_mu = tf.reshape(mean, [config.patch_size, config.patch_size, config.num_channels])

    return train_set, test_set, tf.constant(train_mu, dtype=tf.float32)


def create_model(config, gen_bias_init=0.0, data_type='binary'):
    """Creates a tf.keras.Model object.

    Args:
        config: A configuration object with config values accessible as properties.
            Most likely a FLAGS object.
        gen_bias_init: Bias initialisation of generative model. Usually set to 
            something sensible like the mean of the training set.
        data_type: Whether modelling real-valued or binary data.

    Returns:
        model: The constructed deep generative model.
    """

    if config.model == 'discvae':
        # Create a discvae.DiSCVAE object
        model = discvae.DiSCVAE(static_latent_size=config.latent_size,
                                dynamic_latent_size=config.dynamic_latent_size,
                                mix_components=config.mixture_components,
                                hidden_size=config.hidden_size,
                                rnn_size=config.rnn_size,
                                num_channels=config.num_channels,
                                decoder_type=data_type,
                                encoded_data_size=config.encoded_data_size,
                                encoded_latent_size=config.encoded_latent_size,
                                sigma_min=1e-5,
                                raw_sigma_bias=0.5,
                                gen_bias_init=gen_bias_init,
                                beta=config.beta,
                                temperature=config.init_temp)
    elif config.model == 'vrnn':
        # Create a vrnn.VRNN object
        model = vrnn.VRNN(latent_size=config.latent_size,
                          hidden_size=config.hidden_size,
                          rnn_size=config.rnn_size,
                          num_channels=config.num_channels,
                          decoder_type=data_type,
                          encoded_data_size=config.encoded_data_size,
                          encoded_latent_size=config.encoded_latent_size,
                          sigma_min=1e-5,
                          raw_sigma_bias=0.5,
                          gen_bias_init=gen_bias_init)
    elif config.model == 'gmvae':
        # Create a gmvae.GMVAE object
        model = gmvae.GMVAE(latent_size=config.latent_size,
                            mix_components=config.mixture_components,
                            hidden_size=config.hidden_size,
                            num_channels=config.num_channels,
                            decoder_type=data_type,
                            encoded_data_size=config.encoded_data_size,
                            sigma_min=1e-5,
                            raw_sigma_bias=0.5,
                            temperature=config.init_temp)
    elif config.model == 'vae':
        # Create a vae.VAE object
        model = vae.VAE(latent_size=config.latent_size,
                        hidden_size=config.hidden_size,
                        num_channels=config.num_channels,
                        decoder_type=data_type,
                        encoded_data_size=config.encoded_data_size,
                        sigma_min=1e-5,
                        raw_sigma_bias=0.5)
    else:
        logging.error("No tf.keras.Model available by the name {}".format(config.model))

    return model


def run_train(config):
    """Runs the training of a deep generative model.

    Args:
        config: A configuration object with config values accessible as properties.
    """

    # Set the random seed for shuffling and sampling
    tf.random.set_seed(config.random_seed)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    # Extract integer GPU IDs
    gpu_ids = list(map(int, config.gpu_num.split(',')))
    # Set the accessible GPUs for training
    try:
        for i in gpu_ids:
            tf.config.experimental.set_memory_growth(gpus[i], True)
            tf.config.experimental.set_visible_devices(gpus[i], 'GPU')

        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

    if config.dataset == 'moving_mnist':
        logging.info("Loading the Moving MNIST dataset...")
        train_ds, train_mu = create_moving_mnist(config, split='train', shuffle=True)
        test_ds, _ = create_moving_mnist(config, split=config.split, shuffle=False)
        # Convert training set mean to logit space for bias initialisation of generative model
        gen_bias_init = -tf.math.log(1. / tf.clip_by_value(train_mu, 0.0001, 0.9999) - 1)
        data_type = 'binary'
    elif config.dataset == 'sprites':
        logging.info("Loading the Sprites dataset...")
        train_ds, test_ds, _ = create_lpc_sprites(config, shuffle=True)
        gen_bias_init = 0.0
        data_type = 'real'

    logging.info("Constructing the unsupervised generative model...")
    model = create_model(config, gen_bias_init, data_type)

    # Set up the optimiser
    opt = tf.keras.optimizers.Adam(config.learning_rate,
                                   clipnorm=config.clip_norm)

    # Set up log directory for saving checkpoints
    logdir = '{}/{}/train/{}/h{}_r{}_f{}_z{}/run{}'.format(
        config.logdir,
        config.dataset,
        config.model,
        config.hidden_size,
        config.rnn_size,
        config.latent_size,
        config.dynamic_latent_size,
        config.random_seed)
    if not tf.io.gfile.exists(logdir):
        logging.info("Creating log directory at {}".format(logdir))
        tf.io.gfile.makedirs(logdir)

    # Checkpoint management
    ckpt = tf.train.Checkpoint(model=model,
                               epoch=tf.Variable(0, trainable=False, dtype=tf.int64),
                               step=tf.Variable(0, trainable=False, dtype=tf.int64),
                               optimizer=opt)
    manager = tf.train.CheckpointManager(ckpt, directory=logdir, max_to_keep=5)
    ckpt.restore(manager.latest_checkpoint)

    if manager.latest_checkpoint:
        logging.info("Restored from {}".format(manager.latest_checkpoint))
    else:
        logging.info("Initialising from scratch...")

    # Create summary writer
    summary_writer = tf.summary.create_file_writer(logdir + '/summaries')
    summary_writer.set_as_default()

    # Boolean flags to switch between models
    is_clustering = (config.model == 'discvae') or (config.model == 'gmvae')
    logging.info("Clustering? {}".format(is_clustering))

    # Training aggregate metrics
    train_elbo = tf.keras.metrics.Mean(name='train_elbo', dtype=tf.float32)
    test_elbo = tf.keras.metrics.Mean(name=config.split + '_elbo', dtype=tf.float32)

    for i in range(config.num_epochs):
        # Reset the metrics at the start of the next epoch
        train_elbo.reset_states()
        test_elbo.reset_states()
        # Lists for predictions and labels
        train_predictions = []
        train_labels = []
        test_predictions = []
        test_labels = []

        # Loop over training set
        for imgs, tgts, lens, labs in train_ds:
            with tf.GradientTape() as tape:
                # Run the model to compute the ELBO objective and reconstructions
                if config.model == 'discvae':
                    elbo, infer_c, _, _, recons = model.run_model(imgs, tgts, lens, ckpt.step,
                                                                  num_samples=config.num_samples)
                    train_predictions.extend(infer_c)
                    train_labels.extend(labs)
                elif config.model == 'vrnn':
                    elbo, _, _, recons = model.run_model(imgs, tgts, lens, ckpt.step, num_samples=config.num_samples)
                elif config.model == 'gmvae':
                    elbo, infer_c, _, recons = model.run_model(tgts, ckpt.step, num_samples=config.num_samples)
                    train_predictions.extend(infer_c)
                    train_labels.extend(labs)
                else:
                    elbo, _, recons = model.run_model(tgts, ckpt.step, num_samples=config.num_samples)

                # Compute gradients of operations with respect to model variables
                grads = tape.gradient(-elbo, model.variables)
                # Maximise ELBO objective
                opt.apply_gradients(list(zip(grads, model.variables)))

                # Update metrics
                train_elbo.update_state(elbo)

                if (ckpt.step % config.summarise_every == 0):
                    # Transpose for summary visualisations
                    inputs_viz = tf.transpose(tgts, perm=[1, 0, 2, 3, 4])
                    recons_viz = tf.transpose(recons, perm=[1, 0, 2, 3, 4])

                    # Only take 4 example reconstructions
                    combined = tf.concat(
                        (inputs_viz[:4], recons_viz[:4]),
                        axis=0)

                    utils.image_seq_summary(combined, 'reconstructions', step=ckpt.step)

            # Increment global step
            ckpt.step.assign_add(1)

        # Loop over test set
        for imgs, tgts, lens, labs in test_ds:
            # Acquire test set metrics from computed loss tensors
            if config.model == 'discvae':
                elbo, infer_c, _, _, _ = model.run_model(imgs, tgts, lens, num_samples=config.num_samples)
                test_predictions.extend(infer_c)
                test_labels.extend(labs)
            elif config.model == 'vrnn':
                elbo, _, _, _ = model.run_model(imgs, tgts, lens, num_samples=config.num_samples)
            elif config.model == 'gmvae':
                elbo, infer_c, _, _ = model.run_model(tgts, num_samples=config.num_samples)
                test_predictions.extend(infer_c)
                test_labels.extend(labs)
            else:
                elbo, _, _ = model.run_model(tgts, num_samples=config.num_samples)

            test_elbo.update_state(elbo)

        # Logging phase
        if is_clustering:
            train_predictions_np = np.array(train_predictions)
            train_labels_np = np.array(train_labels)
            train_acc = utils.cluster_acc(train_predictions_np, train_labels_np[:, 0])

            test_predictions_np = np.array(test_predictions)
            test_labels_np = np.array(test_labels)
            test_acc = utils.cluster_acc(test_predictions_np, test_labels_np[:, 0])

            template = "Epoch {:d}, ELBO: {:.2f}, Test ELBO: {:.2f}, Acc: {:.2f}, Test Acc: {:.2f}"
            aggreg_results = [train_elbo.result(), test_elbo.result(),
                              train_acc * 100, test_acc * 100]
            print(template.format(int(ckpt.epoch),
                                  aggreg_results[0],
                                  aggreg_results[1],
                                  aggreg_results[2],
                                  aggreg_results[3]))
        else:
            template = "Epoch {:d}, ELBO: {:.2f}, Test ELBO: {:.2f}"
            aggreg_results = [train_elbo.result(), test_elbo.result()]
            print(template.format(int(ckpt.epoch), aggreg_results[0], aggreg_results[1]))

        # Save aggregate summaries for logging stage
        with tf.name_scope('aggregates'):
            tf.summary.scalar('train_elbo', aggreg_results[0], step=ckpt.epoch)
            tf.summary.scalar(config.split + '_elbo', aggreg_results[1], step=ckpt.epoch)

            if is_clustering:
                tf.summary.scalar('train_acc', aggreg_results[2], step=ckpt.epoch)
                tf.summary.scalar(config.split + '_acc', aggreg_results[3], step=ckpt.epoch)

        # Checkpoint phase
        is_final_epoch = ((i + 1) == config.num_epochs)
        is_save_epoch = (i % config.save_every == 0)
        if is_save_epoch or is_final_epoch:
            save_path = manager.save()
            print("Saving checkpoint for step {}: {}".format(int(ckpt.step), save_path))

        # Increment epoch
        ckpt.epoch.assign_add(1)
        # Force flush the summary writer during training
        summary_writer.flush()


def run_eval(config):
    """Runs the evaluation of a deep generative model.

    Args:
        config: A configuration object with config values accessible as properties.
    """

    # Set the random seed for shuffling and sampling
    tf.random.set_seed(config.random_seed)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    # Extract integer GPU IDs
    gpu_ids = list(map(int, config.gpu_num.split(',')))
    # Set the accessible GPUs for training
    try:
        for i in gpu_ids:
            tf.config.experimental.set_memory_growth(gpus[i], True)
            tf.config.experimental.set_visible_devices(gpus[i], 'GPU')

        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

    if config.dataset == 'moving_mnist':
        logging.info("Loading the Moving MNIST dataset...")
        dataset, train_mu = create_moving_mnist(config, split=config.split, shuffle=False)
        # Convert training set mean to logit space for bias initialisation of generative model
        gen_bias_init = -tf.math.log(1. / tf.clip_by_value(train_mu, 0.0001, 0.9999) - 1)
        data_type = 'binary'
    elif config.dataset == 'sprites':
        logging.info("Loading the Sprites dataset...")
        _, dataset, _ = create_lpc_sprites(config, shuffle=False)
        train_mu = tf.zeros([config.patch_size, config.patch_size, config.num_channels], dtype=np.float32)
        gen_bias_init = 0.0
        data_type = 'real'

    logging.info("Constructing the unsupervised generative model...")
    model = create_model(config, gen_bias_init, data_type)

    # Set up log directory for loading the pre-trained model
    logdir = '{}/{}/train/{}/h{}_r{}_f{}_z{}/run{}'.format(
        config.logdir,
        config.dataset,
        config.model,
        config.hidden_size,
        config.rnn_size,
        config.latent_size,
        config.dynamic_latent_size,
        config.random_seed)
    if not tf.io.gfile.exists(logdir):
        logging.error("No directory {}".format(logdir))
        sys.exit(1)

    # Checkpoint management
    ckpt = tf.train.Checkpoint(model=model,
                               epoch=tf.Variable(0, trainable=False, dtype=tf.int64))
    manager = tf.train.CheckpointManager(ckpt, directory=logdir, max_to_keep=5)
    ckpt.restore(manager.latest_checkpoint).expect_partial()

    if manager.latest_checkpoint:
        logging.info("Successfully restored from {}".format(manager.latest_checkpoint))
        step = int(ckpt.epoch)
        logging.info("At epoch: {}".format(step))
    else:
        logging.error("Failed to restore the model checkpoint.")
        sys.exit(1)

    # Summary directory for evaluation results
    summary_dir = '{}/{}/{}/{}/h{}_r{}_f{}_z{}/run{}'.format(
        config.logdir,
        config.dataset,
        config.split,
        config.model,
        config.hidden_size,
        config.rnn_size,
        config.latent_size,
        config.dynamic_latent_size,
        config.random_seed)
    qualitative_dir = summary_dir + '/qualitative_results'
    if not tf.io.gfile.exists(qualitative_dir):
        tf.io.gfile.makedirs(qualitative_dir)

    # Create summary writer
    summary_writer = tf.summary.create_file_writer(summary_dir)
    summary_writer.set_as_default()

    # Boolean flags to switch between models
    is_clustering = (config.model == 'discvae') or (config.model == 'gmvae')
    logging.info("Clustering? {}".format(is_clustering))
    is_predictive = (config.model == 'discvae') or (config.model == 'vrnn')
    logging.info("Predictive? {}".format(is_predictive))

    # Evaluation metrics
    elbo = tf.keras.metrics.Mean(name='elbo', dtype=tf.float32)
    bce_loss = torch.nn.BCELoss()
    mse_loss = torch.nn.MSELoss()
    bce_results = []
    mse_results = []

    latents = []
    predictions = []
    labels = []
    # Loop over dataset for a single epoch
    for imgs, tgts, lens, labs_all in dataset:
        # Compute bound estimates
        if config.model == 'discvae':
            elbo_per_batch, infer_c, latent_per_batch, _, _ = model.run_model(imgs, tgts, lens,
                                                                              num_samples=config.num_samples)
            predictions.extend(infer_c)
            # Sample from the inferred clusters of the GMM
            prior_f, _ = model.sample_static_prior(infer_c, num_samples=config.num_samples)
        elif config.model == 'vrnn':
            elbo_per_batch, latent_per_batch, _, _ = model.run_model(imgs, tgts, lens, num_samples=config.num_samples)
        elif config.model == 'gmvae':
            elbo_per_batch, infer_c, _, _ = model.run_model(tgts, num_samples=config.num_samples)
            predictions.extend(infer_c)
            latent_per_batch, _ = model.sample_prior(infer_c, num_samples=config.num_samples)
        else:
            elbo_per_batch, latent_per_batch, _ = model.run_model(tgts, num_samples=config.num_samples)

        # Mean integration of MC samples
        latent = tf.reduce_mean(latent_per_batch, axis=0)
        latents.extend(latent)

        # Extend labels for all models
        labels.extend(labs_all)

        # Update elbo metric
        elbo.update_state(elbo_per_batch)

        # Future prediction evaluation if relevant
        if is_predictive:
            input_prefixes = imgs[:config.prefix_length]
            target_prefixes = tgts[:config.prefix_length]
            prefix_lengths = tf.ones_like(lens) * config.prefix_length

            sample_inputs = imgs[config.prefix_length]
            # Run model on prefix input sequences and then conditionally sample forward in time
            if config.model == 'discvae':
                _, _, prefix_f, final_state, _ = model.run_model(input_prefixes, target_prefixes, prefix_lengths,
                                                                 num_samples=config.num_samples)
                # (sample_length, num_samples, batch_size, patch_size, patch_size, num_channels)
                forecasts = model.sample_model(sample_inputs, final_state, inject_f=prefix_f,
                                               sample_length=config.sample_length, train_mu=train_mu)
            elif config.model == 'vrnn':
                _, _, final_state, _ = model.run_model(input_prefixes, target_prefixes, prefix_lengths,
                                                       num_samples=config.num_samples)
                # (sample_length, num_samples, batch_size, patch_size, patch_size, num_channels)
                forecasts = model.sample_model(sample_inputs, final_state, sample_length=config.sample_length,
                                               train_mu=train_mu)

            # (sample_length, batch_size, patch_size, patch_size, num_channels)
            forecasts = tf.reduce_mean(forecasts, axis=1)
            ground_truth = tgts[config.prefix_length:config.prefix_length + config.sample_length]

            forecasts_torch = torch.from_numpy(np.array(forecasts))
            ground_truth_torch = torch.from_numpy(np.array(ground_truth))

            mse_score = mse_loss(forecasts_torch, ground_truth_torch)
            eps = 1e-4
            forecasts_torch[forecasts_torch < eps] = eps
            forecasts_torch[forecasts_torch > 1 - eps] = 1 - eps
            bce_score = bce_loss(forecasts_torch, ground_truth_torch)
            bce_score = bce_score.item() * config.patch_size * config.patch_size * config.num_channels
            mse_score = mse_score.item() * config.patch_size * config.patch_size * config.num_channels

            bce_results.append(bce_score)
            mse_results.append(mse_score)

    latents_np = np.array(latents)
    labels_np = np.array(labels)
    primary_labels = labels_np[:, 0]

    logging.info("Plotting latent code for inferred latent variable...")
    latent_two = utils.reduce_dimensionality(latents_np)
    if config.dataset == 'moving_mnist':
        utils.tsne_visualise(qualitative_dir, step, latent_two, primary_labels, num_colours=10)
    else:
        utils.tsne_visualise(qualitative_dir, step, latent_two, primary_labels, num_colours=9)

    # Save summaries following evaluation over 'split' set
    tf.summary.scalar(config.split + '/elbo', elbo.result(), step=ckpt.epoch)
    # If a clustering model then report on the metric
    if is_clustering:
        predictions_np = np.array(predictions)
        test_acc = utils.cluster_acc(predictions_np, primary_labels)
        tf.summary.scalar(config.split + '/acc', test_acc * 100, step=ckpt.epoch)
        test_nmi = utils.compute_NMI(predictions_np, primary_labels)
        tf.summary.scalar(config.split + '/nmi', test_nmi, step=ckpt.epoch)

    # If a predictive model then report on relevant metrics
    if is_predictive:
        tf.summary.scalar(config.split + '/bce', np.mean(bce_results), step=ckpt.epoch)
        tf.summary.scalar(config.split + '/mse', np.mean(mse_results), step=ckpt.epoch)

    # Perform full qualitiative analysis only if the model is DiSCVAE
    if config.model == 'discvae':
        logging.info("Plotting density estimates of component samples from model prior...")
        component_f, learnt_prior = model.sample_static_prior(num_samples=250)
        flattened_component_f = tf.reshape(component_f, [-1, config.latent_size])
        component_f_two = utils.reduce_dimensionality(flattened_component_f)
        utils.plot_density(qualitative_dir, step, component_f_two)

        # Create plots of sampled states from fixed 'f' samples
        for imgs, tgts, lens, _ in dataset.take(1):
            # Take random batch example but maintain batch dimension
            rand_batch = np.random.randint(config.batch_size)
            inputs = tf.expand_dims(imgs[:, rand_batch], axis=1)
            targets = tf.expand_dims(tgts[:, rand_batch], axis=1)

            # Run model through this single batched prefix sequence
            input_prefixes = inputs[:config.prefix_length]
            target_prefixes = targets[:config.prefix_length]
            _, infer_c, prefix_f, final_state, _ = model.run_model(input_prefixes, target_prefixes,
                                                                   [config.prefix_length])

            # Sample forward from model to obtain conditionally generated predictions
            sample_inputs = inputs[config.prefix_length]
            ground_truth = targets[config.prefix_length:config.prefix_length + config.sample_length]

            # (sample_length, 1, batch_size, patch_size, patch_size, num_channels)
            forecast_samples = model.sample_model(sample_inputs, final_state, inject_f=prefix_f,
                                                  sample_length=config.sample_length, train_mu=train_mu)

            # Extract mean of 'num_samples' from each cluster (1, K, latent_size)
            inject_f = tf.reduce_mean(component_f[:config.num_samples], axis=0, keepdims=True)
            # Reshape to have single batch size (1, 1, K, latent_size)
            inject_f = tf.expand_dims(inject_f, axis=1)

            # Sampled states from each cluster
            inject_samples = [None] * config.mixture_components
            for k in range(config.mixture_components):
                inject_samples[k] = model.sample_model(sample_inputs, final_state, inject_f=inject_f[:, :, k],
                                                       sample_length=config.sample_length, train_mu=train_mu).numpy()

            logging.info("Plotting sampled sequence ground truth and forecasts...")
            utils.plot_video_sequence(qualitative_dir, step, target_prefixes.numpy(), name='prefixes')
            utils.plot_video_sequence(qualitative_dir, step, ground_truth.numpy(), name='ground_truth')
            utils.plot_video_sequence(qualitative_dir, step, forecast_samples.numpy(), name='forecasts')

            logging.info("Plotting forecasted states of fixed samples from each cluster...")
            utils.plot_k_samples(qualitative_dir, step, ground_truth.numpy(), inject_samples, infer_c[0].numpy(),
                                 num_k_display=config.mixture_components)

            # Feature swapping and fixing for reconstruction only
            recons = model.reconstruct(imgs, tgts, lens, num_samples=config.num_samples)
            swapped_f = model.reconstruct(imgs, tgts, lens, swap_f=True, num_samples=config.num_samples)
            swapped_z = model.reconstruct(imgs, tgts, lens, swap_z=True, num_samples=config.num_samples)

            logging.info("Plotting reconstructions and swapped features...")
            utils.plot_batch_sequence(qualitative_dir, step, tgts.numpy(), name='original')
            utils.plot_batch_sequence(qualitative_dir, step, recons.numpy(), name='reconstructions')
            utils.plot_batch_sequence(qualitative_dir, step, swapped_f.numpy(), name='swapped_f')
            utils.plot_batch_sequence(qualitative_dir, step, swapped_z.numpy(), name='swapped_z')

    # Force flush the summary writer after testing
    summary_writer.flush()
