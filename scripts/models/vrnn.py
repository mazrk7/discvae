from collections import namedtuple

import tensorflow as tf
from models import base


# Define named tuple for VRNN state
VRNNState = namedtuple('VRNNState', 'rnn_out rnn_state latent_encoded')

class VRNN(tf.keras.Model):
    """Variational Recurrent Neural Network (VRNN)."""

    def __init__(self, latent_size, hidden_size, rnn_size, num_channels=1, decoder_type='binary',
                 encoded_data_size=None, encoded_latent_size=None, activation_fn=tf.nn.leaky_relu, sigma_min=0.0,
                 raw_sigma_bias=0.25, gen_bias_init=0.0, name='vrnn'):
        """Constructs a VRNN.

        Args:
            latent_size: The size of the latent variable z_t.
            hidden_size: The size of the hidden layers of the intermediate networks used to
                parameterise the conditional distributions of the VRNN.
            rnn_size: The internal state sizes of the dynamic RNN used for the VRNN backbone.
            num_channels: Number of channels in the output image.
            decoder_type: Whether the decoder should model a Bernoulli or Normal distribution.
            encoded_data_size: The size of the output of the data encoding network.
            encoded_latent_size: The size of the output of the latent encoding network.
            activation_fn: The activation operation applied to intermediate layers (e.g. convolutions).
            sigma_min: The minimum standard deviation of the distribution over the latent state.
            raw_sigma_bias: A scalar that is added to the raw standard deviation output from neural networks.
                Useful for preventing standard deviations close to zero.
            gen_bias_init: A bias to added to the raw output of the decoder convolutional network 
                that parameterises the generative distribution.
            name: The name of the generative tf.keras.Model. Used for scoping.
        """

        super(VRNN, self).__init__()

        # If None for encoder sizes then provide default values
        if encoded_data_size is None:
            encoded_data_size = hidden_size
        if encoded_latent_size is None:
            encoded_latent_size = rnn_size

        # Properties
        self._name = name
        self._latent_size = latent_size

        # The encoder networks used to extract features from inputs x_t and z_t
        self.data_encoder = base.ConvEncoder(encoded_data_size,
                                             filters=64,
                                             activation_fn=activation_fn,
                                             name='data_encoder')
        self.latent_encoder = base.MLPEncoder(encoded_latent_size,
                                              hidden_size=hidden_size,
                                              activation_fn=activation_fn,
                                              name='latent_encoder')

        # The generative distribution p(x_t | z_t, h_t) is conditioned 
        # on the latent state z_t and the RNN hidden state h_t
        if decoder_type == 'binary':
            self.decoder = base.BernoulliDecoder(
                hidden_size=hidden_size,
                filters=64,
                activation_fn=activation_fn,
                bias_init=gen_bias_init,
                channels=num_channels,
                name='decoder')
        else:
            self.decoder = base.NormalDecoder(
                hidden_size=hidden_size,
                filters=64,
                activation_fn=activation_fn,
                channels=num_channels,
                name='decoder')

        # A deterministic RNN 
        self._rnn_cell = tf.compat.v1.keras.layers.LSTMCell(rnn_size)

        # Prior p(z_t | h_t) is a learned Normal, where mu_t and
        # sigma_t are output from a fully connected network that accepts
        # the RNN hidden state h_t as input
        self.prior = base.ConditionalMVN(
            size=latent_size,
            hidden_layer_size=hidden_size,
            hidden_activation_fn=activation_fn,
            sigma_min=sigma_min,
            raw_sigma_bias=raw_sigma_bias,
            name='prior')

        # A callable that implements the inference distribution q(z_t | x_t, h_t)
        self.encoder_z = base.ConditionalMVN(
            size=latent_size,
            hidden_layer_size=hidden_size,
            hidden_activation_fn=activation_fn,
            sigma_min=sigma_min,
            raw_sigma_bias=raw_sigma_bias,
            name='encoder_z')

    def zero_state(self, sample_batch_shape=()):
        """Returns the initial state for the VRNN cell."""

        combined_out_shape = tf.concat(
            (tf.convert_to_tensor(sample_batch_shape, dtype=tf.int32),
             [self._rnn_cell.output_size]), axis=-1)
        combined_latent_shape = tf.concat(
            (tf.convert_to_tensor(sample_batch_shape, dtype=tf.int32),
             [self.latent_encoder.output_size]), axis=-1)

        h0 = tf.zeros(combined_out_shape)
        c0 = tf.zeros(combined_out_shape)

        return VRNNState(
            rnn_out=tf.zeros(combined_out_shape),
            rnn_state=[h0, c0],
            latent_encoded=tf.zeros(combined_latent_shape))

    def sample_prior(self, features, prev_state):
        """Sample the latent prior for a single timestep.

        Args:
            features: A Tensor of shape [num_samples, batch_size, encoded_data_size], 
                batched feature representation of the previous inputs.
            prev_state: The previous state of the model, a VRNNState containing the
                previous RNN output, RNN state and the encoded latent.

        Returns:
            A tuple of a float Tensor of shape [num_samples, batch_size, latent_size]
            representing z_t samples drawn from the latent prior, the next VRNNState, and
            a MultivariateNormalDiag distribution modelling the prior distribution p(z_t | h_t).
        """

        # RNNs conditioned on previous inputs and latent states
        rnn_inputs = tf.concat([features, prev_state.latent_encoded], axis=-1)
        rnn_out, rnn_state = self._rnn_cell(rnn_inputs, prev_state.rnn_state)

        p_zt = self.prior(rnn_out)
        # (num_samples, batch_size, latent_size)
        zt = p_zt.sample()

        # (num_samples, batch_size, latent_size)
        zt_encoded = self.latent_encoder(zt)

        # Update VRNN state
        new_state = VRNNState(rnn_out=rnn_out,
                              rnn_state=rnn_state,
                              latent_encoded=zt_encoded)

        return zt, new_state, p_zt

    def sample_posterior(self, features, rnn_out):
        """Sample the latent posterior for a single timestep.

        Args:
            features: A Tensor of shape [num_samples, batch_size, encoded_data_size], 
                batched feature representation of the current targets.
            rnn_out: The output of the VRNN cell for the current timestep.

        Returns:
            A tuple of a float Tensor of shape [num_samples, batch_size, latent_size]
            representing z_t samples drawn from the encoder and a MultivariateNormalDiag
            distribution modelling the inference distribution q(z_t | x_t, h_t).
        """

        # Encoder accept h_t as input and implements q(z_t | x_t, h_t)
        q_zt = self.encoder_z(features, rnn_out)
        # Sample Gaussian variable z_t
        zt = q_zt.sample()

        return zt, q_zt

    def sample_model(self, initial_inputs, prev_state, sample_length, train_mu):
        """Sample predicted outputs from the model.

        Args:
            initial_inputs: A batch of images represented as a dense Tensor of shape
                [batch_size, image_width, image_height, num_channels] used as initial inputs for the model.
            prev_state: The previous state of the model, a VRNNState containing the
                previous RNN output, RNN state and the encoded latent.
            sample_length: The number of discrete time steps to forward sample for.
            train_mu: Training mean used to centre samples at each time step. Necessary for
                sampled states to be correct.

        Returns:
            sample_predictions: A Tensor of shape [sample_length, batch_size, image_width, image_height] 
                containing the mean sampled predictions from the model.
        """

        # Shapes for sample broadcasting
        sample_shape = tf.shape(input=prev_state.latent_encoded)[:-2]
        broadcast_shape_inputs = tf.concat((sample_shape, [1, 1]), 0)

        # Encoded initial inputs (batch_size, encoded_data_size)
        initial_inputs_encoded = self.data_encoder(initial_inputs)
        # Broadcast to shape (num_samples, batch_size, encoded_data_size)
        initial_inputs_broadcasted = initial_inputs_encoded + tf.zeros(broadcast_shape_inputs)
        _, new_state, _ = self.sample_prior(initial_inputs_broadcasted, prev_state)

        p_xt_given_zt = self.decoder(new_state.latent_encoded, new_state.rnn_out)

        xt = tf.math.subtract(tf.cast(p_xt_given_zt.sample(), dtype=tf.float32), train_mu)

        # The predicted sample array, initialised with samples from first timestep
        sample_predictions_ta = tf.TensorArray(dtype=tf.float32,
                                               size=sample_length,
                                               clear_after_read=True)
        sample_predictions_ta = sample_predictions_ta.write(0, p_xt_given_zt.mean())
        for t in range(sample_length - 1):
            xt_encoded = self.data_encoder(xt)
            _, new_state, _ = self.sample_prior(xt_encoded, new_state)

            # Generative distribution p(x_t | z_t, h_t)
            p_xt_given_zt = self.decoder(new_state.latent_encoded, new_state.rnn_out)
            # Re-centre for each new sampled state
            xt = tf.math.subtract(tf.cast(p_xt_given_zt.sample(), dtype=tf.float32), train_mu)

            sample_predictions_ta = sample_predictions_ta.write(t + 1, p_xt_given_zt.mean())

        return sample_predictions_ta.stack()

    def run_model(self, inputs, targets, lengths, training_step=None, num_samples=1):
        """Runs the model and computes importance weights across entire sequences.

        Args:
            inputs: A batch of image sequences represented as a dense Tensor 
                of shape [time, batch_size, image_width, image_height, num_channels].
            targets: A batch of target sequences represented as a dense Tensor 
                of shape [time, batch_size, image_width, image_height, num_channels].
            lengths: An int Tensor of shape [batch_size] representing the lengths of 
                each sequence in the batch.
            training_step: An int Tensor representing the step in the training
                process. If None, then no logging takes place.
            num_samples: Number of samples to use for posterior approximations.

        Returns:
            elbo: A weight Tensor of shape [batch_size] representing multi-sample
                bound estimates of the log marginal probability on observations.
            infer_z: The inferred latent variable determined by q(z_t | x_t, h_t).
            final_state: The final VRNN state returned by running the model.
            recons: A Tensor of shape [max_seq_len, batch_size, image_width, image_height, num_channels]
                representing the reconstructions of the input image sequences.
        """

        # Encoded inputs & targets (time, batch_size, encoded_data_size)
        inputs_encoded = self.data_encoder(inputs)
        targets_encoded = self.data_encoder(targets)

        # Batch size and number of timesteps for dynamic unrolling in encoder
        batch_size = tf.shape(lengths)[0]
        max_seq_len = tf.reduce_max(lengths)
        sample_batch_shape = [num_samples, batch_size]
        combined_latent_shape = tf.concat((sample_batch_shape, [self._latent_size]), axis=-1)

        # Initial VRNN and latent states
        vrnn_state = self.zero_state(sample_batch_shape)
        zt = tf.zeros(combined_latent_shape)

        # Loss tensors and reconstruction evaluations
        nkld_zt = tf.zeros(sample_batch_shape)
        log_lld = tf.zeros(sample_batch_shape)
        recon_ta = tf.TensorArray(dtype=tf.float32, size=max_seq_len)

        broadcast_shape_inputs = tf.concat(([num_samples], [1, 1, 1]), 0)
        inputs_broadcasted = inputs_encoded + tf.zeros(broadcast_shape_inputs)
        targets_broadcasted = targets_encoded + tf.zeros(broadcast_shape_inputs)
        for t in range(max_seq_len):
            # RNNs conditioned on previous inputs and latent states
            rnn_inputs = tf.concat([inputs_broadcasted[:, t], vrnn_state.latent_encoded], axis=-1)
            # RNN outputs of shape (num_samples, batch_size, rnn_size)
            rnn_out, rnn_state = self._rnn_cell(rnn_inputs, vrnn_state.rnn_state)

            p_zt = self.prior(rnn_out)
            # (num_samples, batch_size, latent_size)
            zt, q_zt = self.sample_posterior(targets_broadcasted[:, t], rnn_out)

            # Encode z_t samples (num_samples, batch_size, encoded_latent_size)
            zt_encoded = self.latent_encoder(zt)

            # Generative distribution p(x_t | z_t, h_t)
            p_xt_given_zt = self.decoder(zt_encoded, rnn_out)

            # Summed across time
            nkld_zt += (p_zt.log_prob(zt) - q_zt.log_prob(zt))
            log_lld += p_xt_given_zt.log_prob(targets[t])

            # Write array of integrated sample reconstructions
            recon_ta = recon_ta.write(t, tf.reduce_mean(p_xt_given_zt.mean(), axis=0))

            # Update VRNN state
            vrnn_state = VRNNState(rnn_out=rnn_out,
                                   rnn_state=rnn_state,
                                   latent_encoded=zt_encoded)

        # Need to maximise the ELBO with respect to these weights over sequences
        elbo_local = log_lld + nkld_zt
        elbo = tf.reduce_mean(elbo_local)

        # Set summaries for independent terms of ELBO during training
        if training_step is not None:
            with tf.name_scope('train'):
                tf.summary.scalar('nll', -tf.reduce_mean(log_lld), step=training_step)
                tf.summary.scalar('kld_zt', -tf.reduce_mean(nkld_zt), step=training_step)

        return elbo, zt, vrnn_state, recon_ta.stack()
