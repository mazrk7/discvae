"""Disentangled Sequence Clustering Variational Autoencoder (DiSCVAE).

A probabilistic framework for unsupervised clustering of disentangled 
sequence representations, where disentangling sequences refers to modelling
separate latent variables for the static and dynamic features i.e. 'f' and 'z'.

Generative model:
p(x_{<=T}, z_{<=T}, f, y) = p(y) p(f | y) prod_{t=1}^T p(z_t | x_{<t}, z_{<t}) p(x_t | z_{<=t}, x_{<t}, f)

Inference model:
q(z_{<=T}, f, y | x_{<=T}) = q(y | x_{<=T}) q(f | x_{<=T}, y) prod_{t=1}^T q(z_t | x_{<=t}, z_{<t})
"""

from collections import namedtuple

import tensorflow as tf
import utils
from models import base


# Define named tuple for VRNN backbone of DiSCVAE
VRNNState = namedtuple('VRNNState', 'rnn_out rnn_state latent_encoded')

class DiSCVAE(tf.keras.Model):
    """Disentangled Sequence Clustering Variational Autoencoder (DiSCVAE)."""

    def __init__(self, static_latent_size, dynamic_latent_size, mix_components, hidden_size, rnn_size, num_channels=1,
                 decoder_type='binary', encoded_data_size=None, encoded_latent_size=None, activation_fn=tf.nn.leaky_relu,
                 sigma_min=0.0, raw_sigma_bias=0.25, gen_bias_init=0.0, beta=1.0, temperature=1.0, name='discvae'):
        """Constructs a DiSCVAE.

        Args:
            static_latent_size: The size of the static latent variable f.
            dynamic_latent_size: The size of the dynamic latent variable z_t.
            mix_components: The number of mixture components, also size of variable y.
            hidden_size: The size of the hidden layers of the intermediate networks used to
                parameterise the conditional distributions of the DiSCVAE.
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
            beta: Beta parameter in the b-VAE.
            temperature: Degree of how approximately discrete the Gumbel-Softmax distribution 
                that models the discrete latent variable y should be.
            name: The name of the generative tf.keras.Model. Used for scoping.
        """

        super(DiSCVAE, self).__init__()

        # If None for encoder sizes then provide default values
        if encoded_data_size is None:
            encoded_data_size = hidden_size
        if encoded_latent_size is None:
            encoded_latent_size = rnn_size

        # Properties
        self._name = name
        self._static_latent_size = static_latent_size
        self._dynamic_latent_size = dynamic_latent_size
        self._K = mix_components
        self._beta = beta
        self._tau = temperature

        # The encoder networks used to extract features from inputs x_t and z_t
        self.data_encoder = base.ConvEncoder(encoded_data_size,
                                             filters=64,
                                             activation_fn=activation_fn,
                                             name='data_encoder')
        self.latent_encoder = base.MLPEncoder(encoded_latent_size,
                                              hidden_size=hidden_size,
                                              activation_fn=activation_fn,
                                              name='latent_encoder')

        # The generative distribution p(x_t | z_t, h_t, f) is conditioned on the latent 
        # state z_t, the RNN hidden state h_t and the static latent variable f
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

        # The DiSCVAE has a VRNN as its backbone, represented by a deterministic RNN
        self._rnn_cell = tf.compat.v1.keras.layers.LSTMCell(rnn_size)
        # Bi-directional LSTM that feeds its encoded outputs into the static encoder
        self._bi_lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(hidden_size, time_major=True),
            merge_mode='sum')

        # Prior p(f | y) is a learnt mixture of Gaussians
        self.static_prior = base.ConditionalMVN(
            size=static_latent_size,
            hidden_layer_size=None,
            sigma_min=sigma_min,
            raw_sigma_bias=raw_sigma_bias,
            name='static_prior')
        # Prior p(z_t | h_t) is a learned Normal, where mu_t and
        # sigma_t are output from a fully connected network that accepts
        # the RNN hidden state h_t as input
        self.dynamic_prior = base.ConditionalMVN(
            size=dynamic_latent_size,
            hidden_layer_size=hidden_size,
            hidden_activation_fn=activation_fn,
            sigma_min=sigma_min,
            raw_sigma_bias=raw_sigma_bias,
            name='dynamic_prior')

        # A callable that implements the inference distribution q(y | x_{<=T})
        # Use the Gumbel-Softmax distribution to model the categorical variable y
        self.static_encoder_y = base.ConditionalRelaxedCategorical(
            size=mix_components,
            hidden_layer_size=None,
            temperature=temperature,
            name='static_encoder_y')
        # A callable that implements the inference distribution q(f | x_{<=T}, y)
        self.static_encoder_f = base.ConditionalMVN(
            size=static_latent_size,
            hidden_layer_size=hidden_size,
            hidden_activation_fn=activation_fn,
            sigma_min=sigma_min,
            raw_sigma_bias=raw_sigma_bias,
            name='static_encoder_f')
        # A callable that implements the inference distribution q(z_t | x_t, h_t)
        self.dynamic_encoder = base.ConditionalMVN(
            size=dynamic_latent_size,
            hidden_layer_size=hidden_size,
            hidden_activation_fn=activation_fn,
            sigma_min=sigma_min,
            raw_sigma_bias=raw_sigma_bias,
            name='dynamic_encoder')

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

    def sample_static_prior(self, modes=None, num_samples=1):
        """Sample the GMM static latent prior.
            
        Args:
            modes: Inferred modes of shape [batch_size] to define prior on y.
                If None, then sample sequences from each component.
            num_samples: Number of samples to draw from the prior.

        Returns:
            A tuple of a float Tensor of shape [num_samples, batch_size, static_latent_size]
            representing f samples drawn from components of the GMM, and a MultivariateNormalDiag 
            distribution modelling the GMM prior distribution p(f | y).
        """

        if modes is None:
            # Generate outputs over each component in GMM
            modes = tf.range(0, self._K)

        # batch_size either K or dependent on 'modes' shape
        y = tf.cast(tf.one_hot(modes, self._K), dtype=tf.float32)

        p_f_given_y = self.static_prior(y)
        # (num_samples, batch_size, static_latent_size)
        f = p_f_given_y.sample(num_samples)

        return f, p_f_given_y

    def sample_static_posterior_y(self, bi_lstm_out, num_samples=1):
        """Sample the static latent posterior for variable y.

        Args:
            bi_lstm_out: The output of the bi-directional LSTM encoding over
                the entire input sequence, shape [batch_size, hidden_size].
            num_samples: Number of samples to draw from the latent distribution.

        Returns:
            A tuple of y samples with shape [num_samples, batch_size, mix_components], 
            as well as the inference distribution q(y | x_{<=T}).
        """

        # Encoder for y accepts last bi-directional RNN output 
        # and implements q(y | x_{<=T})
        q_y = self.static_encoder_y(bi_lstm_out)
        # Sample categorical variable y from the Gumbel-Softmax distribution
        y = q_y.sample(num_samples)

        return y, q_y

    def sample_static_posterior_f(self, bi_lstm_out, y):
        """Sample the static latent posterior for variable f.

        Args:
            bi_lstm_out: The output of the bi-directional LSTM encoding over
                the entire input sequence, shape [batch_size, hidden_size].
            y: Relaxed one-hot samples of shape [num_samples, batch_size, mix_components].
                Required in order to select the components of the GMM prior. 

        Returns:
            A tuple of f samples with shape [num_samples, batch_size, static_latent_size],
            as well as the inference distribution q(f | x_{<=T}, y).
        """

        sample_shape_y = tf.shape(input=y)[:-2]
        broadcast_shape_inputs = tf.concat((sample_shape_y, [1, 1]), 0)
        bi_lstm_out = bi_lstm_out + tf.zeros(broadcast_shape_inputs)

        # Encoder for f accepts both bi-directional RNN outputs and y
        # as inputs to implement q(f | x_{<=T}, y)
        q_f = self.static_encoder_f(bi_lstm_out, y)
        # Sample Gaussian variable f to be used in later conditioning
        f = q_f.sample()

        return f, q_f

    def sample_dynamic_prior(self, features, prev_state):
        """Sample the dynamic latent prior for a single timestep.

        Args:
            features: A Tensor of shape [num_samples, batch_size, encoded_data_size], 
                batched feature representation of the previous inputs.
            prev_state: The previous state of the model, a VRNNState containing the
                previous RNN output, RNN state and the encoded latent.

        Returns:
            A tuple of a float Tensor of shape [num_samples, batch_size, dynamic_latent_size]
            representing z_t samples drawn from the dynamic prior, the next VRNNState,
            and a MultivariateNormalDiag distribution modelling the prior distribution p(z_t | h_t).
        """

        # RNNs conditioned on previous inputs and latent states
        rnn_inputs = tf.concat([features, prev_state.latent_encoded], axis=-1)
        rnn_out, rnn_state = self._rnn_cell(rnn_inputs, prev_state.rnn_state)

        p_zt = self.dynamic_prior(rnn_out)
        # (num_samples, batch_size, dynamic_latent_size)
        zt = p_zt.sample()

        # (num_samples, batch_size, encoded_latent_size)
        zt_encoded = self.latent_encoder(zt)

        # Update VRNN state
        new_state = VRNNState(rnn_out=rnn_out,
                              rnn_state=rnn_state,
                              latent_encoded=zt_encoded)

        return zt, new_state, p_zt

    def sample_dynamic_posterior(self, features, rnn_out):
        """Sample the dynamic latent posterior for a single timestep.

        Args:
            features: A Tensor of shape [num_samples, batch_size, encoded_data_size], 
                batched feature representation of the current targets.
            rnn_out: The output of the VRNN cell for the current timestep.

        Returns:
            A tuple of a float Tensor of shape [num_samples, batch_size, dynamic_latent_size]
            representing z_t samples drawn from the dynamic encoder and a MultivariateNormalDiag
            distribution modelling the inference distribution q(z_t | x_t, h_t).
        """

        # Dynamic encoder accept h_t as input and implements q(z_t | x_t, h_t)
        q_zt = self.dynamic_encoder(features, rnn_out)
        # Sample Gaussian variable z_t
        zt = q_zt.sample()

        return zt, q_zt

    def sample_model(self, initial_inputs, prev_state, inject_f, sample_length, train_mu):
        """Sample predicted outputs from the model.

        Args:
            initial_inputs: A batch of images represented as a dense Tensor of shape
                [batch_size, image_width, image_height, num_channels] used as initial inputs for the model.
            prev_state: The previous state of the model, a VRNNState containing the
                previous RNN output, RNN state and the encoded latent.
            inject_f: The latent static variable f of shape [num_samples, batch_size, static_latent_size],
                has either been sampled from a specific cluster or inferred from a prefix sequence.
            sample_length: The number of discrete time steps to forward sample for.
            train_mu: Training mean used to centre samples at each time step. Necessary for
                sampled states to be correct.

        Returns:
            sample_predictions: A Tensor of shape [sample_length, batch_size, image_width, image_height] 
                containing the mean sampled predictions from the model.
        """

        # Shapes for sample broadcasting
        sample_shape_f = tf.shape(input=inject_f)[:-2]
        broadcast_shape_inputs = tf.concat((sample_shape_f, [1, 1]), 0)

        # Encoded initial inputs (batch_size, encoded_data_size)
        initial_inputs_encoded = self.data_encoder(initial_inputs)
        # Broadcast to shape (num_samples, batch_size, encoded_data_size)
        initial_inputs_broadcasted = initial_inputs_encoded + tf.zeros(broadcast_shape_inputs)
        _, new_state, _ = self.sample_dynamic_prior(initial_inputs_broadcasted, prev_state)

        p_xt_given_f_zt = self.decoder(new_state.latent_encoded, new_state.rnn_out, inject_f)

        # Mean-centring required to feed inputs into model
        xt = tf.math.subtract(tf.cast(p_xt_given_f_zt.sample(), dtype=tf.float32), train_mu)

        # The predicted sample array, initialised with samples from first timestep
        sample_predictions_ta = tf.TensorArray(dtype=tf.float32,
                                               size=sample_length,
                                               clear_after_read=True)
        sample_predictions_ta = sample_predictions_ta.write(0, p_xt_given_f_zt.mean())
        for t in range(sample_length - 1):
            xt_encoded = self.data_encoder(xt)
            _, new_state, _ = self.sample_dynamic_prior(xt_encoded, new_state)

            # Generative distribution p(x_t | z_t, h_t, f)
            p_xt_given_f_zt = self.decoder(new_state.latent_encoded, new_state.rnn_out, inject_f)
            # Re-centre for each new sampled state
            xt = tf.math.subtract(tf.cast(p_xt_given_f_zt.sample(), dtype=tf.float32), train_mu)

            sample_predictions_ta = sample_predictions_ta.write(t + 1, p_xt_given_f_zt.mean())

        return sample_predictions_ta.stack()

    def reconstruct(self, inputs, targets, lengths, num_samples=1, swap_f=False, swap_z=False):
        """Reconstructs in the input sequences.

        Args:
            inputs: A batch of image sequences represented as a dense Tensor
                of shape [time, batch_size, image_width, image_height, num_channels].
            targets: A batch of target sequences represented as a dense Tensor
                of shape [time, batch_size, image_width, image_height, num_channels].
            lengths: An int Tensor of shape [batch_size] representing the lengths of
                each sequence in the batch.
            num_samples: Number of samples to use for posterior approximations.

        Returns:
            recons: A Tensor of shape [max_seq_len, batch_size, image_width, image_height, num_channels]
                representing the reconstructions of the input image sequences.
        """

        # Encoded inputs & targets (time, batch_size, encoded_data_size)
        inputs_encoded = self.data_encoder(inputs)
        targets_encoded = self.data_encoder(targets)

        # (batch_size, hidden_size)
        bi_lstm_out = self._bi_lstm(targets_encoded)

        # (num_samples, batch_size, mix_components)
        y, q_y = self.sample_static_posterior_y(bi_lstm_out, num_samples)
        y = tf.cast(tf.one_hot(tf.argmax(y, axis=-1), self._K), y.dtype)

        # (num_samples, batch_size, static_latent_size)
        f, q_f = self.sample_static_posterior_f(bi_lstm_out, y)

        if swap_f:
            f = tf.reverse(f, axis=[1])

        # Batch size and number of timesteps for dynamic unrolling in encoder
        batch_size = tf.shape(lengths)[0]
        max_seq_len = tf.reduce_max(lengths)
        sample_batch_shape = [num_samples, batch_size]

        # Initial VRNN state
        vrnn_state = self.zero_state(sample_batch_shape)

        # Loss tensors and reconstruction evaluations
        recon_ta = tf.TensorArray(dtype=tf.float32, size=max_seq_len)

        broadcast_shape_inputs = tf.concat(([num_samples], [1, 1, 1]), 0)
        inputs_broadcasted = inputs_encoded + tf.zeros(broadcast_shape_inputs)
        targets_broadcasted = targets_encoded + tf.zeros(broadcast_shape_inputs)
        for t in range(max_seq_len):
            # RNNs conditioned on previous inputs and latent states
            rnn_inputs = tf.concat([inputs_broadcasted[:, t], vrnn_state.latent_encoded], axis=-1)
            # RNN outputs of shape (num_samples, batch_size, rnn_size)
            rnn_out, rnn_state = self._rnn_cell(rnn_inputs, vrnn_state.rnn_state)

            # (num_samples, batch_size, dynamic_latent_size)
            zt, q_zt = self.sample_dynamic_posterior(targets_broadcasted[:, t], rnn_out)

            if swap_z:
                zt = tf.reverse(zt, axis=[1])

            # Encode z_t samples (num_samples, batch_size, encoded_latent_size)
            zt_encoded = self.latent_encoder(zt)

            # Generative distribution p(x_t | z_t, h_t, f)
            p_xt_given_f_zt = self.decoder(zt_encoded, rnn_out, f)

            # Write array of integrated sample reconstructions
            recon_ta = recon_ta.write(t, tf.reduce_mean(p_xt_given_f_zt.mean(), axis=0))

            # Update VRNN state
            vrnn_state = VRNNState(rnn_out=rnn_out,
                                   rnn_state=rnn_state,
                                   latent_encoded=zt_encoded)

        return recon_ta.stack()

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
                process. If None, then no logging takes place and straight-through
                Gumbel-Softmax samples are drawn.
            num_samples: Number of samples to use for posterior approximations.

        Returns:
            elbo: A weight Tensor of shape [batch_size] representing multi-sample
                bound estimates of the log marginal probability on observations.
            infer_c: The inferred clusters of input sequences, determined by
                taking the argmax{ q(y | x_{<=T}) }.
            infer_f: The inferred static variable determined by q(f | x_{<=T}, y).
            final_state: The final VRNN state returned by running the model.
            recons: A Tensor of shape [max_seq_len, batch_size, image_width, image_height, num_channels]
                representing the reconstructions of the input image sequences.
        """

        # Encoded inputs & targets (time, batch_size, encoded_data_size)
        inputs_encoded = self.data_encoder(inputs)
        targets_encoded = self.data_encoder(targets)

        # (batch_size, hidden_size)
        bi_lstm_out = self._bi_lstm(targets_encoded)

        # (num_samples, batch_size, mix_components)
        y, q_y = self.sample_static_posterior_y(bi_lstm_out, num_samples)
        q_y_logits = q_y.distribution.logits
        # p_y_logits = tf.ones_like(q_y_logits) * 1./self._K

        # Take 'hard' y if not training
        if training_step is None:
            y = tf.cast(tf.one_hot(tf.argmax(y, axis=-1), self._K), y.dtype)
            # Analytical KL with Categorical prior
            '''p_cat_y = tfp.distributions.OneHotCategorical(logits=p_y_logits)
            q_cat_y = tfp.distributions.OneHotCategorical(logits=q_y_logits)
            
            nkld_y = tfp.distributions.kl_divergence(p_cat_y, q_cat_y)
        else:
            # Monte Carlo KL with Relaxed prior
            p_y = tfp.distributions.ExpRelaxedOneHotCategorical(self._tau, logits=p_y_logits)
            nkld_y = p_y.log_prob(y) - q_y.log_prob(y)'''

        p_f_given_y = self.static_prior(y)
        # (num_samples, batch_size, static_latent_size)
        f, q_f = self.sample_static_posterior_f(bi_lstm_out, y)

        # Batch size and number of timesteps for dynamic unrolling in encoder
        batch_size = tf.shape(lengths)[0]
        max_seq_len = tf.reduce_max(lengths)
        sample_batch_shape = [num_samples, batch_size]

        # Initial VRNN state
        vrnn_state = self.zero_state(sample_batch_shape)

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

            p_zt = self.dynamic_prior(rnn_out)
            # (num_samples, batch_size, dynamic_latent_size)
            zt, q_zt = self.sample_dynamic_posterior(targets_broadcasted[:, t], rnn_out)

            # Encode z_t samples (num_samples, batch_size, encoded_latent_size)
            zt_encoded = self.latent_encoder(zt)

            # Generative distribution p(x_t | z_t, h_t, f)
            p_xt_given_f_zt = self.decoder(zt_encoded, rnn_out, f)

            # Summed across time
            nkld_zt += (p_zt.log_prob(zt) - q_zt.log_prob(zt))
            log_lld += p_xt_given_f_zt.log_prob(targets[t])

            # Write array of integrated sample reconstructions
            recon_ta = recon_ta.write(t, tf.reduce_mean(p_xt_given_f_zt.mean(), axis=0))

            # Update VRNN state
            vrnn_state = VRNNState(rnn_out=rnn_out,
                                   rnn_state=rnn_state,
                                   latent_encoded=zt_encoded)

        # Compute regularisation term for static variable y
        entropy = utils.entropy(q_y_logits, tf.nn.softmax(q_y_logits))

        # Compute KL-divergence term for variational approximation on the f posterior
        nkld_f = p_f_given_y.log_prob(f) - q_f.log_prob(f)

        # Need to maximise the ELBO with respect to these weights over sequences
        elbo_local = log_lld + self._beta * (nkld_zt + nkld_f) + entropy
        elbo = tf.reduce_mean(elbo_local)

        # Set summaries for independent terms of ELBO during training
        if training_step is not None:
            with tf.name_scope('train'):
                tf.summary.scalar('nll', -tf.reduce_mean(log_lld), step=training_step)
                tf.summary.scalar('kld_zt', -tf.reduce_mean(nkld_zt), step=training_step)
                tf.summary.scalar('kld_f', -tf.reduce_mean(nkld_f), step=training_step)
                tf.summary.scalar('entropy', tf.reduce_mean(entropy), step=training_step)
                # tf.summary.scalar('kld_y', -tf.reduce_mean(nkld_y), step=training_step)

        return elbo, tf.argmax(q_y_logits, axis=-1), f, vrnn_state, recon_ta.stack()
