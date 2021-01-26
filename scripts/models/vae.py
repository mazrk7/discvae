import tensorflow as tf
import tensorflow_probability as tfp

from models import base

tfd = tfp.distributions


class VAE(tf.keras.Model):
    """Variational Autoencoder (VAE) over sequential data."""

    def __init__(self, latent_size, hidden_size, num_channels=1, decoder_type='binary', encoded_data_size=None,
                 activation_fn=tf.nn.leaky_relu, sigma_min=0.0, raw_sigma_bias=0.25, gen_bias_init=0.0, name='vae'):
        """Constructs a VAE.

        Args:
            latent_size: The size of the latent variable z.
            hidden_size: The size of the hidden layers of the intermediate networks used to
                parameterise the conditional distributions of the VAE.
            num_channels: Number of channels in the output image.
            decoder_type: Whether the decoder should model a Bernoulli or Normal distribution.
            encoded_data_size: The size of the output of the data encoding network.
            activation_fn: The activation operation applied to intermediate layers (e.g. convolutions).
            sigma_min: The minimum standard deviation of the distribution over the latent state.
            raw_sigma_bias: A scalar that is added to the raw standard deviation output from neural networks.
                Useful for preventing standard deviations close to zero.
            gen_bias_init: A bias to added to the raw output of the decoder convolutional network 
                that parameterises the generative distribution.
            name: The name of the generative tf.keras.Model. Used for scoping.
        """

        super(VAE, self).__init__()

        # If None for encoder sizes then provide default values
        if encoded_data_size is None:
            encoded_data_size = hidden_size

        # Properties
        self._name = name
        self._latent_size = latent_size

        # The encoder networks used to extract features from inputs x
        self.data_encoder = base.ConvEncoder(encoded_data_size,
                                             filters=64,
                                             activation_fn=activation_fn,
                                             name='data_encoder')

        # The generative distribution p(x | z)
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

        # Bi-directional LSTM that feeds its outputs into the stochastic encoder
        self._bi_lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(hidden_size, time_major=True, return_sequences=True),
            merge_mode='sum')

        # Prior p(z) is a multivariate Gaussian prior
        untransformed_stddev = tf.random.normal([latent_size], mean=raw_sigma_bias, stddev=0.1)
        self.prior = tfd.MultivariateNormalDiag(
            loc=tf.random.normal([latent_size], stddev=0.1),
            scale_diag=tf.nn.softplus(untransformed_stddev) + sigma_min)

        # A callable that implements the inference distribution q(z | x)
        self.encoder_z = base.ConditionalMVN(
            size=latent_size,
            hidden_layer_size=hidden_size,
            hidden_activation_fn=activation_fn,
            sigma_min=sigma_min,
            raw_sigma_bias=raw_sigma_bias,
            name='encoder_z')

    def sample_prior(self, num_samples=1):
        """Sample the Gaussian prior.
            
        Args:
            num_samples: Number of samples to draw from the prior.

        Returns:
            A tuple of a float Tensor of shape [num_samples, batch_size, latent_size]
            representing z samples drawn from the Gaussian prior, and a 
            MultivariateNormalDiag distribution modelling the prior distribution p(z).
        """

        p_z = self.prior
        # (num_samples, batch_size, latent_size)
        z = p_z.sample(num_samples)

        return z, p_z

    def sample_posterior(self, bi_lstm_out, num_samples=1):
        """Sample the latent posterior for variable z.

        Args:
            bi_lstm_out: The output of the bi-directional LSTM encoding over
                the entire input sequence, shape [time, batch_size, hidden_size].
            num_samples: Number of samples to draw from the latent distribution.

        Returns:
            A tuple of z samples with shape [num_samples, time, batch_size, latent_size], 
            as well as the inference distribution q(z | x).
        """

        # Encoder for z accepts single image in video sequence
        q_z = self.encoder_z(bi_lstm_out)
        # Sample Gaussian variable z
        z = q_z.sample(num_samples)

        return z, q_z

    def run_model(self, inputs, training_step=None, num_samples=1):
        """Runs the model and computes importance weights across entire sequences.

        Args:
            inputs: A batch of image sequences represented as a dense Tensor 
                of shape [time, batch_size, image_width, image_height, num_channels].
            training_step: An int Tensor representing the step in the training
                process. If None, then no logging takes place.
            num_samples: Number of samples to use when running through the model.

        Returns:
            elbo: A weight Tensor of shape [batch_size] representing multi-sample
                bound estimates of the log marginal probability on observations.
            infer_z: The inferred latent variable determined by q(z | x).
            recons: A Tensor of shape [batch_size, image_width, image_height, num_channels]
                representing the reconstructions of the input images.
        """

        # Encoded inputs (time, batch_size, encoded_data_size)
        inputs_encoded = self.data_encoder(inputs)

        # (time, batch_size, hidden_size)
        bi_lstm_out = self._bi_lstm(inputs_encoded)

        # (num_samples, time, batch_size, latent_size)
        z, q_z = self.sample_posterior(bi_lstm_out, num_samples)

        # Generative distribution p(x | z)
        p_x_given_z = self.decoder(z)

        # Compute KL-divergence term for variational approximation on the z posterior
        nkld_zt = tf.reduce_sum(self.prior.log_prob(z) - q_z.log_prob(z), axis=1)

        # Compute log-likelihood term for reconstruction loss
        log_lld = tf.reduce_sum(p_x_given_z.log_prob(inputs), axis=1)

        # Need to maximise the ELBO with respect to these weights over sequences
        elbo_local = log_lld + nkld_zt
        elbo = tf.reduce_mean(elbo_local)

        # Set summaries for independent terms of ELBO during training
        if training_step is not None:
            with tf.name_scope('train'):
                tf.summary.scalar('nll', -tf.reduce_mean(log_lld), step=training_step)
                tf.summary.scalar('kld_zt', -tf.reduce_mean(nkld_zt), step=training_step)

        return elbo, tf.reduce_sum(z, axis=1), tf.reduce_mean(p_x_given_z.mean(), axis=0)
