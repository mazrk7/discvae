import functools

import tensorflow as tf
import tensorflow_probability as tfp

tfk = tf.keras
tfkl = tfk.layers
tfd = tfp.distributions


class ConditionalMVN(tfk.Model):
    """A MultivariateNormalDiag distribution conditioned on outputs of a fully connected network."""

    def __init__(self, size, hidden_layer_size=None, hidden_activation_fn=tf.nn.relu,
                 sigma_min=1e-5, raw_sigma_bias=0.25, name='cond_mvn'):
        """Creates a conditional MultivariateNormalDiag distribution.

        Args:
            size: The dimension of the random variable.
            hidden_layer_size: The size of the hidden layer of the fully connected
                network used to condition the distribution on the inputs.
            hidden_activation_fn: The activation function to use on the hidden layers
                of the fully connected network.
            sigma_min: The minimum standard deviation allowed, a scalar.
            raw_sigma_bias: A scalar that is added to the raw standard deviation
                output to prevent standard deviations close to 0.
            name: The name of this Model, used for scoping.
        """

        super(ConditionalMVN, self).__init__()

        self._name = name

        self._size = size
        self._hidden_size = hidden_layer_size

        self._sigma_min = sigma_min
        self._raw_sigma_bias = raw_sigma_bias

        if hidden_layer_size is not None:
            self._feature_layer = tfkl.Dense(hidden_layer_size, activation=hidden_activation_fn)

        self._out_layer = tfkl.Dense(2 * size)

    def condition(self, tensor_list, **unused_kwargs):
        """Computes the parameters of a MultivariateNormalDiag based on inputs."""

        inputs = tf.concat(tensor_list, axis=-1)

        if self._hidden_size is not None:
            inputs = self._feature_layer(inputs)

        outs = self._out_layer(inputs)

        mu, sigma = tf.split(outs, 2, axis=-1)
        sigma = tf.maximum(tf.nn.softplus(sigma + self._raw_sigma_bias), self._sigma_min)

        return mu, sigma

    def call(self, *args, **kwargs):
        """Creates a MultivariateNormalDiag distribution conditioned on the inputs."""

        mu, sigma = self.condition(args, **kwargs)

        return tfd.MultivariateNormalDiag(
            loc=mu,
            scale_diag=sigma)

    @property
    def output_size(self):
        """The output size of the random variable."""
        return self._size


class ConditionalRelaxedCategorical(tfk.Model):
    """A RelaxedOneHotCategorical distribution conditioned on outputs of a fully connected network."""

    def __init__(self, size, hidden_layer_size=None, hidden_activation_fn=tf.nn.relu,
                 temperature=1.0, name='cond_relaxed_categorical'):
        """Creates a conditional RelaxedOneHotCategorical distribution.

        Args:
            size: The dimension of the random variable.
            hidden_layer_size: The size of the hidden layer of the fully connected
                network used to condition the distribution on the inputs.
            hidden_activation_fn: The activation function to use on the hidden layers
                of the fully connected network.
            temperature: Degree of how approximately discrete the distribution is. The closer 
                to 0, the more discrete and the closer to infinity, the more uniform.
            name: The name of this Model, used for scoping.
        """

        super(ConditionalRelaxedCategorical, self).__init__()

        self._name = name

        self._size = size
        self._hidden_size = hidden_layer_size

        self._temperature = temperature

        if hidden_layer_size is not None:
            self._feature_layer = tfkl.Dense(hidden_layer_size, activation=hidden_activation_fn)

        self._out_layer = tfkl.Dense(size)

    def condition(self, tensor_list, **unused_kwargs):
        """Computes the logits of a RelaxedOneHotCategorical distribution."""

        inputs = tf.concat(tensor_list, axis=-1)

        if self._hidden_size is not None:
            inputs = self._feature_layer(inputs)

        return self._out_layer(inputs)

    def call(self, *args, **kwargs):
        """Creates a RelaxedOneHotCategorical distribution conditioned on the inputs."""

        logits = self.condition(args, **kwargs)

        # return tfd.ExpRelaxedOneHotCategorical(
        return tfd.RelaxedOneHotCategorical(
            self._temperature,
            logits=logits)

    @property
    def output_size(self):
        """The output size of the random variable."""
        return self._size


class MLPEncoder(tfk.Model):
    """A multilayer perceptron used to extract features from the inputs."""

    def __init__(self, size, hidden_size, activation_fn=tf.nn.relu, name='mlp_encoder'):
        """Creates a multilayer perceptron to apply on time-independent data inputs.

        Args:
            size: The resulting size of the feature code vector.
            hidden_size: The size of the intermediate hidden layer.
            activation_fn: The activation function to use on the hidden layer.
            name: The name of this Model, used for scoping.
        """

        super(MLPEncoder, self).__init__()

        self._name = name

        self._size = size
        self._hidden_size = hidden_size

        self._feature_layer = tfkl.Dense(hidden_size, activation=activation_fn)

        self._out_layer = tfkl.Dense(size)

    def call(self, inputs):
        """Runs the inputs through an MLP."""

        outs = self._feature_layer(inputs)

        return self._out_layer(outs)

    @property
    def output_size(self):
        """The output size of the code vector."""
        return self._size


class ConvEncoder(tfk.Model):
    """A convolutional encoder used to extract features from the inputs."""

    def __init__(self, size, filters, activation_fn=tf.nn.relu, name='conv_encoder'):
        """Creates a CNN to apply on time-independent data inputs.

        Args:
            size: The resulting size of the feature code vector.
            filters: The number of filters to use in the convolutional layers.
            activation_fn: The activation function to use on the convolutional layers.
            name: The name of this Model, used for scoping.
        """

        super(ConvEncoder, self).__init__()

        self._name = name

        self._size = size
        self._filters = filters

        # Spatial sizes: (64,64) -> (32,32) -> (16,16) -> (8,8) -> (1,1)
        conv = functools.partial(
            tfkl.Conv2D, padding='SAME', activation=activation_fn)
        self._conv1 = conv(filters, 3, 2)
        self._conv2 = conv(filters * 2, 3, 2)
        self._conv3 = conv(filters * 4, 3, 2)
        self._conv4 = conv(size, 8, padding='VALID')

    def call(self, inputs):
        """Runs the inputs through the CNN encoder."""

        image_shape = tf.shape(inputs)[-3:]
        collapsed_shape = tf.concat(([-1], image_shape), axis=0)

        outs = tf.reshape(inputs, collapsed_shape)  # (..., h, w, c)
        outs = self._conv1(outs)
        outs = self._conv2(outs)
        outs = self._conv3(outs)
        outs = self._conv4(outs)
        expanded_shape = tf.concat((tf.shape(input=inputs)[:-3], [-1]), axis=0)

        # Original shape with (h, w, c) encoded into 'size'
        return tf.reshape(outs, expanded_shape)

    @property
    def output_size(self):
        """The output size of the code vector."""
        return self._size


class BernoulliDecoder(tfk.Model):
    """A Bernoulli decoder distribution conditioned on the outputs of a CNN decoder."""

    def __init__(self, hidden_size, filters, activation_fn=tf.nn.relu, channels=1, bias_init=0.0, name='bern_decoder'):
        """Creates a conditional Bernoulli decoder distribution.

        Args:
            hidden_size: The size of the intermediate code layer to decode from.
            filters: The number of filters to use in the transposed convolutional layers.
            activation_fn: The activation function to use on the intermediate layers.
            bias_init: A Tensor that is added to the output and parameterises the distribution mean.
            channels: Number of channels in images of sequence.
            name: The name of this Model, used for scoping.
        """

        super(BernoulliDecoder, self).__init__()

        self._name = name

        self._hidden_size = hidden_size
        self._filters = filters
        self._bias_init = bias_init

        self._feature_layer = tfkl.Dense(hidden_size, activation=activation_fn)

        # Spatial sizes: (1,1) -> (8,8) -> (16,16) -> (32,32) -> (64,64)
        conv_transpose = functools.partial(
            tfkl.Conv2DTranspose, padding='SAME', activation=activation_fn)
        self._conv_transpose1 = conv_transpose(filters * 4, 8, 1, padding='VALID')
        self._conv_transpose2 = conv_transpose(filters * 2, 3, 2)
        self._conv_transpose3 = conv_transpose(filters, 3, 2)
        self._conv_transpose4 = conv_transpose(channels, 3, 2, activation=None)

    def condition(self, tensor_list, **unused_kwargs):
        """Computes the logits of a Bernoulli distribution."""

        inputs = tf.concat(tensor_list, axis=-1)

        outs = self._feature_layer(inputs)
        outs = tf.reshape(outs, [-1, 1, 1, self._hidden_size])

        outs = self._conv_transpose1(outs)
        outs = self._conv_transpose2(outs)
        outs = self._conv_transpose3(outs)
        outs = self._conv_transpose4(outs)  # (..., h, w, c)

        expanded_shape = tf.concat(
            (tf.shape(input=inputs)[:-1], tf.shape(input=outs)[1:]), axis=0)
        # Complete expanded shape with (h, w, c) dimensions
        outs = tf.reshape(outs, expanded_shape)

        return outs + self._bias_init

    def call(self, *args, **kwargs):
        """Creates a Bernoulli distribution conditioned on the inputs."""

        logits = self.condition(args, **kwargs)

        return tfd.Independent(
            distribution=tfd.Bernoulli(logits=logits),
            reinterpreted_batch_ndims=3)  # Wrap (h, w, c)


class NormalDecoder(tfk.Model):
    """A Normal decoder distribution conditioned on the outputs of a CNN decoder."""

    def __init__(self, hidden_size, filters, activation_fn=tf.nn.relu, channels=1, name='norm_decoder'):
        """Creates a conditional Normal decoder distribution.

        Args:
            hidden_size: The size of the intermediate code layer to decode from.
            filters: The number of filters to use in the transposed convolutional layers.
            activation_fn: The activation function to use on the intermediate layers.
            channels: Number of channels in images of sequence.
            name: The name of this Model, used for scoping.
        """

        super(NormalDecoder, self).__init__()

        self._name = name

        self._hidden_size = hidden_size
        self._filters = filters

        self._feature_layer = tfkl.Dense(hidden_size, activation=activation_fn)

        # Spatial sizes: (1,1) -> (8,8) -> (16,16) -> (32,32) -> (64,64)
        conv_transpose = functools.partial(
            tfkl.Conv2DTranspose, padding='SAME', activation=activation_fn)
        self._conv_transpose1 = conv_transpose(filters * 4, 8, 1, padding='VALID')
        self._conv_transpose2 = conv_transpose(filters * 2, 3, 2)
        self._conv_transpose3 = conv_transpose(filters, 3, 2)
        self._conv_transpose4 = conv_transpose(channels, 3, 2, activation=None)

    def condition(self, tensor_list, **unused_kwargs):
        """Computes the logits of a Normal distribution."""

        inputs = tf.concat(tensor_list, axis=-1)

        outs = self._feature_layer(inputs)
        outs = tf.reshape(outs, [-1, 1, 1, self._hidden_size])

        outs = self._conv_transpose1(outs)
        outs = self._conv_transpose2(outs)
        outs = self._conv_transpose3(outs)
        outs = self._conv_transpose4(outs)  # (..., h, w, c)

        expanded_shape = tf.concat(
            (tf.shape(input=inputs)[:-1], tf.shape(input=outs)[1:]), axis=0)
        # Complete expanded shape with (h, w, c) dimensions
        outs = tf.reshape(outs, expanded_shape)

        return outs

    def call(self, *args, **kwargs):
        """Creates a Normal distribution conditioned on the inputs."""

        outs = self.condition(args, **kwargs)

        return tfd.Independent(
            distribution=tfd.Normal(loc=outs, scale=1.),
            reinterpreted_batch_ndims=3)  # Wrap (h, w, c)
