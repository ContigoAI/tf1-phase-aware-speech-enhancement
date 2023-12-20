import tensorflow as tf

class ComplexConv3D(tf.layers.Conv3D):
    """
    3D Complex Convolutional Layer.

    Args:
        filters: Number of filters.
        kernel_size: Size of the convolutional kernel.
        strides: Stride size.
        dilation_rate: Dilation rate.
        activation: Activation function.
        use_bias: Whether to use bias.
        padding: Padding type.
        kernel_initializer: Initializer for the kernel weights.
        bias_initializer: Initializer for the bias.
        kernel_regularizer: Regularizer for the kernel weights.
        bias_regularizer: Regularizer for the bias.
        activity_regularizer: Regularizer for the layer activity.
        kernel_constraint: Constraint for the kernel weights.
        bias_constraint: Constraint for the bias.
        trainable: Whether the layer is trainable.
        name: Name of the layer.
        **kwargs: Additional keyword arguments.
    """
    def __init__(self, filters, kernel_size, strides=(1, 1, 1), dilation_rate=(1, 1, 1),
                 activation=None, use_bias=True, padding='same',
                 kernel_initializer=None, bias_initializer=tf.zeros_initializer(),
                 kernel_regularizer=None, bias_regularizer=None,
                 activity_regularizer=None, kernel_constraint=None,
                 bias_constraint=None, trainable=True, name=None, **kwargs):

        super(ComplexConv3D, self).__init__(
            filters=filters, kernel_size=kernel_size,
            strides=strides, padding=padding, data_format='channels_last',
            dilation_rate=dilation_rate, activation=activation,
            use_bias=use_bias, kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint, bias_constraint=bias_constraint,
            trainable=trainable, name=name, **kwargs)

        self.real_conv = tf.layers.Conv2D(filters=filters, kernel_size=kernel_size[:2],
                                                    strides=strides[:2], padding=padding, data_format='channels_last',
                                                    dilation_rate=dilation_rate[:2], activation=activation,
                                                    use_bias=use_bias, kernel_regularizer=kernel_regularizer,
                                                    bias_regularizer=bias_regularizer,
                                                    activity_regularizer=activity_regularizer,
                                                    kernel_constraint=kernel_constraint,
                                                    bias_constraint=bias_constraint,
                                                    trainable=trainable, name=name, **kwargs)

        self.img_conv = tf.layers.Conv2D(filters=filters, kernel_size=kernel_size[:2],
                                                   strides=strides[:2], padding=padding, data_format='channels_last',
                                                   dilation_rate=dilation_rate[:2], activation=activation,
                                                   use_bias=use_bias, kernel_regularizer=kernel_regularizer,
                                                   bias_regularizer=bias_regularizer,
                                                   activity_regularizer=activity_regularizer,
                                                   kernel_constraint=kernel_constraint, bias_constraint=bias_constraint,
                                                   trainable=trainable, name=name, **kwargs)

    def call(self, inputs):
        real = self.real_conv(inputs[:, 0]) - self.img_conv(inputs[:, 1])
        imaginary = self.real_conv(inputs[:, 1]) + self.img_conv(inputs[:, 0])
        output = tf.stack((real, imaginary), axis=1)
        return output


class ComplexTransposeConv3D(tf.layers.Conv3DTranspose):
    """
    3D Complex Transposed Convolutional Layer.

    Args:
        filters: Number of filters.
        kernel_size: Size of the convolutional kernel.
        strides: Stride size.
        dilation_rate: Dilation rate.
        activation: Activation function.
        use_bias: Whether to use bias.
        padding: Padding type.
        kernel_initializer: Initializer for the kernel weights.
        bias_initializer: Initializer for the bias.
        kernel_regularizer: Regularizer for the kernel weights.
        bias_regularizer: Regularizer for the bias.
        activity_regularizer: Regularizer for the layer activity.
        kernel_constraint: Constraint for the kernel weights.
        bias_constraint: Constraint for the bias.
        trainable: Whether the layer is trainable.
        name: Name of the layer.
        **kwargs: Additional keyword arguments.
    """
    def __init__(self, filters, kernel_size, strides=(1, 1, 1), dilation_rate=(1, 1, 1),
                 activation=None, use_bias=True, padding='same',
                 kernel_initializer=None, bias_initializer=tf.zeros_initializer(),
                 kernel_regularizer=None, bias_regularizer=None,
                 activity_regularizer=None, kernel_constraint=None,
                 bias_constraint=None, trainable=True, name=None, **kwargs):

        super(ComplexTransposeConv3D, self).__init__(
            filters=filters, kernel_size=kernel_size,
            strides=strides, padding=padding, data_format='channels_last',
            dilation_rate=dilation_rate, activation=activation,
            use_bias=use_bias, kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint, bias_constraint=bias_constraint,
            trainable=trainable, name=name, **kwargs)

        self.real_conv = tf.layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size[:2],
                                                    strides=strides[:2], padding=padding, data_format='channels_last',
                                                    dilation_rate=dilation_rate[:2], activation=activation,
                                                    use_bias=use_bias, kernel_regularizer=kernel_regularizer,
                                                    bias_regularizer=bias_regularizer,
                                                    activity_regularizer=activity_regularizer,
                                                    kernel_constraint=kernel_constraint,
                                                    bias_constraint=bias_constraint,
                                                    trainable=trainable, name=name, **kwargs)

        self.img_conv = tf.layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size[:2],
                                                   strides=strides[:2], padding=padding, data_format='channels_last',
                                                   dilation_rate=dilation_rate[:2], activation=activation,
                                                   use_bias=use_bias, kernel_regularizer=kernel_regularizer,
                                                   bias_regularizer=bias_regularizer,
                                                   activity_regularizer=activity_regularizer,
                                                   kernel_constraint=kernel_constraint, bias_constraint=bias_constraint,
                                                   trainable=trainable, name=name, **kwargs)

    def call(self, inputs):
        real = self.real_conv(inputs[:, 0]) - self.img_conv(inputs[:, 1])
        imaginary = self.real_conv(inputs[:, 1]) + self.img_conv(inputs[:, 0])
        output = tf.stack((real, imaginary), axis=1)
        return output


def up_depth_concat(inputA, input_B, name, concat=True):
    """
    Depth-wise concatenation after upsampling.

    Args:
        inputA: Input tensor A.
        input_B: Input tensor B.
        name: Name of the operation.
        concat: Whether to concatenate.

    Returns:
        Concatenated tensor.
    """
    up_conv = tf.depth_to_space(inputA, 2)
    if not concat:
        return up_conv
    return tf.concat([up_conv, input_B], axis=-1, name="concat_{}".format(name))


def upconv_3D(tensor, n_filter, flags, name):
    """
    3D Complex Transposed Convolution with Upsampling.

    Args:
        tensor: Input tensor.
        n_filter: Number of filters.
        flags: Flags object.
        name: Name of the operation.

    Returns:
        Upsampled tensor.
    """ 
    layer_name = "upsample_{}".format(name)
    return ComplexTransposeConv3D(filters=n_filter, kernel_size=2, strides=2,
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(flags.regulizer_weight),
                                 name=layer_name)(tensor)


def conv_block_3d(net, F, dilation, i, flags, kernel=(3, 3, 3), bias=True, padding='same'):
    """
    3D Complex Convolution Block.

    Args:
        net: Input tensor.
        F: Number of filters.
        dilation: Dilation rate.
        i: Block index.
        flags: Flags object.
        kernel: Size of the convolutional kernel.
        bias: Whether to use bias.
        padding: Padding type.

    Returns:
        Convolved tensor.
    """
    return ComplexConv3D(F, kernel, activation=None, padding=padding,
                         kernel_regularizer=tf.contrib.layers.l2_regularizer(flags.regulizer_weight),
                         name="conv_{}".format(i), dilation_rate=dilation, use_bias=bias)(net)


def res_pool_3D(input_, n_filters, training, flags, name, pool=True, activation=tf.nn.leaky_relu, dilation=[1, 1]):
    """
    3D Complex Residual Pooling.

    Args:
        input_: Input tensor.
        n_filters: List of numbers of filters for each layer.
        training: Whether the model is in training mode.
        flags: Flags object.
        name: Name of the operation.
        pool: Whether to include pooling.
        activation: Activation function.
        dilation: Dilation rates.

    Returns:
        Pooled tensor.
    """
    net = input_
    layer_name = "layer{}".format(name)
    with tf.variable_scope(layer_name): #, reuse = tf.AUTO_REUSE):
        for i, F in enumerate(n_filters):
            net = conv_block_3d(net, F, dilation[i], i, flags)
            net = tf.layers.batch_normalization(
                net, training=training, name="bn_{}".format(i + 1))
            net = activation(net, name="relu_{}".format(i))

        shortcut = input_
        shortcut = conv_block_3d(shortcut, n_filters[0], dilation[0], 'shortcut', flags)
        shortcut = tf.layers.batch_normalization(shortcut, training=training, name='shortcut_bn')
        net = tf.add_n([shortcut, net])
        if not pool:
            return net
        pool = tf.layers.max_pooling3d(net, (1, 2, 2), strides=(1, 2, 2), name="pool", padding='valid')
        return net, pool


def upconv_concat_3D(inputA, input_B, n_filter, flags, name):
    """
    3D Complex Concatenation after Upsampling.

    Args:
        inputA: Input tensor A.
        input_B: Input tensor B.
        n_filter: Number of filters.
        flags: Flags object.
        name: Name of the operation.

    Returns:
        Concatenated tensor.
    """
    up_conv = upconv_3D(inputA, n_filter, flags, name)
    if input_B is None:
        input_B = up_conv
    return tf.concat([up_conv, input_B], axis=-1, name="concat_{}".format(name))


def make_asppunet_3D(X, training, flags=None, features=1, last_pad=False, mask=False):
    """
    Build a 3D Complex U-Net with ASPP.

    Args:
        X: Input tensor.
        training: Whether the model is in training mode.
        flags: Flags object.
        features: Number of features.
        last_pad: Whether to add padding in the final convolution.
        mask: Whether to apply a mask.

    Returns:
        Tuple containing the output tensor, magnitude tensor, and list of intermediate tensors.
    """
        
    input_ = X
    net = input_
    frame_step = flags.stft_freq_samples - flags.noverlap - 2
    pre_name = 'pre_process'
    mag = None

    with tf.variable_scope(pre_name): #,reuse = tf.AUTO_REUSE):
        pre = net
        pre = tf.contrib.signal.stft(pre, frame_length=flags.stft_freq_samples,frame_step=frame_step)
        pre_stacked = tf.stack([tf.math.real(pre), tf.math.imag(pre)], axis=1)
        pre_stacked = tf.expand_dims(pre_stacked, axis=-1)

    conv1, pool1 = res_pool_3D(pre_stacked, [8 * features, 8 * features], training, flags, name=1)
    conv2, pool2 = res_pool_3D(pool1, [16 * features, 16 * features], training, flags, name=2)
    conv3, pool3 = res_pool_3D(pool2, [32 * features, 32 * features], training, flags, name=3)
    conv4, pool4 = res_pool_3D(pool3, [64 * features, 64 * features], training, flags, name=4)
    conv5 = res_pool_3D(pool4, [128 * features, 128 * features], training, flags,
                        name=5, pool=False, dilation=[2, 2])

    up6 = upconv_concat_3D(conv5, conv4, 64 * features, flags, name=6)
    conv6 = res_pool_3D(up6, [64 * features, 64 * features], training, flags, name=6, pool=False)

    up7 = upconv_concat_3D(conv6, conv3, 32 * features, flags, name=7)
    conv7 = res_pool_3D(up7, [32 * features, 32 * features], training, flags, name=7, pool=False)

    up8 = upconv_concat_3D(conv7, conv2, 16 * features, flags, name=8)
    conv8 = res_pool_3D(up8, [16 * features, 16 * features], training, flags, name=8, pool=False)

    up9 = upconv_concat_3D(conv8, None, 8 * features, flags, name=9)
    conv9 = res_pool_3D(up9, [8 * features, 8 * features], training, flags, name=9, pool=False)

    final_name = 'final'
    with tf.variable_scope(final_name): #, reuse = tf.AUTO_REUSE):
        if last_pad:
            final = tf.pad(conv9, [[0, 0], [0, 0], [1, 0], [1, 0], [0, 0]])
        final = ComplexConv3D(
            1, (1, 1, 1),
            name=final_name,
            activation=None,
            padding='same')(final)
    post_name = 'post_process'
    with tf.variable_scope(post_name): #, reuse = tf.AUTO_REUSE):
        post = tf.squeeze(final, axis=-1)
        post = tf.complex(post[:, 0], post[:, 1])
        if mask:
            phase = post / (tf.complex(tf.abs(post), 0.) + tf.complex(0.000001, 0.))
            mag = tf.tanh(tf.abs(post))
            post = phase * tf.complex(mag, 0.) * pre
        post = tf.contrib.signal.inverse_stft(post, frame_length=flags.stft_freq_samples,
                                              frame_step=frame_step)
        post = tf.real(post)

    return (post, mag, [conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9, final])

