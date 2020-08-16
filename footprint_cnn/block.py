import tensorflow as tf
import numpy as np

#
# Basic 1D convolutional block -> activation, convolution, maxpool
#
def conv_block(inputs, filters=None, kernel_size=1, activation='relu', strides=1,
	dilation_rate=1, l2_scale=0, dropout=0, conv_type='standard', residual=False,
	pool_size=1, batch_norm=False, bn_momentum=0.99, bn_gamma=None,
	kernel_initializer='he_normal'):

	current = inputs
	conv_layer = tf.keras.layers.Conv1D

	if filters is None:
		filters = inputs.shape[-1]

	current = conv_layer(
		filters,
		kernel_size,
		strides=strides,
		padding='same',
		use_bias=False,
		activation='relu',
		dilation_rate=dilation_rate,
		kernel_initializer=kernel_initializer,
		kernel_regularizer=tf.keras.regularizers.l2(l2_scale))(current)

	#activation function; need to customize for other activations
	#current = tf.keras.layers.ReLU(current)

	if residual:
		current = tf.keras.layers.Add([inputs, current])

	if pool_size > 1:
		current = tf.keras.layers.MaxPool1D(
			pool_size=pool_size,
			padding='same')(current)

	return current

def conv_tower(inputs, filters_init, filters_mult=1, repeat=1, **kwargs):

	current = inputs

	rep_filters = filters_init

	for i in range(repeat):
		current = conv_block(current,
			filters=int(np.round(rep_filters)),
			**kwargs)

		rep_filters *= filters_mult

	return current

# dense layer

def dense(inputs, units, activation='linear', kernel_initializer='he_normal',
	l2_scale=0, l1_scale=0, **kwargs):
	current = tf.keras.layers.Dense(
		units=units,
		activation=activation,
		use_bias=True,
		kernel_initializer=kernel_initializer,
		kernel_regularizer=tf.keras.regularizers.l1_l2(l1_scale, l2_scale)
		)(inputs)
	return current

