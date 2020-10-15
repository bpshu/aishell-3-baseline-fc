"""Attention file for location based attention (compatible with tensorflow attention wrapper)"""

from math import tanh
from numpy.core.arrayprint import dtype_is_implied
from six import u
import tensorflow as tf
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import BahdanauAttention
from tensorflow.python.layers import core as layers_core
from tensorflow.python.ops import array_ops, math_ops, nn_ops, variable_scope

import numpy as np 


#From https://github.com/tensorflow/tensorflow/blob/r1.7/tensorflow/contrib/seq2seq/python/ops/attention_wrapper.py
def _compute_attention(attention_mechanism, cell_output, attention_state,
					   attention_layer):
	"""Computes the attention and alignments for a given attention_mechanism."""
	alignments, next_attention_state = attention_mechanism(
		cell_output, state=attention_state)

	# Reshape from [batch_size, memory_time] to [batch_size, 1, memory_time]
	expanded_alignments = array_ops.expand_dims(alignments, 1)
	# Context is the inner product of alignments and values along the
	# memory time dimension.
	# alignments shape is
	#   [batch_size, 1, memory_time]
	# attention_mechanism.values shape is
	#   [batch_size, memory_time, memory_size]
	# the batched matmul is over memory_time, so the output shape is
	#   [batch_size, 1, memory_size].
	# we then squeeze out the singleton dim.
	context = math_ops.matmul(expanded_alignments, attention_mechanism.values)
	context = array_ops.squeeze(context, [1])

	if attention_layer is not None:
		attention = attention_layer(array_ops.concat([cell_output, context], 1))
	else:
		attention = context

	return attention, alignments, next_attention_state


def _location_sensitive_score(W_query, W_fil, W_keys):
	"""Impelements Bahdanau-style (cumulative) scoring function.
	This attention is described in:
		J. K. Chorowski, D. Bahdanau, D. Serdyuk, K. Cho, and Y. Ben-
	  gio, “Attention-based models for speech recognition,” in Ad-
	  vances in Neural Information Processing Systems, 2015, pp.
	  577–585.

	#############################################################################
			  hybrid attention (content-based + location-based)
							   f = F * α_{i-1}
	   energy = dot(v_a, tanh(W_keys(h_enc) + W_query(h_dec) + W_fil(f) + b_a))
	#############################################################################

	Args:
		W_query: Tensor, shape "[batch_size, 1, attention_dim]" to compare to location features.
		W_location: processed previous alignments into location features, shape "[batch_size, max_time, attention_dim]"
		W_keys: Tensor, shape "[batch_size, max_time, attention_dim]", typically the encoder outputs.
	Returns:
		A "[batch_size, max_time]" attention score (energy)
	"""
	# Get the number of hidden units from the trailing dimension of keys
	dtype = W_query.dtype
	num_units = W_keys.shape[-1].value or array_ops.shape(W_keys)[-1]

	v_a = tf.get_variable(
		"attention_variable_projection", shape=[num_units], dtype=dtype,
		initializer=tf.contrib.layers.xavier_initializer())
	b_a = tf.get_variable(
		"attention_bias", shape=[num_units], dtype=dtype,
		initializer=tf.zeros_initializer())

	return tf.reduce_sum(v_a * tf.tanh(W_keys + W_query + W_fil + b_a), [2])


def _Dynamic_convolution_score(W_f, W_g, W_p) : 

	"""
	:param W_f: shape (bcsz, time, attn_dim), location information through static convolution
	:param W_g: shape (bcsz, time, attn_dim), location information through dynamic convolution
	:param W_p: shape (bcsz, time, 1),        location information through priror convolution

	:return: (bcsz, time) attention energy (before normalization)
	"""
	dtype = W_f.dtype
	num_units = W_f.shape[-1].value 

	v_a = tf.get_variable(
		"attention_variable_projection", 
		shape=[num_units], 
		dtype=dtype,
		initializer = tf.contrib.layers.xavier_initializer()
	)

	b_a = tf.get_variable(
		"attention_bias",
		shape=[num_units],
		dtype=dtype,
		initializer = tf.zeros_initializer()
	)
	W_p = tf.squeeze(W_p, axis=-1)

	# return tf.reduce_sum((v_a * tf.tanh(W_f + W_g + b_a) + W_p), [2])
	return tf.reduce_sum((v_a * tf.tanh(W_f + W_g + b_a)), [2]) + W_p



def _smoothing_normalization(e):
	"""Applies a smoothing normalization function instead of softmax
	Introduced in:
		J. K. Chorowski, D. Bahdanau, D. Serdyuk, K. Cho, and Y. Ben-
	  gio, “Attention-based models for speech recognition,” in Ad-
	  vances in Neural Information Processing Systems, 2015, pp.
	  577–585.

	############################################################################
						Smoothing normalization function
				a_{i, j} = sigmoid(e_{i, j}) / sum_j(sigmoid(e_{i, j}))
	############################################################################

	Args:
		e: matrix [batch_size, max_time(memory_time)]: expected to be energy (score)
			values of an attention mechanism
	Returns:
		matrix [batch_size, max_time]: [0, 1] normalized alignments with possible
			attendance to multiple memory time steps.
	"""
	return tf.nn.sigmoid(e) / tf.reduce_sum(tf.nn.sigmoid(e), axis=-1, keepdims=True)


def design_prior_filter(a, b, n) : 
	from scipy.stats import betabinom
	beta = betabinom(a = a, b = b, n = n)
	taps = beta.pmf([i for i in range(n, -1, -1)])
	filter_length = 2 * (len(taps) - 1) + 1
	
	filter_coef = np.zeros((filter_length, 1, 1))
	filter_coef[ : (n + 1), 0,0] = taps

	print(f'priror filter : {filter_coef.flatten()}')

	return filter_coef	

def prior_filter_coef_gen(speed, n, A = 0.5) : 
    a = A
    b = a * (n - speed) / speed
    return (a,b,n)


class LocationSensitiveAttention(BahdanauAttention):
	"""Impelements Bahdanau-style (cumulative) scoring function.
	Usually referred to as "hybrid" attention (content-based + location-based)
	Extends the additive attention described in:
	"D. Bahdanau, K. Cho, and Y. Bengio, “Neural machine transla-
  tion by jointly learning to align and translate,” in Proceedings
  of ICLR, 2015."
	to use previous alignments as additional location features.

	This attention is described in:
	J. K. Chorowski, D. Bahdanau, D. Serdyuk, K. Cho, and Y. Ben-
  gio, “Attention-based models for speech recognition,” in Ad-
  vances in Neural Information Processing Systems, 2015, pp.
  577–585.
	"""

	def __init__(self,
				 num_units,
				 memory,
				 hparams,
				 mask_encoder=True,
				 memory_sequence_length=None,
				 smoothing=False,
				 cumulate_weights=True,
				 name="LocationSensitiveAttention"):
		"""Construct the Attention mechanism.
		Args:
			num_units: The depth of the query mechanism.
			memory: The memory to query; usually the output of an RNN encoder.  This
				tensor should be shaped `[batch_size, max_time, ...]`.
			mask_encoder (optional): Boolean, whether to mask encoder paddings.
			memory_sequence_length (optional): Sequence lengths for the batch entries
				in memory.  If provided, the memory tensor rows are masked with zeros
				for values past the respective sequence lengths. Only relevant if mask_encoder = True.
			smoothing (optional): Boolean. Determines which normalization function to use.
				Default normalization function (probablity_fn) is softmax. If smoothing is
				enabled, we replace softmax with:
						a_{i, j} = sigmoid(e_{i, j}) / sum_j(sigmoid(e_{i, j}))
				Introduced in:
					J. K. Chorowski, D. Bahdanau, D. Serdyuk, K. Cho, and Y. Ben-
				  gio, “Attention-based models for speech recognition,” in Ad-
				  vances in Neural Information Processing Systems, 2015, pp.
				  577–585.
				This is mainly used if the model wants to attend to multiple input parts
				at the same decoding step. We probably won"t be using it since multiple sound
				frames may depend on the same character/phone, probably not the way around.
				Note:
					We still keep it implemented in case we want to test it. They used it in the
					paper in the context of speech recognition, where one phoneme may depend on
					multiple subsequent sound frames.
			name: Name to use when creating ops.
		"""
		#Create normalization function
		#Setting it to None defaults in using softmax
		normalization_function = _smoothing_normalization if (smoothing == True) else None
		memory_length = memory_sequence_length if (mask_encoder==True) else None
		super(LocationSensitiveAttention, self).__init__(
				num_units=num_units,
				memory=memory,
				memory_sequence_length=memory_length,
				probability_fn=normalization_function,
				name=name)

		# (shiyao): performs static filtering on cumulative attention_weights
		self.location_convolution = tf.layers.Conv1D(filters=hparams.attention_filters,
			kernel_size=hparams.attention_kernel, padding="same", use_bias=True,
			bias_initializer=tf.zeros_initializer(), name="location_features_convolution")

		# (shiyao): projects static filtered attentions to attention_dim
		self.location_layer = tf.layers.Dense(units=num_units, use_bias=False,
			dtype=tf.float32, name="location_features_layer")

		# (shiyao): projects decoder state of shape (bcsz, query_depth) to (bcsz, attn_depth)
		self.dynamic_project = tf.layers.Dense(
			units=num_units, 
			use_bias=True, 
			dtype=tf.float32, 
			name='dca_filter_project_inner',
			activation=tf.nn.tanh
			)

		self.dynamic_filter_project = tf.layers.Dense(
			units = hparams.attention_kernel * hparams.attention_filters,
			use_bias = False,
			dtype = tf.float32,
			name='dca_filter_project_outter',
			activation=None
		)

		self.dynamic_layer = tf.layers.Dense(
			units=num_units,
			use_bias=False,
			dtype=tf.float32,
			name='dca_dynamic_features_layer'
		)

		# (shiyao): if use cumulative weights
		self._cumulate = cumulate_weights
		self.num_units = num_units
		self.hparams = hparams

		self.my_batch_size = hparams.tacotron_batch_size // hparams.tacotron_num_gpus

		self.batch_flat_pattern = self.get_batch_flat_pattern(hparams, self.my_batch_size)

		self.prior_filter_kernel = tf.constant(
			design_prior_filter(*prior_filter_coef_gen(
				hparams.prior_speed, 
				hparams.attention_kernel//2, 
				A=hparams.prior_alpha)
			),
			dtype=tf.float32
		)

		self.prior_clamper = tf.constant(
			10e-6, dtype=tf.float32
		)

	def __call__(self, query, state) :
		"""Score the query based on the keys and values.
		Args:
			query: Tensor of dtype matching `self.values` and shape
				`[batch_size, query_depth]`.
			state (previous alignments): Tensor of dtype matching `self.values` and shape
				`[batch_size, alignments_size]`
				(`alignments_size` is memory"s `max_time`).
		Returns:
			alignments: Tensor of dtype matching `self.values` and shape
				`[batch_size, alignments_size]` (`alignments_size` is memory's
				`max_time`).
		"""
		previous_alignments = state
		with variable_scope.variable_scope(None, "Location_Sensitive_Attention", [query]):

			# processed_query shape [batch_size, query_depth] -> [batch_size, attention_dim]
			processed_query = self.query_layer(query) if self.query_layer else query
			# -> [batch_size, 1, attention_dim]
			processed_query = tf.expand_dims(processed_query, 1)

			# processed_location_features shape [batch_size, max_time, attention dimension]
			# [batch_size, max_time] -> [batch_size, max_time, 1]
			expanded_alignments = tf.expand_dims(previous_alignments, axis=2)
			# location features [batch_size, max_time, filters]
			f = self.location_convolution(expanded_alignments)
			# Projected location features [batch_size, max_time, attention_dim]
			processed_location_features = self.location_layer(f)


			###############################################################################
			# (shiyao): DCA dynamic filtering:
			# step one : create convolution kernels using current decoder state

			# query (bcsz, query_dim) -> (bcsz, attn_dim) -> (bcsz, n_filt*kernel_size) -> flat_filter
			flat_filter = self.dynamic_filter_project(self.dynamic_project(query))

			# desiried filter size for a 2D-kernel is (kernel_size, in_channel(1) * bcsz, out_channel * bcsz)
			flat_filter = tf.reshape(flat_filter, (self.hparams.attention_filters * self.my_batch_size, 1, self.hparams.attention_kernel))
			# expanded_filer is of shape (n_filt, in_channel, out_channel)
			expanded_filter = tf.transpose(self.batch_flat_pattern * flat_filter, perm=(2,1,0))


			# step two: flatten the batched input tensors into a bcsz=1 tensor,
			# spreading batched data into multiple channels
			# expanded_alignments is of shape (bcsz, time, 1)
			# need to go to (1, time, bcsz)
			spreaded_alignments = tf.transpose(expanded_alignments, perm=(2,1,0))

			# step three : perform 2D convolution on flattend inputs :

			# result flat-g should have the shape of
			# (1, time, bcsz * n_filter)
			flat_g = tf.nn.conv1d(
				input = spreaded_alignments,
				filters = expanded_filter,
				padding = 'SAME',
				name = 'dca_dynamic_convolution'
			)

			# (time, bcsz, n_filt)
			flat_g = tf.reshape(flat_g, (-1, self.my_batch_size, self.hparams.attention_filters))
			g = tf.transpose(flat_g, perm=(1,0,2))  # g : (bcsz, time, n_filter)

			# step four : project to feature dim
			# (bcsz, time, attn_dim)
			dynamic_features = self.dynamic_layer(g)	# * to be used

			###############################################################################
			# (shiyao): DCA prior filter
			
			# (bcsz, time, 1)
			prior_features = tf.log(tf.maximum((tf.nn.conv1d(
				input = expanded_alignments,
				filters = self.prior_filter_kernel,
				padding = 'SAME',
				name = 'dca_dynamic_convolution'
			)), self.prior_clamper))




			# (bcsz, time)
			energy = _Dynamic_convolution_score(processed_location_features, dynamic_features, prior_features)

			# energy shape [batch_size, max_time]
			# energy = _location_sensitive_score(processed_query, processed_location_features, self.keys)


		# alignments shape = energy shape = [batch_size, max_time]
		alignments = self._probability_fn(energy, previous_alignments)

		# Cumulate alignments
		if self._cumulate:
			next_state = alignments + previous_alignments
		else:
			next_state = alignments

		return alignments, next_state

	def get_batch_flat_pattern(self, hparams, bcsz) : 
		# returned : (n_filt * bcsz, bcsz, n_taps)

		np_ones = np.zeros((hparams.attention_filters * bcsz, bcsz, hparams.attention_kernel))
		for i in range(bcsz) : 
			np_ones[8 * (i) : 8 * (i + 1), i, :] = 1
		ones = tf.constant(np_ones, dtype=tf.float32)
		return ones