import numpy as np
import tensorflow as tf
from keras_llm_light.utils import mb_used
import keras

VariableType = keras.Variable | tf.Variable


def get_variable_name(variable: VariableType) -> str:
    return variable.path if isinstance(variable, keras.Variable) else variable.name


class Variable:
    def __init__(
        self,
        value: tf.Variable,
        dtype: tf.DType = tf.float32,
        device: str = "GPU",
        trainable: bool = False,
    ):
        self._dtype = dtype
        self._device = device
        self._value = None
        self._trainable = trainable
        self.set_value(value)

    @property
    def trainable(self) -> bool:
        return self._trainable

    @property
    def device(self) -> str:
        return self._value.device

    @property
    def shape(self) -> tf.TensorShape:
        return self._value.shape

    @property
    def dtype(self) -> tf.DType:
        return self._value.dtype

    @property
    def value(self) -> tf.Variable:
        return self._value

    def set_value(self, variable: VariableType):
        with tf.device(self._device):
            self._value = tf.Variable(
                tf.cast(variable, self._dtype), name=get_variable_name(variable)
            )

    def get_value(self) -> tf.Tensor:
        return tf.cast(self._value, tf.float32)

    def get_memory_usage(self) -> float:
        return mb_used(self._value)


class FP16Variable(Variable):
    def __init__(
        self, value: tf.Variable, device: str = "GPU", trainable: bool = False
    ):
        super().__init__(value, tf.float16, device, trainable)


class FP32Variable(Variable):
    def __init__(
        self, value: tf.Variable, device: str = "GPU", trainable: bool = False
    ):
        super().__init__(value, tf.float32, device, trainable)


class Int8Variable(Variable):
    """Int8 quantization inspired by https://arxiv.org/pdf/2208.07339 paper"""

    def __init__(
        self,
        value: tf.Variable,
        percentile: float = 0.1,
        device: str = "GPU",
        trainable: bool = False,
        percentile_method: str = "nearest",
    ):
        self._percentile = percentile
        self._percentile_method = percentile_method
        self._scale = None
        self._bias = None
        self._min_values = None
        self._max_values = None
        self._min_indices = None
        self._max_indices = None
        super().__init__(value, tf.uint8, device, trainable)

    def set_value(self, variable: VariableType):
        name = get_variable_name(variable)
        w_src = variable.numpy()
        min_val = np.percentile(w_src, self._percentile, method=self._percentile_method)
        max_val = np.percentile(w_src, 100 - self._percentile, method=self._percentile_method)
        scale = np.maximum(max_val - min_val, 1e-6)
        w = np.clip(w_src, min_val, max_val)
        w_uint8 = np.round((w - min_val) / scale * 255).astype(np.uint8)

        with tf.device(self._device):
            self._value = tf.Variable(tf.cast(w_uint8, self._dtype), name=name)
            self._scale = tf.Variable(
                scale / 255.0, dtype=tf.float32, name=name + "_scale"
            )
            self._bias = tf.Variable(min_val, dtype=tf.float32, name=name + "_bias")

        if self._percentile == 0:
            return
        min_indices = tf.where(variable < min_val)
        max_indices = tf.where(variable > max_val)
        min_values = tf.cast(tf.gather_nd(variable, min_indices), tf.float16)
        max_values = tf.cast(tf.gather_nd(variable, max_indices), tf.float16)

        with tf.device(self._device):
            self._min_values = tf.Variable(
                min_values, dtype=tf.float16, name=name + "_min_values"
            )
            self._max_values = tf.Variable(
                max_values, dtype=tf.float16, name=name + "_max_values"
            )
            self._min_indices = tf.Variable(min_indices, name=name + "_min_indices")
            self._max_indices = tf.Variable(max_indices, name=name + "_max_indices")

    @tf.function(jit_compile=True)
    def get_value(self) -> tf.Tensor:
        var = self._bias + tf.cast(self._value, tf.float32) * self._scale
        if self._percentile > 0:
            min_values = tf.cast(self._min_values, tf.float32)
            max_values = tf.cast(self._max_values, tf.float32)
            var = tf.tensor_scatter_nd_update(var, self._min_indices, min_values)
            var = tf.tensor_scatter_nd_update(var, self._max_indices, max_values)
        return var

    def get_memory_usage(self) -> float:
        used_memory = mb_used(self._value)
        if self._percentile > 0:
            used_memory += mb_used(self._min_values)
            used_memory += mb_used(self._max_values)
            used_memory += mb_used(self._min_indices)
            used_memory += mb_used(self._max_indices)
        return used_memory


class Int4Variable(Variable):
    """
    Int4 quantization inspired by https://arxiv.org/pdf/2208.07339 paper
    It packs two 4-bit values into a single byte.
    """

    def __init__(
        self,
        value: tf.Variable,
        percentile: float = 0.1,
        device: str = "GPU",
        trainable: bool = False,
    ):
        self._percentile = percentile
        self._scale = None
        self._bias = None
        self._min_values = None
        self._max_values = None
        self._min_indices = None
        self._max_indices = None
        self._ndim = value.numpy().ndim

        super().__init__(value, tf.uint8, device, trainable)

    def set_value(self, variable: VariableType):

        name = get_variable_name(variable)
        w_src = variable.numpy()
        min_val = np.percentile(w_src, self._percentile, method="nearest")
        max_val = np.percentile(w_src, 100 - self._percentile, method="nearest")
        scale = np.maximum(max_val - min_val, 1e-6)
        w = np.clip(w_src, min_val, max_val)
        num_buckets = 2**4 - 1
        array = np.round((w - min_val) / scale * num_buckets).astype(np.uint8)

        array = pack_int4_to_int8(array)

        with tf.device(self._device):
            self._value = tf.Variable(tf.cast(array, self._dtype), name=name)
            self._scale = tf.Variable(
                scale / num_buckets, dtype=tf.float32, name=name + "_scale"
            )
            self._bias = tf.Variable(min_val, dtype=tf.float32, name=name + "_bias")

        if self._percentile == 0:
            return
        min_indices = tf.where(variable < min_val)
        max_indices = tf.where(variable > max_val)
        min_values = tf.cast(tf.gather_nd(variable, min_indices), tf.float16)
        max_values = tf.cast(tf.gather_nd(variable, max_indices), tf.float16)

        with tf.device(self._device):
            self._min_values = tf.Variable(
                min_values, dtype=tf.float16, name=name + "_min_values"
            )
            self._max_values = tf.Variable(
                max_values, dtype=tf.float16, name=name + "_max_values"
            )
            self._min_indices = tf.Variable(min_indices, name=name + "_min_indices")
            self._max_indices = tf.Variable(max_indices, name=name + "_max_indices")

    @tf.function(jit_compile=True)
    def get_value(self) -> tf.Tensor:

        array = unpack_int8_to_int4(tf.cast(self._value, tf.uint8), self._ndim)

        var = self._bias + tf.cast(array, tf.float32) * self._scale
        if self._percentile > 0:
            min_values = tf.cast(self._min_values, tf.float32)
            max_values = tf.cast(self._max_values, tf.float32)
            var = tf.tensor_scatter_nd_update(var, self._min_indices, min_values)
            var = tf.tensor_scatter_nd_update(var, self._max_indices, max_values)
        return var

    def get_memory_usage(self) -> float:
        used_memory = mb_used(self._value)
        if self._percentile > 0:
            used_memory += mb_used(self._min_values)
            used_memory += mb_used(self._max_values)
            used_memory += mb_used(self._min_indices)
            used_memory += mb_used(self._max_indices)
        return used_memory


def pack_int4_to_int8(array: np.ndarray) -> tf.Tensor:
    nbits = 4
    bits_values = []

    for bit in range(nbits):
        bits_values.append((array >> bit) & 1)

    array_bits = tf.cast(tf.stack(bits_values, axis=-1), tf.bool)

    size = array_bits.shape[-2]
    array1 = array_bits[..., : size // 2, :]
    array2 = array_bits[..., size // 2 :, :]

    array = tf.concat([array1, array2], axis=-1)

    powers = tf.reshape(2 ** np.arange(8), (*([1] * (array.ndim - 1)), 8))
    powers = tf.cast(powers, tf.uint8)
    array = tf.reduce_sum(tf.cast(array, tf.uint8) * powers, axis=-1)
    return array


def unpack_int8_to_int4(array: tf.Tensor, ndim: int) -> tf.Tensor:
    nbits = 4
    powers = tf.reshape(2 ** np.arange(nbits), (*([1] * ndim), nbits))
    powers = tf.cast(powers, tf.uint8)
    uncompressed_array = tf.cast(array, tf.uint8)
    uncompressed_array = tf.expand_dims(uncompressed_array, axis=-1)
    uncompressed_array = tf.bitwise.right_shift(uncompressed_array, np.arange(8))
    uncompressed_array = tf.bitwise.bitwise_and(uncompressed_array, 1)

    array1 = uncompressed_array[..., :4]
    array2 = uncompressed_array[..., 4:]
    values = tf.concat([array1, array2], axis=-2)

    array = tf.reduce_sum(tf.cast(values, tf.uint8) * powers, axis=-1)
    return array
