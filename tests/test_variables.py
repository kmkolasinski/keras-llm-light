import numpy as np
import tensorflow as tf
import keras_llm_light.variables as vars


class TestVariables(tf.test.TestCase):
    def test_FP32Variable(self):
        input_variable = tf.Variable([-1.0, 0.0, -1.0])
        variable = vars.FP32Variable(input_variable)
        reconstructed_value = variable.get_value()
        self.assertEqual(reconstructed_value.dtype, tf.float32)

        self.assertAllClose(reconstructed_value, input_variable)
        self.assertAllEqual(variable._value, tf.Variable([-1.0, 0.0, -1.0]))
        self.assertEqual(variable._value.dtype, tf.float32)

    def test_Int8Variable(self):
        input_variable = tf.Variable([-1, 0, -1])
        variable = vars.Int8Variable(input_variable, percentile=0)
        reconstructed_value = variable.get_value()
        self.assertEqual(reconstructed_value.dtype, tf.float32)

        self.assertAllClose(reconstructed_value, input_variable)
        self.assertAllEqual(variable._value, tf.Variable([0, 255, 0], dtype=tf.uint8))
        self.assertEqual(variable._value.dtype, tf.uint8)
        self.assertAllEqual(variable._scale, tf.Variable(1.0 / 255.0))
        self.assertAllEqual(variable._bias, tf.Variable(-1.0))

    def test_Int8Variable_random_tensor(self):
        values = tf.cast(tf.random.uniform([32, 1000], 0, 1) * 256, dtype=tf.uint8)
        input_variable = tf.Variable(values)
        variable = vars.Int8Variable(input_variable, percentile=0)
        reconstructed_value = variable.get_value()
        self.assertAllClose(reconstructed_value, input_variable)

    def test_Int8Variable_quantization_error(self):
        input_variable = tf.Variable([-1, 0, -1, 1000.0])
        variable = vars.Int8Variable(input_variable, percentile=0)
        reconstructed_value = variable.get_value()

        self.assertAllClose(
            reconstructed_value, tf.Variable([-1.0, -1.0, -1.0, 1000.0])
        )
        self.assertAllEqual(
            variable._value, tf.Variable([0, 0, 0, 255], dtype=tf.uint8)
        )
        self.assertEqual(variable._value.dtype, tf.uint8)
        self.assertAllEqual(variable._scale, tf.Variable(1001.0 / 255.0))
        self.assertAllEqual(variable._bias, tf.Variable(-1.0))

    def test_Int8Variable_with_percentile(self):
        input_variable = tf.Variable([-1, 0, -1, 0, -1, 0, -1, 0, -1, 1000.0])
        variable = vars.Int8Variable(input_variable, percentile=10.0)
        reconstructed_value = variable.get_value()

        self.assertAllClose(reconstructed_value, input_variable)
        self.assertAllEqual(
            variable._value,
            tf.Variable([0, 255, 0, 255, 0, 255, 0, 255, 0, 255], dtype=tf.uint8),
        )
        self.assertEqual(variable._value.dtype, tf.uint8)
        self.assertAllEqual(variable._scale, tf.Variable(1.0 / 255.0))
        self.assertAllEqual(variable._bias, tf.Variable(-1.0))

        self.assertAllEqual(variable._min_values, tf.Variable([], dtype=tf.float16))
        self.assertAllEqual(
            variable._max_values, tf.Variable([1000.0], dtype=tf.float16)
        )
        self.assertAllEqual(
            variable._min_indices, tf.Variable(tf.zeros([0, 1], dtype=tf.int64))
        )
        self.assertAllEqual(variable._max_indices, tf.Variable([[9]], dtype=tf.int64))

    def test_Int8Variable_with_percentile_error(self):
        array = np.random.uniform(0, 1, size=(1000, 1000)).astype(np.float16)
        array[array > 0.99] *= 100.0
        test_vectors = np.random.uniform(-1, 1, size=(1000, 1000)).astype(np.float16)

        input_variable = tf.Variable(array)
        variable_no_outliers = vars.Int8Variable(input_variable, percentile=0)
        variable_with_outliers = vars.Int8Variable(input_variable, percentile=1)

        exact_result = array @ test_vectors
        no_outliers_result = variable_no_outliers.get_value().numpy() @ test_vectors
        with_outliers_result = variable_with_outliers.get_value().numpy() @ test_vectors
        error_0 = np.abs(exact_result - no_outliers_result).mean()
        error_1 = np.abs(exact_result - with_outliers_result).mean()
        self.assertGreater(error_0 / error_1, 10)

    def test_pack_and_unpack_int4_to_int8(self):

        array = np.array(range(16), dtype=np.uint8)
        for shape in [(-1, ), (1, -1)]:
            array = array.reshape(shape)
            array_packed = vars.pack_int4_to_int8(array)
            reconstructed_array = vars.unpack_int8_to_int4(array_packed, array.ndim)
            self.assertEqual(array_packed.dtype, tf.uint8)
            self.assertAllEqual(array, reconstructed_array)

    def test_Int4Variable(self):
        input_variable = tf.Variable([-1, 0, -1, 0])
        variable = vars.Int4Variable(input_variable, percentile=0)
        reconstructed_value = variable.get_value()
        self.assertEqual(reconstructed_value.dtype, tf.float32)

        self.assertAllClose(reconstructed_value, input_variable)
        self.assertAllEqual(variable._value, tf.Variable([0, 255], dtype=tf.uint8))
        self.assertEqual(variable._value.dtype, tf.uint8)
        self.assertAllEqual(variable._scale, tf.Variable(1.0 / 15.0))
        self.assertAllEqual(variable._bias, tf.Variable(-1.0))

    def test_Int4Variable_random_tensor(self):
        values = tf.cast(tf.random.uniform([32, 1000], 0, 1) * 16, dtype=tf.uint8)
        input_variable = tf.Variable(values)
        variable = vars.Int4Variable(input_variable, percentile=0)
        reconstructed_value = variable.get_value()
        self.assertAllClose(reconstructed_value, input_variable)