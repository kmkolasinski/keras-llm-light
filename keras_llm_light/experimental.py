import keras
import numpy as np
import scann
import tensorflow as tf



EmbeddingsShape = tf.TensorSpec(shape=[None, None, None], dtype=tf.float16)
LabelsShape = tf.TensorSpec(shape=[None, None], dtype=tf.int64)
LabelsWeightShape = tf.TensorSpec(shape=[None, None], dtype=tf.bool)
VectorsShape = tf.TensorSpec(shape=[None, None], dtype=tf.float32)


class ApproxSofmaxLoss:
    def __init__(self, token_embeddings, num_leaves: int = 100):
        num_train_embeddings = token_embeddings.shape[0]
        num_leaves_to_search = num_leaves // 10
        self.num_train_embeddings = num_train_embeddings
        self.token_embeddings = token_embeddings

        scann_builder = scann.scann_ops_pybind.builder(
            token_embeddings,  # / np.linalg.norm(token_embeddings, axis=-1, keepdims=True),
            10,
            "dot_product",
        )
        scann_builder = scann_builder.tree(
            num_leaves=num_leaves,
            num_leaves_to_search=num_leaves_to_search,
            training_sample_size=num_train_embeddings,
            min_partition_size=5,
        )
        # scann_builder = scann_builder.score_ah(
        #     dimensions_per_block=2,
        #     anisotropic_quantization_threshold=0.2,
        #     training_sample_size=num_train_embeddings,
        # )
        scann_builder = scann_builder.score_brute_force()

        # if self.reordering_num_neighbors is not None:
        # scann_builder = scann_builder.reorder(
        #     reordering_num_neighbors=500
        # )
        searcher = scann_builder.build()
        self.searcher = searcher
        self.loss_op = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.accuracy = keras.metrics.SparseCategoricalAccuracy()

    @tf.function(
        input_signature=[EmbeddingsShape, LabelsShape, LabelsWeightShape, VectorsShape]
    )
    def get_fast_loss_grads(
        self, embeddings, new_labels, sample_weight, partial_token_embeddings
    ):
        print("Tracing!")
        # with tf.device("CPU"):
        embeddings = tf.cast(embeddings, tf.float32)
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(embeddings)
            new_logits = tf.matmul(embeddings, tf.transpose(partial_token_embeddings))
            new_loss = self.loss_op(new_labels, new_logits, sample_weight=sample_weight)

        self.accuracy.update_state(new_labels, new_logits, sample_weight=sample_weight)
        gradiens = tape.gradient(new_loss, embeddings)
        return new_loss, self.accuracy.result(), gradiens

    def compute_loss(
        self,
        embeddings,
        labels,
        sample_weight,
        final_num_neighbors: int = 100,
        leaves_to_search: int = 5,
        num_samples: int = 5000,
    ):
        # normed_embeddings = tf.math.l2_normalize(tf.cast(embeddings, tf.float32), axis=-1)
        indices, distances = self.searcher.search_batched_parallel(
            embeddings.numpy().reshape(-1, embeddings.shape[-1]),
            final_num_neighbors=final_num_neighbors,
            leaves_to_search=leaves_to_search,
        )

        unique_target_labels = np.unique(labels.numpy())
        unique_predicted_labels = np.unique(indices)
        unique_indices = np.union1d(unique_target_labels, unique_predicted_labels)

        random_indices = np.unique(
            np.random.randint(0, self.num_train_embeddings, size=(num_samples,))
        )
        unique_indices = np.union1d(unique_indices, random_indices)

        index_to_new_index = {idx: i for i, idx in enumerate(unique_indices)}

        new_labels = np.vectorize(index_to_new_index.get)(labels.numpy())
        new_labels = tf.convert_to_tensor(new_labels)

        new_loss, new_accuracy, new_gradients = self.get_fast_loss_grads(
            embeddings,
            new_labels,
            sample_weight,
            tf.constant(self.token_embeddings[unique_indices]),
        )
        return new_loss, new_accuracy, new_gradients



class TransformerBlock:
    def __init__(
        self,
        block_layer: tf.keras.Model | tf.keras.layers.Layer,
        blocks_weights: list[list[Variable]],
        trainable_weights: list[tf.Variable],
        padding_mask_key: str = "padding_mask",
        max_seq_length: int = 32,
    ):
        self.block_layer = block_layer
        self.blocks_weights = blocks_weights
        self.padding_mask_key = padding_mask_key
        self.trainable_weights = trainable_weights
        self.max_seq_length = max_seq_length
        self._forward_cache = {}

    @property
    def num_blocks(self) -> int:
        return len(self.blocks_weights)

    def print_weights_memory_usage(self):
        block_weights = self.blocks_weights[0]
        total_memory = 0
        for layer_weight, block_weight in zip(self.block_layer.weights, block_weights):
            memory_usage = block_weight.get_memory_usage()
            total_memory += memory_usage
            print(
                f"{layer_weight.path:<60} => {memory_usage:6.3f} MB  {block_weight.dtype.name}"
            )
        print(
            f"Total weights memory usage per single block: {total_memory:6.3f} MB (num blocks = {self.num_blocks})"
        )

    # @tf.function(jit_compile=True)
    def assign_block_weights(self, block_id: int):
        block_weights = self.blocks_weights[block_id]
        assign_ops = [
            # make get_value class method or static method
            weight.value.assign(w.get_value())
            for weight, w in zip(self.block_layer.weights, block_weights)
        ]
        return assign_ops

    @tf.function(jit_compile=True)
    def block_forward(self, x, padding_mask, training=False):
        print("Tracing block_forward")
        outputs = self.block_layer(
            x, **{self.padding_mask_key: padding_mask}, training=training
        )

        return outputs

    @tf.function(jit_compile=True)
    def block_backward(self, x, padding_mask, output_gradients=None):
        print("Tracing block_backward")
        inputs = x
        with tf.GradientTape(False, False) as tape:
            tape.watch(inputs)
            tape.watch([v.value for v in self.trainable_weights])
            outputs = self.block_layer(inputs, **{self.padding_mask_key: padding_mask})
            grads_outputs = tape.gradient(
                outputs,
                [inputs] + [v.value for v in self.trainable_weights],
                output_gradients=output_gradients,
            )
        return outputs, grads_outputs[0], grads_outputs[1:]

    def forward(self, inputs, padding_mask):
        x = inputs
        self._cache = []
        for block_id in range(self.num_blocks):
            self._cache.append(x)
            self.assign_block_weights(block_id)
            x = self.block_forward(x, padding_mask)

        return x

    def backward(self, padding_mask, output_gradients=None):
        variables_gradients = []
        for block_id in reversed(range(self.num_blocks)):
            inputs = self._cache.pop()
            self.assign_block_weights(block_id)
            _, output_gradients, vars_grads = self.block_backward(
                inputs, padding_mask, output_gradients
            )
            variables_gradients.append(vars_grads)

        variables_gradients = variables_gradients[::-1]
        return output_gradients, variables_gradients

    @tf.function(
        jit_compile=True,
        input_signature=[
            tf.TensorSpec(shape=[1, None, 2560], dtype=tf.float16),
            tf.TensorSpec(shape=[1, 32], dtype=tf.bool),
            tf.TensorSpec(shape=[1, 2, 32, 32, 80], dtype=tf.float16),
            tf.TensorSpec(shape=(), dtype=tf.int64),
        ],
    )
    def block_forward_with_cache(self, x, padding_mask, cache, cache_update_index: int):
        print("Tracing block_forward_with_cache")
        outputs, cache = self.block_layer(
            x,
            **{self.padding_mask_key: padding_mask},
            cache=cache,
            cache_update_index=cache_update_index,
        )
        return outputs, cache

    def get_or_create_cache(self, block_id: int):
        if block_id in self._forward_cache:
            return self._forward_cache[block_id]
        head_dim = self.block_layer._self_attention_layer.head_dim
        num_heads = self.block_layer._self_attention_layer.num_heads
        cache = tf.zeros(
            (1, 2, self.max_seq_length, num_heads, head_dim), dtype=tf.float16
        )
        return cache

    def forward_with_cache(self, inputs, padding_mask, cache_update_index: int):
        x = inputs
        for block_id in range(self.num_blocks):
            self.assign_block_weights(block_id)
            cache = self.get_or_create_cache(block_id)
            x, cache = self.block_forward_with_cache(
                x, padding_mask, cache, cache_update_index
            )
            self._forward_cache[block_id] = cache

        return x