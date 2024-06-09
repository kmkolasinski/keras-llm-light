from collections import defaultdict
from itertools import chain
from typing import Optional

import keras
import numpy as np
import tensorflow as tf
from keras_nlp.src.models import Preprocessor
from tqdm import tqdm

import keras_llm_light.variables as vs


class ModelBlock:
    def __init__(
        self,
        block_layer: tf.keras.Model | tf.keras.layers.Layer,
        blocks_weights: list[list[vs.Variable]],
        trainable_weights: list[tf.Variable],
        padding_mask_key: str = "padding_mask",
    ):
        self.block_layer = block_layer
        self.blocks_weights = blocks_weights
        self.padding_mask_key = padding_mask_key
        self.trainable_weights = trainable_weights
        self._activations_cache = []

    @property
    def num_blocks(self) -> int:
        return len(self.blocks_weights)

    def print_weights_memory_usage(self):
        block_weights = self.blocks_weights[0]
        from rich.console import Console
        from rich.table import Table

        table = Table(title=f"Block Memory Usage (Num blocks {self.num_blocks})")
        table.add_column("Tensor", justify="right", style="cyan", no_wrap=True)
        table.add_column("Shape", justify="right", style="magenta")
        table.add_column("dtype", justify="right", style="green")
        table.add_column("Mem Usage [MB]", justify="right", style="green")
        table.add_column("FP32 Mem Usage [MB]", justify="right", style="green")
        table.add_column("Trainable", justify="right", style="green")
        for layer_weight, block_weight in zip(self.block_layer.weights, block_weights):
            table.add_row(
                layer_weight.path,
                str(block_weight.shape.as_list()),
                block_weight.dtype.name,
                f"{block_weight.get_memory_usage():.3f}",
                f"{block_weight.get_fp32_memory_usage():.3f}",
                str(block_weight.trainable),
            )

        console = Console()
        console.print(table)

    @tf.function(jit_compile=True)
    def assign_block_weights(self, block_id: int):
        print("Tracing assign_block_weights")
        block_weights = self.blocks_weights[block_id]
        assign_ops = [
            # make get_value class method or static method
            weight.value.assign(w.get_value())
            for weight, w in zip(self.block_layer.weights, block_weights)
        ]
        return assign_ops

    @tf.function(jit_compile=True)
    def block_forward(
        self, x: tf.Tensor, padding_mask: tf.Tensor, training: bool = False
    ) -> tf.Tensor:
        print("Tracing block_forward")
        outputs = self.block_layer(
            x, **{self.padding_mask_key: padding_mask}, training=training
        )

        return outputs

    @tf.function(jit_compile=True)
    def block_backward(
        self,
        x: tf.Tensor,
        padding_mask: tf.Tensor,
        output_gradients: Optional[tf.Tensor] = None,
    ) -> tuple[tf.Tensor, tf.Tensor, list[tf.Tensor]]:
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

    def forward(
        self, inputs: tf.Tensor, padding_mask: tf.Tensor, training: bool = False
    ) -> tf.Tensor:
        x = inputs
        self._activations_cache = []
        for block_id in range(self.num_blocks):
            self._activations_cache.append(x)
            self.assign_block_weights(block_id)
            x = self.block_forward(x, padding_mask, training=training)

        return x

    def backward(
        self, padding_mask: tf.Tensor, output_gradients: Optional[tf.Tensor] = None
    ) -> tuple[tf.Tensor, list[list[tf.Tensor]]]:
        if not self._activations_cache:
            raise ValueError("Run forward first.")

        variables_gradients = []
        for block_id in reversed(range(self.num_blocks)):
            inputs = self._activations_cache.pop()
            self.assign_block_weights(block_id)
            _, output_gradients, vars_grads = self.block_backward(
                inputs, padding_mask, output_gradients
            )
            variables_gradients.append(vars_grads)

        variables_gradients = variables_gradients[::-1]
        return output_gradients, variables_gradients


class LLM:
    def __init__(
        self,
        preprocessor: Preprocessor,
        token_embeddings: keras.layers.Embedding,
        preprocessing_fn: keras.Layer,
        block_model: ModelBlock,
        postprocessing_fn: keras.Layer,
    ):
        self.preprocessor = preprocessor
        self.token_embeddings = token_embeddings
        self.preprocessing_fn = preprocessing_fn
        self.block_model = block_model
        self.postprocessing_fn = postprocessing_fn

        self.embeddings_vectors = token_embeddings.weights[0].value

    @tf.function
    def preprocess(
        self, token_id_input: tf.Tensor, padding_mask: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor]:
        x = self.preprocessing_fn(token_id_input)
        x = tf.cast(x, tf.float16)
        return x, padding_mask

    def forward(
        self, token_id_input: tf.Tensor, padding_mask: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor]:
        x, padding_mask = self.preprocess(token_id_input, padding_mask)
        outputs = self.block_model.forward(x, padding_mask)
        return outputs, padding_mask

    def backward(
        self, padding_mask: tf.Tensor, initial_gradients: tf.Tensor
    ) -> list[list[tf.Tensor]]:
        _, vars_gradients = self.block_model.backward(
            padding_mask, output_gradients=initial_gradients
        )
        return vars_gradients

    @tf.function(
        jit_compile=True,
        input_signature=[tf.TensorSpec(shape=[None], dtype=tf.float16)],
    )
    def predict_next_token(self, inputs: tf.Tensor) -> tf.Tensor:
        """Speed optimized prediction of the next token using greedy approach."""
        embeddings = self.postprocessing_fn(inputs)
        scores = self.embeddings_vectors @ tf.expand_dims(embeddings, -1)
        scores = tf.squeeze(scores, -1)
        token_id = tf.argmax(scores)
        return token_id

    def generate(
        self, prompt: str, max_length: int = 512, verbose: bool = True
    ) -> tf.Tensor:

        inputs = self.preprocessor.generate_preprocess(
            prompt, sequence_length=max_length
        )
        token_id_input, padding_mask = inputs["token_ids"], inputs["padding_mask"]
        token_id_input = np.expand_dims(token_id_input, axis=0).copy()
        padding_mask = np.expand_dims(padding_mask, axis=0).copy()
        predictions = []
        index = padding_mask[0].sum()
        if index >= max_length:
            print(f"Prompt is too long >{max_length} (Max Length), prompt: {prompt}")
            return tf.constant("")
        if verbose:
            pbar = tqdm(total=max_length, desc="Generating text")
            pbar.update(index)
        while True:
            outputs, _ = self.forward(token_id_input, padding_mask)
            next_token = self.predict_next_token(outputs[0, index - 1]).numpy()
            if next_token == self.preprocessor.tokenizer.end_token_id:
                break
            token_id_input[0, index] = next_token
            padding_mask[0, index] = True
            index += 1

            if index >= max_length:
                break

            predictions.append(next_token)
            if verbose:
                pbar.update()
        if len(predictions) == 0:
            return tf.constant("")
        return self.preprocessor.tokenizer.detokenize(predictions)


class Trainer:
    def __init__(
        self,
        token_embeddings: tf.Variable,
        postprocessing_fn,
        blocks_weights: list[list[vs.Variable]],
        optimizer: keras.optimizers.Optimizer,
    ):
        self.accuracy = keras.metrics.SparseCategoricalAccuracy()
        self.loss_op = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.token_embeddings = token_embeddings
        self.postprocessing_fn = postprocessing_fn
        self.vocabulary_size = self.token_embeddings.shape[0]
        self.hidden_dim = self.token_embeddings.shape[1]

        block_model_trainable_weights = []
        for block_weights in blocks_weights:
            block_model_trainable_weights.append(
                [w.value for w in block_weights if w.trainable]
            )
        self.block_model_trainable_weights = block_model_trainable_weights
        self.optimizer = optimizer

    @tf.function(jit_compile=True)
    def partial_train_loss_step(
        self,
        inputs: tf.Tensor,
        labels: tf.Tensor,
        sample_weight: tf.Tensor,
        update_accuracy: bool = True,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:

        vocabulary_size = self.vocabulary_size
        hidden_dim = self.hidden_dim

        with tf.GradientTape(False, False) as tape:
            tape.watch(inputs)
            outputs = self.postprocessing_fn(inputs)

            seq_len = tf.shape(outputs)[1]
            outputs = tf.reshape(outputs, [-1, hidden_dim])
            logits = self.token_embeddings @ tf.transpose(outputs, [1, 0])
            logits = tf.transpose(logits, [1, 0])
            logits = tf.reshape(logits, [-1, seq_len, vocabulary_size])

            loss = self.loss_op(labels, logits, sample_weight=sample_weight)
            initial_gradients = tape.gradient(loss, inputs)
            initial_gradients = tf.cast(initial_gradients, tf.float16)

        if update_accuracy:
            self.accuracy.update_state(labels, logits)
        return loss, self.accuracy.result(), initial_gradients

    def train_loss_step(
        self,
        inputs: tf.Tensor,
        labels: tf.Tensor,
        sample_weight: tf.Tensor,
        num_splits: int = 2,
        update_accuracy: bool = True,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:

        inputs_splits = tf.split(inputs, num_splits, axis=1)
        labels_splits = tf.split(labels, num_splits, axis=1)
        sample_weight_splits = tf.split(sample_weight, num_splits, axis=1)
        initial_gradients_splits = []
        loss_splits = []
        accuracy_splits = []

        for inputs, labels, sample_weight in zip(
            inputs_splits, labels_splits, sample_weight_splits
        ):
            loss, accuracy, initial_gradients = self.partial_train_loss_step(
                inputs, labels, sample_weight, update_accuracy=update_accuracy
            )
            initial_gradients_splits.append(initial_gradients / num_splits)
            loss_splits.append(loss)
            accuracy_splits.append(accuracy)

        loss_value = tf.reduce_mean(loss_splits)
        accuracy_value = tf.reduce_mean(accuracy_splits)
        initial_gradients = tf.concat(initial_gradients_splits, axis=1)

        return loss_value, accuracy_value, initial_gradients

    @tf.function(jit_compile=True)
    def apply_gradients(self, vars_gradients: list[tf.Tensor]):
        print("Trace apply gradients")
        variables = list(chain.from_iterable(self.block_model_trainable_weights))
        self.optimizer.apply_gradients(zip(vars_gradients, variables))

    def fit(
        self,
        model: LLM,
        train_dataset,
        epochs: int = 1,
        steps_per_epoch: int = 1000,
        num_loss_splits: int = 2,
    ):
        metrics_hist = defaultdict(list)

        for epoch in range(epochs):
            for i, documents in tqdm(enumerate(train_dataset)):
                if i >= steps_per_epoch:
                    break
                features, labels, sample_weight = documents
                token_id_input = features["token_ids"]
                padding_mask = features["padding_mask"]

                outputs, _ = model.forward(token_id_input, padding_mask)

                loss_value, accuracy_value, initial_gradients = self.train_loss_step(
                    outputs, labels, sample_weight, num_splits=num_loss_splits
                )
                vars_gradients = model.backward(padding_mask, initial_gradients)

                # tf function requires flat list of tensors
                self.apply_gradients(list(chain.from_iterable(vars_gradients)))

                if i % 10 == 0:
                    print(epoch, i, loss_value.numpy(), accuracy_value.numpy())
                metrics_hist["loss"].append(loss_value.numpy())
                metrics_hist["accuracy"].append(accuracy_value.numpy())
            print(epoch, loss_value.numpy(), accuracy_value.numpy())
        return metrics_hist
