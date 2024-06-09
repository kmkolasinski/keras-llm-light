# keras-llm-light

## Description

This project provides simple method of training LLM on limited GPU resources. With
this code I was able to run example finetuning of `gemma_2b_en` model with about ~6GB
of GPU RAM with sequence length 512 and batch size 1. I used Tensorflow 2.16.1 and Keras 3.3.3
was used for these experiments.

Techniques used:
* LoRA - to reduce the number of trainable parameters
* Custom Int8 quantization with outliers compensation - to reduce memory usage of large kernels
* Gradient checkpointing - to reduce memory usage while training the model
* XLA - to make computations faster
* Mixed Precision training - weights are in fp32 (after de-quantization) and activations are in fp16
* Memory saving loss splitting along sequence axis - to reduce memory requirements for computing final loss and gradients

## Methodology

Everything is in the code, but this section shortly explains the content of this project and tricks/methods used to fit 
training of LLM model into ~6GB of GPU RAM.

### Block-wise inference and model training

Standard LLM can be divided into three parts:
* preprocessing step which takes tokens, apply some normalization and create token embeddings tensor of shape (batch_size, seq_len, embedding_dim)
* LLM interior build from a list of block which operates on token embeddings (batch_size, seq_len, embedding_dim)
* postprocessing step which takes token embeddings and predict next token probabilities, producing tensor (batch_size, seq_len, vocab_size)

In order to train the model simple block-wise gradient checkpointing is implemented. Here is a
pseudo code which describe how forward and backward steps are computed manually for LLM using standard back-propagation

```python
x = preprocess(input_tokens)
# block wise forward pass
self._activations_cache = []
for block_id in range(self.num_blocks):
    self._activations_cache.append(x) # cache input activations used later to recompute gradients
    # to save GPU memory usage only one block is store in GPU
    self.assign_block_weights(block_id)
    x = self.block_forward(x, padding_mask)

# compute loss and loss gradient w.r.t. x tensor, compute_loss_and_gradients also perform postprocessing step
loss, output_gradients = self.compute_loss_and_gradients(labels, x, padding_mask)

# run gradients back-propagation using gradient checkpointing
variables_gradients = []
for block_id in reversed(range(self.num_blocks)):
    inputs = self._activations_cache.pop()
    self.assign_block_weights(block_id)
    output_gradients, vars_grads = self.block_backward(inputs, padding_mask, output_gradients)
    variables_gradients.append(vars_grads)

variables_gradients = variables_gradients[::-1]
```

Notes:
* gradients are computed only for selected weights, in my case these are only LoRA parameters, which take a little memory as compared to other parameters,
* only transformers blocks input activations are stored on GPU in float16 prediction,
* token embeddings layer is not trained, also to save memory it is converted to fp16 dtype
* gradient checkpointing requires to compute forward pass for each block two times while training: forward and backward step.


### Weights quantization

* only LLM blocks weights are quantized to reduce the memory footprint
* large kernels are quantized with simplified version of [LLM.int8()](https://arxiv.org/abs/2208.07339) quantization method
* smaller kernels are stored in fp16
* LoRA weights are **stored in fp32** which improves training stability. I was getting nan when LoRA weights were fp16.
* Here is an example block summary for Gemma 2B
```
                                        Block Memory Usage (Num blocks 18)
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━┓
┃                                                 ┃             ┃         ┃   Mem Usage ┃    FP32 Mem ┃           ┃
┃                                          Tensor ┃       Shape ┃   dtype ┃        [MB] ┃  Usage [MB] ┃ Trainable ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━┩
│          decoder_block/pre_attention_norm/scale │      [2048] │ float16 │       0.004 │       0.008 │     False │
│            decoder_block/attention/query/kernel │   [8, 2048, │ float16 │       8.000 │      16.000 │     False │
│                                                 │        256] │         │             │             │           │
│              decoder_block/attention/key/kernel │   [1, 2048, │ float16 │       1.000 │       2.000 │     False │
│                                                 │        256] │         │             │             │           │
│            decoder_block/attention/value/kernel │   [1, 2048, │ float16 │       1.000 │       2.000 │     False │
│                                                 │        256] │         │             │             │           │
│ decoder_block/attention/attention_output/kernel │    [8, 256, │ float16 │       8.000 │      16.000 │     False │
│                                                 │       2048] │         │             │             │           │
│     decoder_block/attention/query/lora_A/kernel │   [2048, 4] │ float32 │       0.031 │       0.031 │      True │
│     decoder_block/attention/query/lora_B/kernel │ [8, 4, 256] │ float32 │       0.031 │       0.031 │      True │
│     decoder_block/attention/value/lora_A/kernel │   [2048, 4] │ float32 │       0.031 │       0.031 │      True │
│     decoder_block/attention/value/lora_B/kernel │ [1, 4, 256] │ float32 │       0.004 │       0.004 │      True │
│                decoder_block/pre_ffw_norm/scale │      [2048] │   uint8 │       0.002 │       0.008 │     False │
│                 decoder_block/ffw_gating/kernel │      [2048, │   uint8 │      33.130 │     128.000 │     False │
│                                                 │      16384] │         │             │             │           │
│               decoder_block/ffw_gating_2/kernel │      [2048, │   uint8 │      33.131 │     128.000 │     False │
│                                                 │      16384] │         │             │             │           │
│                 decoder_block/ffw_linear/kernel │     [16384, │   uint8 │      33.138 │     128.000 │     False │
│                                                 │       2048] │         │             │             │           │
└─────────────────────────────────────────────────┴─────────────┴─────────┴─────────────┴─────────────┴───────────┘
```

### Simplified Int8Variable quantization

* In the original int8 paper the authors decompose the layer and activations into two matrices of int8 and fp16 types and they use fp16 part to compensate for potential outlier values.
* My Int8 implementation is inspired by the authors.

    1) weights are quantized by removing outliers using 99.9 percentile cutoff.
    2) outlier values are stored in a sparse format (row, col, value) using higher fp16 precision
    3) before computing layer call method weights are reconstructed using following code

    ```python
    # standard de-quantization from int8 to fp32
    var = self._bias + tf.cast(self._value, tf.float32) * self._scale
    # replace outliers with exact values
    min_values = tf.cast(self._min_values, tf.float32)
    max_values = tf.cast(self._max_values, tf.float32)
    var = tf.tensor_scatter_nd_update(var, self._min_indices, min_values)
    var = tf.tensor_scatter_nd_update(var, self._max_indices, max_values)
    ```
* See [tests/test_variables.py](tests/test_variables.py) for more details


### Simplified Int4Variable quantization

* To further reduce the memory usage, one may use int4 quantization to store model's weights.
* This implementation store two int4 parameters inside single int8 variable.
* This variable consumes twice less memory than int8 at cost of the computation speed.


### Fitting loss computation step in the GPU RAM

Large portion of the total model GPU memory usage is consumed by the token embedding layer, which is used in the final step of the model. Computing the final layer tokens logits and CE loss can allocate a lot of memory when used naively. In order to reduce the memory usage for this step I split tokens sequence into N parts and compute output gradients for each part separately. For example, if the final output layer activations tensor is of shape [1, 1024, 2048] and num_splits=2, then the loss and gradients is computed first for [1, :512, 2048] tokens and then for [1, 512:, 2048]. This requires half less memory. A pseudo code which realizes this logic can be written in following form:

```python
# split embeddings and target labels along the second axis
inputs_splits = tf.split(inputs, num_splits, axis=1)
labels_splits = tf.split(labels, num_splits, axis=1)
sample_weight_splits = tf.split(sample_weight, num_splits, axis=1)

output_gradients_splits = []
loss_splits = []
for inputs, labels, sample_weight in zip(inputs_splits, labels_splits, sample_weight_splits):
    loss, output_gradients = self.partial_train_loss_step(inputs, labels, sample_weight)
    initial_gradients_splits.append(output_gradients / num_splits) # gradients have to be scaled
    loss_splits.append(loss)

loss_value = tf.reduce_mean(loss_splits)
output_gradients = tf.concat(initial_gradients_splits, axis=1)
```



## Example finetuning of Gemma 2B model

See example notebook in [(notebooks/train-gemma-2b-imdb_reviews.ipynb)](notebooks/train-gemma-2b-imdb_reviews.ipynb)

```python
from keras_llm_light.models import gemma
import keras
import tensorflow_datasets as tfds

trainer, llm = gemma.build_gemma_llm(
    lora_rank=4,
    lora_alpha=32.,
    max_sequence_len=512,
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
)

ds = tfds.load("imdb_reviews", split="train")
train_ds = ds.map(lambda x: x["text"]).repeat(-1).shuffle(2000).batch(1).map(llm.preprocessor)
metrics = trainer.fit(llm, train_ds_processed, epochs=1, steps_per_epoch=1000)
generated_text = llm.generate("Example text ...")
```

## Limitations / Simplifications

* This implementation has very limited capabilities and was implemented only for educational purposes
* `generate` function uses very basic greedy top-1 sampler
* For simplicity KV-cache is not implemented
* I tested single model only `gemma_2b_en`, however initial experiments were performed on Bloom LLM
* Quantization layers were implemented manually and for example int8 quantization does not follow the original paper
* The code does not provide any model save/load functionality

## Resources

* https://keras.io/examples/nlp/parameter_efficient_finetuning_of_gpt2_with_lora/
* https://www.answer.ai/posts/2024-03-06-fsdp-qlora.html
* https://huggingface.co/docs/transformers/v4.18.0/en/performance#gradient-checkpointing
* https://www.kaggle.com/code/nilaychauhan/fine-tune-gemma-models-in-keras-using-lora
* https://medium.com/tensorflow/fitting-larger-networks-into-memory-583e3c758ff9
* https://huggingface.co/blog/hf-bitsandbytes-integration
