import logging
from typing import Union

from keras_llm_light import blocks_ops
from keras_llm_light import variables
from keras_llm_light import lora
import keras
from tqdm import tqdm

from keras_nlp.layers import ReversibleEmbedding
from keras_nlp.src.models.gemma.gemma_decoder_block import GemmaDecoderBlock
from keras_nlp.src.models.gemma.rms_normalization import RMSNormalization
import keras_nlp
import tensorflow as tf
import numpy as np

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

DUMMY_SEQUENCE_LENGTH = 64


def build_gemma_llm(
    lora_rank: int = 4,
    lora_alpha: float = 32.0,
    max_sequence_len: int = 512,
    optimizer: keras.optimizers.Optimizer = keras.optimizers.Adam(learning_rate=0.0001),
    preset: str = "gemma_2b_en",
) -> tuple[blocks_ops.Trainer, blocks_ops.LLM]:
    gemma_llm = build_gemma_llm_cpu(
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        preset=preset,
    )
    gemma_llm.preprocessor.sequence_length = max_sequence_len

    keras.mixed_precision.set_global_policy("mixed_float16")
    preprocessing_fn, token_embedding_layer = build_gemma_preprocessing_fn(gemma_llm)
    transformer_block = build_gemma_decoder_block(
        gemma_llm=gemma_llm,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
    )
    postprocessing_fn = build_gemma_postprocessing_fn(gemma_llm)
    model = build_block_wise_gemma(
        gemma_llm=gemma_llm,
        token_embeddings=token_embedding_layer,
        preprocessing_fn=preprocessing_fn,
        transformer_block=transformer_block,
        postprocessing_fn=postprocessing_fn,
    )
    model.block_model.print_weights_memory_usage()

    trainer = blocks_ops.Trainer(
        model.token_embeddings.weights[0].value,
        postprocessing_fn=model.postprocessing_fn,
        blocks_weights=model.block_model.blocks_weights,
        optimizer=optimizer,
    )

    return trainer, model


def build_block_wise_gemma(
    gemma_llm: keras_nlp.models.GemmaCausalLM,
    token_embeddings: Union[keras.layers.Embedding, ReversibleEmbedding],
    preprocessing_fn: keras.layers.Layer,
    transformer_block: Union[keras.layers.Layer, GemmaDecoderBlock],
    postprocessing_fn: Union[keras.layers.Layer, RMSNormalization],
) -> blocks_ops.LLM:
    trainable_variables_names = [
        "decoder_block/attention/query/lora_A/kernel",
        "decoder_block/attention/query/lora_B/kernel",
        "decoder_block/attention/value/lora_A/kernel",
        "decoder_block/attention/value/lora_B/kernel",
    ]

    trainable_weights = [
        v for v in transformer_block.weights if v.path in trainable_variables_names
    ]
    print("Single block trainable weights:")
    for v in trainable_weights:
        print(f"- {v.path}")

    print("Collecting transformer block weights ...")
    blocks_weights = []
    for block in tqdm(gemma_llm.backbone.transformer_layers, desc="Preparing/Quantizing weights"):
        block_weights = []
        for w in block.weights:
            if "lora_" in w.path:
                block_weights.append(variables.FP32Variable(w, trainable=True))
            elif "ffw_" in w.path:
                block_weights.append(variables.Int8Variable(w))
            else:
                block_weights.append(variables.FP16Variable(w))

        blocks_weights.append(block_weights)

    trainable_weights = [
        layer_weight
        for layer_weight, block_weight in zip(
            transformer_block.weights, blocks_weights[0]
        )
        if block_weight.trainable
    ]

    block_model = blocks_ops.ModelBlock(
        transformer_block,
        blocks_weights,
        trainable_weights=trainable_weights,
        padding_mask_key="padding_mask",
    )

    return blocks_ops.LLM(
        gemma_llm.preprocessor,
        token_embeddings=token_embeddings,
        preprocessing_fn=preprocessing_fn,
        block_model=block_model,
        postprocessing_fn=postprocessing_fn,
    )


def build_gemma_llm_cpu(
    lora_rank: int = 4,
    lora_alpha: float = 32.0,
    preset: str = "gemma_2b_en",
) -> keras_nlp.models.GemmaCausalLM:
    # Prepare CPU version of the model to later copy pretrained model weights to GPU
    print("Building Gemma 2B model on CPU ...")

    with tf.device("cpu"):
        llm = keras_nlp.models.GemmaCausalLM.from_preset(preset)
        llm.summary()

        for layer_idx in range(llm.backbone.num_layers):
            decoder_layer = llm.backbone.get_layer(f"decoder_block_{layer_idx}")
            self_attention_layer = decoder_layer.attention
            self_attention_layer._tracker.locked = False
            self_attention_layer.query_dense = lora.LoraLayer(
                self_attention_layer.query_dense, rank=lora_rank, alpha=lora_alpha
            )
            self_attention_layer.value_dense = lora.LoraLayer(
                self_attention_layer.value_dense, rank=lora_rank, alpha=lora_alpha
            )

        input_data = {
            "token_ids": np.ones(shape=(1, 12), dtype="int32"),
            "padding_mask": np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]),
        }
        llm.backbone(input_data)
    return llm


def build_gemma_preprocessing_fn(
    gemma_llm: keras_nlp.models.GemmaCausalLM,
) -> tuple[keras.Model, ReversibleEmbedding]:
    hidden_dim = gemma_llm.backbone.hidden_dim
    vocabulary_size = gemma_llm.preprocessor.tokenizer.vocabulary_size()

    token_id_input = keras.Input(shape=(None,), dtype="int32", name="token_ids")
    token_embedding = ReversibleEmbedding(
        input_dim=vocabulary_size,
        output_dim=hidden_dim,
        tie_weights=True,
        dtype="float16",
        name="token_embedding",
    )
    x = token_embedding(token_id_input)
    x = keras.layers.Lambda(lambda _: tf.cast(_, tf.float16))(x)
    x = x * keras.ops.cast(keras.ops.sqrt(hidden_dim), x.dtype)
    embeddings_op_layer = tf.keras.Model(
        inputs={"token_ids": token_id_input}, outputs=x
    )
    with tf.device("CPU"):
        embeddings_op_layer(tf.zeros((1, DUMMY_SEQUENCE_LENGTH)))
        embeddings_op_layer.set_weights(
            gemma_llm.backbone.token_embedding.get_weights()
        )

    return embeddings_op_layer, token_embedding


def build_gemma_decoder_block(
    gemma_llm: keras_nlp.models.GemmaCausalLM,
    lora_rank: int = 4,
    lora_alpha: float = 32.0,
) -> GemmaDecoderBlock:
    hidden_dim = gemma_llm.backbone.hidden_dim
    intermediate_dim = gemma_llm.backbone.intermediate_dim
    num_query_heads = gemma_llm.backbone.num_query_heads
    head_dim = gemma_llm.backbone.head_dim
    num_key_value_heads = gemma_llm.backbone.num_key_value_heads
    dropout = gemma_llm.backbone.dropout

    transformer_block_layer = GemmaDecoderBlock(
        intermediate_dim=intermediate_dim,
        hidden_dim=hidden_dim,
        num_query_heads=num_query_heads,
        head_dim=head_dim,
        num_key_value_heads=num_key_value_heads,
        dropout=dropout,
        name="decoder_block",
    )

    transformer_block_layer(tf.zeros((1, DUMMY_SEQUENCE_LENGTH, hidden_dim)))

    self_attention_layer = transformer_block_layer.attention
    self_attention_layer._tracker.locked = False
    self_attention_layer.query_dense = lora.LoraLayer(
        self_attention_layer.query_dense, rank=lora_rank, alpha=lora_alpha
    )
    self_attention_layer.value_dense = lora.LoraLayer(
        self_attention_layer.value_dense, rank=lora_rank, alpha=lora_alpha
    )
    transformer_block_layer(tf.zeros((1, DUMMY_SEQUENCE_LENGTH, hidden_dim)))
    return transformer_block_layer


def build_gemma_postprocessing_fn(
    gemma_llm: keras_nlp.models.GemmaCausalLM,
) -> RMSNormalization:
    hidden_dim = gemma_llm.backbone.hidden_dim
    postprocessing_op_layer = RMSNormalization(
        name="final_normalization",
        epsilon=1e-06,
    )
    postprocessing_op_layer(tf.zeros((1, DUMMY_SEQUENCE_LENGTH, hidden_dim)))
    postprocessing_op_layer.set_weights(gemma_llm.backbone.layer_norm.get_weights())
    return postprocessing_op_layer
