import math
import tensorflow as tf


class LoraLayer(tf.keras.layers.Layer):
    """
    LoRA layer copied from https://keras.io/examples/nlp/parameter_efficient_finetuning_of_gpt2_with_lora/
    Title: Parameter-efficient fine-tuning of GPT-2 with LoRA
    Author: [Abheesht Sharma](https://github.com/abheesht17/), [Matthew Watson](https://github.com/mattdangerw/)
    Date created: 2023/05/27
    Last modified: 2023/05/27
    Description: Use KerasNLP to fine-tune a GPT-2 LLM with LoRA.
    Accelerator: GPU
    """
    def __init__(
        self,
        original_layer,
        rank: int = 8,
        alpha: float = 32.0,
        trainable: bool = True,
        **kwargs,
    ):
        # We want to keep the name of this layer the same as the original
        # dense layer.
        original_layer_config = original_layer.get_config()
        name = original_layer_config["name"]

        kwargs.pop("name", None)

        super().__init__(name=name, trainable=trainable, **kwargs)

        self.rank = rank
        self.alpha = alpha

        self._scale = alpha / rank

        self._num_heads = original_layer_config["output_shape"][-2]
        self._hidden_dim = self._num_heads * original_layer_config["output_shape"][-1]

        # Original dense layer.
        self.original_layer = original_layer
        # No matter whether we are training the model or are in inference mode,
        # this layer should be frozen.
        self.original_layer.trainable = False

        # LoRA dense layers.
        self.A = tf.keras.layers.Dense(
            units=rank,
            use_bias=False,
            # Note: the original paper mentions that normal distribution was
            # used for initialization. However, the official LoRA implementation
            # uses "Kaiming/He Initialization".
            kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=math.sqrt(5), mode="fan_in", distribution="uniform"
            ),
            trainable=trainable,
            name="lora_A",
        )
        # B has the same `equation` and `output_shape` as the original layer.
        # `equation = abc,cde->abde`, where `a`: batch size, `b`: sequence
        # length, `c`: `hidden_dim`, `d`: `num_heads`,
        # `e`: `hidden_dim//num_heads`. The only difference is that in layer `B`,
        # `c` represents `rank`.
        self.B = tf.keras.layers.EinsumDense(
            equation=original_layer_config["equation"],
            output_shape=original_layer_config["output_shape"],
            kernel_initializer="zeros",
            trainable=trainable,
            name="lora_B",
        )

    def get_config(self):
        config = super().get_config()
        config.update({"rank": self.rank, "alpha": self.alpha})
        return config

    def call(self, inputs):
        original_output = self.original_layer(inputs)
        if self.trainable:
            # If we are fine-tuning the model, we will add LoRA layers' output
            # to the original layer's output.
            lora_output = self.B(self.A(inputs)) * self._scale
            return original_output + lora_output

        raise NotImplementedError("Lora Layer inference mode is not implemented")
        # TODO requires merge implementation: https://keras.io/examples/nlp/parameter_efficient_finetuning_of_gpt2_with_lora/#merge-weights-and-generate-text
        # return original_output
