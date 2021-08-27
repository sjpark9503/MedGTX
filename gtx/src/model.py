# coding=utf-8
# Copyright 2018 Hao Tan, Mohit Bansal, and the HuggingFace team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch GTX model. """


import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss, SmoothL1Loss
import torch.nn.functional as F

from configuration import GTXConfig
from transformers.activations import ACT2FN, gelu
from transformers.file_utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    #add_start_docstrings_to_callable,
    replace_return_docstrings,
)
from transformers.modeling_utils import PreTrainedModel
# Logging tool
from utils.notifier import logging, log_formatter
notifier = logging.getLogger(__name__)
notifier.addHandler(log_formatter())

class GeLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return gelu(x)


@dataclass
class GTXModelOutput(ModelOutput):
    """
    GTX's outputs that contain the last hidden states, pooled outputs, and attention probabilites for the language,
    visual, and, cross-modality encoders. (note: the visual encoder in GTX is referred to as the "relation-ship"
    encoder")


    Args:
        language_output (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the language encoder.
        vision_output (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the visual encoder.
        pooled_output (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification, CLS, token) further processed
            by a Linear layer and a Tanh activation function. The Linear
        language_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for input features + one for the output of each cross-modality
            layer) of shape :obj:`(batch_size, sequence_length, hidden_size)`.
        vision_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for input features + one for the output of each cross-modality
            layer) of shape :obj:`(batch_size, sequence_length, hidden_size)`.
        language_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the
            weighted average in the self-attention heads.
        vision_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the
            weighted average in the self-attention heads.
        cross_encoder_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the
            weighted average in the self-attention heads.
    """

    language_output: Optional[torch.FloatTensor] = None
    kg_output: Optional[torch.FloatTensor] = None
    pooled_output: Optional[torch.FloatTensor] = None
    language_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    kg_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    language_attentions: Optional[Tuple[torch.FloatTensor]] = None
    kg_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class GTXForPreTrainingOutput(ModelOutput):
    """
    Output type of :class:`~transformers.GTXForPreTrainingModel`.

    Args:
        loss (`optional`, returned when ``labels`` is provided, ``torch.FloatTensor`` of shape :obj:`(1,)`):
            Total loss as the sum of the masked language modeling loss and the next sequence prediction
            (classification) loss.
        prediction_logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        cross_relationship_score: (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, 2)`):
            Prediction scores of the textual matching objective (classification) head (scores of True/False
            continuation before SoftMax).
        question_answering_score: (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, n_qa_answers)`):
            Prediction scores of question answering objective (classification).
        language_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for input features + one for the output of each cross-modality
            layer) of shape :obj:`(batch_size, sequence_length, hidden_size)`.
        vision_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for input features + one for the output of each cross-modality
            layer) of shape :obj:`(batch_size, sequence_length, hidden_size)`.
        language_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the
            weighted average in the self-attention heads.
        vision_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the
            weighted average in the self-attention heads.
        cross_encoder_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the
            weighted average in the self-attention heads.

    """

    loss: [torch.FloatTensor] = None
    loss_dict: Optional[dict] = None
    lang_prediction_logits: Optional[torch.FloatTensor] = None
    kg_prediction_logits: Optional[torch.FloatTensor] = None
    cross_relationship_score: Optional[torch.FloatTensor] = None
    language_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    kg_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    language_attentions: Optional[Tuple[torch.FloatTensor]] = None
    kg_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class GTXForDownstreamOutput(ModelOutput):
    """
    Output type of :class:`~transformers.GTXForPreTrainingModel`.

    Args:
        loss (`optional`, returned when ``labels`` is provided, ``torch.FloatTensor`` of shape :obj:`(1,)`):
            Total loss as the sum of the masked language modeling loss and the next sequence prediction
            (classification) loss.
        prediction_logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        pooled_logits: (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, 2)`):
            Pooled logit from two [CLS] pooling token
        question_answering_score: (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, n_qa_answers)`):
            Prediction scores of question answering objective (classification).
        language_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for input features + one for the output of each cross-modality
            layer) of shape :obj:`(batch_size, sequence_length, hidden_size)`.
        vision_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for input features + one for the output of each cross-modality
            layer) of shape :obj:`(batch_size, sequence_length, hidden_size)`.
        language_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the
            weighted average in the self-attention heads.
        vision_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the
            weighted average in the self-attention heads.
        cross_encoder_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the
            weighted average in the self-attention heads.

    """

    loss: [torch.FloatTensor] = None
    loss_dict: Optional[dict] = None
    lang_prediction_logits: Optional[torch.FloatTensor] = None
    kg_prediction_logits: Optional[torch.FloatTensor] = None
    pooled_logits: Optional[torch.FloatTensor] = None
    language_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    kg_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    language_attentions: Optional[Tuple[torch.FloatTensor]] = None
    kg_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None

def load_tf_weights_in_GTX(model, config, tf_checkpoint_path):
    """Load tf checkpoints in a pytorch model."""
    try:
        import re

        import numpy as np
        import tensorflow as tf
    except ImportError:
        notifier.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    notifier.info("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        notifier.info("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split("/")
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(
            n
            in [
                "adam_v",
                "adam_m",
                "AdamWeightDecayOptimizer",
                "AdamWeightDecayOptimizer_1",
                "global_step",
            ]
            for n in name
        ):
            notifier.info("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == "kernel" or scope_names[0] == "gamma":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "output_weights":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "squad":
                pointer = getattr(pointer, "classifier")
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    notifier.info("Skipping {}".format("/".join(name)))
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        elif m_name == "kernel":
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        notifier.info("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)
    return model


class GTXEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config, input_type=None):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size[input_type], config.hidden_size)
        if config.max_position_embeddings[input_type]>0:
            self.position_embeddings = nn.Embedding(config.max_position_embeddings[input_type], config.hidden_size)
        else:
            self.position_embeddings = None
        if config.type_vocab_size[input_type]>0:
            self.token_type_embeddings = nn.Embedding(config.type_vocab_size[input_type], config.hidden_size)
        else:
            self.token_type_embeddings = None
        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
            device = input_ids.device
        else:
            input_shape = inputs_embeds.size()[:-1]
            device = inputs_embeds.device
        seq_length = input_shape[1]

        position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(input_shape)

        if token_type_ids is None and self.token_type_embeddings is not None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=position_ids.device)
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        embeddings = inputs_embeds
        if self.position_embeddings:
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        if self.token_type_embeddings:
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embeddings += token_type_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class GTXAttention(nn.Module):
    def __init__(self, config, ctx_dim=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.head_size = self.num_attention_heads * self.attention_head_size

        # visual_dim = 2048
        if ctx_dim is None:
            ctx_dim = config.hidden_size
        self.query = nn.Linear(config.hidden_size, self.head_size)
        self.key = nn.Linear(ctx_dim, self.head_size)
        self.value = nn.Linear(ctx_dim, self.head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, context, attention_mask=None, output_attentions=False):
        # notifier.warning(hidden_states.size())
        # notifier.warning(attention_mask.size())
        # notifier.warning(context.size())
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(context)
        mixed_value_layer = self.value(context)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class GTXAttentionOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor, KnowMix_indices=None):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if KnowMix_indices is None:
            hidden_states = input_tensor + hidden_states
        else:
            if isinstance(KnowMix_indices,int):
                input_tensor[:,KnowMix_indices] = input_tensor[:,KnowMix_indices]+hidden_states.squeeze(1)
            else:
                input_tensor[KnowMix_indices,:] = input_tensor[KnowMix_indices,:] + hidden_states.squeeze(1)
            hidden_states = input_tensor
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class GTXCrossAttentionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.att = GTXAttention(config)
        self.output = GTXAttentionOutput(config)

    def forward(self, input_tensor, ctx_tensor, ctx_att_mask=None, KnowMix_indices=None, output_attentions=False):
        if KnowMix_indices is None:
            output = self.att(input_tensor, ctx_tensor, ctx_att_mask, output_attentions=output_attentions)
        else:
            if isinstance(KnowMix_indices,int):
                output = self.att(input_tensor[:,KnowMix_indices].unsqueeze(1), ctx_tensor, ctx_att_mask, output_attentions=output_attentions)
            else:
                output = self.att(input_tensor[KnowMix_indices,:].unsqueeze(1), ctx_tensor[KnowMix_indices,:], ctx_att_mask[KnowMix_indices.unsqueeze(1),:].unsqueeze(1).unsqueeze(2), output_attentions=output_attentions)
        if output_attentions:
            attention_probs = output[1]
        attention_output = self.output(output[0], input_tensor, KnowMix_indices)
        outputs = (attention_output, attention_probs) if output_attentions else (attention_output,)
        return outputs


class GTXSelfAttentionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = GTXAttention(config)
        self.output = GTXAttentionOutput(config)

    def forward(self, input_tensor, attention_mask, output_attentions=False):
        # Self attention attends to itself, thus keys and querys are the same (input_tensor).
        output = self.self(
            input_tensor,
            input_tensor,
            attention_mask,
            output_attentions=output_attentions,
        )
        if output_attentions:
            attention_probs = output[1]
        attention_output = self.output(output[0], input_tensor)
        outputs = (attention_output, attention_probs) if output_attentions else (attention_output,)
        return outputs

class GTXIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class GTXOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class GTXLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = GTXSelfAttentionLayer(config)
        self.intermediate = GTXIntermediate(config)
        self.output = GTXOutput(config)

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        outputs = self.attention(hidden_states, attention_mask, output_attentions=output_attentions)
        attention_output = outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs[1:]  # add attentions if we output them
        return outputs

class GTXXLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.cross_att_type = config.cross_att_type if 'cross_att_type' in vars(config).keys() else 'cross'

        # The cross-attention Layer
        self.cross_attention = GTXCrossAttentionLayer(config)

        # Self-attention Layers
        self.lang_self_att = GTXSelfAttentionLayer(config)
        self.visn_self_att = GTXSelfAttentionLayer(config)

        # Intermediate and Output Layers (FFNs)
        self.lang_inter = GTXIntermediate(config)
        self.lang_output = GTXOutput(config)
        self.visn_inter = GTXIntermediate(config)
        self.visn_output = GTXOutput(config)

    def cross_att(
        self,
        lang_input,
        lang_attention_mask,
        visual_input,
        visual_attention_mask,
        output_x_attentions=False,
    ):
        lang_att_output = self.cross_attention(
            lang_input,
            visual_input,
            ctx_att_mask=visual_attention_mask,
            output_attentions=output_x_attentions,
        )
        visual_att_output = self.cross_attention(
            visual_input,
            lang_input,
            ctx_att_mask=lang_attention_mask,
            output_attentions=output_x_attentions,
        )
        return lang_att_output, visual_att_output
    
    def no_cross_att(
        self,
        lang_input,
        lang_attention_mask,
        visual_input,
        visual_attention_mask,
        output_x_attentions=False,
    ):
        lang_att_output = self.cross_attention(
            lang_input,
            lang_input,
            ctx_att_mask=lang_attention_mask,
            output_attentions=output_x_attentions,
        )
        visual_att_output = self.cross_attention(
            visual_input,
            visual_input,
            ctx_att_mask=visual_attention_mask,
            output_attentions=output_x_attentions,
        )
        return lang_att_output, visual_att_output
    
    def unilm_cross_att(
        self,
        lang_input,
        lang_attention_mask,
        visual_input,
        visual_attention_mask,
        output_x_attentions=False,
    ):
        lang_att_output = self.cross_attention(
            lang_input,
            visual_input,
            ctx_att_mask=visual_attention_mask, # in fact, visual_padding_masks
            output_attentions=output_x_attentions,
        )
        visual_att_output = (visual_input, None) if output_x_attentions else (visual_input,)
        return lang_att_output, visual_att_output

    def self_att(self, lang_input, lang_attention_mask, visual_input, visual_attention_mask):
        # Self Attention
        lang_att_output = self.lang_self_att(lang_input, lang_attention_mask, output_attentions=False)
        visual_att_output = self.visn_self_att(visual_input, visual_attention_mask, output_attentions=False)
        return lang_att_output[0], visual_att_output[0]

    def output_fc(self, lang_input, visual_input):
        # FC layers
        lang_inter_output = self.lang_inter(lang_input)
        visual_inter_output = self.visn_inter(visual_input)

        # Layer output
        lang_output = self.lang_output(lang_inter_output, lang_input)
        visual_output = self.visn_output(visual_inter_output, visual_input)

        return lang_output, visual_output

    def forward(
        self,
        lang_feats,
        lang_attention_mask,
        visual_feats,
        visual_attention_mask,
        visual_padding_mask,
        output_attentions=False,
    ):
        if self.cross_att_type == 'single':
            lang_att_output, visual_att_output = self.no_cross_att(
                lang_input=lang_feats,
                lang_attention_mask=lang_attention_mask,
                visual_input=visual_feats,
                visual_attention_mask=visual_padding_mask,
                output_x_attentions=output_attentions,
            )
        elif self.cross_att_type == 'unilm':
            lang_att_output, visual_att_output = self.unilm_cross_att(
                lang_input=lang_feats,
                lang_attention_mask=lang_attention_mask,
                visual_input=visual_feats,
                visual_attention_mask=visual_padding_mask,
                output_x_attentions=output_attentions,
            )
        else: # original cross attention
            lang_att_output, visual_att_output = self.cross_att(
                lang_input=lang_feats,
                lang_attention_mask=lang_attention_mask,
                visual_input=visual_feats,
                visual_attention_mask=visual_padding_mask,
                output_x_attentions=output_attentions,
            )
        attention_probs = {'txt->kg':lang_att_output[-1],
                           'kg->txt':visual_att_output[-1]}
        
        lang_att_output, visual_att_output = self.self_att(
            lang_att_output[0],
            lang_attention_mask,
            visual_att_output[0],
            visual_attention_mask,
        )

        lang_output, visual_output = self.output_fc(lang_att_output, visual_att_output)
        return (
            (
                lang_output,
                visual_output,
                attention_probs,
            )
            if output_attentions
            else (lang_output, visual_output)
        )

class GTXKnowMixLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # The cross-attention Layer
        self.cross_attention = GTXCrossAttentionLayer(config)

        # Self-attention Layers
        self.self_attention = GTXSelfAttentionLayer(config)

        # Intermediate and Output Layers (FFNs)
        self.inter = GTXIntermediate(config)
        self.output = GTXOutput(config)

    def mixup(
        self,
        inputs,
        attention_masks,
        contexts,
        KnowMix_indices,
        output_x_attentions=False,
    ):
        att_output = self.cross_attention(
            inputs,
            contexts,
            ctx_att_mask=attention_masks,
            KnowMix_indices=KnowMix_indices,
            output_attentions=output_x_attentions,
        )

        return att_output

    def self_att(self, inputs, attention_masks):
        # Self Attention
        att_output = self.self_attention(inputs, attention_masks, output_attentions=False)
        return att_output[0]

    def output_fc(self, inputs):
        # FC layers
        inter_output = self.inter(inputs)

        # Layer output
        output = self.output(inter_output, inputs)

        return output

    def forward(
        self,
        inputs,
        attention_masks,
        contexts,
        context_attention_masks,
        KnowMix_indices,
        output_attentions=False,
    ):  
        # Select fusion level
        if ("lit" in self.config.KnowMix) or ("abs" in self.config.KnowMix):
            KnowMix_indices = KnowMix_indices.bool().any(-1)
        elif "adm" in self.config.KnowMix:
            KnowMix_indices = 1
        else:
            raise NotImplementedError(f"{self.config.Knowmix} is an invalid fusion level.")

        att_output = self.mixup(
                inputs=inputs,
                attention_masks=context_attention_masks,
                contexts=contexts,
                KnowMix_indices=KnowMix_indices,
                output_x_attentions=output_attentions,
            )
        attention_probs = {'kg->txt':att_output[-1]}
        
        att_output = self.self_att(
            att_output[0],
            attention_masks,
        )

        output = self.output_fc(att_output)
        return (
            (
                output,
                attention_probs,
            )
            if output_attentions
            else (output,)
        )


# class GTXKGFeatureEncoder(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#
#     def forward(self):
#         """
#         To-Do : Some GCNs will be integrated in future
#         """
#         return None

class GTXEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        if 'encoder_type' not in vars(config).keys():
            notifier.warning("You have not specific encoder type in config, so that you don't use any kinds of LSTM")
            self.config.encoder_type = {'lang': ''}
        self.encoder_type = self.config.encoder_type['lang'].lower()

        # Number of layers
        self.num_l_layers = config.l_layers
        self.num_x_layers = config.x_layers
        self.num_r_layers = config.r_layers

        # Layers
        # Using self.layer instead of self.l_layer to support loading BERT weights.
        self.layer = nn.ModuleList([GTXLayer(config) for _ in range(self.num_l_layers)])
        notifier.warning(f"This model has a {config.cross_att_type if 'cross_att_type' in vars(config).keys() else 'cross'} type of x_attention architecture.")
        self.x_layers = nn.ModuleList([GTXXLayer(config) for _ in range(self.num_x_layers)])
        if ("lit" in self.config.KnowMix) or ("abs" in self.config.KnowMix) or ("adm" in self.config.KnowMix):
            notifier.critical(f"Use Knowledge Mixup Layer on {config.KnowMix} nodes")
            self.r_layers = nn.ModuleList([GTXKnowMixLayer(config) for _ in range(self.num_r_layers)])
        else:
            notifier.critical("Use Standard GAT Layer")
            self.r_layers = nn.ModuleList([GTXLayer(config) for _ in range(self.num_r_layers)])            
        
        # Lang Encoder Architecture
        # LSTM for generation, BiLSTM for pretraining/other donwstream tasks
        if self.encoder_type in ['bilstm', 'lstm']:
            self.convert_lang_encoder_to_RNN()

    def re_init_to_pretrained_lang_model(self):
        if isinstance(self.layer, nn.LSTM):
            notifier.warning("You've already used RNN-Style Architecture so that cannot re-init with PLMs.")
        else:
            """ If we use lm to language part, then we re-init our encoder.layer """
            plm_usage = self.config.pretrained_lang_model
            from transformers import AutoModel, AutoConfig
            if plm_usage['use_weight']:
                notifier.warning("Warm start for language part")
                self.layer = AutoModel.from_pretrained(plm_usage['model_name']).encoder.layer
            else:
                notifier.warning("Cold start for language part")
                plm_config = AutoConfig.from_pretrained(plm_usage['model_name'])
                self.layer = AutoModel.from_config(plm_config).encoder.layer
            
    def convert_lang_encoder_to_RNN(self):
        if self.encoder_type == 'lstm':
            notifier.critical("Use LSTM Decoder instead of BERT")
            self.layer = nn.LSTM(input_size=self.config.hidden_size,
                                 hidden_size=self.config.hidden_size,
                                 num_layers=1,
                                 dropout=self.config.hidden_dropout_prob,
                                 batch_first=True,
                                 bidirectional=False)
        elif self.encoder_type == 'bilstm':
            notifier.critical("Use BiLSTM Encoder instead of BERT")
            self.layer = nn.LSTM(input_size=self.config.hidden_size,
                                 hidden_size=self.config.hidden_size,
                                 num_layers=1,
                                 dropout=self.config.hidden_dropout_prob,
                                 batch_first=True,
                                 bidirectional=True)
        else:
            raise NotImplementedError("not implemented yet, a such kind of architecture for language encoder:", self.encoder_type)
        
    def forward(
        self,
        lang_feats,
        lang_attention_mask,
        kg_feats,
        kg_attention_mask,
        kg_padding_mask,
        kg_ext_input_ids=None,
        kg_ext_attention_mask=None,
        output_attentions=None,
    ):

        kg_hidden_states = ()
        language_hidden_states = ()
        kg_attentions = () if output_attentions or self.config.output_attentions else None
        language_attentions = () if output_attentions or self.config.output_attentions else None
        cross_encoder_attentions = {'txt->kg':(),'kg->txt':()} if output_attentions or self.config.output_attentions else None
        
        # Run language layers
        ## use RNN Encoder
        if self.encoder_type in ['bilstm', 'lstm']:
            l_outputs = self.layer(lang_feats)
            lang_feats = l_outputs[0]
            if self.layer.bidirectional:
                bsz, seq_len = lang_feats.shape[0], lang_feats.shape[1]
                lang_feats = lang_feats.view(bsz, seq_len, 2, -1).sum(axis=2)
            language_hidden_states = language_hidden_states + (lang_feats,)
        ## use BERT Encoder
        else:
            for layer_module in self.layer:
                l_outputs = layer_module(lang_feats, lang_attention_mask, output_attentions=output_attentions)
                lang_feats = l_outputs[0]
                language_hidden_states = language_hidden_states + (lang_feats,)
                if language_attentions is not None:
                    language_attentions = language_attentions + (l_outputs[1],)

        # Run relational layers
        ## Process the KG attention mask
        if kg_ext_attention_mask is not None:
            if len(kg_ext_attention_mask.shape) == 2:
                extended_kg_ext_attention_mask = kg_ext_attention_mask.unsqueeze(1).unsqueeze(2)
            elif len(kg_ext_attention_mask.shape) == 3:
                extended_kg_ext_attention_mask = kg_ext_attention_mask.unsqueeze(1)    
            extended_kg_ext_attention_mask = extended_kg_ext_attention_mask.to(dtype=lang_attention_mask.dtype)
            extended_kg_ext_attention_mask = (1.0 - extended_kg_ext_attention_mask) * -10000.0
        else:
            extended_kg_ext_attention_mask = None

        for layer_module in self.r_layers:
            if ("lit" in self.config.KnowMix) or ("abs" in self.config.KnowMix) or ("adm" in self.config.KnowMix):
                kg_outputs = layer_module(
                    kg_feats,
                    kg_attention_mask,
                    contexts=kg_ext_input_ids,
                    context_attention_masks=extended_kg_ext_attention_mask,
                    KnowMix_indices=kg_ext_attention_mask,
                    output_attentions=output_attentions
                )
            else:
                kg_outputs = layer_module(kg_feats, kg_attention_mask, output_attentions=output_attentions)
            kg_feats = kg_outputs[0]
            kg_hidden_states = kg_hidden_states + (kg_feats,)
            if kg_attentions is not None:
                kg_attentions = kg_attentions + (kg_outputs[1],)

        # Run cross-modality layers
        for layer_module in self.x_layers:
            x_outputs = layer_module(
                lang_feats,
                lang_attention_mask,
                kg_feats,
                kg_padding_mask,
                kg_padding_mask,
                output_attentions=output_attentions,
            )
            lang_feats, kg_feats = x_outputs[:2]
            kg_hidden_states = kg_hidden_states + (kg_feats,)
            language_hidden_states = language_hidden_states + (lang_feats,)
            if cross_encoder_attentions is not None:
                cross_encoder_attentions = {k:cross_encoder_attentions[k] + (x_outputs[2][k],) for k in cross_encoder_attentions}
        kg_encoder_outputs = (
            kg_hidden_states,
            kg_attentions if output_attentions else None,
        )
        lang_encoder_outputs = (
            language_hidden_states,
            language_attentions if output_attentions else None,
        )
        return (
            kg_encoder_outputs,
            lang_encoder_outputs,
            cross_encoder_attentions if output_attentions else None,
        )

class GTXPooler(nn.Module):
    def __init__(self, config):
        super(GTXPooler, self).__init__()
        self.multi_pooler = nn.Sequential(nn.Linear(config.hidden_size*2, config.hidden_size*2),
                                    nn.Tanh(),
                                    nn.Linear(config.hidden_size*2, config.num_labels))
        self.ce_pooler = nn.Sequential(nn.Linear(config.hidden_size*2, config.hidden_size*2),
                                    nn.Tanh(),
                                    nn.Linear(config.hidden_size*2, 2))
        self.use_ce_pooler = config.use_ce_pooler
    #def forward(self, hidden_states):
    def forward(self, kg_hidden_states, lang_hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensors = torch.cat([kg_hidden_states[:, 0],lang_hidden_states[:, 0]],dim=1)
        if self.use_ce_pooler:
            pooled_output = self.ce_pooler(first_token_tensors)
        else:
            pooled_output = self.multi_pooler(first_token_tensors)
        return pooled_output


class GTXPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(GTXPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = ACT2FN[config.hidden_act]
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class GTXLMPredictionHead(nn.Module):
    def __init__(self, config, GTX_model_embedding_weights):
        super(GTXLMPredictionHead, self).__init__()
        self.transform = GTXPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(
            GTX_model_embedding_weights.size(1),
            GTX_model_embedding_weights.size(0),
            bias=False,
        )
        self.decoder.weight = GTX_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(GTX_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states

class GTXPreTrainingHeads(nn.Module):
    def __init__(self, config, GTX_model_embedding_weights):
        super(GTXPreTrainingHeads, self).__init__()
        self.predictions = GTXLMPredictionHead(config, GTX_model_embedding_weights)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)

        return prediction_scores

class GTXPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = GTXConfig
    load_tf_weights = load_tf_weights_in_GTX
    base_model_prefix = "GTX"

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


GTX_START_DOCSTRING = r"""

    The GTX model was proposed in `GTX: Learning Cross-Modality Encoder Representations from Transformers
    <https://arxiv.org/abs/1908.07490>`__ by Hao Tan and Mohit Bansal. It's a vision and language transformer model,
    pretrained on a variety of multi-modal datasets comprising of GQA, VQAv2.0, MCSCOCO captions, and Visual genome,
    using a combination of masked language modeling, region of interest feature regression, cross entropy loss for
    question answering attribute prediction, and object tag predicition.

    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)

    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.

    Parameters:
        config (:class:`~transformers.GTXConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
"""

GTX_INPUTS_DOCSTRING = r"""

    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`~transformers.GTXTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__
        visual_feats: (:obj:`torch.FloatTensor` of shape :obj:՝(batch_size, num_visual_features, visual_feat_dim)՝):
            This input represents visual features. They ROI pooled object features from bounding boxes using a
            faster-RCNN model)

            These are currently not provided by the transformers library.
        visual_pos: (:obj:`torch.FloatTensor` of shape :obj:՝(batch_size, num_visual_features, visual_pos_dim)՝):
            This input represents spacial features corresponding to their relative (via index) visual features. The
            pre-trained GTX model expects these spacial features to be normalized bounding boxes on a scale of 0 to
            1.

            These are currently not provided by the transformers library.
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`({0})`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        visual_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`({0})`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`, `optional`):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in ``[0,
            1]``:

            - 0 corresponds to a `sentence A` token,
            - 1 corresponds to a `sentence B` token.

            `What are token type IDs? <../glossary.html#token-type-ids>`__
        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`({0}, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare GTX Model transformer outputting raw hidden-states without any specific head on top.",
    GTX_START_DOCSTRING,
)
class GTXModel(GTXPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.lang_embeddings = GTXEmbeddings(config,input_type='lang')
        self.kg_embeddings = GTXEmbeddings(config,input_type='kg')

        self.encoder = GTXEncoder(config)
        self.pooler = GTXPooler(config)

        self.init_weights()

    def get_lang_embeddings(self):
        return self.lang_embeddings.word_embeddings

    def set_lang_embeddings(self, new_embeddings):
        self.lang_embeddings.word_embeddings = new_embeddings

    def get_kg_embeddings(self):
        return self.kg_embeddings.word_embeddings

    def set_kg_embeddings(self, new_embedding, lit2word=None):
        if lit2word is None:
            if len(self.config.kg_special_token_ids)>0:
                self.kg_embeddings.word_embeddings.weight.data[len(self.config.kg_special_token_ids):,:] = new_embedding.data[:-len(self.config.kg_special_token_ids)]
            else:
                self.kg_embeddings.word_embeddings.weight.data = new_embedding.data
        else:
            litwords = torch.tensor(lit2word['input_ids'])
            litwords_notpad = torch.tensor(lit2word['attention_mask'])
            literal_idx = litwords_notpad.any(1)
            kg_emb = self.get_lang_embeddings()(litwords)
            kg_emb[~litwords_notpad] = 0
            literal_emb = kg_emb.sum(1)[literal_idx]/litwords_notpad.sum(1)[literal_idx].unsqueeze(-1)
            self.kg_embeddings.word_embeddings.weight.data[literal_idx] = literal_emb
            notifier.warning(f"Successfully initialize the KG embdding w/ subword embddings. {literal_idx.float().mean():.2f} indices initialized.")
    def forward(
        self,
        lang_input_ids=None,
        kg_input_ids=None,
        lang_inputs_embeds=None,
        kg_inputs_embeds=None,
        lang_attention_mask=None,
        kg_attention_mask=None,
        kg_padding_mask=None,
        kg_ext_input_ids = None,
        kg_ext_attention_mask = None,
        token_type_ids=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if lang_input_ids is not None and lang_inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif kg_input_ids is not None and kg_inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        if lang_attention_mask is not None:
            if len(lang_attention_mask.shape)==2: # (batch_size, seq_length)
                extended_lang_attention_mask = lang_attention_mask.unsqueeze(1).unsqueeze(2)
            elif len(lang_attention_mask.shape)==3: # (batch_size, seq_length, seq_length)
                extended_lang_attention_mask = lang_attention_mask.unsqueeze(1)
            elif len(lang_attention_mask.shape)==4: # (batch_size, 1, seq_length, seq_length)
                extended_lang_attention_mask = lang_attention_mask
            else:
                raise ValueError("Only supports (batch_size, seq_length) or (batch_size, seq_length, seq_length) or even full extended")    
        else:
            raise ValueError("there is no attention mask for langauge part")

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_lang_attention_mask = extended_lang_attention_mask.to(dtype=self.dtype)
        extended_lang_attention_mask = (1.0 - extended_lang_attention_mask) * -10000.0

        # Process the KG attention mask
        if kg_attention_mask is not None:
            if len(kg_attention_mask.shape)==3:
                # Process KG-side self attention mask
                extended_kg_attention_mask = kg_attention_mask.unsqueeze(1)
                extended_kg_attention_mask = extended_kg_attention_mask.to(dtype=self.dtype)
                extended_kg_attention_mask = (1.0 - extended_kg_attention_mask) * -10000.0
                # Process KG padding mask for cross attention
                extended_kg_padding_mask = kg_padding_mask.unsqueeze(1).unsqueeze(2)
                extended_kg_padding_mask = extended_kg_padding_mask.to(dtype=self.dtype)
                extended_kg_padding_mask = (1.0 - extended_kg_padding_mask) * -10000.0

            elif len(kg_attention_mask.shape)==4:
                # Process KG-side self attention mask
                extended_kg_attention_mask = kg_attention_mask
                extended_kg_attention_mask = extended_kg_attention_mask.to(dtype=self.dtype)
                extended_kg_attention_mask = (1.0 - extended_kg_attention_mask) * -10000.0
                # Process KG padding mask for cross attention
                extended_kg_padding_mask = kg_padding_mask.unsqueeze(1).unsqueeze(2)
                extended_kg_padding_mask = extended_kg_padding_mask.to(dtype=self.dtype)
                extended_kg_padding_mask = (1.0 - extended_kg_padding_mask) * -10000.0
            else:
                raise ValueError("Only supports seq_len X seq_len mask or batch_size X # head X seq_len X seq_len")
        else:
            # Process KG padding mask for cross attention
            extended_kg_padding_mask = kg_padding_mask.unsqueeze(1).unsqueeze(2)
            extended_kg_padding_mask = extended_kg_padding_mask.to(dtype=self.dtype)
            extended_kg_padding_mask = (1.0 - extended_kg_padding_mask) * -10000.0
            extended_kg_attention_mask = extended_kg_padding_mask.clone().detach()

        # Positional Word Embeddings
        lang_embedding_output = self.lang_embeddings(lang_input_ids, token_type_ids, lang_inputs_embeds)
        kg_embedding_output = self.kg_embeddings(kg_input_ids, None, kg_inputs_embeds)
        if kg_ext_input_ids is not None:
            kg_ext_embedding_output = self.lang_embeddings.word_embeddings(kg_ext_input_ids)
        else:
            kg_ext_embedding_output = None
        # Run GTX encoder
        encoder_outputs = self.encoder(
            lang_feats=lang_embedding_output,
            lang_attention_mask=extended_lang_attention_mask,
            kg_feats=kg_embedding_output,
            kg_attention_mask=extended_kg_attention_mask,
            kg_padding_mask=extended_kg_padding_mask,
            kg_ext_input_ids = kg_ext_embedding_output,
            kg_ext_attention_mask = kg_ext_attention_mask,
            output_attentions=output_attentions,
        )

        kg_encoder_outputs, lang_encoder_outputs = encoder_outputs[:2]
        kg_hidden_states = kg_encoder_outputs[0]
        language_hidden_states = lang_encoder_outputs[0]

        all_attentions = ()
        if output_attentions:
            language_attentions = lang_encoder_outputs[1]
            kg_attentions = kg_encoder_outputs[1]
            cross_encoder_attentions = encoder_outputs[2]
            all_attentions = (
                language_attentions,
                kg_attentions,
                cross_encoder_attentions,
            )

        hidden_states = (language_hidden_states, kg_hidden_states) if output_hidden_states else ()

        kg_output = kg_hidden_states[-1]
        lang_output = language_hidden_states[-1]
        #pooled_output = self.pooler(lang_output)
        pooled_output = self.pooler(kg_output, lang_output)

        if not return_dict:
            return (lang_output, kg_output, pooled_output) + hidden_states + all_attentions

        return GTXModelOutput(
            pooled_output=pooled_output,
            language_output=lang_output,
            kg_output=kg_output,
            language_hidden_states=language_hidden_states if output_hidden_states else None,
            kg_hidden_states=kg_hidden_states if output_hidden_states else None,
            language_attentions=language_attentions if output_attentions else None,
            kg_attentions=kg_attentions if output_attentions else None,
            cross_encoder_attentions=cross_encoder_attentions if output_attentions else None,
        )

class GTXForKGTokPredAndMaskedLM(GTXPreTrainedModel):
    def __init__(self, config, lit2word=None):
        super().__init__(config, lit2word)
        # Configuration
        self.config = config
        self.num_labels = config.num_labels
        self.num_kg_labels = config.num_kg_labels

        # GTX backbone
        self.GTX = GTXModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_kg_labels)
        self.edge_classifier = nn.Sequential(nn.Linear(config.hidden_size*2, config.hidden_size*2),
                                            nn.Tanh(),
                                            nn.Linear(config.hidden_size*2, config.num_relations))

        # Pre-training heads
        self.lm_head = GTXPreTrainingHeads(config, self.GTX.lang_embeddings.word_embeddings.weight)

        # Weight initialization
        self.init_weights()

        # Warm start KG embedding
        if not config.gcn and config.pretrained_kg_embedding:
            notifier.critical("Load pretrained embedding for translation based KG-GTX")
            new_embedding = torch.load(config.pretrained_kg_embedding)
            self.GTX.set_kg_embeddings(new_embedding)
            del new_embedding
        elif lit2word:
            self.GTX.set_kg_embeddings(None, lit2word)

        # Use Pretrained-LM in Language Part
        self.GTX.encoder.re_init_to_pretrained_lang_model()

        # Loss functions
        self.loss_fcts = {
            "l2": SmoothL1Loss(reduction="none"),
            "mse": MSELoss(reduction="none"),
            "ce": CrossEntropyLoss(),
            "tri": nn.TripletMarginLoss()#(margin=config.margin)
        }

    def forward(
        self,
        lang_input_ids=None,
        kg_input_ids=None,
        lang_inputs_embeds=None,
        kg_inputs_embeds=None,
        lang_attention_mask=None,
        kg_attention_mask=None,
        kg_padding_mask=None,
        kg_label_mask=None,
        lm_label=None,
        kg_label=None,
        cross_label=None,
        token_type_ids=None,
        rc_indeces=None,
        kg_ext_input_ids = None,
        kg_ext_attention_mask = None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
    ):
        r"""
        masked_lm_labels (``torch.LongTensor`` of shape ``(batch_size, sequence_length)``, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        obj_labels: (``Dict[Str: Tuple[Torch.FloatTensor, Torch.FloatTensor]]``, `optional`):
            each key is named after each one of the visual losses and each element of the tuple is of the shape
            ``(batch_size, num_features)`` and ``(batch_size, num_features, visual_feature_dim)`` for each the label id
            and the label score respectively
        matched_label (``torch.LongTensor`` of shape ``(batch_size,)``, `optional`):
            Labels for computing the whether or not the text input matches the image (classification) loss. Input
            should be a sequence pair (see :obj:`input_ids` docstring) Indices should be in ``[0, 1]``:

            - 0 indicates that the sentence does not match the image,
            - 1 indicates that the sentence does match the image.
        ans: (``Torch.Tensor`` of shape ``(batch_size)``, `optional`):
            a one hot representation hof the correct answer `optional`

        Returns:
        """

        device = lang_input_ids.device #if lang_input_ids is not None else inputs_embeds.device
        GTX_output = self.GTX(
            lang_input_ids=lang_input_ids,
            kg_input_ids=kg_input_ids,
            lang_inputs_embeds=lang_inputs_embeds,
            kg_inputs_embeds=kg_inputs_embeds,
            lang_attention_mask=lang_attention_mask,
            kg_attention_mask=kg_attention_mask,
            kg_padding_mask=kg_padding_mask,
            token_type_ids=token_type_ids,
            kg_ext_input_ids = kg_ext_input_ids,
            kg_ext_attention_mask = kg_ext_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        lang_output, kg_output, cross_relationship_score = (
            GTX_output.language_output,
            GTX_output.kg_output,
            GTX_output.pooled_output,
        )
        lang_prediction_scores = self.lm_head(lang_output)
        kg_prediction_scores = self.classifier(self.dropout(kg_output))

        # Loss calculation
        total_loss = (
            None
            if (lm_label is None and kg_label is None)
            else torch.tensor(0.0, device=device)
        )
        loss_dict = dict()
        if lm_label is not None:
            _lm_label = lm_label.view(-1)
            positive_batch_size = _lm_label.size(0)
            masked_lm_loss = self.loss_fcts["ce"](
                lang_prediction_scores.view(-1, self.config.vocab_size['lang'])[:positive_batch_size],
                _lm_label,
            )
            total_loss += masked_lm_loss
            loss_dict['lm_loss']=masked_lm_loss.mean().detach()

        if kg_label is not None:
            if kg_label_mask is not None:
                active_logits = kg_prediction_scores.view(-1, self.num_kg_labels)
                active_labels = torch.where(
                    kg_label_mask.view(-1), kg_label.view(-1), torch.tensor(self.loss_fcts['ce'].ignore_index).type_as(kg_label)
                )
                positive_batch_size = active_labels.size(0)
                kg_loss = self.loss_fcts['ce'](active_logits[:positive_batch_size], active_labels)
            else:
                kg_loss = self.loss_fcts['ce'](kg_prediction_scores.view(-1, self.num_kg_labels), kg_label.view(-1))
            total_loss += kg_loss
            loss_dict['kg_loss']=kg_loss.mean().detach()
        if cross_label is not None:
            cross_loss = self.loss_fcts["ce"](cross_relationship_score, cross_label)
            total_loss += cross_loss
            loss_dict['align_loss']=cross_loss.mean().detach()
        
        if rc_indeces is not None:
            rc_labels = list()
            rc_inputs = list()
            for idx, sample_rc_indeces in enumerate(rc_indeces):
                for rc_idx in sample_rc_indeces:
                    rc_inputs.append(torch.cat([kg_output[idx, rc_idx[0]],kg_output[idx, rc_idx[1]]],dim=-1))
                    rc_labels.append(rc_idx[2])
            rc_outputs = self.edge_classifier(torch.stack(rc_inputs,dim=0))
            rc_loss = self.loss_fcts['ce'](rc_outputs,torch.tensor(rc_labels,dtype=torch.long, device=device))
            total_loss += rc_loss
            loss_dict['rc_loss']=rc_loss.mean().detach()
            
        loss_dict['loss'] = total_loss.mean().detach()
        if not return_dict:
            output = (
                loss_dict,
                lang_prediction_scores,
                kg_prediction_scores,
                cross_relationship_score,

            ) + GTX_output[3:]
            return ((total_loss,) + output) if total_loss is not None else output

        return GTXForPreTrainingOutput(
            loss=total_loss,
            loss_dict=loss_dict,
            lang_prediction_logits=lang_prediction_scores.detach(),
            kg_prediction_logits=kg_prediction_scores.detach(),
            cross_relationship_score=cross_relationship_score.detach(),
            language_hidden_states=GTX_output.language_hidden_states,
            kg_hidden_states=GTX_output.kg_hidden_states,
            language_attentions=GTX_output.language_attentions,
            kg_attentions=GTX_output.kg_attentions,
            cross_encoder_attentions=GTX_output.cross_encoder_attentions,
        )

class GTXForRanking(GTXPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # Configuration
        self.config = config

        # GTX backbone
        self.GTX = GTXModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Weight initialization
        self.init_weights()

        # Use Pretrained-LM in Language Part
        self.GTX.encoder.re_init_to_pretrained_lang_model()

        # Warm start KG embedding
        if not config.gcn and config.pretrained_kg_embedding:
            notifier.critical("Load pretrained embedding for translation based KG-GTX")
            new_embedding = torch.load(config.pretrained_kg_embedding)
            # new_embedding = loaded_state_dict['ent_embeddings.weight']
            self.GTX.set_kg_embeddings(new_embedding)
            del new_embedding
            #torch.cuda.empty_cache()

        # Loss functions
        self.loss_fcts = {
            "ce": nn.CrossEntropyLoss(),
            "tri": nn.TripletMarginLoss(),
        }

    def forward(
        self,
        lang_input_ids=None,
        kg_input_ids=None,
        lang_inputs_embeds=None,
        kg_inputs_embeds=None,
        lang_attention_mask=None,
        kg_attention_mask=None,
        kg_padding_mask=None,
        label=None,
        token_type_ids=None,
        kg_ext_input_ids = None,
        kg_ext_attention_mask = None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
    ):
        r"""
        masked_lm_labels (``torch.LongTensor`` of shape ``(batch_size, sequence_length)``, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        obj_labels: (``Dict[Str: Tuple[Torch.FloatTensor, Torch.FloatTensor]]``, `optional`):
            each key is named after each one of the visual losses and each element of the tuple is of the shape
            ``(batch_size, num_features)`` and ``(batch_size, num_features, visual_feature_dim)`` for each the label id
            and the label score respectively
        matched_label (``torch.LongTensor`` of shape ``(batch_size,)``, `optional`):
            Labels for computing the whether or not the text input matches the image (classification) loss. Input
            should be a sequence pair (see :obj:`input_ids` docstring) Indices should be in ``[0, 1]``:

            - 0 indicates that the sentence does not match the image,
            - 1 indicates that the sentence does match the image.
        ans: (``Torch.Tensor`` of shape ``(batch_size)``, `optional`):
            a one hot representation hof the correct answer `optional`

        Returns:
        """

        device = lang_input_ids.device #if lang_input_ids is not None else inputs_embeds.device

        loss_dict = dict()

        GTX_output = self.GTX(
            lang_input_ids=lang_input_ids,
            kg_input_ids=kg_input_ids,
            lang_inputs_embeds=lang_inputs_embeds,
            kg_inputs_embeds=kg_inputs_embeds,
            lang_attention_mask=lang_attention_mask,
            kg_attention_mask=kg_attention_mask,
            kg_padding_mask=kg_padding_mask,
            token_type_ids=token_type_ids,
            kg_ext_input_ids = kg_ext_input_ids,
            kg_ext_attention_mask = kg_ext_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        lang_output, kg_output, pooled_output = (
            GTX_output.language_output,
            GTX_output.kg_output,
            GTX_output.pooled_output,
        )
        cross_relationship_score = pooled_output.squeeze()
        if label is not None:
            total_loss = self.loss_fcts["ce"](cross_relationship_score, label)
            loss_dict['loss']=total_loss.mean().detach()
        else:
            total_loss = None
        

        if not return_dict:
            output = (
                loss_dict,
            ) + GTX_output[3:]
            return ((total_loss,) + output) if total_loss is not None else output

        return GTXForDownstreamOutput(
            loss=total_loss,
            loss_dict=loss_dict,
            pooled_logits=cross_relationship_score.detach(),
            language_hidden_states=GTX_output.language_hidden_states,
            kg_hidden_states=GTX_output.kg_hidden_states,
            language_attentions=GTX_output.language_attentions,
            kg_attentions=GTX_output.kg_attentions,
            cross_encoder_attentions=GTX_output.cross_encoder_attentions,
        )

class GTXForAdmLvlPrediction(GTXPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # Configuration
        self.config = config

        # GTX backbone
        self.GTX = GTXModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Weight initialization
        self.init_weights()

        # Use Pretrained-LM in Language Part
        self.GTX.encoder.re_init_to_pretrained_lang_model()

        # Warm start KG embedding
        if not config.gcn and config.pretrained_kg_embedding:
            notifier.critical("Load pretrained embedding for translation based KG-GTX")
            new_embedding = torch.load(config.pretrained_kg_embedding)
            # new_embedding = loaded_state_dict['ent_embeddings.weight']
            self.GTX.set_kg_embeddings(new_embedding)
            del new_embedding
            #torch.cuda.empty_cache()

        # Loss functions
        self.loss_fcts = {
            "bce": nn.BCEWithLogitsLoss(reduction='none'),
            # "ce": nn.CrossEntropyLoss(),
            # "nll": nn.NLLLoss(),
        }
        self.class_weight = None

    def forward(
        self,
        lang_input_ids=None,
        kg_input_ids=None,
        lang_inputs_embeds=None,
        kg_inputs_embeds=None,
        lang_attention_mask=None,
        kg_attention_mask=None,
        kg_padding_mask=None,
        label=None,
        token_type_ids=None,
        kg_ext_input_ids = None,
        kg_ext_attention_mask = None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
    ):
        r"""
        masked_lm_labels (``torch.LongTensor`` of shape ``(batch_size, sequence_length)``, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        obj_labels: (``Dict[Str: Tuple[Torch.FloatTensor, Torch.FloatTensor]]``, `optional`):
            each key is named after each one of the visual losses and each element of the tuple is of the shape
            ``(batch_size, num_features)`` and ``(batch_size, num_features, visual_feature_dim)`` for each the label id
            and the label score respectively
        matched_label (``torch.LongTensor`` of shape ``(batch_size,)``, `optional`):
            Labels for computing the whether or not the text input matches the image (classification) loss. Input
            should be a sequence pair (see :obj:`input_ids` docstring) Indices should be in ``[0, 1]``:

            - 0 indicates that the sentence does not match the image,
            - 1 indicates that the sentence does match the image.
        ans: (``Torch.Tensor`` of shape ``(batch_size)``, `optional`):
            a one hot representation hof the correct answer `optional`

        Returns:
        """

        device = lang_input_ids.device #if lang_input_ids is not None else inputs_embeds.device

        loss_dict = dict()

        GTX_output = self.GTX(
            lang_input_ids=lang_input_ids,
            kg_input_ids=kg_input_ids,
            lang_inputs_embeds=lang_inputs_embeds,
            kg_inputs_embeds=kg_inputs_embeds,
            lang_attention_mask=lang_attention_mask,
            kg_attention_mask=kg_attention_mask,
            kg_padding_mask=kg_padding_mask,
            token_type_ids=token_type_ids,
            kg_ext_input_ids = kg_ext_input_ids,
            kg_ext_attention_mask = kg_ext_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        lang_output, kg_output, multi_label_score = (
            GTX_output.language_output,
            GTX_output.kg_output,
            GTX_output.pooled_output,
        )
        # multi_label_score = multi_label_score.softmax(dim=1)
        if label is not None:
            total_loss = self.loss_fcts["bce"](multi_label_score, label)
            if self.class_weight is not None:
                total_loss = total_loss*torch.tensor(self.class_weight, requires_grad=False, device=label.device)
                #focal_weight = (multi_label_score.sigmoid()-label).abs().detach().clone()
                #total_loss = total_loss*focal_weight

            total_loss = total_loss.mean()
            loss_dict['loss']=total_loss.detach()
        else:
            total_loss = None
        

        if not return_dict:
            output = (
                loss_dict,
            ) + GTX_output[3:]
            return ((total_loss,) + output) if total_loss is not None else output

        return GTXForDownstreamOutput(
            loss=total_loss,
            loss_dict=loss_dict,
            pooled_logits=multi_label_score.detach(),
            language_hidden_states=GTX_output.language_hidden_states,
            kg_hidden_states=GTX_output.kg_hidden_states,
            language_attentions=GTX_output.language_attentions,
            kg_attentions=GTX_output.kg_attentions,
            cross_encoder_attentions=GTX_output.cross_encoder_attentions,
        )

class GTXForErrorDetection(GTXPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # Configuration
        self.config = config

        # GTX backbone
        self.GTX = GTXModel(config)
        self.multilabel_classifier = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                                nn.ReLU(),
                                                nn.Linear(config.hidden_size, config.num_labels))
        self.token_classifier = nn.Linear(config.hidden_size, 1)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Weight initialization
        self.init_weights()

        # Use Pretrained-LM in Language Part
        self.GTX.encoder.re_init_to_pretrained_lang_model()

        # Warm start KG embedding
        if not config.gcn and config.pretrained_kg_embedding:
            notifier.critical("Load pretrained embedding for translation based KG-GTX")
            new_embedding = torch.load(config.pretrained_kg_embedding)
            # new_embedding = loaded_state_dict['ent_embeddings.weight']
            self.GTX.set_kg_embeddings(new_embedding)
            del new_embedding
            #torch.cuda.empty_cache()

        # Loss functions
        self.loss_fcts = {
            "bce": nn.BCEWithLogitsLoss()
        }

    def forward(
        self,
        lang_input_ids=None,
        kg_input_ids=None,
        lang_inputs_embeds=None,
        kg_inputs_embeds=None,
        lang_attention_mask=None,
        kg_attention_mask=None,
        kg_padding_mask=None,
        kg_label=None,
        lm_label=None,
        token_type_ids=None,
        kg_ext_input_ids = None,
        kg_ext_attention_mask = None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
    ):
        r"""
        masked_lm_labels (``torch.LongTensor`` of shape ``(batch_size, sequence_length)``, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        obj_labels: (``Dict[Str: Tuple[Torch.FloatTensor, Torch.FloatTensor]]``, `optional`):
            each key is named after each one of the visual losses and each element of the tuple is of the shape
            ``(batch_size, num_features)`` and ``(batch_size, num_features, visual_feature_dim)`` for each the label id
            and the label score respectively
        matched_label (``torch.LongTensor`` of shape ``(batch_size,)``, `optional`):
            Labels for computing the whether or not the text input matches the image (classification) loss. Input
            should be a sequence pair (see :obj:`input_ids` docstring) Indices should be in ``[0, 1]``:

            - 0 indicates that the sentence does not match the image,
            - 1 indicates that the sentence does match the image.
        ans: (``Torch.Tensor`` of shape ``(batch_size)``, `optional`):
            a one hot representation hof the correct answer `optional`

        Returns:
        """

        device = lang_input_ids.device #if lang_input_ids is not None else inputs_embeds.device

        loss_dict = dict()

        GTX_output = self.GTX(
            lang_input_ids=lang_input_ids,
            kg_input_ids=kg_input_ids,
            lang_inputs_embeds=lang_inputs_embeds,
            kg_inputs_embeds=kg_inputs_embeds,
            lang_attention_mask=lang_attention_mask,
            kg_attention_mask=kg_attention_mask,
            kg_padding_mask=kg_padding_mask,
            token_type_ids=token_type_ids,
            kg_ext_input_ids = kg_ext_input_ids,
            kg_ext_attention_mask = kg_ext_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        lang_output, kg_output, multi_label_score = (
            GTX_output.language_output,
            GTX_output.kg_output,
            GTX_output.pooled_output,
        )
        # total_loss = 0
        if kg_label is not None:
            if kg_input_ids.size(-1)==kg_label.size(-1):
                _size = kg_output.shape[:-1]
                score = self.token_classifier(kg_output.view(-1, self.config.hidden_size)).view(_size)
                total_loss = self.loss_fcts["bce"](score, kg_label)
                loss_dict['loss']=total_loss.mean().detach()
            else:
                score = self.multilabel_classifier(kg_output[:,0])
                total_loss = self.loss_fcts["bce"](score, kg_label)
                loss_dict['loss']=total_loss.mean().detach()
        elif lm_label is not None:
                _size = lang_output.shape[:-1]
                score = self.token_classifier(lang_output.view(-1, self.config.hidden_size)).view(_size)
                total_loss = self.loss_fcts["bce"](score, lm_label)
                focal_weight = (score.sigmoid()-lm_label).square().abs().detach().clone()
                total_loss = (total_loss*focal_weight).mean()
                loss_dict['loss']=total_loss.detach()
        else:
            total_loss = None

        if not return_dict:
            output = (
                loss_dict,
            ) + GTX_output[3:]
            return ((total_loss,) + output) if total_loss is not None else output

        return GTXForDownstreamOutput(
            loss=total_loss,
            loss_dict=loss_dict,
            pooled_logits=score.detach(),
            language_hidden_states=GTX_output.language_hidden_states,
            kg_hidden_states=GTX_output.kg_hidden_states,
            language_attentions=GTX_output.language_attentions,
            kg_attentions=GTX_output.kg_attentions,
            cross_encoder_attentions=GTX_output.cross_encoder_attentions,
        )

class GTXForTemporalPred(GTXPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # Configuration
        self.config = config

        # GTX backbone
        self.GTX = GTXModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Weight initialization
        self.init_weights()

        # Use Pretrained-LM in Language Part
        self.GTX.encoder.re_init_to_pretrained_lang_model()

        # Warm start KG embedding
        if not config.gcn and config.pretrained_kg_embedding:
            notifier.critical("Load pretrained embedding for translation based KG-GTX")
            new_embedding = torch.load(config.pretrained_kg_embedding)
            self.GTX.set_kg_embeddings(new_embedding)
            del new_embedding

        # Loss functions
        self.loss_fcts = {
            "bce": nn.BCEWithLogitsLoss(reduction='none'),
            # "ce": nn.CrossEntropyLoss(),
        }
        self.class_weight = None

    def forward(
        self,
        lang_input_ids=None,
        kg_input_ids=None,
        lang_inputs_embeds=None,
        kg_inputs_embeds=None,
        lang_attention_mask=None,
        kg_attention_mask=None,
        kg_padding_mask=None,
        label=None,
        token_type_ids=None,
        kg_ext_input_ids = None,
        kg_ext_attention_mask = None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
    ):
        r"""
        masked_lm_labels (``torch.LongTensor`` of shape ``(batch_size, sequence_length)``, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        obj_labels: (``Dict[Str: Tuple[Torch.FloatTensor, Torch.FloatTensor]]``, `optional`):
            each key is named after each one of the visual losses and each element of the tuple is of the shape
            ``(batch_size, num_features)`` and ``(batch_size, num_features, visual_feature_dim)`` for each the label id
            and the label score respectively
        matched_label (``torch.LongTensor`` of shape ``(batch_size,)``, `optional`):
            Labels for computing the whether or not the text input matches the image (classification) loss. Input
            should be a sequence pair (see :obj:`input_ids` docstring) Indices should be in ``[0, 1]``:

            - 0 indicates that the sentence does not match the image,
            - 1 indicates that the sentence does match the image.
        ans: (``Torch.Tensor`` of shape ``(batch_size)``, `optional`):
            a one hot representation hof the correct answer `optional`

        Returns:
        """

        device = lang_input_ids.device #if lang_input_ids is not None else inputs_embeds.device

        loss_dict = dict()

        GTX_output = self.GTX(
            lang_input_ids=lang_input_ids,
            kg_input_ids=kg_input_ids,
            lang_inputs_embeds=lang_inputs_embeds,
            kg_inputs_embeds=kg_inputs_embeds,
            lang_attention_mask=lang_attention_mask,
            kg_attention_mask=kg_attention_mask,
            kg_padding_mask=kg_padding_mask,
            token_type_ids=token_type_ids,
            kg_ext_input_ids = kg_ext_input_ids,
            kg_ext_attention_mask = kg_ext_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        lang_output, kg_output, multi_label_score = (
            GTX_output.language_output,
            GTX_output.kg_output,
            GTX_output.pooled_output,
        )
        # multi_label_score = multi_label_score.softmax(dim=1)
        if label is not None:
            total_loss = self.loss_fcts["bce"](multi_label_score.squeeze(dim=1), label.float())
            total_loss = total_loss.mean()
            loss_dict['loss']=total_loss.detach()
        else:
            total_loss = None
    
        if not return_dict:
            output = (
                loss_dict,
            ) + GTX_output[3:]
            return ((total_loss,) + output) if total_loss is not None else output

        return GTXForDownstreamOutput(
            loss=total_loss,
            loss_dict=loss_dict,
            pooled_logits=multi_label_score.detach(),
            language_hidden_states=GTX_output.language_hidden_states,
            kg_hidden_states=GTX_output.kg_hidden_states,
            language_attentions=GTX_output.language_attentions,
            kg_attentions=GTX_output.kg_attentions,
            cross_encoder_attentions=GTX_output.cross_encoder_attentions,
        )

class GTXForGeneration(GTXPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # Configuration
        self.config = config
        self.num_labels = config.num_labels

        # GTX backbone
        self.GTX = GTXModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        # self.edge_classifier = nn.Sequential(nn.Linear(config.hidden_size*2, config.hidden_size*2),
        #                                     nn.Tanh(),
        #                                     nn.Linear(config.hidden_size*2, config.num_relations))

        # Pre-training heads
        self.lm_head = GTXPreTrainingHeads(config, self.GTX.lang_embeddings.word_embeddings.weight)

        # Weight initialization
        self.init_weights()

        # Warm start KG embedding
        if not config.gcn and config.pretrained_kg_embedding:
            notifier.critical("Load pretrained embedding for translation based KG-GTX")
            new_embedding = torch.load(config.pretrained_kg_embedding)
            # new_embedding = loaded_state_dict['ent_embeddings.weight']
            self.GTX.set_kg_embeddings(new_embedding)
            del new_embedding
            #torch.cuda.empty_cache()

        # Use Pretrained-LM in Language Part
        self.GTX.encoder.re_init_to_pretrained_lang_model()

        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.loss_fcts = {
            "l2": SmoothL1Loss(reduction="none"),
            "mse": MSELoss(reduction="none"),
            "ce": CrossEntropyLoss(),
            "tri": nn.TripletMarginLoss()#(margin=config.margin)
        }
        
        # Rules for special_token_ids: bert-base-uncased
        self.sep_token_id = 102
        self.mask_token_id = 103
        self.pad_token_id = 0

    def forward(
        self,
        lang_input_ids=None,
        kg_input_ids=None,
        lang_inputs_embeds=None,
        kg_inputs_embeds=None,
        lang_attention_mask=None,
        kg_attention_mask=None,
        kg_padding_mask=None,
        kg_label_mask=None,
        lm_label=None,
        kg_label=None,
        cross_label=None,
        token_type_ids=None,
        rc_indeces=None,
        kg_ext_input_ids = None,
        kg_ext_attention_mask = None,
        token_type_ids=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
    ):
        device = lang_input_ids.device #if lang_input_ids is not None else inputs_embeds.device
        GTX_output = self.GTX(
            lang_input_ids=lang_input_ids,
            kg_input_ids=kg_input_ids,
            lang_inputs_embeds=lang_inputs_embeds,
            kg_inputs_embeds=kg_inputs_embeds,
            lang_attention_mask=lang_attention_mask,
            kg_attention_mask=kg_attention_mask,
            kg_padding_mask=kg_padding_mask,
            token_type_ids=token_type_ids,
            kg_ext_input_ids = kg_ext_input_ids,
            kg_ext_attention_mask = kg_ext_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        lang_output, kg_output, cross_relationship_score = (
            GTX_output.language_output,
            GTX_output.kg_output,
            GTX_output.pooled_output,
        )
        lang_prediction_scores = self.lm_head(lang_output)

        # Loss calculation
        total_loss = (
            None
            if (lm_label is None and kg_label is None)
            else torch.tensor(0.0, device=device)
        )
        loss_dict = dict()
        if lm_label is not None:
            _lm_label = lm_label.view(-1)
            positive_batch_size = _lm_label.size(0)
            masked_lm_loss = self.loss_fcts["ce"](
                lang_prediction_scores.view(-1, self.config.vocab_size['lang'])[:positive_batch_size],
                _lm_label,
            )
            total_loss += masked_lm_loss
            loss_dict['lm_loss']=masked_lm_loss.mean().detach()
        
        if lm_label is None and kg_label is None:
            total_loss = torch.tensor(0.0, device=device)
            
        loss_dict['loss'] = total_loss.mean().detach()
        if not return_dict:
            output = (
                loss_dict,
                lang_prediction_scores,
                # kg_prediction_scores,
                cross_relationship_score,

            ) + GTX_output[3:]
            return ((total_loss,) + output) if total_loss is not None else output

        # return GTXForDownstreamOutput(
        #     loss=total_loss,
        #     loss_dict=loss_dict,
        #     pooled_logits=cross_relationship_score.detach(),
        #     language_hidden_states=GTX_output.language_hidden_states,
        #     kg_hidden_states=GTX_output.kg_hidden_states,
        #     language_attentions=GTX_output.language_attentions,
        #     kg_attentions=GTX_output.kg_attentions,
        #     cross_encoder_attentions=GTX_output.cross_encoder_attentions,
        # )
        return GTXForPreTrainingOutput(
            loss=total_loss,
            loss_dict=loss_dict,
            lang_prediction_logits=lang_prediction_scores.detach(),
            # kg_prediction_logits=kg_prediction_scores.detach(),
            cross_relationship_score=cross_relationship_score.detach(),
            language_hidden_states=GTX_output.language_hidden_states,
            kg_hidden_states=GTX_output.kg_hidden_states,
            language_attentions=GTX_output.language_attentions,
            kg_attentions=GTX_output.kg_attentions,
            cross_encoder_attentions=GTX_output.cross_encoder_attentions,
        )
        
    def decode_for_ppl(
        self,
        lang_input_ids=None,
        kg_input_ids=None,
        lang_inputs_embeds=None,
        kg_inputs_embeds=None,
        lang_attention_mask=None,
        kg_attention_mask=None,
        kg_padding_mask=None,
        kg_label_mask=None,
        lm_label=None,
        kg_label=None,
        kg_ext_input_ids = None,
        kg_ext_attention_mask = None,
        token_type_ids=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
        ):
        
        total_ppl = 0.0
        
        # when model is trained, we treat the original language input ids as the labels
        if lm_label is not None:
            ignore_index = -100
            lang_input_ids = lm_label.eq(ignore_index) * lang_input_ids.clone() + \
                lm_label.not_equal(ignore_index) * lm_label.clone()
            lm_label = lang_input_ids.clone()
        else:
            lm_label = lang_input_ids.clone()
        
        assert len(lang_input_ids.shape) == 2
        batch_size, max_seq_length = lang_input_ids.shape
        
        # set `max_output_length` for max length within a batch
        gt_length = torch.sum(lang_input_ids.not_equal(self.pad_token_id), axis=1)
        max_output_length = torch.max(gt_length).detach()
        assert max_output_length <= max_seq_length
        
        # construct initial settings [[CLS], [MASK]]
        mask_ids = lang_input_ids.new(batch_size, 1).fill_(self.mask_token_id)
        
        next_pos = 1
        while next_pos < max_output_length:
            
            curr_ids = torch.cat([lang_input_ids[:, :next_pos], mask_ids], axis=1)
            curr_length = list(curr_ids.size())[1]
            curr_attention_mask = lang_attention_mask[:, :curr_length, :curr_length]
            curr_token_type_ids = token_type_ids[:, :curr_length]
            
            assert curr_ids.shape[-1] == curr_attention_mask.shape[-1]
            GTX_output = self.GTX(
                lang_input_ids=curr_ids,
                kg_input_ids=kg_input_ids,
                lang_inputs_embeds=lang_inputs_embeds,
                kg_inputs_embeds=kg_inputs_embeds,
                lang_attention_mask=curr_attention_mask,
                kg_attention_mask=kg_attention_mask,
                kg_padding_mask=kg_padding_mask,
                token_type_ids=curr_token_type_ids,
                kg_ext_input_ids = kg_ext_input_ids,
                kg_ext_attention_mask = kg_ext_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            
            lang_output, _, _ = (
                GTX_output.language_output,
                GTX_output.kg_output,
                GTX_output.pooled_output,
            )
        
            last_hidden = lang_output[:, -1, :]
            prediction_scores = self.lm_head(last_hidden)
            _lm_label = lm_label[:, next_pos]
            # log_scores = F.log_softmax(prediction_scores, dim=-1)
            
            masked_lm_loss = self.ce_loss(
                prediction_scores.view(-1, self.config.vocab_size['lang']),
                _lm_label,
            )
            loss_mask = (curr_length <= gt_length)
            total_ppl += masked_lm_loss * loss_mask
            
            next_pos += 1
            
        total_ppl = torch.exp(total_ppl/gt_length)
        batch_mean_ppl = total_ppl.mean().detach()
        return batch_mean_ppl, lm_label
    
    def decode(
        self,
        lang_input_ids=None,
        kg_input_ids=None,
        lang_inputs_embeds=None,
        kg_inputs_embeds=None,
        lang_attention_mask=None,
        kg_attention_mask=None,
        kg_padding_mask=None,
        kg_label_mask=None,
        lm_label=None,
        kg_label=None,
        kg_ext_input_ids = None,
        kg_ext_attention_mask = None,
        token_type_ids=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
        given_lang_tokens=1,
        clean_outputs=True,
        given_gt_length=False,
        search_beam_size=1,
        ):
        
        device = lang_input_ids.device if lang_input_ids is not None else lang_inputs_embeds.device

        # if lm_label is not None:
        #     ignore_index = -100
        #     lm_label = lm_label.eq(ignore_index) * lang_input_ids.clone() + \
        #         lm_label.not_equal(ignore_index) * lm_label.clone()
                
        # Choose the database (dx/px, rx)
        num_db = len(torch.bincount(token_type_ids[0]))
        # Calcuate ground truth length
        gt_length = torch.sum(lang_input_ids.not_equal(self.pad_token_id), axis=1)
        
        # 1. Greedy Decoding
        if search_beam_size == 1:
            output_ids = self._greedy_decode(
                lang_input_ids=lang_input_ids,
                kg_input_ids=kg_input_ids,
                lang_inputs_embeds=lang_inputs_embeds,
                kg_inputs_embeds=kg_inputs_embeds,
                lang_attention_mask=lang_attention_mask,
                kg_attention_mask=kg_attention_mask,
                kg_padding_mask=kg_padding_mask,
                # kg_label_mask=None,
                # lm_label=None,
                # kg_label=None,
                kg_ext_input_ids=kg_ext_input_ids,
                kg_ext_attention_mask=kg_ext_attention_mask,
                token_type_ids=token_type_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
                num_db=num_db,
                gt_length=gt_length,
                given_lang_tokens=1,
            )
            
        # 2. Beam Search Decoding    
        else:
            output_ids = self._beam_search(
                lang_input_ids=lang_input_ids,
                kg_input_ids=kg_input_ids,
                # lang_inputs_embeds=lang_inputs_embeds,
                kg_inputs_embeds=kg_inputs_embeds,
                lang_attention_mask=lang_attention_mask,
                kg_attention_mask=kg_attention_mask,
                kg_padding_mask=kg_padding_mask,
                token_type_ids=token_type_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                search_beam_size=search_beam_size,      
                num_db=num_db,
                gt_length=gt_length,
                given_lang_tokens=1,
            )
        
        if clean_outputs:
            output_ids = self.clean_output_ids(output_ids=output_ids,
                                               gt_length=gt_length,
                                               num_sep_id=num_db,
                                               given_gt_length=given_gt_length)
        
        outputs = (output_ids, kg_input_ids, lang_input_ids)
        return outputs
    
    def _greedy_decode(
        self,
        lang_input_ids=None,
        kg_input_ids=None,
        lang_inputs_embeds=None,
        kg_inputs_embeds=None,
        lang_attention_mask=None,
        kg_attention_mask=None,
        kg_padding_mask=None,
        kg_label_mask=None,
        lm_label=None,
        kg_label=None,
        kg_ext_input_ids=None,
        kg_ext_attention_mask=None,
        token_type_ids=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
        num_db=1,
        gt_length=None,
        given_lang_tokens=1,
        top_p_sampling=False, # NOTE: Should be changed!
        ):
        
        assert len(lang_input_ids.shape) == 2
        batch_size, max_seq_length = lang_input_ids.shape
        
        # set output_length for max_length within a batch
        max_output_length = torch.max(gt_length).detach()
        assert max_output_length <= max_seq_length
        
        # construct initial settings [[CLS], [MASK]]
        output_ids = []
        curr_ids = lang_input_ids[:, :given_lang_tokens]
        mask_ids = lang_input_ids.new(batch_size, 1).fill_(self.mask_token_id)
        output_ids.append(curr_ids)
        
        next_pos = given_lang_tokens
        while next_pos < max_output_length:
            
            curr_length = list(curr_ids.size())[1]
            curr_ids = torch.cat([curr_ids, mask_ids], axis=1)
            curr_attention_mask = lang_attention_mask[:, :curr_length+1, :curr_length+1]
            curr_token_type_ids = token_type_ids[:, :curr_length+1]
            assert curr_ids.shape[-1] == curr_attention_mask.shape[-1]
            
            # when dx, prx case, we should consider token_type_ids
            if num_db == 2:
                curr_token_type_ids = self.convert_token_type_ids(curr_ids, curr_token_type_ids)
                
            GTX_output = self.GTX(
                lang_input_ids=curr_ids,
                kg_input_ids=kg_input_ids,
                lang_inputs_embeds=lang_inputs_embeds,
                kg_inputs_embeds=kg_inputs_embeds,
                lang_attention_mask=curr_attention_mask,
                kg_attention_mask=kg_attention_mask,
                kg_padding_mask=kg_padding_mask,
                kg_ext_input_ids=kg_ext_input_ids,
                kg_ext_attention_mask=kg_ext_attention_mask,
                token_type_ids=curr_token_type_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
                
            lang_output, _, _ = (
                GTX_output.language_output,
                GTX_output.kg_output,
                GTX_output.pooled_output,
            )
            
            # predict [MASK] by greedy infer.
            last_hidden = lang_output[:, -1, :]
            prediction_scores = self.lm_head(last_hidden)  # `prediction_scores` \approx `logits`
            
            if top_p_sampling:
                
                def fixed_top_k_top_p_filtering(logits, top_k=0, top_p=0.9, filter_value=-float('Inf')):
                    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
                    https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
                        Args:
                            logits: logits distribution shape (..., vocabulary size)
                            top_k >0: keep only top k tokens with highest probability (top-k filtering).
                            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                    """
                    top_k = min(top_k, logits.size(-1))  # Safety check
                    if top_k > 0:
                        # Remove all tokens with a probability less than the last token of the top-k
                        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                        logits[indices_to_remove] = filter_value

                    if top_p > 0.0:
                        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                        # Remove tokens with cumulative probability above the threshold
                        sorted_indices_to_remove = cumulative_probs >= top_p
                        # Shift the indices to the right to keep also the first token above the threshold
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                                                
                        indices_to_remove = torch.zeros_like(logits, dtype=torch.bool).scatter_(
                            dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
                        logits[indices_to_remove] = filter_value
                    return logits
                
                prediction_scores = fixed_top_k_top_p_filtering(logits=prediction_scores)
                probs = F.softmax(prediction_scores, dim=-1)
                max_ids = torch.multinomial(probs, num_samples=1).squeeze(1)
                output_ids.append(max_ids.unsqueeze(1))
            else:
                _, max_ids = torch.max(prediction_scores, dim=1)
                output_ids.append(max_ids.unsqueeze(1))
            
            # setup for next loop
            curr_ids[:, curr_length] = max_ids
            next_pos += 1
            
        output_ids = torch.cat(output_ids, dim=1)
        return output_ids
    
    def _beam_search(
        self,
        lang_input_ids=None,
        kg_input_ids=None,
        # lang_inputs_embeds=None,
        kg_inputs_embeds=None,
        lang_attention_mask=None,
        kg_attention_mask=None,
        kg_padding_mask=None,
        kg_label_mask=None,
        lm_label=None,
        kg_label=None,
        kg_ext_input_ids = None,
        kg_ext_attention_mask = None,
        token_type_ids=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
        search_beam_size=5,
        num_db=1,
        gt_length=None,
        given_lang_tokens=1,
        ):
        
        batch_size, max_length = lang_input_ids.shape

        # find maximum output_length
        output_length = torch.max(gt_length).detach()
        assert output_length <= max_length
        
        # init settings
        curr_ids = lang_input_ids[:, :1]
        mask_ids = curr_ids.new(batch_size, 1).fill_(self.mask_token_id)
        # output_ids = []
        # output_ids.append(curr_ids)

        next_pos = 1
        K = search_beam_size

        total_scores = []
        beam_masks = []
        step_ids = []
        step_back_ptrs = []
        # partial_seqs = []
        # forbid_word_mask = None
        # buf_matrix = None

        while next_pos < output_length:
            
            # construct current inputs
            curr_length = list(curr_ids.size())[1]
            curr_ids = torch.cat([curr_ids, mask_ids], axis=1)    
            curr_token_type_ids = token_type_ids[:, :curr_length+1]
            curr_attention_mask = lang_attention_mask[:, :curr_length+1, :curr_length+1]
            
            # when dx,prx case, we should consider token_type_ids
            if num_db == 2:
                curr_token_type_ids = self.convert_token_type_ids(curr_ids, curr_token_type_ids)
            
            # output for current inputs
            GTX_output = self.GTX(
                lang_input_ids=curr_ids,
                kg_input_ids=kg_input_ids,
                lang_attention_mask=curr_attention_mask,
                kg_padding_mask=kg_padding_mask,
                token_type_ids=curr_token_type_ids,
                kg_ext_input_ids = kg_ext_input_ids,
                kg_ext_attention_mask = kg_ext_attention_mask,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
            lang_output, _, _ = (
                GTX_output.language_output,
                GTX_output.kg_output,
                GTX_output.pooled_output,
            )
            last_hidden = lang_output[:, -1, :]
            prediction_scores = self.lm_head(last_hidden)
            log_scores = F.log_softmax(prediction_scores, dim=-1)
            
            # choose top-k logits and indices
            kk_scores, kk_ids = torch.topk(log_scores, k=K) # (batch_size, beam_size) / (batch_size*beam_size, beam_size)
            
            def first_expand(x):
                input_shape = list(x.size())
                expanded_shape = input_shape[:1] + [1] + input_shape[1:]
                x = torch.reshape(x, expanded_shape)
                repeat_count = [1, K] + [1] * (len(input_shape) - 1)
                x = x.repeat(*repeat_count)
                x = torch.reshape(x, [input_shape[0] * K] + input_shape[1:])
                return x
            
            def select_beam_items(x, ids):
                id_shape = list(ids.size())
                id_rank = len(id_shape)
                assert len(id_shape) == 2
                x_shape = list(x.size())
                x = torch.reshape(x, [batch_size, K] + x_shape[1:])
                x_rank = len(x_shape) + 1
                assert x_rank >= 2
                if id_rank < x_rank:
                    ids = torch.reshape(
                        ids, id_shape + [1] * (x_rank - id_rank))
                    ids = ids.expand(id_shape + x_shape[1:])
                y = torch.gather(x, 1, ids)
                y = torch.reshape(y, x_shape)
                return y
            
            if len(total_scores) == 0: # for the first time,
                k_ids = torch.reshape(kk_ids, [batch_size, K])
                back_ptrs = torch.zeros(batch_size, K, dtype=torch.long)
                k_scores = torch.reshape(kk_scores, [batch_size, K])
                print('start:', k_scores, k_ids)
            else:
                last_eos = torch.reshape(beam_masks[-1], [batch_size * K, 1])
                last_seq_scores = torch.reshape(total_scores[-1], [batch_size * K, 1])
                kk_scores += last_eos * (-10000.0) + last_seq_scores
                kk_scores = torch.reshape(kk_scores, [batch_size, K * K])
                k_scores, k_ids = torch.topk(kk_scores, k=K)
                back_ptrs = torch.floor_divide(k_ids, K)
                kk_ids = torch.reshape(kk_ids, [batch_size, K * K])
                print('kk_ids', kk_ids[0])
                print(k_ids[0])
                k_ids = torch.gather(kk_ids, 1, k_ids)
                print('gather 이후:', k_ids[0])
                curr_ids = select_beam_items(curr_ids, back_ptrs.long())
                print(curr_ids.shape)
                for idx in range(curr_ids.shape[0]):
                    print(self.tokenizer.decode(curr_ids[idx]))
                print()
                
            step_back_ptrs.append(back_ptrs)
            step_ids.append(k_ids)
            beam_masks.append(torch.eq(k_ids, self.sep_token_id).float())
            total_scores.append(k_scores)

            if next_pos == 1: # for the first time,
                curr_ids = first_expand(curr_ids)
                kg_input_ids = first_expand(kg_input_ids)
                token_type_ids = first_expand(token_type_ids)
                lang_attention_mask = first_expand(lang_attention_mask)
                kg_padding_mask = first_expand(kg_padding_mask)
                mask_ids = first_expand(mask_ids)
            
            # fill out the [MASK]'s position with stretched ids
            curr_ids[:, curr_length] = torch.reshape(k_ids, [batch_size * K])
            next_pos += 1
            
        output_ids = curr_ids.reshape(batch_size, K, -1)
        
        for idx in range(K):
            print(f'batch_idx:0, beam_idx:{idx}:', self.tokenizer.decode(output_ids[0][idx], skip_special_tokens=True))
            print()
        print('original text:', self.tokenizer.decode(lang_input_ids[0], skip_special_tokens=True))
        
        return output_ids
    
    def clean_output_ids(self, output_ids, gt_length, num_sep_id, given_gt_length):
        c_output_ids = []
        if given_gt_length:
            for i, o in enumerate(output_ids):
                l_gt = gt_length[i]
                sep_idx = o.eq(self.sep_token_id)
                l_sep = torch.nonzero(sep_idx)[num_sep_id-1] if sep_idx.sum() >= num_sep_id else 512
                c_output_ids.append(o[:min(l_gt, l_sep)])
        else:
            for o in output_ids:
                if len(o.shape) == 1: # greedy
                    sep_idx = o.eq(self.sep_token_id)
                    l_sep = torch.nonzero(sep_idx)[num_sep_id-1] if sep_idx.sum() >= num_sep_id else 512
                    c_output_ids.append(o[:min(512, l_sep)])
                else: # beam search
                    c_output_id = []
                    for oo in o:
                        sep_idx = oo.eq(self.sep_token_id)
                        l_sep = torch.nonzero(sep_idx)[num_sep_id-1] if sep_idx.sum() >= num_sep_id else 512
                        c_output_id.append(oo[:min(512, l_sep)])
                    assert len(c_output_id) == o.shape[0] # beam size
                    c_output_ids.append(c_output_id)
        return c_output_ids
    
    def convert_token_type_ids(self, curr_ids, curr_token_type_ids):
        for idx in range(len(curr_ids)):
            if self.sep_token_id in curr_ids[idx][-2]:
                first_sep_idx = torch.nonzero(curr_ids[idx].eq(self.sep_token_id))[0]
                curr_token_type_ids[idx][first_sep_idx+1:] = 1        
        return curr_token_type_ids