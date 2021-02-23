import torch

def format_attention(attention):
    formatted_attention = dict()
    for key in attention:
        squeezed = []
        for layer_attention in attention[key]:
            # 1 x num_heads x seq_len x seq_len
            if len(layer_attention.shape) != 4:
                raise ValueError("The attention tensor does not have the correct number of dimensions. Make sure you set "
                                 "output_attentions=True when initializing your model.")
            squeezed.append(layer_attention.squeeze(0))
        formatted_attention[key] = torch.stack(squeezed)
    # num_layers x num_heads x seq_len x seq_len
    return formatted_attention

def format_special_chars(tokens):
    return [t.replace('Ġ', ' ').replace('▁', ' ').replace('</w>', '') for t in tokens]
