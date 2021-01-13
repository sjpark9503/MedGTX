import json
from IPython.core.display import display, HTML, Javascript
import os
from .util import format_special_chars, format_attention


def head_view(attention, sample, prettify_tokens=True):
    """Render head view

        Args:
            attention: list of ``torch.FloatTensor``(one for each layer) of shape
                ``(batch_size(must be 1), num_heads, sequence_length, sequence_length)``
            sample : dictionary of sample sentences and KGs
            prettify_tokens: indicates whether to remove special characters in wordpieces, e.g. Ġ
    """
    modality = list(sample.keys())[0]
    if len(sample)>1:
        vis_html = """
        <span style="user-select:none">
            Layer: <select id="layer"></select>
            Attention: <select id="filter">
              <option value="kg->txt">kg->txt</option>
              <option value="txt->kg">txt->kg</option>
              <option value="txt->txt">txt->txt</option>
              <option value="kg->kg">kg->kg</option>
            </select>
            </span>
        <div id='vis'></div>
        """
    else:

        vis_html = """
        <span style="user-select:none">
            Layer: <select id="layer"></select>
            Attention: <select id="filter">
              <option value="{0}->{0}">{0}->{0}</option>
            </select>
            </span>
        <div id='vis'></div> 
        """.format(modality)
        
    display(HTML(vis_html))
    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))
    vis_js = open(os.path.join(__location__, 'head_view.js')).read()

    if prettify_tokens:
        for k in sample:
            sample[k] = format_special_chars(sample[k])

    attn = format_attention(attention)
    attn_data = dict()
    for k in attn:
        attn_data[k] = {
            'attn': attn[k].tolist(),
            'left_text': sample[k.split('->')[0]],
            'right_text': sample[k.split('->')[-1]],
        }
    params = {
        'attention': attn_data,
        'default_filter': "{0}->{0}".format(modality)
    }

    display(Javascript('window.params = %s' % json.dumps(params)))
    display(Javascript(vis_js))