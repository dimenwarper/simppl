import base64
import hashlib
from io import BytesIO
from typing import Tuple, Any, List, Dict, Optional, Union

from matplotlib import pyplot as plt


def fig2html(fig: plt.Figure) -> str:
    io = BytesIO()
    fig.savefig(io, format='png', dpi=200)
    encoded = base64.b64encode(io.getvalue()).decode('utf-8')
    plt.close()
    return '<img style="display: inline" src=\'data:image/png;base64,{}\', height="15%", width="15%">'.format(encoded)

def color_from_string(string: str) -> Tuple[int]:
    cm = plt.get_cmap('tab20')
    N = 300
    i = int(hashlib.sha1(string.encode('utf-8')).hexdigest(), 16) % N
    return cm(i / N)


def fracbar(
        categories: List[Any],
        fractions: Dict[Any, float],
        colors: Optional[List[Union[str, Tuple[int]]]] = None,
        label_colors: Optional[List[Union[str, Tuple[int]]]] = None,
        title: Optional[str] = None
):
    fig = plt.figure(figsize=(2, 0.8))

    offset = 0
    if colors is None:
        cm = plt.get_cmap('Paired')
        colors = [cm(i) for i in range(len(fractions))]
    if label_colors is None:
        label_colors = ['black'] * len(fractions)
    for i in range(len(categories)):
        plt.barh([0], [fractions[i]], color=colors[i], left=offset)
        plt.text(offset + (fractions[i]) / 2 - 0.05, -0.2, str(categories[i]), color=label_colors[i], fontweight='bold')
        offset += fractions[i]

    plt.axis('off')
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    return fig2html(fig)

def sparkbars(*args, title: Optional[str] = None, xticklabels: Optional[List[str]] = None, **kwargs):
    fig, ax = plt.subplots(1, 1, figsize=(2, 1))
    if 'color' not in kwargs:
        kwargs['color'] = color_from_string(title if title is not None else 'none')
    ax.bar(*args, **kwargs)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.set_yticklabels([])
    ax.set_yticks([])
    if xticklabels is not None:
        ax.set_xticks(range(len(xticklabels)))
        ax.set_xticklabels(xticklabels)
    ax.xaxis.set_ticks_position('bottom')
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    return fig2html(fig)

