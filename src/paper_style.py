"""Shared Matplotlib style for paper figures.

Reviewer requirement: figure font must match the main text (Times New Roman).
`set_paper_font()` sets a serif font stack that prefers the real Times New Roman
when installed (e.g. on the authors' machine) and falls back to Liberation Serif
/ Nimbus Roman, which are metric-compatible with Times New Roman and render
essentially identically. Call it once, right after importing matplotlib, in every
figure-producing notebook or script.
"""

import matplotlib as mpl

# Preferred serif chain: real Times New Roman first, then metric-compatible clones.
_SERIF = ['Times New Roman', 'Liberation Serif', 'Nimbus Roman', 'DejaVu Serif']


def set_paper_font():
    """Apply the Times-New-Roman-consistent serif font to all Matplotlib text."""
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = _SERIF
    mpl.rcParams['mathtext.fontset'] = 'stix'   # serif math, matches Times
    mpl.rcParams['axes.unicode_minus'] = False  # ASCII minus renders in serif fonts
