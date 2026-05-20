#!/usr/bin/env python3
"""
Create a 3-panel butterfly plot (PageRec@5 left, DocRec@5 right)
by question type (MG, DR, NG) for the provided methods.

Saves output to `butterfly_by_qtype.png` in the current working dir.
"""
import matplotlib.pyplot as plt
import numpy as np


def make_butterfly(methods, docrec, pagerec, categories, out_path="butterfly_by_qtype.png"):
    n_methods = len(methods)
    # larger figure and more left margin so method names are fully visible
    fig, axs = plt.subplots(1, len(categories), figsize=(20, 10), sharey=False)
    fig.subplots_adjust(left=0.56, wspace=0.28)

    # styling
    page_color = "#48b0a8"
    doc_color = "#6f5fa3"
    hatch_page = "///"
    hatch_doc = "\\\\\\"

    # use a consistent row index and invert the axis so the first method appears at the top
    y = np.arange(n_methods)
    height = 0.7

    # compute a symmetric x-limit based on the data and add padding
    all_docs = np.array(docrec).max()
    all_pages = np.array(pagerec).max()
    max_val = max(all_docs, all_pages)
    lim = min(1.4, max(1.1, max_val + 0.30))

    for i, cat in enumerate(categories):
        ax = axs[i]
        docs = np.array([d[i] for d in docrec])
        pages = np.array([p[i] for p in pagerec])

        # plot page recall to the LEFT as negative values
        ax.barh(y, -pages, height=height, color=page_color, edgecolor='k', hatch=hatch_page, label='PageRec@5')
        # plot doc recall to the RIGHT as positive values
        ax.barh(y, docs, height=height, color=doc_color, edgecolor='k', hatch=hatch_doc, label='DocRec@5')

        ax.set_title(cat, fontsize=14)

        # x limits symmetric around zero based on data
        ax.set_xlim(-lim, lim)
        ax.axvline(0, color='k', linewidth=0.8)

        # show absolute tick labels scaled to lim
        ticks = np.linspace(-lim, lim, 5)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{abs(t):.2f}" for t in ticks], fontsize=10)

        # add numeric labels on the bars with a larger proportional offset and clamp
        offset = 0.06 * lim
        for yi, pair in zip(y, zip(pages, docs)):
            pv, dv = pair
            x_page = -pv - offset
            x_doc = dv + offset
            # clamp to axis to avoid overflow
            x_page = max(x_page, -lim + 0.02)
            x_doc = min(x_doc, lim - 0.02)
            ax.text(x_page, yi, f"{pv:.2f}", va='center', ha='right', fontsize=10, clip_on=False)
            ax.text(x_doc, yi, f"{dv:.2f}", va='center', ha='left', fontsize=10, clip_on=False)

        # enforce identical y-limits so rows align across panels
        ax.set_ylim(n_methods - 0.5, -0.5)
        ax.set_yticks(y)
        # show method names on the leftmost axis only (clear and readable)
        if i == 0:
            ax.set_yticklabels(methods, fontsize=14, fontweight='medium')
            ax.tick_params(axis='y', which='major', pad=14, labelleft=True, labelsize=14)
        else:
            ax.set_yticklabels(["" for _ in methods])
            ax.tick_params(axis='y', labelleft=False)

        # slightly larger gridlines for readability
        ax.xaxis.grid(True, linestyle=':', linewidth=0.7)

    # legend (single combined)
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=page_color, edgecolor='k', hatch=hatch_page),
        plt.Rectangle((0, 0), 1, 1, facecolor=doc_color, edgecolor='k', hatch=hatch_doc),
    ]
    fig.legend(handles, ["PageRec@5", "DocRec@5"], loc='lower center', ncol=2, fontsize=11)

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"Saved butterfly plot to {out_path}")


if __name__ == '__main__':
    # Methods (rows)
    methods = [
        'Dense BGE-M3',
        'BGE-M3 + ReRanker',
        'BGE-M3 + Multi-HyDE + ReRanker',
        'BGE-M3 + Multi-HyDE + FT ReRanker',
        'Page-then-chunk',
        'Oracle document',
    ]

    # categories: MG (metrics-generated), DR (domain-relevant), NG (novel-generated)
    categories = ['MG', 'DR', 'NG']

    # DocRec arrays: list of length n_methods, each is (MG, DR, NG)
    docrec = [
        (0.98, 0.92, 0.92),  # Dense BGE-M3
        (1.00, 0.86, 0.98),  # BGE-M3 + ReRanker
        (1.00, 0.84, 0.94),  # BGE-M3 + Multi-HyDE + ReRanker
        (0.98, 0.90, 0.96),  # BGE-M3 + Multi-HyDE + FT ReRanker
        (1.00, 0.92, 0.90),  # Page-then-chunk (user-provided)
        (1.00, 1.00, 1.00),  # Oracle document
    ]

    # PageRec arrays: same layout
    pagerec = [
        (0.49, 0.22, 0.35),  # Dense BGE-M3
        (0.77, 0.18, 0.38),  # BGE-M3 + ReRanker
        (0.78, 0.29, 0.38),  # BGE-M3 + Multi-HyDE + ReRanker
        (0.80, 0.32, 0.54),  # BGE-M3 + Multi-HyDE + FT ReRanker
        (0.81, 0.41, 0.32),  # Page-then-chunk (user-provided)
        (0.68, 0.53, 0.58),  # Oracle document
    ]

    make_butterfly(methods, docrec, pagerec, categories)
