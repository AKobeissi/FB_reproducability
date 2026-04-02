#!/usr/bin/env python3
"""
pdf_ex.py
=========
EDGAR PDF Downloader for FinanceBench / SECQA datasets.

Input:  A .jsonl file where each line has an "evidences" field containing
        doc_name in the format  {TICKER}_{YEAR}_{FORM_TYPE}
        e.g. "SLG_2011_10K"

Output: ./pdfs/{doc_name}.pdf  — one PDF per unique filing

Strategy per filing:
  1. Find the filing on EDGAR (CIK → accession number)
  2. Parse the filing index to list all documents
  3. If a .pdf exists → download it directly
  4. If only .htm  → download HTML + all embedded images, inline images as
     base64 data-URIs, inject page-break CSS, render to PDF via Playwright.

HTM → PDF approach (v2 — page-break injection):
  - JavaScript detects logical page boundaries in the HTML (div.page
    containers, <hr> separators, or "Page N" text markers).
  - CSS page-break-before rules are injected at each boundary so Playwright's
    native pagination produces one PDF page per logical source page.
  - Sections taller than one letter page are scaled down with CSS transform
    so they fit on a single page without overflow.
  - This replaces the previous "one tall page + pypdf crop" strategy, which
    produced blank white pages because Chromium silently stops rendering
    content beyond its internal height limit (~32 000 px).

Install:
    pip install requests playwright beautifulsoup4 lxml pypdf
    python -m playwright install chromium
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests

# ── SEC access ────────────────────────────────────────────────────────────────
EDGAR_USER_AGENT = "Amine Kobeissi amine.kobeissi@umontreal.ca"
HEADERS = {
    "User-Agent": EDGAR_USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}
SLEEP = 0.5   # SEC allows ~10 req/s; 0.5 s is comfortable
MAX_RETRIES = 4
RETRY_STATUS_CODES = {429, 500, 502, 503, 504}

# ── JavaScript run inside Playwright to locate logical page boundaries ────────
#
# Returns { method: str, boundaries: [{top: px, bottom: px}, ...] }
# where top/bottom are absolute pixel offsets from the top of the document.
#
# Detection priority:
#   1. div.page / div.Page / etc.  — modern EDGAR 10-K / 10-Q / 8-K filings
#   2. Full-width <hr> separators  — older EDGAR inline filings
#   3. Centred "Page N" text nodes — plain-text style filings
#   4. Single-page fallback        — no detectable structure
_JS_FIND_BOUNDARIES = """
() => {
    // ── helper: absolute y-offset of an element from the document top ──────
    function absTop(el) {
        let y = 0;
        while (el) { y += (el.offsetTop || 0); el = el.offsetParent; }
        return y;
    }
    function absBottom(el) { return absTop(el) + (el.offsetHeight || 0); }

    const bodyH = document.body.scrollHeight;

    // ── 1. Structured div.page containers ──────────────────────────────────
    const PAGE_SELECTORS = [
        'div.page', 'div.Page', 'div.PAGE',
        'div.document-page', 'div.filing-page',
        'div[class*="page-container"]',
        'section.page', 'article.page',
    ];
    for (const sel of PAGE_SELECTORS) {
        const els = Array.from(document.querySelectorAll(sel));
        if (els.length > 1) {
            const boundaries = els.map(el => ({
                top:    absTop(el),
                bottom: absBottom(el),
            }));
            return { method: sel, boundaries };
        }
    }

    // ── 2. Full-width <hr> separators ──────────────────────────────────────
    const bodyW = document.body.offsetWidth || document.body.scrollWidth || 816;
    const hrEls = Array.from(document.querySelectorAll('hr')).filter(hr => {
        const w = hr.offsetWidth || hr.getBoundingClientRect().width || 0;
        return w >= bodyW * 0.70;
    });
    if (hrEls.length > 0) {
        const splitYs = hrEls.map(hr => absBottom(hr));
        const boundaries = [];
        let prev = 0;
        for (const y of splitYs) {
            if (y - prev > 20) { boundaries.push({ top: prev, bottom: y }); }
            prev = y;
        }
        if (bodyH - prev > 20) { boundaries.push({ top: prev, bottom: bodyH }); }
        if (boundaries.length > 1) return { method: 'hr', boundaries };
    }

    // ── 3. Page-marker text lines (expanded pattern set) ──────────────────
    //
    // Matches all common EDGAR page-end patterns:
    //   • Plain number:          "5"    "5."   " 5 "
    //   • Number + bar:          "5 | Company Name | 2023"
    //   • Page word:             "Page 5"   "PAGE 5"
    //   • Dashes:                "- 5 -"    "– 5 –"
    //   • Roman numerals (lc):   "iii"   "iv"   "xii"
    //   • Roman numerals (uc):   "III"   "IV"   "XII"
    //   • Letter pages:          "A"  "B"  (appendix)
    //   • Letter-number:         "A-1"  "B-2"
    //
    // Alignment: must be centred, right-aligned, or left-aligned but short.
    // We explicitly exclude style.display==='block' because EVERY block element
    // has display:block — it is NOT a centering signal.
    const pageNumRe = /^\s*(?:page\s+[\d]+|[-\u2013]\s*\d+\s*[-\u2013]|\d+\.?|[ivxlcdm]{1,8}|[IVXLCDM]{1,8}|[A-Z]-?\d+)(?:\s*[|\u00B7\u2022\u00BA]\s*[^\\n]{0,80})?\s*$/;
    const walker = document.createTreeWalker(
        document.body, NodeFilter.SHOW_TEXT, null
    );
    const splitYs = [];
    let node;
    while ((node = walker.nextNode())) {
        const txt = node.textContent.trim();
        if (!txt || txt.length > 120) continue;
        if (!pageNumRe.test(txt)) continue;
        const el = node.parentElement;
        if (!el) continue;
        const style = window.getComputedStyle(el);
        // Centred, right-aligned, or short left-aligned line (≤ 60 chars)
        // Do NOT use style.display === 'block' — that matches everything.
        const isCentered = (
            style.textAlign === 'center' || style.textAlign === 'right' ||
            el.align === 'center'        || el.align === 'right' ||
            (style.textAlign !== 'left' && txt.length <= 60)
        );
        if (isCentered) splitYs.push(absBottom(el));
    }
    if (splitYs.length > 0) {
        const boundaries = [];
        let prev = 0;
        for (const y of splitYs) {
            if (y - prev > 20) { boundaries.push({ top: prev, bottom: y }); }
            prev = y;
        }
        if (bodyH - prev > 20) { boundaries.push({ top: prev, bottom: bodyH }); }
        if (boundaries.length > 1) return { method: 'page-text', boundaries };
    }

    // ── 4. No structure found — treat entire document as one page ──────────
    return { method: 'single', boundaries: [{ top: 0, bottom: bodyH }] };
}
"""

# ── JavaScript to inject page-break CSS and scale oversized sections ──────────
#
# Called AFTER _JS_FIND_BOUNDARIES.  Receives the detection result and the
# target page height (in CSS px).  Modifies the live DOM so that Playwright's
# standard @page pagination produces exactly one PDF page per logical page.
#
# For each logical page boundary it:
#   1. Finds the DOM element closest to the boundary and adds
#      class="pdf-page-break" (→ page-break-before: always).
#   2. Measures the content height of the section.
#   3. If the section is taller than the target, wraps it in a <div> with
#      CSS transform: scale(factor) so it shrinks to fit one PDF page.
#
_JS_INJECT_PAGE_BREAKS = """
([info, targetPageH]) => {
    const method = info.method;
    const boundaries = info.boundaries;
    const PAGE_SELECTORS = [
        'div.page', 'div.Page', 'div.PAGE',
        'div.document-page', 'div.filing-page',
        'div[class*="page-container"]',
        'section.page', 'article.page',
    ];

    // ── helper: absolute y-offset of an element ──────────────────────────
    function absTop(el) {
        let y = 0;
        while (el) { y += (el.offsetTop || 0); el = el.offsetParent; }
        return y;
    }

    // ── helper: true if an element's text content is only a bare page number ─
    // Used to avoid injecting page-break-before on a page-number row, which
    // would orphan it on its own PDF page.
    const _pageNumOnlyRe = /^\s*(?:\d{1,4}|[ivxlcdmIVXLCDM]{1,8}|[A-Z]-?\d{1,3})\s*$/;
    function isPageNumElement(el) {
        const t = (el.textContent || '').trim();
        // Short text that is only a page number — and the element is small
        return t.length <= 8 && _pageNumOnlyRe.test(t) && (el.offsetHeight || 99) < 50;
    }

    // ── helper: find the first block-level element whose top is AT OR AFTER y ─
    //
    // Using "nearest" is wrong: if the boundary is at Y=5000 and the last
    // element of the previous section ends at Y=4992, "nearest" picks it and
    // injects page-break-before it, pulling that element onto the NEXT PDF
    // page (backflow).  We want the FIRST element that STARTS on the new page,
    // i.e. top >= y.  Fall back to the closest-before element only when nothing
    // is found at or after y.
    //
    // skipPageNums: when true, skip elements whose entire text is a bare page
    // number so we don't inject a break that orphans the page number alone.
    function elementNearY(y, skipPageNums) {
        const candidates = document.body.querySelectorAll(
            'div, section, article, table, p, h1, h2, h3, h4, h5, h6, ul, ol, dl, pre, blockquote, hr, form'
        );
        let firstAfter = null, firstAfterTop = Infinity;
        let fallback = null, fallbackDist = Infinity;
        for (const el of candidates) {
            const top = absTop(el);
            if (skipPageNums && isPageNumElement(el)) continue;
            if (top >= y - 2) {
                // Element starts at or just at the boundary → first of new page
                if (top < firstAfterTop) { firstAfterTop = top; firstAfter = el; }
            } else {
                // Before boundary — keep only as last-resort fallback
                const d = y - top;
                if (d < fallbackDist) { fallbackDist = d; fallback = el; }
            }
        }
        return firstAfter || fallback;
    }

    // ── PRE-STRIP: remove ALL native page-break CSS before injecting ours ──
    //
    // Many EDGAR HTM filings ship their own page-break-before:always rules
    // (on div.page containers, section headers, or page-number rows).
    // If we add a SECOND page-break on the same element (or the element just
    // after one that already has a break), Chromium inserts a blank page
    // between every pair of content pages.  Stripping native breaks first
    // gives us clean, exclusive control over pagination.
    //
    // We strip via inline style overrides (not by editing stylesheets) so
    // that the override survives specificity battles from the source CSS.
    {
        const stripProps = [
            ['pageBreakBefore', 'auto'],
            ['pageBreakAfter',  'auto'],
            ['pageBreakInside', 'auto'],
            ['breakBefore',     'auto'],
            ['breakAfter',      'auto'],
            ['breakInside',     'auto'],
        ];
        const BREAK_VALS = new Set(['always', 'page', 'left', 'right', 'column']);
        for (const el of document.body.querySelectorAll('*')) {
            const cs = window.getComputedStyle(el);
            let needsStrip = false;
            for (const [prop] of stripProps) {
                const val = cs[prop] || '';
                if (BREAK_VALS.has(val)) { needsStrip = true; break; }
            }
            if (needsStrip) {
                for (const [prop, reset] of stripProps) {
                    el.style[prop] = reset;
                }
            }
        }
    }

    let injected = 0;

    // ── METHOD A: div.page containers ────────────────────────────────────
    if (PAGE_SELECTORS.some(sel => method === sel)) {
        const els = Array.from(document.querySelectorAll(method));
        els.forEach((el, i) => {
            if (i > 0) {
                el.classList.add('pdf-page-break');
                injected++;
            }
            // Scale if oversized.
            // Use zoom (not transform:scale) — zoom shrinks the layout
            // footprint so content doesn't bleed into the next PDF page.
            // transform:scale is purely visual; the browser still reserves
            // the original height in the layout flow.
            const h = el.scrollHeight || el.offsetHeight || 0;
            if (h > targetPageH * 1.05) {
                const factor = targetPageH / h;
                el.style.zoom = factor;
                el.style.height = targetPageH + 'px';
                el.style.overflow = 'hidden';
                el.style.pageBreakInside = 'avoid';
                el.style.breakInside = 'avoid';
            }
        });
        return { injected, scaled: 0 };
    }

    // ── METHOD B: <hr> separators ────────────────────────────────────────
    if (method === 'hr') {
        const bodyW = document.body.offsetWidth || document.body.scrollWidth || 816;
        const hrs = Array.from(document.querySelectorAll('hr')).filter(hr => {
            const w = hr.offsetWidth || hr.getBoundingClientRect().width || 0;
            return w >= bodyW * 0.70;
        });

        // Collapse HRs to zero height (don't display:none — that shifts
        // the layout and invalidates the boundary Y coords that were measured
        // before this function was called).
        hrs.forEach(hr => {
            hr.style.cssText += '; visibility:hidden !important; height:0 !important; margin:0 !important; padding:0 !important; border:none !important;';
        });

        // Use boundary coordinates to find the start element of each page
        // and inject page-break-before there.
        let scaled = 0;
        for (let i = 1; i < boundaries.length; i++) {
            const el = elementNearY(boundaries[i].top, /*skipPageNums=*/true);
            if (el && !el.classList.contains('pdf-page-break')) {
                el.classList.add('pdf-page-break');
                injected++;
            }
        }

        _scaleOversizedSections(targetPageH);
        return { injected, scaled };
    }

    // ── METHOD C: "Page N" text boundaries ──────────────────────────────
    //
    // boundaries[i].top is absBottom() of the page-number element that ends
    // logical page i-1.  We look strictly AFTER that bottom edge (+ 5 px)
    // so we never accidentally inject page-break-before on the page-number
    // element itself, which would orphan it alone on a PDF page.
    if (method === 'page-text') {
        for (let i = 1; i < boundaries.length; i++) {
            const el = elementNearY(boundaries[i].top + 5, /*skipPageNums=*/true);
            if (el) {
                el.classList.add('pdf-page-break');
                injected++;
            }
        }
        _scaleOversizedSections(targetPageH);
        return { injected, scaled: 0 };
    }

    // ── METHOD D: single page — let Playwright paginate naturally ─────────
    //
    // Previously this zoomed the entire body to fit one page, which produced
    // a 1-page PDF and lost all content beyond the first screen.  The correct
    // fallback is to do NOTHING: Playwright's native @page pagination will
    // split the document at letter-size boundaries, giving a multi-page PDF
    // that is imperfect (breaks mid-paragraph) but complete and readable.
    if (method === 'single') {
        return { injected: 0, scaled: 0 };
    }

    // ── Fallback: inject breaks at boundary coordinates ─────────────────
    for (let i = 1; i < boundaries.length; i++) {
        const el = elementNearY(boundaries[i].top + 5, /*skipPageNums=*/true);
        if (el) {
            el.classList.add('pdf-page-break');
            injected++;
        }
    }
    _scaleOversizedSections(targetPageH);
    return { injected, scaled: 0 };
}
"""

# ── Helper JS function injected into page scope for scaling ───────────────────
_JS_SCALE_HELPER = """
window._scaleOversizedSections = function(targetH) {
    // ── helpers ────────────────────────────────────────────────────────────
    function absTop(el) {
        let y = 0, cur = el;
        while (cur) { y += (cur.offsetTop || 0); cur = cur.offsetParent; }
        return y;
    }

    const breakEls = Array.from(document.querySelectorAll('.pdf-page-break'));

    // ── No page breaks: scale the whole body if it overflows ───────────────
    if (breakEls.length === 0) {
        const h = document.body.scrollHeight;
        if (h > targetH * 1.05) {
            document.body.style.zoom = targetH / h;
        }
        return;
    }

    // ── First section (before the first page-break) ────────────────────────
    // Measure from the top of <body> to the absolute top of the first break
    // element.  We collect all direct children of <body> that come before
    // the first break (or its nearest body-level ancestor), because the
    // break might be nested several levels deep.
    (function scaleFirstSection() {
        const firstBreak = breakEls[0];

        // Walk up from firstBreak to find its body-level ancestor
        let bodyChild = firstBreak;
        while (bodyChild.parentNode && bodyChild.parentNode !== document.body) {
            bodyChild = bodyChild.parentNode;
        }

        // Collect body children that precede the body-level ancestor of firstBreak
        const firstEls = [];
        for (const child of Array.from(document.body.children)) {
            if (child === bodyChild) break;
            firstEls.push(child);
        }
        if (firstEls.length === 0) return;

        // Measure: from y=0 (body top) to the absolute top of the first break
        const sectionH = absTop(firstBreak);
        if (sectionH <= targetH * 1.05) return;

        const factor = targetH / sectionH;
        const wrapper = document.createElement('div');
        wrapper.style.cssText = [
            'height:' + targetH + 'px',
            'max-height:' + targetH + 'px',
            'overflow:hidden',
            'box-sizing:border-box',
            'page-break-inside:avoid',
            'break-inside:avoid',
        ].join(';');
        const inner = document.createElement('div');
        inner.style.cssText = [
            'zoom:' + factor,
            'width:' + Math.round(100 / factor) + '%',
            'box-sizing:border-box',
        ].join(';');
        document.body.insertBefore(wrapper, firstEls[0]);
        wrapper.appendChild(inner);
        for (const el of firstEls) inner.appendChild(el);
        // Also move bodyChild (the ancestor of firstBreak) into the scaled section
        // only if it's different from the already-collected elements
        if (!firstEls.includes(bodyChild)) inner.appendChild(bodyChild);
    })();

    // ── Middle + last sections (each starts at a .pdf-page-break element) ──
    // For each page-break element, collect all DOM siblings until the next
    // page-break (or end of parent), measure their combined height, and if
    // they overflow zoom them inside a fixed-height wrapper.
    for (let i = 0; i < breakEls.length; i++) {
        const startEl = breakEls[i];
        const nextEl  = breakEls[i + 1] || null;
        const parent  = startEl.parentNode;
        if (!parent) continue;

        // Only handle the case where start and next are siblings.
        // (When they're not, the page-break CSS alone has to suffice.)
        if (nextEl && startEl.parentNode !== nextEl.parentNode) {
            continue;
        }

        // Collect sibling elements that belong to this section
        const sectionEls = [];
        let cur = startEl;
        while (cur && cur !== nextEl) {
            sectionEls.push(cur);
            cur = cur.nextElementSibling;
        }
        if (sectionEls.length === 0) continue;

        // Measure total height of the section
        const topY    = absTop(startEl);
        const bottomY = nextEl ? absTop(nextEl) : (document.documentElement.scrollHeight);
        const sectionH = bottomY - topY;
        if (sectionH <= targetH * 1.05) continue;

        const factor = targetH / sectionH;

        // Build: wrapper (page-break + fixed height + overflow:hidden)
        //          └── inner (zoom applied here)
        //                └── all section elements
        const wrapper = document.createElement('div');
        wrapper.style.cssText = [
            'height:' + targetH + 'px',
            'max-height:' + targetH + 'px',
            'overflow:hidden',
            'box-sizing:border-box',
            'page-break-before:always',
            'break-before:page',
            // Critical: tell Chromium's native paginator NOT to split this
            // wrapper across two PDF pages — the zoom makes it fit exactly
            // one page, and a split would produce a partial-content page.
            'page-break-inside:avoid',
            'break-inside:avoid',
        ].join(';');
        wrapper.classList.add('pdf-page-break');

        const inner = document.createElement('div');
        // zoom shrinks the layout footprint (unlike transform:scale which is
        // purely visual and leaves the original height in the flow).
        inner.style.cssText = [
            'zoom:' + factor,
            'width:' + Math.round(100 / factor) + '%',
            'box-sizing:border-box',
        ].join(';');

        parent.insertBefore(wrapper, startEl);
        wrapper.appendChild(inner);

        for (const el of sectionEls) {
            // The page-break class moves to the wrapper — remove from startEl
            el.classList.remove('pdf-page-break');
            inner.appendChild(el);
        }
    }
};
"""

# ── CSS injected before rendering ─────────────────────────────────────────────
# Uses standard @page rules + page-break classes.  Splitting is handled by
# Playwright's native pagination (NOT by pypdf crop of a tall page, which
# breaks because Chromium silently stops rendering content beyond ~32 000 px).
_RENDER_CSS = """
<meta charset="utf-8">
<style>
@page { size: 8.5in 11in; margin: 0; }
* { box-sizing: border-box; }
body { margin: 0; padding: 0; }

/* ── Images ─────────────────────────────────────────────────────────────────
   Keep display:inline-block so that text-align:center / align="center" from
   parent <center>, <td>, or <p> elements correctly centres the image.
   display:block would require explicit margin:auto and breaks legacy <center>
   tag centering used extensively in older EDGAR filings.               */
img { max-width: 100%; height: auto; display: inline-block; vertical-align: middle; }

/* Ensure <center> and align="center" attributes are honoured */
center, [align="center"] { text-align: center !important; display: block; }
[align="right"]           { text-align: right  !important; }
[align="left"]            { text-align: left   !important; }
td[align="center"] img, th[align="center"] img { margin: 0 auto; display: block; }

/* ── Unicode / symbol font fallback ─────────────────────────────────────── */
/* Ensure checkboxes (U+2610–2612), bullets (U+2022), dashes (U+2013/2014),
   and other common SEC-filing symbols render from a bundled system font
   rather than falling back to a glyph-less placeholder box.
   Font stack: prefer common sans-serif → DejaVu (bundled with most Linux
   distros including the cluster) → generic sans-serif.              */
body, td, th, p, div, span, li {
    font-family: Arial, Helvetica, "DejaVu Sans", "Liberation Sans",
                 "Noto Sans", FreeSans, sans-serif;
}

/* ── Page-break utility ──────────────────────────────────────────────────── */
.pdf-page-break {
    page-break-before: always !important;
    break-before: page !important;
}

/* ── Scaled-section wrapper ──────────────────────────────────────────────── */
.pdf-scaled-wrapper {
    transform-origin: top left;
    page-break-inside: avoid !important;
    break-inside: avoid !important;
    overflow: hidden;
}
</style>
"""


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def sleep_get(url: str, **kwargs) -> requests.Response:
    timeout = kwargs.pop("timeout", 30)
    for attempt in range(1, MAX_RETRIES + 1):
        time.sleep(SLEEP)
        try:
            response = requests.get(url, headers=HEADERS, timeout=timeout, **kwargs)
        except requests.RequestException as e:
            if attempt == MAX_RETRIES:
                raise
            backoff = min(8.0, float(2 ** (attempt - 1)))
            print(
                f"    [warn] request error for {url} ({e.__class__.__name__}); "
                f"retrying in {backoff:.1f}s"
            )
            time.sleep(backoff)
            continue

        if response.status_code in RETRY_STATUS_CODES and attempt < MAX_RETRIES:
            backoff = min(8.0, float(2 ** (attempt - 1)))
            print(
                f"    [warn] HTTP {response.status_code} for {url}; "
                f"retrying in {backoff:.1f}s"
            )
            time.sleep(backoff)
            continue

        return response

    raise RuntimeError(f"Unreachable retry path while fetching {url}")


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Parse doc_name  →  (ticker, year, form_type)
# ══════════════════════════════════════════════════════════════════════════════

def parse_doc_name(doc_name: str):
    """
    'SLG_2011_10K'    → ('SLG', 2011, None, '10-K')
    'AAPL_2020_10K'   → ('AAPL', 2020, None, '10-K')
    'LLY_2024Q2_10Q'  → ('LLY', 2024, 2, '10-Q')
    'MFC_2012_20F'    → ('MFC', 2012, None, '20-F')
    """
    parts = doc_name.rsplit("_", 2)
    if len(parts) != 3:
        raise ValueError(f"Cannot parse doc_name: {doc_name!r}")
    ticker, period_str, form_raw = parts

    m = re.fullmatch(r"(\d{4})(?:Q([1-4]))?", period_str.upper())
    if not m:
        raise ValueError(f"Cannot parse year/quarter token in doc_name: {doc_name!r}")

    year = int(m.group(1))
    quarter = int(m.group(2)) if m.group(2) else None
    form_type = re.sub(r"^(\d+)([A-Z].*)$", r"\1-\2", form_raw)
    return ticker.upper(), year, quarter, form_type


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Ticker → CIK
# ══════════════════════════════════════════════════════════════════════════════

_cik_cache: dict = {}
_ticker_map: dict[str, str] | None = None


def _load_ticker_map() -> dict[str, str]:
    global _ticker_map
    if _ticker_map is not None:
        return _ticker_map

    _ticker_map = {}
    try:
        r = sleep_get("https://www.sec.gov/files/company_tickers.json")
        r.raise_for_status()
        for entry in r.json().values():
            ticker = str(entry.get("ticker", "")).upper().strip()
            cik_raw = entry.get("cik_str")
            if not ticker or cik_raw is None:
                continue
            _ticker_map[ticker] = str(cik_raw).zfill(10)
    except Exception as e:
        print(f"  [warn] ticker map failed: {e}")

    return _ticker_map

def get_cik(ticker: str) -> str | None:
    ticker = ticker.upper()
    if ticker in _cik_cache:
        return _cik_cache[ticker]

    ticker_map = _load_ticker_map()
    cik = ticker_map.get(ticker)
    if cik:
        _cik_cache[ticker] = cik
        print(f"  CIK={cik}")
        return cik

    # Fallback: EDGAR company search
    try:
        r = sleep_get(
            "https://efts.sec.gov/LATEST/search-index",
            params={"q": f'"{ticker}"', "forms": "10-K"},
        )
        r.raise_for_status()
        hits = r.json().get("hits", {}).get("hits", [])
        for hit in hits:
            src = hit.get("_source", {})
            if src.get("ticker_symbol", "").upper() == ticker:
                entity_id = str(src.get("entity_id", "")).zfill(10)
                _cik_cache[ticker] = entity_id
                print(f"  CIK={entity_id}  (via search fallback)")
                return entity_id
    except Exception as e:
        print(f"  [warn] EDGAR search fallback failed: {e}")

    return None


# ══════════════════════════════════════════════════════════════════════════════
# 3.  CIK + year + form_type  →  accession number
# ══════════════════════════════════════════════════════════════════════════════

def _date_to_quarter(date_str: str) -> int | None:
    if len(date_str) < 7:
        return None
    try:
        month = int(date_str[5:7])
    except ValueError:
        return None
    return (month - 1) // 3 + 1


def get_accession_number(
    cik: str,
    year: int,
    form_type: str,
    quarter: int | None = None,
) -> str | None:
    def _search_block(block: dict) -> str | None:
        forms = block.get("form", [])
        dates = block.get("filingDate", [])
        accs  = block.get("accessionNumber", [])
        report_dates = block.get("reportDate", [])
        candidates = []
        for idx, form in enumerate(forms):
            if form.upper() != form_type.upper():
                continue

            date = dates[idx] if idx < len(dates) else ""
            acc = accs[idx] if idx < len(accs) else ""
            report_date = report_dates[idx] if idx < len(report_dates) else ""
            if not acc:
                continue

            if quarter is not None:
                period_date = report_date or date
                if not period_date:
                    continue

                try:
                    period_year = int(period_date[:4])
                except ValueError:
                    continue

                period_quarter = _date_to_quarter(period_date)
                if period_year != year:
                    continue
                if period_quarter is not None and period_quarter != quarter:
                    continue

                preferred = 0 if report_date else 1
                candidates.append((preferred, date, acc))
                continue

            period_date = report_date or date
            if not period_date:
                continue

            try:
                fy = int(period_date[:4])
            except ValueError:
                continue

            if fy == year or fy == year + 1:
                candidates.append((abs(fy - (year + 1)), date, acc))

        if not candidates:
            return None

        candidates.sort(key=lambda x: (x[0], x[1]))
        return candidates[0][2]

    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    try:
        r = sleep_get(url)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"  [error] submissions fetch failed: {e}")
        return None

    acc = _search_block(data.get("filings", {}).get("recent", {}))
    if acc:
        return acc

    for file_info in data.get("filings", {}).get("files", []):
        try:
            r2 = sleep_get(f"https://data.sec.gov/submissions/{file_info['name']}")
            r2.raise_for_status()
            acc = _search_block(r2.json())
            if acc:
                return acc
        except Exception:
            continue

    return None


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Accession number → list of documents
# ══════════════════════════════════════════════════════════════════════════════

def get_filing_documents(cik: str, accession_number: str) -> list[dict]:
    cik_int   = int(cik)
    acc_clean = accession_number.replace("-", "")
    base_url  = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_clean}"

    html = ""
    for idx_url in [
        f"{base_url}/{accession_number}-index.htm",
        f"{base_url}/{accession_number}-index.html",
        f"{base_url}/",
    ]:
        try:
            r = sleep_get(idx_url)
            if r.ok and len(r.text) > 200:
                html = r.text
                print(f"  Index: {idx_url}")
                break
        except Exception:
            continue

    if not html:
        print(f"  [error] Could not fetch filing index for {accession_number}")
        return []

    docs = []
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.IGNORECASE | re.DOTALL)
    for row in rows:
        hrefs = re.findall(r'href="([^"]+)"', row, re.IGNORECASE)
        if not hrefs:
            continue
        filename = hrefs[0].split("/")[-1]
        if not filename or "?" in filename:
            continue
        cells = [re.sub(r"<[^>]+>", "", c).strip()
                 for c in re.findall(r"<td[^>]*>(.*?)</td>", row, re.IGNORECASE | re.DOTALL)]
        docs.append({
            "sequence":    cells[0] if len(cells) > 0 else "",
            "description": cells[1] if len(cells) > 1 else "",
            "document":    filename,
            "type":        cells[3] if len(cells) > 3 else "",
            "url":         f"{base_url}/{filename}",
        })

    return docs


# ══════════════════════════════════════════════════════════════════════════════
# 5a.  Direct PDF download
# ══════════════════════════════════════════════════════════════════════════════

def download_pdf_direct(url: str, dest: Path) -> bool:
    try:
        r = sleep_get(url, stream=True)
        r.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1 << 16):
                f.write(chunk)
        print(f"  ✓ PDF downloaded  ({dest.stat().st_size // 1024} KB)")
        return True
    except Exception as e:
        print(f"  [error] PDF download failed: {e}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
# 5b.  Image inlining helper  (FIX: was the root cause of broken images)
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_as_data_uri(url: str) -> str | None:
    """
    Download a resource via the SEC-compliant requests session and return a
    base64 data-URI string.  Returns None on failure (so the original src is
    left unchanged rather than breaking the whole render).
    """
    try:
        r = sleep_get(url)
        r.raise_for_status()
        raw   = r.content
        ctype = r.headers.get("Content-Type", "").split(";")[0].strip()
        if not ctype:
            # Guess from URL extension
            ext   = Path(urlparse(url).path).suffix.lower()
            ctype = mimetypes.types_map.get(ext, "application/octet-stream")
        b64  = base64.b64encode(raw).decode("ascii")
        return f"data:{ctype};base64,{b64}"
    except Exception as e:
        print(f"    [warn] could not inline resource {url}: {e}")
        return None


def inline_images_and_css(html: str, base_url: str) -> str:
    """
    Parse the HTML, resolve every <img src>, <link rel=stylesheet href>,
    and CSS url() reference to an absolute URL, download it via requests,
    and replace it with a base64 data-URI.

    This ensures Playwright never needs to reach sec.gov from inside headless
    Chromium, which would fail due to CORS / missing SEC User-Agent headers.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        print("  [warn] beautifulsoup4 not installed — images may not render.")
        print("         Run: pip install beautifulsoup4 lxml")
        return html

    soup = BeautifulSoup(html, "lxml")  # html may be str or bytes; lxml handles both

    # ── <img src="..."> ──────────────────────────────────────────────────────
    for img in soup.find_all("img"):
        src = img.get("src", "")
        if not src or src.startswith("data:"):
            continue
        abs_url = urljoin(base_url, src)
        data_uri = _fetch_as_data_uri(abs_url)
        if data_uri:
            img["src"] = data_uri
            # Remove srcset — it would override our inlined src
            if img.get("srcset"):
                del img["srcset"]

    # ── <link rel="stylesheet" href="..."> ──────────────────────────────────
    for link in soup.find_all("link", rel=lambda r: r and "stylesheet" in r):
        href = link.get("href", "")
        if not href or href.startswith("data:"):
            continue
        abs_url = urljoin(base_url, href)
        data_uri = _fetch_as_data_uri(abs_url)
        if data_uri:
            # Replace <link> with inline <style> (Playwright honours both)
            style_tag = soup.new_tag("style")
            # data_uri is the raw CSS bytes encoded; we need the text
            try:
                r2 = sleep_get(abs_url)
                r2.raise_for_status()
                style_tag.string = r2.text
            except Exception:
                continue
            link.replace_with(style_tag)

    # ── CSS url() inside <style> tags ────────────────────────────────────────
    for style_tag in soup.find_all("style"):
        css = style_tag.string or ""
        def _replace_css_url(m: re.Match) -> str:
            raw_ref = m.group(1).strip("'\"")
            if raw_ref.startswith("data:"):
                return m.group(0)
            abs_url = urljoin(base_url, raw_ref)
            data_uri = _fetch_as_data_uri(abs_url)
            return f"url('{data_uri}')" if data_uri else m.group(0)
        css = re.sub(r"url\(([^)]+)\)", _replace_css_url, css)
        style_tag.string = css

    # ── inline style="background-image: url(...)" ────────────────────────────
    for tag in soup.find_all(style=True):
        style_val = tag["style"]
        def _replace_inline_url(m: re.Match) -> str:
            raw_ref = m.group(1).strip("'\"")
            if raw_ref.startswith("data:"):
                return m.group(0)
            abs_url = urljoin(base_url, raw_ref)
            data_uri = _fetch_as_data_uri(abs_url)
            return f"url('{data_uri}')" if data_uri else m.group(0)
        tag["style"] = re.sub(r"url\(([^)]+)\)", _replace_inline_url, style_val)

    return str(soup)


# ══════════════════════════════════════════════════════════════════════════════
# 5c.  Inject minimal render CSS
# ══════════════════════════════════════════════════════════════════════════════

def inject_render_css(html: str) -> str:
    """Inject a minimal CSS block to stabilise layout (no page-break rules)."""
    if "</head>" in html:
        return html.replace("</head>", _RENDER_CSS + "</head>", 1)
    if "<body" in html:
        return re.sub(r"(<body[^>]*>)", r"\1" + _RENDER_CSS, html, count=1)
    return _RENDER_CSS + html


# ══════════════════════════════════════════════════════════════════════════════
# 5d.  HTM → PDF via Playwright  —  tall render + pypdf crop
# ══════════════════════════════════════════════════════════════════════════════

# Letter paper dimensions at 96 dpi.
_VIEWPORT_W_PX = 816
_PAGE_H_PX = 1056       # 11 in × 96 dpi
_PAGE_H_CSS = "11in"
_PAGE_W_CSS = "8.5in"

def convert_htm_to_pdf(htm_url: str, dest: Path) -> bool:
    """
    Render an EDGAR HTM filing to a PDF where each output PDF page contains
    exactly one logical source page — no content is ever cut mid-page.

    Algorithm  (v2 — fixes blank-page bug in the tall-render approach)
    ------------------------------------------------------------------
    1. Fetch the HTML via requests (SEC User-Agent compliant).
    2. Inline all images / stylesheets as base64 data-URIs so Playwright
       never needs to reach sec.gov from inside headless Chromium.
    3. Load the processed HTML in Playwright at a fixed Letter-width viewport.
    4. Run _JS_FIND_BOUNDARIES to locate logical page boundaries.
    5. Run _JS_INJECT_PAGE_BREAKS to add CSS page-break rules at each boundary
       and scale down any section that exceeds one letter page.
    6. Render with Playwright's standard page.pdf() using Letter size.

    Why not "one tall page + pypdf crop"?
    Chromium silently stops rendering content beyond its internal height limit
    (~32 000 px), producing blank white pages for everything below that point.
    Standard pagination avoids this entirely.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("  [error] playwright not installed.")
        print("          pip install playwright && python -m playwright install chromium")
        return False

    import os, tempfile

    print("  Converting HTM → PDF (page-break injection) …")

    # ── 1. Fetch HTML ─────────────────────────────────────────────────────────
    #
    # We use r.content (raw bytes) rather than r.text so that requests' codec
    # guess (which defaults to ISO-8859-1 when the server omits a charset
    # declaration — common on old EDGAR filings) cannot corrupt Unicode chars.
    # Strategy:
    #   1. Honour an explicit charset in the Content-Type header.
    #   2. Look for <meta charset> / <meta http-equiv=Content-Type> in the
    #      first 4 KB of the raw bytes.
    #   3. Try UTF-8 (covers the vast majority of modern EDGAR filings).
    #   4. Fall back to latin-1 (lossless for 8-bit bytes, no UnicodeDecodeError).
    try:
        r = sleep_get(htm_url)
        r.raise_for_status()
        raw_bytes = r.content

        # ── Charset detection ──────────────────────────────────────────────
        import re as _re
        detected_enc: str | None = None

        # 1. Content-Type header
        ct = r.headers.get("Content-Type", "")
        m = _re.search(r"charset\s*=\s*([\w-]+)", ct)
        if m:
            detected_enc = m.group(1)

        # 2. HTML meta tag (check the first 4 KB only)
        if not detected_enc:
            head = raw_bytes[:4096]
            m = _re.search(
                rb'(?:charset[\s]*=[\s]*[\x22\x27]?|charset[\s]*:[\s]*)([\w-]+)',
                head, _re.IGNORECASE
            )
            if m:
                detected_enc = m.group(1).decode("ascii", errors="ignore")

        # 3/4. UTF-8 then latin-1
        for enc in filter(None, [detected_enc, "utf-8", "latin-1"]):
            try:
                html = raw_bytes.decode(enc)
                break
            except (UnicodeDecodeError, LookupError):
                continue
        else:
            html = raw_bytes.decode("latin-1")  # guaranteed lossless

        enc_used = detected_enc or "utf-8"
        print(f"  Encoding: {enc_used}")

        if "undeclared automated tool" in html.lower() or "reference id:" in html.lower():
            print("  [error] SEC bot-block page — check User-Agent")
            return False
    except Exception as e:
        print(f"  [error] HTM fetch failed: {e}")
        return False

    # ── 2. Inline images / CSS as base64 ─────────────────────────────────────
    print("  Inlining images …")
    html = inline_images_and_css(html, base_url=htm_url)

    # ── 3. Inject render CSS (includes @page rules) ──────────────────────────
    html = inject_render_css(html)

    # Write to temp file for Playwright to load via file:// URI
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".html", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(html)
        tmp_path = tmp.name

    try:
        dest.parent.mkdir(parents=True, exist_ok=True)

        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=[
                    # Use system font config so DejaVu/Liberation/Noto fonts
                    # are found and Unicode glyphs (checkboxes, bullets, etc.)
                    # render correctly instead of falling back to ▯ placeholders.
                    "--font-render-hinting=none",
                    "--disable-font-subpixel-positioning",
                    # Allow file:// pages to load local stylesheets / resources
                    "--disable-web-security",
                    "--allow-file-access-from-files",
                    # Larger shared memory for rendering large documents
                    "--disable-dev-shm-usage",
                    "--no-sandbox",
                ],
            )
            ctx = browser.new_context(
                viewport={"width": _VIEWPORT_W_PX, "height": 1080},
            )
            pw_page = ctx.new_page()

            file_uri = Path(tmp_path).as_uri()
            pw_page.goto(file_uri, wait_until="networkidle", timeout=120_000)
            pw_page.wait_for_timeout(1_500)   # let any JS layout settle

            # ── 4. Detect logical page boundaries ────────────────────────────
            result     = pw_page.evaluate(_JS_FIND_BOUNDARIES)
            boundaries = result["boundaries"]
            method     = result["method"]
            print(f"  Page detection: '{method}' → {len(boundaries)} logical page(s)")

            # ── 5. Inject the scale-helper into the page scope ───────────────
            pw_page.evaluate(_JS_SCALE_HELPER)

            # ── 6. Inject page-break CSS at each boundary ────────────────────
            inject_result = pw_page.evaluate(
                _JS_INJECT_PAGE_BREAKS,
                [result, _PAGE_H_PX],    # packed as [info, targetPageH]; JS destructs with ([info, targetPageH]) =>
            )
            print(f"  Injected {inject_result.get('injected', 0)} page-break(s)")

            # Force a synchronous layout pass by reading a layout property,
            # then wait for async reflow to settle.  Large EDGAR documents
            # (100+ pages) need more than 500ms; 1500ms is safer.
            pw_page.evaluate("() => document.body.scrollHeight")  # forced reflow
            pw_page.wait_for_timeout(1_500)

            # ── 7. Render PDF with standard letter-size pagination ────────────
            pw_page.emulate_media(media="print")
            pw_page.wait_for_timeout(300)   # brief extra settle after media switch

            pdf_bytes: bytes = pw_page.pdf(
                width=_PAGE_W_CSS,
                height=_PAGE_H_CSS,
                print_background=True,
                prefer_css_page_size=False,
                margin={"top": "0px", "right": "0px",
                        "bottom": "0px", "left": "0px"},
            )
            browser.close()

        # ── 8. Write final PDF ────────────────────────────────────────────────
        with open(dest, "wb") as f:
            f.write(pdf_bytes)

        size_kb = dest.stat().st_size // 1024
        # Count pages via pypdf if available, else estimate
        try:
            from pypdf import PdfReader
            import io
            n_pages = len(PdfReader(io.BytesIO(pdf_bytes)).pages)
        except Exception:
            n_pages = "?"
        print(f"  ✓ PDF written: {n_pages} pages, {size_kb} KB")
        return True

    except Exception as e:
        print(f"  [error] render failed: {e}")
        import traceback; traceback.print_exc()
        return False
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# 6.  Orchestrate: process one doc_name end-to-end
# ══════════════════════════════════════════════════════════════════════════════

def process_doc(doc_name: str, out_dir: Path, no_convert: bool = False, overwrite: bool = False) -> bool:
    dest = out_dir / f"{doc_name}.pdf"
    if dest.exists() and not overwrite:
        print(f"  ✓ Already exists: {dest.name}")
        return True
    if dest.exists() and overwrite:
        print(f"  Overwriting: {dest.name}")

    print(f"\n{'─'*60}\n  {doc_name}\n{'─'*60}")
    try:
        ticker, year, quarter, form_type = parse_doc_name(doc_name)
    except ValueError as e:
        print(f"  [error] {e}")
        return False

    cik = get_cik(ticker)
    if not cik:
        print(f"  [error] CIK not found for ticker {ticker!r}")
        return False

    acc = get_accession_number(cik, year, form_type, quarter=quarter)
    if not acc:
        print(f"  [error] No accession number found for {doc_name}")
        return False
    print(f"  Accession: {acc}")

    docs = get_filing_documents(cik, acc)
    if not docs:
        print(f"  [error] Filing index is empty for {acc}")
        return False

    # Prefer direct PDF; fall back to the largest HTM
    pdf_docs = [d for d in docs if d["document"].lower().endswith(".pdf")]
    htm_docs = [d for d in docs
                if d["document"].lower().endswith((".htm", ".html"))
                and not re.search(r"(?:ex|exhibit|_ex\d)", d["document"], re.I)]

    if pdf_docs:
        # Pick primary doc PDF first, else largest
        primary = next(
            (d for d in pdf_docs if "10-k" in d["description"].lower()
             or "annual" in d["description"].lower()),
            pdf_docs[0],
        )
        print(f"  PDF: {primary['document']}")
        return download_pdf_direct(primary["url"], dest)

    if htm_docs:
        if no_convert:
            print("  [skip] HTML-only filing and --no-convert is set")
            return False
        # Largest HTM is most likely the full filing body
        htm_docs.sort(key=lambda d: len(d.get("description", "")), reverse=True)
        primary_htm = htm_docs[0]
        print(f"  HTM: {primary_htm['document']} (→ PDF via Playwright)")
        return convert_htm_to_pdf(primary_htm["url"], dest)

    print(f"  [error] No usable PDF or HTM document found")
    return False


# ══════════════════════════════════════════════════════════════════════════════
# 7.  Entry point
# ══════════════════════════════════════════════════════════════════════════════

def collect_doc_names(jsonl_path: str) -> list[str]:
    seen: set[str] = set()
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            # Support both FinanceBench-style and SECQA-style evidence fields
            for ev in row.get("evidences", row.get("evidence", [])):
                doc = ev.get("doc_name", ev.get("document", ""))
                if doc:
                    seen.add(doc)
    return sorted(seen)


def main():
    parser = argparse.ArgumentParser(
        description="Download EDGAR PDFs for a FinanceBench / SECQA-style JSONL dataset"
    )
    parser.add_argument("--jsonl",      required=True,
                        help="Path to the .jsonl dataset file")
    parser.add_argument("--out_dir",    default="pdfs",
                        help="Output directory (default: ./pdfs)")
    parser.add_argument("--only",       nargs="*",
                        help="Only process these doc_names (space-separated)")
    parser.add_argument("--failed-list", default="failed_downloads.txt",
                        help="Filename for failed doc_names inside out_dir")
    parser.add_argument("--no-convert", action="store_true",
                        help="Skip Playwright HTM→PDF conversion; report failure instead")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-download / re-render PDFs even if they already exist")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    doc_names = collect_doc_names(args.jsonl)
    if args.only:
        doc_names = args.only

    print(f"Found {len(doc_names)} unique filings → {out_dir}/")

    ok, fail = 0, 0
    failed_docs: list[str] = []

    for doc_name in doc_names:
        success = process_doc(doc_name, out_dir, no_convert=args.no_convert, overwrite=args.overwrite)
        if success:
            ok += 1
        else:
            fail += 1
            failed_docs.append(doc_name)

    print(f"\n{'='*55}")
    print(f"Done.  ✓ {ok} downloaded   ✗ {fail} failed")

    if failed_docs:
        print("\nFailed filings:")
        for d in failed_docs:
            print(f"  {d}")
        retry_path = out_dir / args.failed_list
        retry_path.write_text("\n".join(failed_docs) + "\n")
        print(f"\nFailed list saved to: {retry_path}")
    else:
        retry_path = out_dir / args.failed_list
        if retry_path.exists():
            retry_path.unlink()
        print("All filings downloaded successfully!")


if __name__ == "__main__":
    main()