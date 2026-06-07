"""Build the BBM462 final-project A0 portrait academic poster — V2 redesign.

Stronger visual hierarchy, fewer panels, real conference-poster feel:
  - Hero problem strip across the top body
  - Asymmetric 3-column body: Context (sidebar) | Method (widest, centerpiece) | Results (large)
  - Horizontal pipeline diagram with feature-family chips
  - Hero result number callout in the Results column
  - Findings + Conclusion merged into one full-width strip
  - Compact footer for validation / references / reproducibility

Generates  report/BBM462_poster_v2.pptx  as a single A0-portrait slide.

Run:
    python -m scripts.build_poster_v2
"""
from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Cm, Pt

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "results" / "figures"
OUT = ROOT / "report" / "BBM462_poster_v2.pptx"

# ----------------------------------------------------------------------
# Palette — one accent color, restrained chrome
# ----------------------------------------------------------------------
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
NAVY = RGBColor(0x10, 0x2A, 0x55)
ACCENT = RGBColor(0xC0, 0x39, 0x2B)
BODY = RGBColor(0x2B, 0x2B, 0x2B)
MUTED = RGBColor(0x6E, 0x6E, 0x6E)
RULE = RGBColor(0xCF, 0xD5, 0xDE)
HERO_BG = RGBColor(0xF6, 0xF8, 0xFB)
RESULTS_BG = RGBColor(0xFD, 0xF5, 0xF3)
CHIP_S = RGBColor(0xEE, 0xEE, 0xEE)
CHIP_TRUST = RGBColor(0xE7, 0xEC, 0xF4)
CHIP_T = RGBColor(0xFB, 0xE5, 0xDF)
CHIP_C = RGBColor(0xE3, 0xEE, 0xE4)
BANNER_SUBTITLE = RGBColor(0xCF, 0xD8, 0xE6)
BANNER_MUTED = RGBColor(0xE8, 0xEC, 0xF2)
NAVY_SOFT = RGBColor(0x35, 0x4C, 0x76)

FONT = "Calibri"

# A0 portrait
W = 84.1
H = 118.9

# Margins / band heights
MX = 2.0
BANNER_TOP = 0.0
BANNER_H = 11.5
HERO_TOP = BANNER_H + 1.5
HERO_H = 9.5
BODY_TOP = HERO_TOP + HERO_H + 2.0
FINDINGS_H = 18.0
FOOTER_H = 8.0
FOOTER_TOP = H - 1.5 - FOOTER_H
FINDINGS_TOP = FOOTER_TOP - 2.0 - FINDINGS_H
BODY_BOTTOM = FINDINGS_TOP - 2.0
BODY_H = BODY_BOTTOM - BODY_TOP

# Three body columns (sum to USABLE_W; Method is widest)
USABLE_W = W - 2 * MX
GUTTER = 1.8
COL_CTX_W = 18.0
COL_RES_W = 26.0
COL_MTH_W = USABLE_W - COL_CTX_W - COL_RES_W - 2 * GUTTER  # ~34.5 cm
COL_CTX_X = MX
COL_MTH_X = COL_CTX_X + COL_CTX_W + GUTTER
COL_RES_X = COL_MTH_X + COL_MTH_W + GUTTER


# ----------------------------------------------------------------------
# Drawing helpers
# ----------------------------------------------------------------------

def textbox(slide, x, y, w, h):
    tb = slide.shapes.add_textbox(Cm(x), Cm(y), Cm(w), Cm(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = Cm(0.0)
    tf.margin_right = Cm(0.0)
    tf.margin_top = Cm(0.0)
    tf.margin_bottom = Cm(0.0)
    return tb


def run(p, text, *, size=24, bold=False, italic=False,
        color=BODY, font=FONT):
    r = p.add_run()
    r.text = text
    r.font.name = font
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.italic = italic
    r.font.color.rgb = color
    return r


def add_rect(slide, x, y, w, h, *, fill=None, line=None, line_w=0.75):
    rect = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Cm(x), Cm(y), Cm(w), Cm(h),
    )
    if fill is None:
        rect.fill.background()
    else:
        rect.fill.solid()
        rect.fill.fore_color.rgb = fill
    if line is None:
        rect.line.fill.background()
    else:
        rect.line.color.rgb = line
        rect.line.width = Pt(line_w)
    return rect


def add_hline(slide, x, y, w, *, color=RULE, height=0.04):
    return add_rect(slide, x, y, w, height, fill=color)


def section_header(slide, x, y, w, text, *, size=32, color=NAVY,
                    rule=True, rule_color=ACCENT):
    tb = textbox(slide, x, y, w, 1.7)
    p = tb.text_frame.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    run(p, text, size=size, bold=True, color=color)
    if rule:
        add_hline(slide, x, y + 1.55, w * 0.42, color=rule_color, height=0.10)
    return y + 2.1


def paragraph_block(slide, x, y, w, items, *, default_size=22,
                    line_spacing=1.18, space_after=6, max_h=60):
    tb = textbox(slide, x, y, w, max_h)
    tf = tb.text_frame
    first = True
    for item in items:
        if first:
            p = tf.paragraphs[0]
            first = False
        else:
            p = tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        p.line_spacing = line_spacing
        p.space_after = Pt(space_after)
        if item == "":
            continue
        if isinstance(item, str):
            run(p, item, size=default_size)
        else:
            for chunk in item:
                if isinstance(chunk, str):
                    text, opts = chunk, {}
                else:
                    text, opts = chunk
                run(p, text,
                    size=opts.get("size", default_size),
                    bold=opts.get("bold", False),
                    italic=opts.get("italic", False),
                    color=opts.get("color", BODY))
    return tb


def add_image(slide, path, x, y, *, w=None, h=None):
    if w is not None and h is None:
        slide.shapes.add_picture(str(path), Cm(x), Cm(y), width=Cm(w))
    elif h is not None and w is None:
        slide.shapes.add_picture(str(path), Cm(x), Cm(y), height=Cm(h))
    else:
        slide.shapes.add_picture(str(path), Cm(x), Cm(y),
                                  width=Cm(w), height=Cm(h))


# ----------------------------------------------------------------------
# Special composites
# ----------------------------------------------------------------------

def horizontal_pipeline(slide, x, y, w, h):
    """5 stages × right-arrows. Used as the visual centerpiece."""
    n = 5
    arrow_w = 1.0
    box_w = (w - (n - 1) * arrow_w) / n
    labels = [
        ("Raw events",      "u → v, rating, ts"),
        ("Snapshots",       "monthly, cumulative"),
        ("Candidates",      "+ labels (2-hop, core)"),
        ("36 features",     "5 interpretable families"),
        ("Models",          "LR  /  RF"),
    ]
    for i, (lab, sub) in enumerate(labels):
        lx = x + i * (box_w + arrow_w)
        rect = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE,
            Cm(lx), Cm(y), Cm(box_w), Cm(h),
        )
        rect.adjustments[0] = 0.10
        rect.fill.solid()
        rect.fill.fore_color.rgb = HERO_BG
        rect.line.color.rgb = NAVY
        rect.line.width = Pt(1.2)
        tf = rect.text_frame
        tf.margin_left = Cm(0.2); tf.margin_right = Cm(0.2)
        tf.margin_top = Cm(0.25); tf.margin_bottom = Cm(0.25)
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        run(p, lab, size=24, bold=True, color=NAVY)
        p2 = tf.add_paragraph()
        p2.alignment = PP_ALIGN.CENTER
        p2.space_before = Pt(4)
        run(p2, sub, size=14, italic=True, color=MUTED)
        if i < n - 1:
            ax = lx + box_w + 0.05
            arrow = slide.shapes.add_shape(
                MSO_SHAPE.RIGHT_ARROW,
                Cm(ax), Cm(y + h / 2 - 0.5),
                Cm(arrow_w - 0.1), Cm(1.0),
            )
            arrow.fill.solid()
            arrow.fill.fore_color.rgb = NAVY
            arrow.line.fill.background()


def feature_chips(slide, x, y, w, h):
    """4 colored chips horizontally - feature families."""
    chips = [
        ("S",  "Structural · 12",      "CN · Jaccard · AA · PA · signed degrees · reciprocity",  CHIP_S),
        ("+",  "Trust-summary · 4",    "mean rating given/received · negative ratios",           CHIP_TRUST),
        ("+T", "Temporal · 12",        "recency · recent counts · decayed aggregates (τ = 60 d)", CHIP_T),
        ("+C", "Community · 8",        "Louvain · same_community · neighborhood overlap · boundary", CHIP_C),
    ]
    gap = 0.5
    n = len(chips)
    cw = (w - (n - 1) * gap) / n
    for i, (tag, name, content, color) in enumerate(chips):
        lx = x + i * (cw + gap)
        rect = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE,
            Cm(lx), Cm(y), Cm(cw), Cm(h),
        )
        rect.adjustments[0] = 0.08
        rect.fill.solid()
        rect.fill.fore_color.rgb = color
        rect.line.color.rgb = RULE
        rect.line.width = Pt(0.5)
        tf = rect.text_frame
        tf.margin_left = Cm(0.3); tf.margin_right = Cm(0.3)
        tf.margin_top = Cm(0.25); tf.margin_bottom = Cm(0.25)
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.LEFT
        run(p, tag, size=22, bold=True, color=ACCENT)
        run(p, "  " + name, size=20, bold=True, color=NAVY)
        p2 = tf.add_paragraph()
        p2.alignment = PP_ALIGN.LEFT
        p2.space_before = Pt(4)
        run(p2, content, size=14, color=BODY)


def results_table(slide, x, y, w, h):
    """Compact RF-only results table with the S+T row highlighted."""
    rows = 4
    cols = 3
    ts = slide.shapes.add_table(rows, cols, Cm(x), Cm(y), Cm(w), Cm(h))
    t = ts.table
    t.columns[0].width = Cm(w * 0.42)
    t.columns[1].width = Cm(w * 0.29)
    t.columns[2].width = Cm(w * 0.29)
    data = [
        ["Feature set", "ROC-AUC", "PR-AUC"],
        ["S",            "0.88",    "0.02"],
        ["S + T",        "0.95",    "0.04"],
        ["S + T + C",    "0.94",    "0.07"],
    ]
    for r in range(rows):
        for c in range(cols):
            cell = t.cell(r, c)
            if r == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = HERO_BG
            elif r == 2:
                cell.fill.solid()
                cell.fill.fore_color.rgb = RGBColor(0xFD, 0xE7, 0xE3)
            else:
                cell.fill.solid()
                cell.fill.fore_color.rgb = WHITE
            tf = cell.text_frame
            tf.margin_left = Cm(0.25); tf.margin_right = Cm(0.25)
            tf.margin_top = Cm(0.1); tf.margin_bottom = Cm(0.1)
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER
            color = (NAVY if r == 0 else
                     ACCENT if r == 2 else BODY)
            run(p, data[r][c],
                size=24,
                bold=(r == 0 or r == 2),
                color=color)


# ----------------------------------------------------------------------
# Build
# ----------------------------------------------------------------------

def build() -> None:
    prs = Presentation()
    prs.slide_width = Cm(W)
    prs.slide_height = Cm(H)
    blank = prs.slide_layouts[6]
    s = prs.slides.add_slide(blank)

    # white background
    bg = s.background
    bg.fill.solid()
    bg.fill.fore_color.rgb = WHITE

    # ==================================================================
    # [A] TITLE BANNER  (deep navy band, full width)
    # ==================================================================
    add_rect(s, 0, BANNER_TOP, W, BANNER_H, fill=NAVY)
    tb = textbox(s, MX, BANNER_TOP + 1.4, W - 2 * MX, BANNER_H - 1.4)
    tf = tb.text_frame
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    run(p, "Early Warning of Distrust in Bitcoin-OTC",
        size=78, bold=True, color=WHITE)
    p2 = tf.add_paragraph()
    p2.alignment = PP_ALIGN.LEFT
    p2.space_before = Pt(10)
    run(p2,
        "An Interpretable Temporal-Network Pipeline for "
        "Predicting New Negative Trust Edges",
        size=30, italic=True, color=BANNER_SUBTITLE)
    p3 = tf.add_paragraph()
    p3.alignment = PP_ALIGN.LEFT
    p3.space_before = Pt(18)
    run(p3,
        "Ömer Faruk Güler   ·   Hacettepe University   ·   "
        "BBM462 Final Project   ·   omer55glr@gmail.com",
        size=20, color=BANNER_MUTED)

    # ==================================================================
    # [B] PROBLEM HERO STRIP (light bg, large prose — visual anchor #1)
    # ==================================================================
    add_rect(s, 0, HERO_TOP, W, HERO_H, fill=HERO_BG)
    # Big one-line message
    tb = textbox(s, MX, HERO_TOP + 0.8, W - 2 * MX, 3.2)
    p = tb.text_frame.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    run(p, "Forecasting where distrust will appear, ",
        size=54, bold=True, color=NAVY)
    run(p, "one month before it happens.",
        size=54, bold=True, color=ACCENT)
    # Supporting prose, two short lines
    tb = textbox(s, MX, HERO_TOP + 4.6, W - 2 * MX, 4.5)
    tf = tb.text_frame
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    p.space_after = Pt(6)
    p.line_spacing = 1.25
    run(p,
        "Negative ratings in trust marketplaces are rare, "
        "time-sensitive, and high-stakes — yet there is no principled "
        "framework for predicting, at the pair level, where new distrust "
        "will emerge next.",
        size=24, color=BODY)
    p2 = tf.add_paragraph()
    p2.alignment = PP_ALIGN.LEFT
    p2.line_spacing = 1.25
    run(p2, "We ask: ", size=24, bold=True, color=NAVY)
    run(p2,
        "at the end of month  t,  can we rank user pairs  (u, v)  by "
        "their probability of receiving a new negative directed rating "
        "in the window  (t, t + 1] ?",
        size=24, italic=True, color=BODY)

    # ==================================================================
    # [C] CONTEXT  (left sidebar)
    # ==================================================================
    cx = COL_CTX_X
    cw = COL_CTX_W
    y = BODY_TOP

    inner_y = section_header(s, cx, y, cw, "Context")

    paragraph_block(
        s, cx, inner_y, cw,
        [
            [("Why this is hard",
              {"size": 23, "bold": True, "color": NAVY})],
            "",
            "•  Sign-dependent — any edge ≠ negative edge",
            "•  Heavily imbalanced — new monthly negatives are rare",
            "•  Strict temporal causality",
            "•  Ranking metrics: ROC-AUC, PR-AUC",
        ],
        default_size=20, line_spacing=1.25, space_after=4,
    )

    paragraph_block(
        s, cx, inner_y + 11.5, cw,
        [
            [("Dataset", {"size": 23, "bold": True, "color": NAVY})],
            "",
            [("Bitcoin-OTC", {"size": 22, "bold": True}),
             ("  (SNAP)", {"size": 18, "italic": True, "color": MUTED})],
            "35,592 ratings · 5,881 users",
            "Nov 2010 — Jan 2016",
            "",
            [("Edge:  ", {"size": 18}),
             ("(u → v, rating ∈ [−10, +10], ts)",
              {"size": 18, "italic": True})],
            "",
            "≈ 90 % positive · ≈ 10 % negative",
            "reciprocity 0.79",
        ],
        default_size=20, line_spacing=1.25, space_after=4,
    )

    # Small inline figure under the dataset block
    fig_y = inner_y + 27.0
    add_image(s, FIG_DIR / "edges_per_month.png", cx, fig_y, w=cw)
    cap_y = fig_y + cw * 0.45
    tb = textbox(s, cx, cap_y, cw, 1.4)
    p = tb.text_frame.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    run(p, "Monthly interaction volume (2010 – 2016).",
        size=14, italic=True, color=MUTED)

    # ==================================================================
    # [D] OUR APPROACH  (center, visual centerpiece)
    # ==================================================================
    mx = COL_MTH_X
    mw = COL_MTH_W
    y = BODY_TOP

    inner_y = section_header(s, mx, y, mw, "Our Approach")

    # Sub-header
    paragraph_block(
        s, mx, inner_y, mw,
        [
            [("We developed an interpretable, "
              "leakage-controlled prediction pipeline",
              {"size": 26, "bold": True, "color": ACCENT})],
            [("for new negative trust edges in a temporal signed network.",
              {"size": 22, "italic": True, "color": NAVY_SOFT})],
        ],
        default_size=24, line_spacing=1.2, space_after=6,
    )

    # Horizontal pipeline
    pipe_y = inner_y + 4.5
    pipe_h = 5.8
    horizontal_pipeline(s, mx, pipe_y, mw, pipe_h)

    # Leakage note centered under pipeline
    tb = textbox(s, mx, pipe_y + pipe_h + 0.2, mw, 1.0)
    p = tb.text_frame.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    run(p, "Leakage boundary:  only events with  ts ≤ end_of_month(t)",
        size=18, italic=True, color=MUTED)

    # Feature chips
    chips_y = pipe_y + pipe_h + 1.8
    chips_h = 5.0
    paragraph_block(
        s, mx, chips_y - 1.3, mw,
        [
            [("Feature families  ",
              {"size": 24, "bold": True, "color": NAVY}),
             ("·  36 features in 4 interpretable groups",
              {"size": 20, "italic": True, "color": MUTED})],
        ],
        default_size=24, space_after=2,
    )
    feature_chips(s, mx, chips_y, mw, chips_h)

    # Models bar
    bar_y = chips_y + chips_h + 1.0
    paragraph_block(
        s, mx, bar_y, mw,
        [
            [("Models  ", {"size": 22, "bold": True, "color": NAVY}),
             ("Logistic Regression  ·  Random Forest",
              {"size": 22}),
             ("    no hyperparameter tuning · seeds = 42",
              {"size": 18, "italic": True, "color": MUTED})],
        ],
        default_size=22, space_after=2,
    )

    # Evaluation summary table — compact, sits under Models in the Method column
    split_y = bar_y + 2.0
    paragraph_block(
        s, mx, split_y, mw,
        [
            [("Evaluation  ",
              {"size": 22, "bold": True, "color": NAVY}),
             ("forward-time split, ranking metrics",
              {"size": 18, "italic": True, "color": MUTED})],
        ],
        default_size=22, space_after=2,
    )
    # Split table
    rows, cols = 4, 4
    table_y = split_y + 2.0
    ts = s.shapes.add_table(rows, cols,
                            Cm(mx), Cm(table_y), Cm(mw), Cm(6.5))
    t = ts.table
    t.columns[0].width = Cm(mw * 0.18)
    t.columns[1].width = Cm(mw * 0.35)
    t.columns[2].width = Cm(mw * 0.24)
    t.columns[3].width = Cm(mw * 0.23)
    split_data = [
        ["Split", "Months", "Pairs", "Positives"],
        ["TRAIN", "t = 13 – 45", "5.0 M", "545"],
        ["VAL",   "t = 46, 47",  "184 k", "16"],
        ["TEST",  "t = 48, 49",  "149 k", "34"],
    ]
    for r in range(rows):
        for c in range(cols):
            cell = t.cell(r, c)
            if r == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = HERO_BG
            else:
                cell.fill.solid()
                cell.fill.fore_color.rgb = WHITE
            tf = cell.text_frame
            tf.margin_left = Cm(0.2); tf.margin_right = Cm(0.2)
            tf.margin_top = Cm(0.08); tf.margin_bottom = Cm(0.08)
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER
            run(p, split_data[r][c],
                size=20,
                bold=(r == 0),
                color=NAVY if r == 0 else BODY)

    # ==================================================================
    # [E] RESULTS (right column)
    # ==================================================================
    rx = COL_RES_X
    rw = COL_RES_W
    y = BODY_TOP

    # subtle results-background panel
    add_rect(s, rx - 0.5, y, rw + 1.0, BODY_H, fill=RESULTS_BG)

    inner_y = section_header(s, rx, y, rw, "Results")

    # HERO CALLOUT
    hero_callout_y = inner_y + 0.2
    tb = textbox(s, rx, hero_callout_y, rw, 1.8)
    p = tb.text_frame.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    run(p, "Headline finding", size=22, bold=True, color=NAVY)
    # Big number block
    tb = textbox(s, rx, hero_callout_y + 2.0, rw, 6.5)
    tf = tb.text_frame
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    run(p, "ROC-AUC", size=22, italic=True, color=MUTED)
    p2 = tf.add_paragraph()
    p2.alignment = PP_ALIGN.CENTER
    p2.space_before = Pt(4)
    run(p2, "0.88 ", size=92, bold=True, color=NAVY)
    run(p2, "→ ", size=80, color=MUTED)
    run(p2, "0.95", size=92, bold=True, color=ACCENT)
    p3 = tf.add_paragraph()
    p3.alignment = PP_ALIGN.CENTER
    p3.space_before = Pt(6)
    run(p3,
        "structural    →    structural + temporal",
        size=20, italic=True, color=MUTED)
    p4 = tf.add_paragraph()
    p4.alignment = PP_ALIGN.CENTER
    p4.space_before = Pt(6)
    run(p4, "the largest single jump in our ablation",
        size=20, italic=True, color=NAVY_SOFT)

    # Results table
    rt_y = hero_callout_y + 11.0
    paragraph_block(
        s, rx, rt_y, rw,
        [
            [("Random Forest, ", {"size": 22, "bold": True, "color": NAVY}),
             ("held-out test fold",
              {"size": 20, "italic": True, "color": MUTED})],
        ],
        default_size=22, space_after=4,
    )
    results_table(s, rx, rt_y + 2.0, rw, 6.5)

    # Bullets
    bullets_y = rt_y + 9.0
    paragraph_block(
        s, rx, bullets_y, rw,
        [
            [("+ T",
              {"size": 22, "bold": True, "color": ACCENT}),
             ("    largest, most consistent improvement",
              {"size": 20})],
            [("+ C",
              {"size": 22, "bold": True, "color": ACCENT}),
             ("    further PR-AUC gain; ROC near saturation",
              {"size": 20})],
            [("LR shows the same ordering at lower absolute PR-AUC.",
              {"size": 17, "italic": True, "color": MUTED})],
        ],
        default_size=20, line_spacing=1.2, space_after=6,
    )

    # PR curves figure
    fig_y = bullets_y + 6.5
    fig_w = rw
    add_image(s, FIG_DIR / "test_rf_pr.png", rx, fig_y, w=fig_w)
    tb = textbox(s, rx, fig_y + fig_w * 0.46, rw, 1.0)
    p = tb.text_frame.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    run(p,
        "Random Forest precision-recall curves on the test fold.",
        size=16, italic=True, color=MUTED)

    # ==================================================================
    # [F] FINDINGS + CONCLUSION (full-width strip)
    # ==================================================================
    fy = FINDINGS_TOP
    inner_y = section_header(s, MX, fy, USABLE_W, "Findings & Conclusion")

    # 3 sub-columns
    sub_w = (USABLE_W - 2 * GUTTER) / 3
    sub_x = [MX,
             MX + sub_w + GUTTER,
             MX + 2 * (sub_w + GUTTER)]
    sub_y = inner_y + 0.3

    # Col 1: Temporal dominates
    paragraph_block(
        s, sub_x[0], sub_y, sub_w,
        [
            [("Temporal information is the dominant signal.",
              {"size": 22, "bold": True, "color": NAVY})],
            "",
            "Recency and decayed activity drive feature importance "
            "in both models.",
            "",
            [("A simple recency-only ranker reaches  ",
              {"size": 19}),
             ("ROC-AUC ≈ 0.89",
              {"size": 19, "bold": True, "color": ACCENT}),
             (", confirming that the temporal signal is robust and "
              "supported by simple recency information.",
              {"size": 19})],
        ],
        default_size=19, line_spacing=1.25, space_after=4,
    )

    # Col 2: Community is a nuanced refinement
    paragraph_block(
        s, sub_x[1], sub_y, sub_w,
        [
            [("Community-aware features add a nuanced refinement.",
              {"size": 22, "bold": True, "color": NAVY})],
            "",
            "Community features mostly improve PR-AUC under Random Forest; "
            "ROC-AUC is already saturated.",
            "",
            [("The naive cross-community-distrust intuition is not "
              "supported by the data.  At the pair level, ",
              {"size": 19}),
             ("neighborhood overlap",
              {"size": 19, "bold": True, "color": ACCENT}),
             (" is the strongest community signal.",
              {"size": 19})],
        ],
        default_size=19, line_spacing=1.25, space_after=4,
    )

    # Col 3: Conclusion + future work
    paragraph_block(
        s, sub_x[2], sub_y, sub_w,
        [
            [("Conclusion · Future work",
              {"size": 22, "bold": True, "color": NAVY})],
            "",
            [("We developed", {"size": 19, "bold": True})],
            "•  a precise temporal signed-network task",
            "•  an interpretable, leakage-controlled pipeline",
            "•  a reproducible feature-ablation evaluation",
            "",
            [("Future work", {"size": 19, "bold": True})],
            "•  external replication on Bitcoin-Alpha",
            "•  stronger learning baselines",
            "•  conditional community-signal analyses",
        ],
        default_size=18, line_spacing=1.22, space_after=2,
    )

    # ==================================================================
    # [G] FOOTER STRIP (compact)
    # ==================================================================
    foot_y = FOOTER_TOP
    add_rect(s, 0, foot_y, W, FOOTER_H, fill=HERO_BG)
    add_hline(s, 0, foot_y, W, color=NAVY, height=0.08)

    paragraph_block(
        s, MX, foot_y + 0.6, USABLE_W,
        [
            [("ADDITIONAL VALIDATION    ",
              {"size": 18, "bold": True, "color": NAVY}),
             ("Rolling-origin over 25 monthly test folds (425 positives) "
              "confirms the temporal lift  ",
              {"size": 17}),
             ("(Wilcoxon p ≤ 1.4 × 10⁻⁴, Holm-corrected)",
              {"size": 17, "bold": True, "color": ACCENT}),
             (".  Bootstrap CIs on the locked fold agree with "
              "the across-fold results.",
              {"size": 17})],
            "",
            [("REFERENCES    ",
              {"size": 18, "bold": True, "color": NAVY}),
             ("Kumar et al. (ICDM 2016)   ·   "
              "Blondel et al. (J Stat Mech 2008)   ·   "
              "Bertazzi et al. (SocInfo 2018)   ·   "
              "Leskovec & Krevl, SNAP (2014).",
              {"size": 17})],
            [("REPRODUCIBILITY    ",
              {"size": 18, "bold": True, "color": NAVY}),
             ("Code organized in  src/  and  scripts/  ·  all seeds fixed "
              "at 42  ·  intermediate artifacts persisted under  data/  "
              "and  results/ .",
              {"size": 17})],
        ],
        default_size=17, line_spacing=1.2, space_after=4,
    )

    # ==================================================================
    # SAVE
    # ==================================================================
    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    print(f"Saved {OUT}")
    print(f"Poster v2: A0 portrait  {W} cm × {H} cm  (1 slide)")


if __name__ == "__main__":
    build()
