"""Build the BBM462 final-project A0 portrait academic poster — V3.

V3 brings back the boxed/celled panel style of V1 while keeping V2's
strong asymmetric hierarchy, and adds a dedicated Interpretation panel
that uses the same/cross figure to fill the previous empty space.

Layout:
  [A] Banner                                    full width, navy
  [B] Problem hero                              boxed, full width
  [C] Context        [D] Approach    [E] Results    boxed, 3 columns
  [F] Interpretation                             boxed, full width, with figure
  [G] Conclusion                                 boxed, full width, 2 cols
  [H] Footer                                     compact strip

Run:
    python -m scripts.build_poster_v3
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
OUT = ROOT / "report" / "BBM462_poster_v3.pptx"

# ----------------------------------------------------------------------
# Palette
# ----------------------------------------------------------------------
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
NAVY = RGBColor(0x10, 0x2A, 0x55)
NAVY_SOFT = RGBColor(0x35, 0x4C, 0x76)
ACCENT = RGBColor(0xC0, 0x39, 0x2B)
BODY = RGBColor(0x2B, 0x2B, 0x2B)
MUTED = RGBColor(0x6E, 0x6E, 0x6E)
RULE = RGBColor(0xCF, 0xD5, 0xDE)
PANEL_BG = RGBColor(0xF6, 0xF8, 0xFB)
HERO_BG = RGBColor(0xF1, 0xF4, 0xFA)
RESULTS_BG = RGBColor(0xFD, 0xF5, 0xF3)
RESULTS_BORDER = RGBColor(0xEB, 0xC8, 0xC2)
INTERP_BG = RGBColor(0xF7, 0xF9, 0xF5)
CONCL_BG = RGBColor(0xF8, 0xF8, 0xF8)
BANNER_SUBTITLE = RGBColor(0xCF, 0xD8, 0xE6)
BANNER_MUTED = RGBColor(0xE8, 0xEC, 0xF2)

CHIP_S = RGBColor(0xEE, 0xEE, 0xEE)
CHIP_TRUST = RGBColor(0xE7, 0xEC, 0xF4)
CHIP_T = RGBColor(0xFB, 0xE5, 0xDF)
CHIP_C = RGBColor(0xE3, 0xEE, 0xE4)

FONT = "Calibri"

# A0 portrait
W = 84.1
H = 118.9

# ----------------------------------------------------------------------
# Vertical bands (carefully budgeted; no large empty zones)
# ----------------------------------------------------------------------
MX = 2.0
GAP = 1.4

BANNER_TOP = 0.0
BANNER_H = 11.5

HERO_TOP = BANNER_H + GAP             # 12.9
HERO_H = 9.5

BODY_TOP = HERO_TOP + HERO_H + GAP     # 23.8
BODY_H = 48.0

INTERP_TOP = BODY_TOP + BODY_H + GAP   # 73.2
INTERP_H = 22.0

CONCL_TOP = INTERP_TOP + INTERP_H + GAP  # 96.6
CONCL_H = 12.5

FOOTER_TOP = CONCL_TOP + CONCL_H + GAP   # 110.5
FOOTER_H = H - FOOTER_TOP - 0.6        # ~7.8

# Horizontal column widths in body row
USABLE_W = W - 2 * MX                  # 80.1
GUTTER = 1.8
COL_CTX_W = 17.0
COL_RES_W = 25.5
COL_MTH_W = USABLE_W - COL_CTX_W - COL_RES_W - 2 * GUTTER  # ~34.0
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
    tf.margin_left = Cm(0)
    tf.margin_right = Cm(0)
    tf.margin_top = Cm(0)
    tf.margin_bottom = Cm(0)
    return tb


def run(p, text, *, size=22, bold=False, italic=False,
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


def panel(slide, x, y, w, h, *, fill=PANEL_BG, border=RULE,
          radius=0.025, line_w=0.9):
    """Rounded-rectangle panel background; returns the shape."""
    rect = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Cm(x), Cm(y), Cm(w), Cm(h),
    )
    rect.adjustments[0] = radius
    rect.fill.solid()
    rect.fill.fore_color.rgb = fill
    rect.line.color.rgb = border
    rect.line.width = Pt(line_w)
    return rect


def panel_header(slide, x, y, w, text, *, size=32, color=NAVY,
                  rule=True, rule_color=ACCENT):
    """Section header inside a panel; returns y for the next content row."""
    tb = textbox(slide, x, y, w, 1.7)
    p = tb.text_frame.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    run(p, text, size=size, bold=True, color=color)
    if rule:
        add_rect(slide, x, y + 1.55, w * 0.45, 0.10,
                  fill=rule_color)
    return y + 2.2


def paragraph_block(slide, x, y, w, items, *, default_size=22,
                     line_spacing=1.2, space_after=6, max_h=60):
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
# Composite figures
# ----------------------------------------------------------------------

def horizontal_pipeline(slide, x, y, w, h):
    n = 5
    arrow_w = 1.0
    box_w = (w - (n - 1) * arrow_w) / n
    labels = [
        ("Raw events",   "u → v, rating, ts"),
        ("Snapshots",    "monthly, cumulative"),
        ("Candidates",   "+ labels (2-hop, core)"),
        ("36 features",  "5 interpretable families"),
        ("Models",       "LR  /  RF"),
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
        run(p, lab, size=22, bold=True, color=NAVY)
        p2 = tf.add_paragraph()
        p2.alignment = PP_ALIGN.CENTER
        p2.space_before = Pt(4)
        run(p2, sub, size=13, italic=True, color=MUTED)
        if i < n - 1:
            ax = lx + box_w + 0.05
            arrow = slide.shapes.add_shape(
                MSO_SHAPE.RIGHT_ARROW,
                Cm(ax), Cm(y + h / 2 - 0.45),
                Cm(arrow_w - 0.1), Cm(0.9),
            )
            arrow.fill.solid()
            arrow.fill.fore_color.rgb = NAVY
            arrow.line.fill.background()


def feature_chips(slide, x, y, w, h):
    chips = [
        ("S",  "Structural · 12",   "CN · Jaccard · AA · PA · signed degrees · reciprocity",  CHIP_S),
        ("+",  "Trust-summary · 4", "mean rating given/received · negative ratios",           CHIP_TRUST),
        ("+T", "Temporal · 12",     "recency · recent counts · decay (τ = 60 d)",             CHIP_T),
        ("+C", "Community · 8",     "Louvain · same_community · neighborhood overlap",        CHIP_C),
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
        tf.margin_top = Cm(0.22); tf.margin_bottom = Cm(0.22)
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.LEFT
        run(p, tag, size=20, bold=True, color=ACCENT)
        run(p, "  " + name, size=18, bold=True, color=NAVY)
        p2 = tf.add_paragraph()
        p2.alignment = PP_ALIGN.LEFT
        p2.space_before = Pt(3)
        run(p2, content, size=13, color=BODY)


def results_table(slide, x, y, w, h):
    rows, cols = 4, 3
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
            tf.margin_top = Cm(0.06); tf.margin_bottom = Cm(0.06)
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER
            color = (NAVY if r == 0 else
                     ACCENT if r == 2 else BODY)
            run(p, data[r][c],
                size=22,
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

    bg = s.background
    bg.fill.solid()
    bg.fill.fore_color.rgb = WHITE

    # ==================================================================
    # [A] BANNER
    # ==================================================================
    add_rect(s, 0, 0, W, BANNER_H, fill=NAVY)
    tb = textbox(s, MX, 1.4, W - 2 * MX, BANNER_H - 1.4)
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
    p3.space_before = Pt(16)
    run(p3,
        "Ömer Faruk Güler   ·   Hacettepe University   ·   "
        "BBM462 Final Project   ·   omer55glr@gmail.com",
        size=20, color=BANNER_MUTED)

    # ==================================================================
    # [B] PROBLEM HERO PANEL (boxed)
    # ==================================================================
    panel(s, MX, HERO_TOP, W - 2 * MX, HERO_H,
          fill=HERO_BG, border=NAVY, line_w=1.0)
    # Big one-line message
    tb = textbox(s, MX + 0.8, HERO_TOP + 0.6, W - 2 * MX - 1.6, 3.0)
    p = tb.text_frame.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    run(p, "Forecasting where distrust will appear, ",
        size=50, bold=True, color=NAVY)
    run(p, "one month before it happens.",
        size=50, bold=True, color=ACCENT)
    # Supporting prose
    tb = textbox(s, MX + 0.8, HERO_TOP + 4.4, W - 2 * MX - 1.6, 4.5)
    tf = tb.text_frame
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    p.space_after = Pt(6); p.line_spacing = 1.25
    run(p,
        "Negative ratings in trust marketplaces are rare, "
        "time-sensitive, and high-stakes — yet there is no principled "
        "framework for predicting, at the pair level, where new distrust "
        "will emerge next.",
        size=22, color=BODY)
    p2 = tf.add_paragraph()
    p2.alignment = PP_ALIGN.LEFT
    p2.line_spacing = 1.25
    run(p2, "We ask: ", size=22, bold=True, color=NAVY)
    run(p2,
        "at the end of month  t,  can we rank user pairs  (u, v)  by their "
        "probability of receiving a new negative directed rating in  "
        "(t, t + 1] ?",
        size=22, italic=True, color=BODY)

    # ==================================================================
    # [C] CONTEXT PANEL (boxed, left column)
    # ==================================================================
    cx = COL_CTX_X
    cw = COL_CTX_W
    panel(s, cx, BODY_TOP, cw, BODY_H, fill=PANEL_BG)
    inner_y = panel_header(s, cx + 0.6, BODY_TOP + 0.5, cw - 1.2, "Context")

    paragraph_block(
        s, cx + 0.6, inner_y, cw - 1.2,
        [
            [("Why this is hard",
              {"size": 22, "bold": True, "color": NAVY})],
            "",
            "•  Sign-dependent — any edge ≠ negative edge",
            "•  Heavily imbalanced — rare events",
            "•  Strict temporal causality",
            "•  Ranking metrics — ROC-AUC, PR-AUC",
        ],
        default_size=18, line_spacing=1.25, space_after=4,
    )

    paragraph_block(
        s, cx + 0.6, inner_y + 11.5, cw - 1.2,
        [
            [("Dataset", {"size": 22, "bold": True, "color": NAVY})],
            "",
            [("Bitcoin-OTC", {"size": 20, "bold": True}),
             ("  (SNAP)", {"size": 16, "italic": True, "color": MUTED})],
            "35,592 ratings",
            "5,881 users",
            "Nov 2010 — Jan 2016",
            "",
            [("Edge:",
              {"size": 16}),
             ("  (u → v, r ∈ [−10, 10], ts)",
              {"size": 16, "italic": True})],
            "",
            "≈ 90 % positive",
            "≈ 10 % negative",
            "reciprocity 0.79",
        ],
        default_size=18, line_spacing=1.25, space_after=3,
    )

    # Inline figure: edges_per_month
    fig_y = inner_y + 28.5
    fig_w = cw - 1.2
    add_image(s, FIG_DIR / "edges_per_month.png",
              cx + 0.6, fig_y, w=fig_w)
    tb = textbox(s, cx + 0.6, fig_y + fig_w * 0.45, fig_w, 1.0)
    p = tb.text_frame.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    run(p, "Monthly interaction volume.",
        size=13, italic=True, color=MUTED)

    # ==================================================================
    # [D] APPROACH PANEL (boxed, center, dominant)
    # ==================================================================
    mx = COL_MTH_X
    mw = COL_MTH_W
    panel(s, mx, BODY_TOP, mw, BODY_H, fill=PANEL_BG)
    inner_y = panel_header(s, mx + 0.7, BODY_TOP + 0.5, mw - 1.4,
                            "Our Approach")

    paragraph_block(
        s, mx + 0.7, inner_y, mw - 1.4,
        [
            [("We developed an interpretable, "
              "leakage-controlled prediction pipeline",
              {"size": 25, "bold": True, "color": ACCENT})],
            [("for new negative trust edges in a temporal signed network.",
              {"size": 21, "italic": True, "color": NAVY_SOFT})],
        ],
        default_size=22, line_spacing=1.2, space_after=4,
    )

    # Horizontal pipeline
    pipe_y = inner_y + 4.0
    pipe_h = 5.5
    horizontal_pipeline(s, mx + 0.7, pipe_y, mw - 1.4, pipe_h)

    # Leakage note centered under pipeline
    tb = textbox(s, mx + 0.7, pipe_y + pipe_h + 0.2, mw - 1.4, 0.9)
    p = tb.text_frame.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    run(p, "Leakage boundary:  only events with  ts ≤ end_of_month(t)",
        size=17, italic=True, color=MUTED)

    # Feature chips label + chips
    chips_label_y = pipe_y + pipe_h + 1.5
    paragraph_block(
        s, mx + 0.7, chips_label_y, mw - 1.4,
        [
            [("Feature families  ",
              {"size": 22, "bold": True, "color": NAVY}),
             ("·  36 features in 4 interpretable groups",
              {"size": 18, "italic": True, "color": MUTED})],
        ],
        default_size=22, space_after=2,
    )
    feature_chips(s, mx + 0.7, chips_label_y + 1.5, mw - 1.4, 4.8)

    # Models bar
    bar_y = chips_label_y + 1.5 + 4.8 + 1.2
    paragraph_block(
        s, mx + 0.7, bar_y, mw - 1.4,
        [
            [("Models  ", {"size": 21, "bold": True, "color": NAVY}),
             ("Logistic Regression  ·  Random Forest",
              {"size": 21}),
             ("    no tuning · seeds = 42",
              {"size": 17, "italic": True, "color": MUTED})],
        ],
        default_size=21, space_after=2,
    )

    # Evaluation header + split table
    eval_y = bar_y + 2.0
    paragraph_block(
        s, mx + 0.7, eval_y, mw - 1.4,
        [
            [("Evaluation  ",
              {"size": 21, "bold": True, "color": NAVY}),
             ("forward-time split, ranking metrics",
              {"size": 17, "italic": True, "color": MUTED})],
        ],
        default_size=21, space_after=2,
    )
    table_y = eval_y + 1.8
    rows, cols = 4, 4
    ts = s.shapes.add_table(rows, cols,
                            Cm(mx + 0.7), Cm(table_y),
                            Cm(mw - 1.4), Cm(6.0))
    t = ts.table
    inner_tw = mw - 1.4
    t.columns[0].width = Cm(inner_tw * 0.18)
    t.columns[1].width = Cm(inner_tw * 0.35)
    t.columns[2].width = Cm(inner_tw * 0.24)
    t.columns[3].width = Cm(inner_tw * 0.23)
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
            tf.margin_top = Cm(0.06); tf.margin_bottom = Cm(0.06)
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER
            run(p, split_data[r][c],
                size=19,
                bold=(r == 0),
                color=NAVY if r == 0 else BODY)

    # ==================================================================
    # [E] RESULTS PANEL (boxed, right column)
    # ==================================================================
    rx = COL_RES_X
    rw = COL_RES_W
    panel(s, rx, BODY_TOP, rw, BODY_H,
          fill=RESULTS_BG, border=RESULTS_BORDER, line_w=1.0)
    inner_y = panel_header(s, rx + 0.6, BODY_TOP + 0.5, rw - 1.2, "Results")

    # Sub-header
    tb = textbox(s, rx + 0.6, inner_y, rw - 1.2, 1.0)
    p = tb.text_frame.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    run(p, "Headline finding", size=20, bold=True, color=NAVY)

    # Hero number callout
    callout_y = inner_y + 1.4
    tb = textbox(s, rx + 0.6, callout_y, rw - 1.2, 6.5)
    tf = tb.text_frame
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    run(p, "ROC-AUC", size=20, italic=True, color=MUTED)
    p2 = tf.add_paragraph()
    p2.alignment = PP_ALIGN.CENTER
    p2.space_before = Pt(4)
    run(p2, "0.88 ", size=82, bold=True, color=NAVY)
    run(p2, "→ ", size=72, color=MUTED)
    run(p2, "0.95", size=82, bold=True, color=ACCENT)
    p3 = tf.add_paragraph()
    p3.alignment = PP_ALIGN.CENTER
    p3.space_before = Pt(4)
    run(p3,
        "structural   →   structural + temporal",
        size=18, italic=True, color=MUTED)
    p4 = tf.add_paragraph()
    p4.alignment = PP_ALIGN.CENTER
    p4.space_before = Pt(4)
    run(p4, "the largest single jump in our ablation",
        size=18, italic=True, color=NAVY_SOFT)

    # Results table
    rt_y = callout_y + 9.5
    paragraph_block(
        s, rx + 0.6, rt_y, rw - 1.2,
        [
            [("Random Forest", {"size": 20, "bold": True, "color": NAVY}),
             (",  held-out test fold",
              {"size": 18, "italic": True, "color": MUTED})],
        ],
        default_size=20, space_after=2,
    )
    results_table(s, rx + 0.6, rt_y + 1.7, rw - 1.2, 5.6)

    # Mini bullets
    bullets_y = rt_y + 7.7
    paragraph_block(
        s, rx + 0.6, bullets_y, rw - 1.2,
        [
            [("+ T", {"size": 20, "bold": True, "color": ACCENT}),
             ("    largest, most consistent improvement",
              {"size": 18})],
            [("+ C", {"size": 20, "bold": True, "color": ACCENT}),
             ("    further PR-AUC gain; ROC near saturation",
              {"size": 18})],
            [("LR shows the same ordering at lower absolute PR-AUC.",
              {"size": 15, "italic": True, "color": MUTED})],
        ],
        default_size=18, line_spacing=1.15, space_after=4,
    )

    # PR curves figure
    fig_y = bullets_y + 5.5
    fig_w = rw - 1.2
    add_image(s, FIG_DIR / "test_rf_pr.png", rx + 0.6, fig_y, w=fig_w)
    tb = textbox(s, rx + 0.6, fig_y + fig_w * 0.46, rw - 1.2, 0.9)
    p = tb.text_frame.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    run(p, "RF precision-recall curves, test fold.",
        size=14, italic=True, color=MUTED)

    # ==================================================================
    # [F] INTERPRETATION PANEL (boxed, full width, with figure)
    # ==================================================================
    panel(s, MX, INTERP_TOP, USABLE_W, INTERP_H, fill=INTERP_BG)
    inner_y = panel_header(s, MX + 0.7, INTERP_TOP + 0.4, USABLE_W - 1.4,
                            "Interpretation",
                            rule_color=RGBColor(0x4D, 0x8B, 0x5C))

    # Left: 3 takeaway blocks   |   Right: figure (same vs cross)
    fig_block_w = 26.0
    text_block_x = MX + 0.7
    text_block_w = USABLE_W - 1.4 - fig_block_w - 1.5

    # Three stacked takeaways
    items_y = inner_y + 0.3
    paragraph_block(
        s, text_block_x, items_y, text_block_w,
        [
            [("①   Temporal information is the dominant signal.",
              {"size": 22, "bold": True, "color": NAVY})],
            "",
            [("Recency and decayed activity drive feature importance in "
              "both models.  A simple recency-only ranker reaches  ",
              {"size": 18}),
             ("ROC-AUC ≈ 0.89",
              {"size": 18, "bold": True, "color": ACCENT}),
             (", confirming that the temporal signal is robust and "
              "supported by simple recency information.",
              {"size": 18})],
        ],
        default_size=18, line_spacing=1.22, space_after=4,
    )

    paragraph_block(
        s, text_block_x, items_y + 5.0, text_block_w,
        [
            [("②   Community-aware features add a nuanced refinement.",
              {"size": 22, "bold": True, "color": NAVY})],
            "",
            "Community features mostly improve PR-AUC under Random "
            "Forest;  ROC-AUC is already saturated.  The improvement is "
            "real but modest — community is a refinement, not the headline.",
        ],
        default_size=18, line_spacing=1.22, space_after=4,
    )

    paragraph_block(
        s, text_block_x, items_y + 10.5, text_block_w,
        [
            [("③   Pair-level overlap matters more than community label.",
              {"size": 22, "bold": True, "color": NAVY})],
            "",
            [("The naive cross-community-distrust intuition is ",
              {"size": 18}),
             ("not supported",
              {"size": 18, "bold": True, "color": ACCENT}),
             (".  Same-community pairs have a slightly higher empirical "
              "positive rate.  At the pair level, ",
              {"size": 18}),
             ("neighborhood overlap",
              {"size": 18, "bold": True, "color": ACCENT}),
             (" is the strongest community signal — distrust appears on "
              "weakly-embedded ties, often within the same broader trust "
              "community.",
              {"size": 18})],
        ],
        default_size=18, line_spacing=1.22, space_after=4,
    )

    # Right: same/cross figure
    fig_x = text_block_x + text_block_w + 1.5
    fig_y = inner_y + 0.6
    fig_render_w = fig_block_w
    add_image(s, FIG_DIR / "same_vs_cross_positive_rate.png",
              fig_x, fig_y, w=fig_render_w)
    tb = textbox(s, fig_x, fig_y + fig_render_w * 0.66, fig_render_w, 1.5)
    p = tb.text_frame.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    run(p,
        "Training positive rate by community membership.  "
        "Same-community pairs are not the safer ones.",
        size=14, italic=True, color=MUTED)

    # ==================================================================
    # [G] CONCLUSION PANEL (boxed, full width, 2-column)
    # ==================================================================
    panel(s, MX, CONCL_TOP, USABLE_W, CONCL_H, fill=CONCL_BG)
    inner_y = panel_header(s, MX + 0.7, CONCL_TOP + 0.4, USABLE_W - 1.4,
                            "Conclusion & Future Work")

    col_w = (USABLE_W - 1.4 - 2.0) / 2
    cx1 = MX + 0.7
    cx2 = cx1 + col_w + 2.0

    paragraph_block(
        s, cx1, inner_y + 0.2, col_w,
        [
            [("We developed",
              {"size": 22, "bold": True, "color": NAVY})],
            "",
            "•  A precise temporal signed-network prediction task",
            "•  An interpretable, leakage-controlled pipeline",
            [("•  ", {"size": 19}),
             ("Temporal features → strongest, most consistent improvement",
              {"size": 19, "bold": True, "color": ACCENT})],
            "•  Community features → smaller, nuanced refinement",
        ],
        default_size=19, line_spacing=1.2, space_after=3,
    )

    paragraph_block(
        s, cx2, inner_y + 0.2, col_w,
        [
            [("Future work",
              {"size": 22, "bold": True, "color": NAVY})],
            "",
            "•  External replication on Bitcoin-Alpha",
            "•  Stronger learning baselines and graph embeddings",
            "•  Conditional analyses of community signal",
            "•  Rolling-origin validation extended across snapshots",
        ],
        default_size=19, line_spacing=1.2, space_after=3,
    )

    # ==================================================================
    # [H] FOOTER STRIP
    # ==================================================================
    add_rect(s, 0, FOOTER_TOP, W, FOOTER_H, fill=HERO_BG)
    add_rect(s, 0, FOOTER_TOP, W, 0.10, fill=NAVY)

    paragraph_block(
        s, MX, FOOTER_TOP + 0.5, USABLE_W,
        [
            [("ADDITIONAL VALIDATION    ",
              {"size": 16, "bold": True, "color": NAVY}),
             ("Rolling-origin over 25 monthly test folds "
              "(425 positives total) confirms the temporal lift  ",
              {"size": 15}),
             ("(Wilcoxon p ≤ 1.4 × 10⁻⁴, Holm-corrected)",
              {"size": 15, "bold": True, "color": ACCENT}),
             (".  Bootstrap CIs on the locked fold agree with the "
              "across-fold results.",
              {"size": 15})],
            "",
            [("REFERENCES    ",
              {"size": 16, "bold": True, "color": NAVY}),
             ("Kumar et al. (ICDM 2016)   ·   "
              "Blondel et al. (J Stat Mech 2008)   ·   "
              "Bertazzi et al. (SocInfo 2018)   ·   "
              "Leskovec & Krevl, SNAP (2014).",
              {"size": 15})],
            [("REPRODUCIBILITY    ",
              {"size": 16, "bold": True, "color": NAVY}),
             ("Code in  src/  and  scripts/  ·  seeds fixed at 42  ·  "
              "intermediate artifacts persisted under  data/  and  results/ .",
              {"size": 15})],
        ],
        default_size=15, line_spacing=1.2, space_after=3,
    )

    # ==================================================================
    # SAVE
    # ==================================================================
    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    print(f"Saved {OUT}")
    print(f"Poster v3: A0 portrait  {W} cm × {H} cm  (1 slide)")


if __name__ == "__main__":
    build()
