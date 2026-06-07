"""Build the BBM462 final-project A0 portrait academic poster.

Generates report/BBM462_poster.pptx as a single A0 portrait slide
(84.1 x 118.9 cm) with the full poster layout: title banner, three
columns of panels, and a bottom additional-validation strip.

Run:
    python -m scripts.build_poster
"""
from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Cm, Emu, Pt

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "results" / "figures"
OUT = ROOT / "report" / "BBM462_poster.pptx"

# ----------------------------------------------------------------------
# Color palette
# ----------------------------------------------------------------------
COLOR_BG = RGBColor(0xFF, 0xFF, 0xFF)
COLOR_TITLE = RGBColor(0x10, 0x2A, 0x55)
COLOR_ACCENT = RGBColor(0xC0, 0x39, 0x2B)
COLOR_ACCENT2 = RGBColor(0x3B, 0x7D, 0xD8)
COLOR_BODY = RGBColor(0x2B, 0x2B, 0x2B)
COLOR_MUTED = RGBColor(0x6E, 0x6E, 0x6E)
COLOR_RULE = RGBColor(0xD0, 0xD0, 0xD0)
COLOR_PANEL_BG = RGBColor(0xF7, 0xF8, 0xFB)
COLOR_HEADER_BG = RGBColor(0xF1, 0xF3, 0xF8)
COLOR_BOX_S = RGBColor(0xF6, 0xF6, 0xF6)
COLOR_BOX_TRUST = RGBColor(0xF1, 0xF3, 0xF8)
COLOR_BOX_T = RGBColor(0xFD, 0xF1, 0xEE)
COLOR_BOX_C = RGBColor(0xEF, 0xF6, 0xEF)

FONT = "Calibri"

# Poster dimensions: A0 portrait
W_CM = 84.1
H_CM = 118.9

# Layout constants
MARGIN_X = 2.0
MARGIN_TOP = 1.5
TITLE_H = 11.0
GUTTER = 1.5
N_COLS = 3

USABLE_W = W_CM - 2 * MARGIN_X
COL_W = (USABLE_W - (N_COLS - 1) * GUTTER) / N_COLS    # ~25.7 cm

# Vertical bands
TITLE_TOP = MARGIN_TOP
CONTENT_TOP = TITLE_TOP + TITLE_H + 1.5                 # ~14.0
BOTTOM_STRIP_H = 8.0
BOTTOM_STRIP_TOP = H_CM - 1.5 - BOTTOM_STRIP_H          # ~109.4
CONTENT_BOTTOM = BOTTOM_STRIP_TOP - 1.5                 # ~107.9
CONTENT_H = CONTENT_BOTTOM - CONTENT_TOP                # ~93.9 cm

# Column x positions
def col_x(i: int) -> float:
    return MARGIN_X + i * (COL_W + GUTTER)


# ----------------------------------------------------------------------
# Drawing helpers
# ----------------------------------------------------------------------

def add_textbox(slide, left_cm, top_cm, width_cm, height_cm):
    tb = slide.shapes.add_textbox(Cm(left_cm), Cm(top_cm),
                                  Cm(width_cm), Cm(height_cm))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = Cm(0.2)
    tf.margin_right = Cm(0.2)
    tf.margin_top = Cm(0.1)
    tf.margin_bottom = Cm(0.1)
    return tb


def set_run(run, text, *, size=24, bold=False, italic=False,
            color=COLOR_BODY, font=FONT):
    run.text = text
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color


def add_panel_bg(slide, left_cm, top_cm, width_cm, height_cm,
                 *, fill=COLOR_PANEL_BG, border=COLOR_RULE):
    rect = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Cm(left_cm), Cm(top_cm), Cm(width_cm), Cm(height_cm),
    )
    # subtle corner radius
    rect.adjustments[0] = 0.04
    rect.fill.solid()
    rect.fill.fore_color.rgb = fill
    rect.line.color.rgb = border
    rect.line.width = Pt(0.75)
    return rect


def add_section_header(slide, left_cm, top_cm, width_cm, text,
                       *, color=COLOR_TITLE):
    """Header with thin underline; returns top of next content area."""
    tb = add_textbox(slide, left_cm + 0.3, top_cm + 0.1,
                     width_cm - 0.6, 1.7)
    p = tb.text_frame.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    set_run(p.add_run(), text, size=34, bold=True, color=color)
    # underline
    rule = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        Cm(left_cm + 0.3), Cm(top_cm + 1.95),
        Cm(width_cm - 0.6), Cm(0.05),
    )
    rule.fill.solid()
    rule.fill.fore_color.rgb = color
    rule.line.fill.background()
    return top_cm + 2.5


def add_paragraph_block(slide, left_cm, top_cm, width_cm, height_cm,
                        items, *, size=22, line_spacing=1.2):
    """Items: list[str] or list[list[(text, opts)]]; '' = spacer."""
    tb = add_textbox(slide, left_cm, top_cm, width_cm, height_cm)
    tf = tb.text_frame
    first = True
    for item in items:
        if first:
            p = tf.paragraphs[0]
            first = False
        else:
            p = tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        p.space_after = Pt(6)
        p.line_spacing = line_spacing
        if isinstance(item, str):
            if item == "":
                p.space_after = Pt(8)
                continue
            r = p.add_run()
            set_run(r, item, size=size)
        else:
            for chunk in item:
                if isinstance(chunk, str):
                    text, opts = chunk, {}
                else:
                    text, opts = chunk
                r = p.add_run()
                set_run(r, text, size=opts.get("size", size),
                        bold=opts.get("bold", False),
                        italic=opts.get("italic", False),
                        color=opts.get("color", COLOR_BODY))
    return tb


def add_image(slide, path, left_cm, top_cm, width_cm=None, height_cm=None):
    if width_cm is not None and height_cm is None:
        slide.shapes.add_picture(str(path), Cm(left_cm), Cm(top_cm),
                                 width=Cm(width_cm))
    elif height_cm is not None and width_cm is None:
        slide.shapes.add_picture(str(path), Cm(left_cm), Cm(top_cm),
                                 height=Cm(height_cm))
    else:
        slide.shapes.add_picture(str(path), Cm(left_cm), Cm(top_cm),
                                 width=Cm(width_cm), height=Cm(height_cm))


def add_caption(slide, left_cm, top_cm, width_cm, text):
    tb = add_textbox(slide, left_cm, top_cm, width_cm, 0.9)
    p = tb.text_frame.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    set_run(p.add_run(), text, size=16, italic=True, color=COLOR_MUTED)


def add_table(slide, left_cm, top_cm, width_cm, height_cm, data,
              *, header_bold=True, col_widths_cm=None, font_size=22,
              accent_rows=None):
    if accent_rows is None:
        accent_rows = set()
    rows, cols = len(data), len(data[0])
    ts = slide.shapes.add_table(rows, cols,
                                Cm(left_cm), Cm(top_cm),
                                Cm(width_cm), Cm(height_cm))
    t = ts.table
    if col_widths_cm is not None:
        for i, w in enumerate(col_widths_cm):
            t.columns[i].width = Cm(w)
    for r, row in enumerate(data):
        for c, val in enumerate(row):
            cell = t.cell(r, c)
            if r == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = COLOR_HEADER_BG
            elif r in accent_rows:
                cell.fill.solid()
                cell.fill.fore_color.rgb = RGBColor(0xFD, 0xF1, 0xEE)
            else:
                cell.fill.solid()
                cell.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            tf = cell.text_frame
            tf.margin_left = Cm(0.2)
            tf.margin_right = Cm(0.2)
            tf.margin_top = Cm(0.08)
            tf.margin_bottom = Cm(0.08)
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER
            r0 = p.add_run()
            in_accent = r in accent_rows
            set_run(r0, str(val),
                    size=font_size,
                    bold=(r == 0 and header_bold) or in_accent,
                    color=(COLOR_ACCENT if in_accent else
                           (COLOR_TITLE if r == 0 else COLOR_BODY)))
    return ts


def add_pipeline_diagram(slide, left_cm, top_cm, width_cm):
    """Vertical pipeline: 5 boxes connected by down arrows."""
    box_h = 2.8
    arrow_h = 0.9
    n = 5
    labels = [
        "Raw events",
        "Monthly snapshots",
        "Candidates + labels",
        "36 features  (5 families)",
        "LR  /  RF",
    ]
    sublabels = [
        "u → v, rating, ts",
        "cumulative; last-rating-wins state",
        "2-hop, core-active, no existing u → v −",
        "structural · trust · temporal · community",
        "fitted, scored, ranked",
    ]
    total_h = n * box_h + (n - 1) * arrow_h
    y = top_cm
    for i, (lab, sub) in enumerate(zip(labels, sublabels)):
        rect = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE,
            Cm(left_cm), Cm(y), Cm(width_cm), Cm(box_h),
        )
        rect.adjustments[0] = 0.10
        rect.fill.solid()
        rect.fill.fore_color.rgb = COLOR_HEADER_BG
        rect.line.color.rgb = COLOR_TITLE
        rect.line.width = Pt(1.0)
        tf = rect.text_frame
        tf.margin_left = Cm(0.4); tf.margin_right = Cm(0.4)
        tf.margin_top = Cm(0.25); tf.margin_bottom = Cm(0.25)
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        set_run(p.add_run(), lab, size=22, bold=True, color=COLOR_TITLE)
        p2 = tf.add_paragraph()
        p2.alignment = PP_ALIGN.CENTER
        p2.space_before = Pt(2)
        set_run(p2.add_run(), sub, size=14, italic=True, color=COLOR_MUTED)
        # Arrow down to next
        if i < n - 1:
            arr = slide.shapes.add_shape(
                MSO_SHAPE.DOWN_ARROW,
                Cm(left_cm + width_cm / 2 - 0.5),
                Cm(y + box_h + 0.05),
                Cm(1.0), Cm(arrow_h - 0.1),
            )
            arr.fill.solid()
            arr.fill.fore_color.rgb = COLOR_TITLE
            arr.line.fill.background()
        y += box_h + arrow_h
    return total_h


def add_feature_family_box(slide, left_cm, top_cm, width_cm, height_cm,
                           *, tag, name, content, color=COLOR_BOX_S):
    rect = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Cm(left_cm), Cm(top_cm), Cm(width_cm), Cm(height_cm),
    )
    rect.adjustments[0] = 0.06
    rect.fill.solid()
    rect.fill.fore_color.rgb = color
    rect.line.color.rgb = COLOR_RULE
    rect.line.width = Pt(0.5)
    tf = rect.text_frame
    tf.margin_left = Cm(0.35); tf.margin_right = Cm(0.35)
    tf.margin_top = Cm(0.25); tf.margin_bottom = Cm(0.25)
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    # Tag in colored chip
    r1 = p.add_run()
    set_run(r1, tag, size=20, bold=True, color=COLOR_ACCENT)
    r2 = p.add_run()
    set_run(r2, "  " + name, size=20, bold=True, color=COLOR_TITLE)
    p2 = tf.add_paragraph()
    p2.alignment = PP_ALIGN.LEFT
    p2.space_before = Pt(4)
    set_run(p2.add_run(), content, size=15, color=COLOR_BODY)


# ----------------------------------------------------------------------
# Build poster
# ----------------------------------------------------------------------

def build() -> None:
    prs = Presentation()
    prs.slide_width = Cm(W_CM)
    prs.slide_height = Cm(H_CM)

    blank = prs.slide_layouts[6]
    s = prs.slides.add_slide(blank)

    # White background
    bg = s.background
    bg.fill.solid()
    bg.fill.fore_color.rgb = COLOR_BG

    # ==================================================================
    # TITLE BANNER (full width)
    # ==================================================================
    banner = s.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        Cm(0), Cm(0), Cm(W_CM), Cm(TITLE_TOP + TITLE_H),
    )
    banner.fill.solid()
    banner.fill.fore_color.rgb = COLOR_TITLE
    banner.line.fill.background()

    # Title text
    tb = add_textbox(s, MARGIN_X, TITLE_TOP + 0.5,
                     W_CM - 2 * MARGIN_X, TITLE_H - 1.0)
    tf = tb.text_frame
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    set_run(p.add_run(),
            "Early Warning of Distrust in Bitcoin-OTC",
            size=80, bold=True, color=RGBColor(0xFF, 0xFF, 0xFF))
    p2 = tf.add_paragraph()
    p2.alignment = PP_ALIGN.LEFT
    p2.space_before = Pt(8)
    set_run(p2.add_run(),
            "Community-Aware Prediction of New Negative Trust Edges "
            "in a Temporal Signed Network",
            size=34, italic=True, color=RGBColor(0xCF, 0xD8, 0xE6))
    p3 = tf.add_paragraph()
    p3.alignment = PP_ALIGN.LEFT
    p3.space_before = Pt(14)
    set_run(p3.add_run(),
            "Ömer Faruk Güler   ·   Hacettepe University   ·   "
            "BBM462 Final Project   ·   omer55glr@gmail.com",
            size=22, color=RGBColor(0xE8, 0xEC, 0xF2))

    # ==================================================================
    # COLUMN 1
    # ==================================================================
    cx = col_x(0)
    y = CONTENT_TOP

    # ---- [1] Motivation ----
    h = 14.5
    add_panel_bg(s, cx, y, COL_W, h)
    inner_y = add_section_header(s, cx, y, COL_W, "1   ·   Motivation")
    add_paragraph_block(
        s, cx + 0.5, inner_y, COL_W - 1.0, h - (inner_y - y) - 0.4,
        [
            "Online trust marketplaces depend on user-to-user ratings.",
            "",
            "Negative ratings carry early information about emerging distrust.",
            "",
            [("Goal — ", {"size": 24, "bold": True, "color": COLOR_ACCENT}),
             ("at end of month t, rank pairs (u, v) by",
              {"size": 24})],
            [("P(new negative directed edge u → v in t + 1)",
              {"size": 22, "italic": True, "bold": True})],
            "",
            [("Applications: ",
              {"size": 22, "italic": True, "color": COLOR_MUTED}),
             ("moderation · fraud surveillance · reputation",
              {"size": 22, "color": COLOR_BODY})],
        ],
        size=24,
    )
    y += h + 1.0

    # ---- [2] Why Not Standard ----
    h = 16.5
    add_panel_bg(s, cx, y, COL_W, h)
    inner_y = add_section_header(s, cx, y, COL_W,
                                  "2   ·   Why It Is Not Generic Link Prediction")
    add_paragraph_block(
        s, cx + 0.5, inner_y, COL_W - 1.0, h - (inner_y - y) - 0.4,
        [
            [("•  Sign-dependent — ",
              {"size": 24, "bold": True, "color": COLOR_ACCENT}),
             ("predicting any edge ≠ predicting a negative edge",
              {"size": 24})],
            "",
            [("•  Heavily imbalanced — ",
              {"size": 24, "bold": True, "color": COLOR_ACCENT}),
             ("~10% of edges negative; new monthly negatives much rarer",
              {"size": 24})],
            "",
            [("•  Time-causal — ",
              {"size": 24, "bold": True, "color": COLOR_ACCENT}),
             ("only pre-snapshot information allowed",
              {"size": 24})],
            "",
            [("→ ranking metrics ", {"size": 22, "italic": True}),
             ("(ROC-AUC, PR-AUC)", {"size": 22, "italic": True,
                                     "bold": True, "color": COLOR_ACCENT2}),
             (", not threshold metrics",
              {"size": 22, "italic": True})],
        ],
        size=24,
    )
    y += h + 1.0

    # ---- [3] Dataset and Task (with figure) ----
    h = CONTENT_BOTTOM - y
    add_panel_bg(s, cx, y, COL_W, h)
    inner_y = add_section_header(s, cx, y, COL_W, "3   ·   Dataset and Task")
    add_paragraph_block(
        s, cx + 0.5, inner_y, COL_W - 1.0, 14,
        [
            [("Bitcoin-OTC (SNAP)",
              {"size": 24, "bold": True, "color": COLOR_TITLE})],
            [("35,592 ratings · 5,881 users · Nov 2010 – Jan 2016",
              {"size": 22})],
            "",
            [("Edge tuple: ", {"size": 22}),
             ("(u → v, rating ∈ [−10, +10], timestamp)",
              {"size": 22, "italic": True})],
            "",
            [("90% positive · 10% negative · reciprocity 0.79",
              {"size": 22})],
            "",
            [("Task formulation",
              {"size": 22, "bold": True, "color": COLOR_ACCENT})],
            [("Predict probability that an ordered pair (u, v) "
              "receives a new negative rating in (t, t + 1].",
              {"size": 21})],
        ],
        size=22,
    )
    # Figure
    img_y = inner_y + 16.5
    img_x = cx + 0.5
    img_w = COL_W - 1.0
    add_image(s, FIG_DIR / "edges_per_month.png", img_x, img_y, width_cm=img_w)
    add_caption(s, img_x, img_y + img_w * 0.45, img_w,
                "Fig 1 — Monthly interaction volume; tail decay motivates the eligible pool.")

    # ==================================================================
    # COLUMN 2
    # ==================================================================
    cx = col_x(1)
    y = CONTENT_TOP

    # ---- [4] Pipeline (with diagram) ----
    h = 28.5
    add_panel_bg(s, cx, y, COL_W, h)
    inner_y = add_section_header(s, cx, y, COL_W, "4   ·   Pipeline")
    add_paragraph_block(
        s, cx + 0.5, inner_y, COL_W - 1.0, 2.5,
        [
            [("Five interpretable stages — no leakage, fixed seeds.",
              {"size": 21, "italic": True, "color": COLOR_MUTED})],
        ],
        size=22,
    )
    add_pipeline_diagram(s, cx + 4.5, inner_y + 2.5, COL_W - 9.0)
    # Leakage note at bottom of panel
    add_paragraph_block(
        s, cx + 0.5, y + h - 1.8, COL_W - 1.0, 1.5,
        [
            [("Leakage boundary: ",
              {"size": 18, "bold": True, "italic": True,
               "color": COLOR_ACCENT}),
             ("only events with  ts ≤ end_of_month(t)  enter features.",
              {"size": 18, "italic": True, "color": COLOR_MUTED})],
        ],
        size=18,
    )
    y += h + 1.0

    # ---- [5] Feature Groups (2x2) ----
    h = 21.0
    add_panel_bg(s, cx, y, COL_W, h)
    inner_y = add_section_header(s, cx, y, COL_W,
                                  "5   ·   Feature Groups   ·   36 features")
    box_w = (COL_W - 1.0 - 0.5) / 2
    box_h = (h - (inner_y - y) - 0.8 - 0.5) / 2
    bx = cx + 0.5
    by = inner_y + 0.1
    add_feature_family_box(s, bx, by, box_w, box_h,
                           tag="(S)", name="Structural · 12",
                           content="CN, Jaccard, Adamic-Adar, PA · "
                                   "signed degrees · reciprocity · "
                                   "prior interaction",
                           color=COLOR_BOX_S)
    add_feature_family_box(s, bx + box_w + 0.5, by, box_w, box_h,
                           tag="(+)", name="Trust-summary · 4",
                           content="Mean rating given/received · "
                                   "negative-given ratio · "
                                   "negative-received ratio",
                           color=COLOR_BOX_TRUST)
    add_feature_family_box(s, bx, by + box_h + 0.5, box_w, box_h,
                           tag="(+T)", name="Temporal · 12",
                           content="Days since last activity · "
                                   "recent-window counts · "
                                   "exponentially decayed aggregates  "
                                   "(τ = 60 d)",
                           color=COLOR_BOX_T)
    add_feature_family_box(s, bx + box_w + 0.5, by + box_h + 0.5,
                           box_w, box_h,
                           tag="(+C)", name="Community-aware · 8",
                           content="Louvain on G⁺ · same_community · "
                                   "sizes · neighborhood overlap · "
                                   "boundary fractions",
                           color=COLOR_BOX_C)
    y += h + 1.0

    # ---- [6] Experimental Setup ----
    h = CONTENT_BOTTOM - y
    add_panel_bg(s, cx, y, COL_W, h)
    inner_y = add_section_header(s, cx, y, COL_W,
                                  "6   ·   Experimental Setup")
    add_paragraph_block(
        s, cx + 0.5, inner_y, COL_W - 1.0, 8,
        [
            [("Models — ",
              {"size": 23, "bold": True, "color": COLOR_ACCENT}),
             ("Logistic Regression  ·  Random Forest",
              {"size": 23})],
            [("(no hyperparameter tuning)",
              {"size": 18, "italic": True, "color": COLOR_MUTED})],
            "",
            [("Forward-time split — no shuffle, no leakage:",
              {"size": 22, "bold": True})],
        ],
        size=23,
    )
    # Split table
    data = [
        ["Split", "Months",      "Pairs", "Positives"],
        ["TRAIN", "t = 13 – 45", "5.0 M", "545"],
        ["VAL",   "t = 46, 47",  "184 k", "16"],
        ["TEST",  "t = 48, 49",  "149 k", "34"],
    ]
    table_y = inner_y + 7.5
    add_table(s, cx + 0.5, table_y, COL_W - 1.0, 7.0, data,
              col_widths_cm=[(COL_W - 1.0) * 0.20,
                             (COL_W - 1.0) * 0.35,
                             (COL_W - 1.0) * 0.22,
                             (COL_W - 1.0) * 0.23],
              font_size=22)
    add_paragraph_block(
        s, cx + 0.5, table_y + 7.5, COL_W - 1.0, 2,
        [
            [("Metrics: ", {"size": 21, "bold": True}),
             ("ROC-AUC  ·  PR-AUC", {"size": 21}),
             ("    Seeds: ",
              {"size": 21, "italic": True, "color": COLOR_MUTED}),
             ("all fixed at 42",
              {"size": 21, "italic": True, "color": COLOR_MUTED})],
        ],
        size=21,
    )

    # ==================================================================
    # COLUMN 3
    # ==================================================================
    cx = col_x(2)
    y = CONTENT_TOP

    # ---- [7] Main Results (with table + figure) ----
    h = 33.0
    add_panel_bg(s, cx, y, COL_W, h)
    inner_y = add_section_header(
        s, cx, y, COL_W,
        "7   ·   Main Results",
        color=COLOR_TITLE,
    )
    add_paragraph_block(
        s, cx + 0.5, inner_y, COL_W - 1.0, 3,
        [
            [("Adding temporal features produces the largest improvement.",
              {"size": 22, "bold": True, "color": COLOR_ACCENT})],
            [("Random Forest, test fold  (t = 48, 49)",
              {"size": 19, "italic": True, "color": COLOR_MUTED})],
        ],
        size=22,
    )
    # Results table
    data = [
        ["Feature set", "ROC-AUC", "PR-AUC"],
        ["S",           "0.88",    "0.02"],
        ["S + T",       "0.95",    "0.04"],
        ["S + T + C",   "0.94",    "0.07"],
    ]
    add_table(s, cx + 0.5, inner_y + 3.0, COL_W - 1.0, 8.0, data,
              col_widths_cm=[(COL_W - 1.0) * 0.42,
                             (COL_W - 1.0) * 0.29,
                             (COL_W - 1.0) * 0.29],
              font_size=24,
              accent_rows={2})
    # Bullets below table
    add_paragraph_block(
        s, cx + 0.5, inner_y + 11.5, COL_W - 1.0, 4.5,
        [
            [("•  + T   → ",
              {"size": 21, "bold": True, "color": COLOR_ACCENT}),
             ("largest, most consistent jump", {"size": 21})],
            [("•  + C   → ",
              {"size": 21, "bold": True, "color": COLOR_ACCENT}),
             ("further PR-AUC gain; ROC near saturation",
              {"size": 21})],
            [("•  LR shows the same ordering at lower absolute PR-AUC.",
              {"size": 19, "italic": True, "color": COLOR_MUTED})],
        ],
        size=21,
    )
    # Figure: PR curves
    img_y = inner_y + 16.5
    add_image(s, FIG_DIR / "test_rf_pr.png",
              cx + 0.5, img_y, width_cm=COL_W - 1.0)
    add_caption(s, cx + 0.5, img_y + (COL_W - 1.0) * 0.45, COL_W - 1.0,
                "Fig 2 — RF precision-recall curves on the test fold.")
    y += h + 1.0

    # ---- [8] Interpretation (with figure) ----
    h = 26.0
    add_panel_bg(s, cx, y, COL_W, h)
    inner_y = add_section_header(s, cx, y, COL_W, "8   ·   Interpretation")
    add_paragraph_block(
        s, cx + 0.5, inner_y, COL_W - 1.0, 9,
        [
            [("Temporal information dominates.",
              {"size": 22, "bold": True, "color": COLOR_ACCENT})],
            [("Recency and decayed activity drive importance in both models.",
              {"size": 19})],
            "",
            [("Community-aware features add a nuanced refinement.",
              {"size": 22, "bold": True, "color": COLOR_ACCENT})],
            [("Mostly on PR-AUC; ROC-AUC is already saturated.",
              {"size": 19})],
            "",
            [("Naive cross-community-distrust is NOT supported.",
              {"size": 22, "bold": True, "color": COLOR_ACCENT})],
            [("At the pair level, neighborhood overlap is the strongest "
              "community signal — distrust appears on weakly-embedded ties.",
              {"size": 19})],
        ],
        size=22,
    )
    # Figure
    img_y = inner_y + 9.5
    add_image(s, FIG_DIR / "same_vs_cross_positive_rate.png",
              cx + 1.5, img_y, width_cm=COL_W - 3.0)
    add_caption(s, cx + 0.5, img_y + (COL_W - 3.0) * 0.65, COL_W - 1.0,
                "Fig 3 — Training positive rate: same vs cross-community pairs.")
    y += h + 1.0

    # ---- [9] Conclusion ----
    h = CONTENT_BOTTOM - y
    add_panel_bg(s, cx, y, COL_W, h)
    inner_y = add_section_header(s, cx, y, COL_W,
                                  "9   ·   Conclusion & Future Work")
    add_paragraph_block(
        s, cx + 0.5, inner_y, COL_W - 1.0, h - (inner_y - y) - 0.5,
        [
            [("This project",
              {"size": 23, "bold": True, "color": COLOR_TITLE})],
            "•  A precise temporal signed-network task, defined end-to-end",
            "•  An interpretable, leakage-controlled feature pipeline",
            [("•  ", {"size": 21}),
             ("Temporal features → strongest, most consistent improvement",
              {"size": 21, "bold": True, "color": COLOR_ACCENT})],
            "•  Community features → smaller, nuanced refinement",
            "",
            [("Future work",
              {"size": 23, "bold": True, "color": COLOR_TITLE})],
            "•  External replication on Bitcoin-Alpha",
            "•  Stronger learning baselines and graph embeddings",
            "•  Conditional analyses of community signal",
        ],
        size=21,
    )

    # ==================================================================
    # BOTTOM STRIP (full width) — Additional Validation + References
    # ==================================================================
    strip_x = MARGIN_X
    strip_w = W_CM - 2 * MARGIN_X
    add_panel_bg(s, strip_x, BOTTOM_STRIP_TOP, strip_w, BOTTOM_STRIP_H,
                 fill=COLOR_HEADER_BG, border=COLOR_TITLE)
    inner_y = add_section_header(s, strip_x, BOTTOM_STRIP_TOP, strip_w,
                                  "10   ·   Additional Validation   ·   "
                                  "Robustness consistent with the main story")
    # Three-column micro-content
    sub_w = (strip_w - 1.5) / 3
    cells = [
        [
            [("Rolling-origin evaluation",
              {"size": 19, "bold": True, "color": COLOR_TITLE})],
            "25 disjoint monthly test folds · 425 positives total.",
            [("Wilcoxon p ≤ 1.4 × 10⁻⁴",
              {"size": 18, "bold": True, "color": COLOR_ACCENT}),
             ("  for the (S+T) − S ROC-AUC delta under both models, "
              "after Holm-Bonferroni correction.",
              {"size": 18})],
        ],
        [
            [("Recency-only baseline",
              {"size": 19, "bold": True, "color": COLOR_TITLE})],
            "A parameter-free recency-only ranker reaches",
            [("ROC-AUC ≈ 0.89",
              {"size": 18, "bold": True, "color": COLOR_ACCENT}),
             ("  — the temporal signal is robust and supported by simple "
              "recency information.",
              {"size": 18})],
        ],
        [
            [("Bootstrap (B = 1000)",
              {"size": 19, "bold": True, "color": COLOR_TITLE})],
            "Paired stratified bootstrap on the locked fold.",
            "95% percentile CIs agree with the across-fold results; "
            "no contradictions.",
        ],
    ]
    for i, items in enumerate(cells):
        add_paragraph_block(
            s, strip_x + 0.3 + i * (sub_w + 0.75),
            inner_y, sub_w, 5,
            items, size=18, line_spacing=1.15,
        )

    # References + contact line at very bottom
    tb = add_textbox(s, MARGIN_X, H_CM - 1.4, W_CM - 2 * MARGIN_X, 1.2)
    tf = tb.text_frame
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    set_run(p.add_run(),
            "References: ",
            size=15, bold=True, color=COLOR_TITLE)
    set_run(p.add_run(),
            "Kumar et al. (ICDM 2016)  ·  "
            "Blondel et al. (J Stat Mech 2008)  ·  "
            "Bertazzi et al. (SocInfo 2018)  ·  "
            "Leskovec & Krevl, SNAP (2014).",
            size=15, color=COLOR_BODY)
    p2 = tf.add_paragraph()
    p2.alignment = PP_ALIGN.LEFT
    set_run(p2.add_run(),
            "Reproducibility: ", size=15, bold=True, color=COLOR_TITLE)
    set_run(p2.add_run(),
            "all seeds fixed at 42 · code organized in src/ and scripts/ · "
            "intermediate artifacts persisted under data/ and results/.",
            size=15, color=COLOR_BODY)

    # ==================================================================
    # Save
    # ==================================================================
    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    print(f"Saved {OUT}")
    print(f"Poster: A0 portrait  {W_CM} cm × {H_CM} cm  (1 slide)")


if __name__ == "__main__":
    build()
