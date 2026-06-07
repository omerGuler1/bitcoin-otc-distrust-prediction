"""Build the BBM462 final-project video presentation deck.

Generates `report/BBM462_presentation.pptx` from the slide content already
designed in the conversation. Each slide uses a clean, presentation-friendly
layout: blank base, large title at the top, body content below, optional
figure on the right.

Run:
    python -m scripts.build_presentation
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
OUT = ROOT / "report" / "BBM462_presentation.pptx"

# ----------------------------------------------------------------------
# Style constants -- single accent color, dark grey on white.
# ----------------------------------------------------------------------
COLOR_BG = RGBColor(0xFF, 0xFF, 0xFF)
COLOR_TITLE = RGBColor(0x10, 0x2A, 0x55)        # deep navy
COLOR_ACCENT = RGBColor(0xC0, 0x39, 0x2B)       # warm accent for emphasis
COLOR_BODY = RGBColor(0x2B, 0x2B, 0x2B)         # near-black
COLOR_MUTED = RGBColor(0x6E, 0x6E, 0x6E)        # footnotes / subtitle
COLOR_RULE = RGBColor(0xD0, 0xD0, 0xD0)         # thin divider line

FONT_TITLE = "Calibri"
FONT_BODY = "Calibri"

# 16:9 widescreen
SLIDE_W_CM = 33.867
SLIDE_H_CM = 19.05


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def add_textbox(slide, left, top, width, height) -> "pptx.shapes.autoshape.Shape":
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = Cm(0.05)
    tf.margin_right = Cm(0.05)
    tf.margin_top = Cm(0.05)
    tf.margin_bottom = Cm(0.05)
    return tb


def set_run(run, text, *, font=FONT_BODY, size=20, bold=False,
            color=COLOR_BODY, italic=False):
    run.text = text
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color


def add_title(slide, text, *, size=36, color=COLOR_TITLE):
    tb = add_textbox(slide, Cm(1.5), Cm(0.9), Cm(SLIDE_W_CM - 3.0), Cm(1.8))
    p = tb.text_frame.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    set_run(p.add_run(), text, font=FONT_TITLE, size=size, bold=True, color=color)
    # Thin rule below the title.
    line = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        Cm(1.5), Cm(2.85), Cm(SLIDE_W_CM - 3.0), Cm(0.04),
    )
    line.fill.solid()
    line.fill.fore_color.rgb = COLOR_RULE
    line.line.fill.background()
    return tb


def add_footer(slide, text):
    tb = add_textbox(slide, Cm(1.5), Cm(SLIDE_H_CM - 1.0),
                     Cm(SLIDE_W_CM - 3.0), Cm(0.7))
    p = tb.text_frame.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    set_run(p.add_run(), text, size=11, italic=True, color=COLOR_MUTED)


def add_slide_number(slide, n, total):
    tb = add_textbox(slide, Cm(SLIDE_W_CM - 3.5), Cm(SLIDE_H_CM - 1.0),
                     Cm(2.5), Cm(0.7))
    p = tb.text_frame.paragraphs[0]
    p.alignment = PP_ALIGN.RIGHT
    set_run(p.add_run(), f"{n} / {total}", size=11, color=COLOR_MUTED)


def add_bullets(slide, left, top, width, height, items,
                *, size=22, line_spacing=1.15, indent=False):
    """Items: list of either str or list[(text, opts)] for mixed-style lines."""
    tb = add_textbox(slide, left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    first = True
    for item in items:
        if first:
            p = tf.paragraphs[0]
            first = False
        else:
            p = tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        p.space_after = Pt(8)
        p.line_spacing = line_spacing
        if indent:
            p.level = 0
        if isinstance(item, str):
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
                        color=opts.get("color", COLOR_BODY),
                        italic=opts.get("italic", False))
    return tb


def add_image(slide, path: Path, left, top, width=None, height=None):
    if not path.exists():
        raise FileNotFoundError(path)
    if width is not None and height is None:
        slide.shapes.add_picture(str(path), left, top, width=width)
    elif height is not None and width is None:
        slide.shapes.add_picture(str(path), left, top, height=height)
    elif width is not None and height is not None:
        slide.shapes.add_picture(str(path), left, top, width=width, height=height)
    else:
        slide.shapes.add_picture(str(path), left, top)


def set_white_background(slide):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = COLOR_BG


def add_table(slide, left, top, width, height, data, *, header_bold=True,
              col_widths=None, font_size=18, header_color=COLOR_TITLE):
    """data: list of rows; each row is a list of strings.
    First row is treated as header."""
    rows, cols = len(data), len(data[0])
    table_shape = slide.shapes.add_table(rows, cols, left, top, width, height)
    table = table_shape.table
    if col_widths is not None:
        for i, w in enumerate(col_widths):
            table.columns[i].width = w
    for r, row in enumerate(data):
        for c, val in enumerate(row):
            cell = table.cell(r, c)
            cell.fill.solid()
            cell.fill.fore_color.rgb = (RGBColor(0xF1, 0xF3, 0xF8)
                                        if r == 0
                                        else RGBColor(0xFF, 0xFF, 0xFF))
            tf = cell.text_frame
            tf.margin_left = Cm(0.15)
            tf.margin_right = Cm(0.15)
            tf.margin_top = Cm(0.05)
            tf.margin_bottom = Cm(0.05)
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER
            r0 = p.add_run()
            set_run(r0, str(val),
                    size=font_size,
                    bold=(r == 0 and header_bold),
                    color=header_color if r == 0 else COLOR_BODY)
    return table_shape


# ----------------------------------------------------------------------
# Build slides
# ----------------------------------------------------------------------

def build() -> None:
    prs = Presentation()
    prs.slide_width = Cm(SLIDE_W_CM)
    prs.slide_height = Cm(SLIDE_H_CM)

    blank = prs.slide_layouts[6]   # truly blank layout
    N = 8

    # ------------------------------------------------------------------
    # Slide 1 -- Title
    # ------------------------------------------------------------------
    s = prs.slides.add_slide(blank)
    set_white_background(s)

    # Centered big title
    tb = add_textbox(s, Cm(1.5), Cm(5.5), Cm(SLIDE_W_CM - 3.0), Cm(4.5))
    tf = tb.text_frame
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    set_run(p.add_run(),
            "Early Warning of Distrust",
            font=FONT_TITLE, size=54, bold=True, color=COLOR_TITLE)
    p2 = tf.add_paragraph()
    p2.alignment = PP_ALIGN.LEFT
    set_run(p2.add_run(),
            "in Bitcoin-OTC",
            font=FONT_TITLE, size=54, bold=True, color=COLOR_TITLE)
    p3 = tf.add_paragraph()
    p3.alignment = PP_ALIGN.LEFT
    p3.space_before = Pt(20)
    set_run(p3.add_run(),
            "Community-Aware Prediction of New Negative Trust Edges",
            font=FONT_TITLE, size=24, italic=True, color=COLOR_MUTED)

    # Author / affiliation block
    tb = add_textbox(s, Cm(1.5), Cm(15.0), Cm(SLIDE_W_CM - 3.0), Cm(2.5))
    tf = tb.text_frame
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    set_run(p.add_run(),
            "Ömer Faruk Güler",
            font=FONT_TITLE, size=22, bold=True, color=COLOR_BODY)
    p = tf.add_paragraph()
    p.alignment = PP_ALIGN.LEFT
    set_run(p.add_run(),
            "Hacettepe University · BBM462 Final Project",
            font=FONT_BODY, size=18, color=COLOR_MUTED)

    add_slide_number(s, 1, N)

    # ------------------------------------------------------------------
    # Slide 2 -- Why not generic link prediction
    # ------------------------------------------------------------------
    s = prs.slides.add_slide(blank)
    set_white_background(s)
    add_title(s, "Why This Is Not Generic Link Prediction")

    bullets = [
        [
            ("Sign-dependent — ",
             {"size": 26, "bold": True, "color": COLOR_ACCENT}),
            ("predicting any edge ≠ predicting a negative edge",
             {"size": 26}),
        ],
        [
            ("Heavily imbalanced — ",
             {"size": 26, "bold": True, "color": COLOR_ACCENT}),
            ("≈ 10 % of edges negative; new monthly negatives much rarer",
             {"size": 26}),
        ],
        [
            ("Time-causal — ",
             {"size": 26, "bold": True, "color": COLOR_ACCENT}),
            ("only pre-snapshot information can enter the model",
             {"size": 26}),
        ],
        " ",
        [
            ("→ ranking metrics (ROC-AUC, PR-AUC), not threshold metrics",
             {"size": 24, "italic": True, "color": COLOR_MUTED}),
        ],
    ]
    add_bullets(s, Cm(1.8), Cm(4.5), Cm(SLIDE_W_CM - 3.6), Cm(12),
                bullets, size=26, line_spacing=1.35)
    add_slide_number(s, 2, N)

    # ------------------------------------------------------------------
    # Slide 3 -- Dataset and Task
    # ------------------------------------------------------------------
    s = prs.slides.add_slide(blank)
    set_white_background(s)
    add_title(s, "Dataset and Task")

    # left text column
    bullets = [
        [
            ("35,592 ratings · 5,881 users · 63 months",
             {"size": 24, "bold": True}),
        ],
        [
            ("Edge: ",
             {"size": 22}),
            ("(u → v, rating ∈ [-10, +10], timestamp)",
             {"size": 22, "italic": True}),
        ],
        [
            ("≈ 90 % positive · 10 % negative · reciprocity 0.79",
             {"size": 22}),
        ],
        " ",
        [
            ("Task — ",
             {"size": 22, "bold": True, "color": COLOR_ACCENT}),
            ("at end of month t, rank pairs (u, v) by",
             {"size": 22}),
        ],
        [
            ("P(new negative directed edge u → v in t + 1)",
             {"size": 22, "italic": True, "bold": True}),
        ],
    ]
    add_bullets(s, Cm(1.8), Cm(4.0), Cm(15.0), Cm(13),
                bullets, size=22, line_spacing=1.35)

    # right image
    add_image(s, FIG_DIR / "edges_per_month.png",
              Cm(17.5), Cm(4.5), width=Cm(14.5))
    add_footer(s, "Bitcoin-OTC monthly interaction volume")
    add_slide_number(s, 3, N)

    # ------------------------------------------------------------------
    # Slide 4 -- Pipeline + Feature families
    # ------------------------------------------------------------------
    s = prs.slides.add_slide(blank)
    set_white_background(s)
    add_title(s, "Interpretable Feature Pipeline")

    # Pipeline ribbon: 5 connected boxes
    ribbon_top = Cm(4.0)
    ribbon_h = Cm(1.6)
    block_w = Cm(5.8)
    gap = Cm(0.4)
    start_x = Cm(1.8)
    labels = ["Raw events", "Monthly snapshots", "Candidates + labels",
              "36 features", "LR / RF"]
    for i, lab in enumerate(labels):
        left = Emu(start_x + i * (block_w + gap))
        rect = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                  left, ribbon_top, block_w, ribbon_h)
        rect.fill.solid()
        rect.fill.fore_color.rgb = RGBColor(0xE8, 0xEE, 0xF6)
        rect.line.color.rgb = COLOR_TITLE
        rect.line.width = Pt(0.75)
        tf = rect.text_frame
        tf.margin_left = Cm(0.15); tf.margin_right = Cm(0.15)
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        set_run(p.add_run(), lab, size=16, bold=True, color=COLOR_TITLE)
        # arrow between boxes
        if i < len(labels) - 1:
            ax = Emu(start_x + i * (block_w + gap) + block_w + Emu(0))
            arrow = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW,
                                       ax, Emu(ribbon_top + Cm(0.4)),
                                       gap, Cm(0.8))
            arrow.fill.solid()
            arrow.fill.fore_color.rgb = COLOR_TITLE
            arrow.line.fill.background()

    # Feature families: four boxes in 2x2 grid
    fam_top = Cm(7.5)
    fam_h = Cm(4.0)
    fam_w = Cm(14.5)
    gap_x = Cm(0.6)
    gap_y = Cm(0.6)
    families = [
        ("(S) Structural", "CN, Jaccard, AA, PA · signed degrees · reciprocity · prior interaction"),
        ("(+ Trust-summary)", "Mean rating given/received · negative ratios"),
        ("(+ T) Temporal", "Days since last activity · recent counts · decayed aggregates (τ = 60 d)"),
        ("(+ C) Community-aware", "Louvain on G⁺ · same-community · neighborhood overlap · boundary fractions"),
    ]
    positions = [
        (Cm(1.8),            fam_top),
        (Cm(1.8 + 14.5 + 0.6), fam_top),
        (Cm(1.8),            Emu(fam_top + fam_h + gap_y)),
        (Cm(1.8 + 14.5 + 0.6), Emu(fam_top + fam_h + gap_y)),
    ]
    fam_colors = [
        RGBColor(0xF3, 0xF6, 0xFB),  # subtle blue
        RGBColor(0xF6, 0xF6, 0xF6),  # neutral grey
        RGBColor(0xFD, 0xF1, 0xEE),  # warm accent
        RGBColor(0xEF, 0xF6, 0xEF),  # subtle green
    ]
    for (title, body), (lx, ly), col in zip(families, positions, fam_colors):
        rect = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                  lx, ly, fam_w, fam_h)
        rect.fill.solid()
        rect.fill.fore_color.rgb = col
        rect.line.color.rgb = COLOR_RULE
        rect.line.width = Pt(0.5)
        tf = rect.text_frame
        tf.margin_left = Cm(0.4); tf.margin_right = Cm(0.4)
        tf.margin_top = Cm(0.25); tf.margin_bottom = Cm(0.25)
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.LEFT
        set_run(p.add_run(), title, size=22, bold=True, color=COLOR_TITLE)
        p2 = tf.add_paragraph()
        p2.alignment = PP_ALIGN.LEFT
        p2.space_before = Pt(6)
        set_run(p2.add_run(), body, size=16, color=COLOR_BODY)

    add_footer(s, "Strict leakage boundary: only events with ts ≤ end_of_month(t).")
    add_slide_number(s, 4, N)

    # ------------------------------------------------------------------
    # Slide 5 -- Models and Evaluation
    # ------------------------------------------------------------------
    s = prs.slides.add_slide(blank)
    set_white_background(s)
    add_title(s, "Models and Evaluation")

    bullets = [
        [
            ("Models — ", {"size": 22, "bold": True, "color": COLOR_ACCENT}),
            ("Logistic Regression and Random Forest (no hyperparameter tuning)",
             {"size": 22}),
        ],
        [
            ("Feature sets — ", {"size": 22, "bold": True, "color": COLOR_ACCENT}),
            ("S → S + T → S + T + C", {"size": 22}),
        ],
        [
            ("Metrics — ", {"size": 22, "bold": True, "color": COLOR_ACCENT}),
            ("ROC-AUC, PR-AUC (heavy class imbalance)", {"size": 22}),
        ],
        [
            ("Verification — ", {"size": 22, "bold": True, "color": COLOR_ACCENT}),
            ("rolling-origin over 25 monthly folds (425 positives total)",
             {"size": 22}),
        ],
    ]
    add_bullets(s, Cm(1.8), Cm(4.0), Cm(20.0), Cm(7),
                bullets, size=22, line_spacing=1.35)

    # Split table
    data = [
        ["Split", "Months", "Pairs", "Positives"],
        ["TRAIN", "t = 13 – 45", "5.0 M", "545"],
        ["VAL",   "t = 46, 47",  "184 k",  "16"],
        ["TEST",  "t = 48, 49",  "149 k",  "34"],
    ]
    col_widths = [Cm(5.0), Cm(7.0), Cm(5.0), Cm(5.0)]
    add_table(s, Cm(1.8), Cm(12.0), Cm(22.0), Cm(5.5),
              data, col_widths=col_widths, font_size=20)

    add_footer(s, "Forward-time split — no shuffle, no leakage.")
    add_slide_number(s, 5, N)

    # ------------------------------------------------------------------
    # Slide 6 -- Main Results
    # ------------------------------------------------------------------
    s = prs.slides.add_slide(blank)
    set_white_background(s)
    add_title(s, "Temporal Features Drive the Largest Improvement")

    # Results table (left)
    data = [
        ["Feature set",      "ROC-AUC", "PR-AUC"],
        ["S",                "0.88",    "0.02"],
        ["S + T",            "0.95",    "0.04"],
        ["S + T + C",        "0.94",    "0.07"],
    ]
    col_widths = [Cm(6.0), Cm(4.5), Cm(4.5)]
    tbl = add_table(s, Cm(1.8), Cm(4.0), Cm(15.0), Cm(6.5),
                    data, col_widths=col_widths, font_size=22)
    # Emphasize the S+T row by bolding it
    for c in range(3):
        cell = tbl.table.cell(2, c)
        for p in cell.text_frame.paragraphs:
            for r in p.runs:
                r.font.bold = True
                r.font.color.rgb = COLOR_ACCENT

    # Bullets below table
    bullets = [
        [
            ("+ T → ", {"size": 22, "bold": True, "color": COLOR_ACCENT}),
            ("largest, most consistent jump", {"size": 22}),
        ],
        [
            ("+ C → ", {"size": 22, "bold": True, "color": COLOR_ACCENT}),
            ("further PR-AUC gain; ROC-AUC near saturation", {"size": 22}),
        ],
        [
            ("Rolling-origin verification confirms temporal lift across folds",
             {"size": 22, "italic": True, "color": COLOR_MUTED}),
        ],
    ]
    add_bullets(s, Cm(1.8), Cm(11.0), Cm(15.0), Cm(7),
                bullets, size=22, line_spacing=1.3)

    # Figure on the right
    add_image(s, FIG_DIR / "test_rf_pr.png",
              Cm(17.5), Cm(4.5), width=Cm(15.0))

    add_footer(s, "Random Forest test-set results; PR-AUC curves on the right.")
    add_slide_number(s, 6, N)

    # ------------------------------------------------------------------
    # Slide 7 -- Interpretation
    # ------------------------------------------------------------------
    s = prs.slides.add_slide(blank)
    set_white_background(s)
    add_title(s, "What the Results Tell Us")

    bullets = [
        [
            ("Temporal information is the strongest signal",
             {"size": 22, "bold": True, "color": COLOR_ACCENT}),
        ],
        [
            ("Recency + decayed activity dominate feature importance",
             {"size": 20}),
        ],
        " ",
        [
            ("Community features add a nuanced, PR-oriented refinement",
             {"size": 22, "bold": True, "color": COLOR_ACCENT}),
        ],
        [
            ("ROC-AUC saturates near 0.95 — most ranking power already in S + T",
             {"size": 20}),
        ],
        " ",
        [
            ("Pair-level neighborhood overlap matters more than community label",
             {"size": 22, "bold": True, "color": COLOR_ACCENT}),
        ],
        [
            ("Same-community pairs are not the safer ones — descriptive, not causal",
             {"size": 20, "italic": True, "color": COLOR_MUTED}),
        ],
    ]
    add_bullets(s, Cm(1.8), Cm(4.0), Cm(17.0), Cm(13),
                bullets, size=22, line_spacing=1.25)

    add_image(s, FIG_DIR / "same_vs_cross_positive_rate.png",
              Cm(20.0), Cm(5.5), width=Cm(12.5))

    add_footer(s, "Same vs. cross-community positive rate (training data).")
    add_slide_number(s, 7, N)

    # ------------------------------------------------------------------
    # Slide 8 -- Conclusion + Future Work
    # ------------------------------------------------------------------
    s = prs.slides.add_slide(blank)
    set_white_background(s)
    add_title(s, "Conclusion and Future Work")

    # Two columns: This project (left) + Future work (right)
    # Left column
    tb = add_textbox(s, Cm(1.8), Cm(4.0), Cm(15.0), Cm(13))
    tf = tb.text_frame
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    set_run(p.add_run(), "This project", size=26, bold=True, color=COLOR_TITLE)
    items_left = [
        "A precise temporal signed-network task, defined end-to-end",
        "An interpretable, leakage-controlled pipeline",
        "Temporal features → strongest, most consistent improvement",
        "Community features → smaller, more nuanced refinement",
    ]
    for it in items_left:
        p = tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        p.space_before = Pt(10)
        set_run(p.add_run(), "• " + it, size=20)

    # Right column
    tb = add_textbox(s, Cm(17.5), Cm(4.0), Cm(15.0), Cm(13))
    tf = tb.text_frame
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    set_run(p.add_run(), "Future work", size=26, bold=True, color=COLOR_TITLE)
    items_right = [
        "External replication on Bitcoin-Alpha",
        "Stronger learning baselines",
        "Deeper community-conditioning analyses",
    ]
    for it in items_right:
        p = tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        p.space_before = Pt(10)
        set_run(p.add_run(), "• " + it, size=20)

    add_footer(s, "Thank you · Questions welcome.")
    add_slide_number(s, 8, N)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    print(f"Saved {OUT}")
    print(f"Slides: {len(prs.slides)}")


if __name__ == "__main__":
    build()
