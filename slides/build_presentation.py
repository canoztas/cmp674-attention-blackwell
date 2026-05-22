"""Build the simplified course-presentation PPTX from the paper content.

Style: Hacettepe red + NVIDIA green theme (matches slides/lesson.pdf).
Output: slides/Attention_Performance_Comparison_Analysis.pptx
"""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

# --- color palette (mirrors slides/lesson.tex) -------------------------------
HACETTEPE_RED       = RGBColor(0xA8, 0x00, 0x2B)
HACETTEPE_RED_DARK  = RGBColor(0x78, 0x00, 0x1E)
HACETTEPE_RED_LIGHT = RGBColor(0xDC, 0x64, 0x82)
NVIDIA_GREEN        = RGBColor(0x76, 0xB9, 0x00)
NVIDIA_GREEN_DARK   = RGBColor(0x4B, 0x78, 0x00)
NVIDIA_GREEN_LIGHT  = RGBColor(0xC8, 0xE6, 0x82)
CHARCOAL            = RGBColor(0x28, 0x28, 0x2D)
LIGHT_GRAY          = RGBColor(0xF0, 0xF0, 0xF2)
WHITE               = RGBColor(0xFF, 0xFF, 0xFF)

# --- presentation geometry ---------------------------------------------------
SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)

prs = Presentation()
prs.slide_width = SLIDE_W
prs.slide_height = SLIDE_H

BLANK = prs.slide_layouts[6]  # blank layout


# --- helpers -----------------------------------------------------------------
def add_rect(slide, left, top, width, height, fill_color, line_color=None):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    if line_color is None:
        shape.line.fill.background()
    else:
        shape.line.color.rgb = line_color
    return shape


def add_text_box(slide, left, top, width, height, text, *,
                 size=18, bold=False, color=CHARCOAL, align=PP_ALIGN.LEFT,
                 anchor=MSO_ANCHOR.TOP, font="Calibri"):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.05)
    tf.margin_right = Inches(0.05)
    tf.margin_top = Inches(0.02)
    tf.margin_bottom = Inches(0.02)
    tf.vertical_anchor = anchor
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = font
    return tb


def add_bullets(slide, left, top, width, height, items,
                *, size=16, color=CHARCOAL, bullet_color=NVIDIA_GREEN,
                space_after=4, font="Calibri"):
    """Add a vertical list of bullet items, each prefixed with a green ▶."""
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.05)
    tf.margin_right = Inches(0.05)

    for i, item in enumerate(items):
        if isinstance(item, str):
            text, sub = item, []
        else:
            text, sub = item

        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        p.space_after = Pt(space_after)

        marker = p.add_run()
        marker.text = "▶  "
        marker.font.size = Pt(size)
        marker.font.color.rgb = bullet_color
        marker.font.bold = True
        marker.font.name = font

        body = p.add_run()
        body.text = text
        body.font.size = Pt(size)
        body.font.color.rgb = color
        body.font.name = font

        for s in sub:
            sp = tf.add_paragraph()
            sp.alignment = PP_ALIGN.LEFT
            sp.space_after = Pt(2)
            sp.level = 1
            sm = sp.add_run()
            sm.text = "·  "
            sm.font.size = Pt(size - 2)
            sm.font.color.rgb = NVIDIA_GREEN_DARK
            sm.font.bold = True
            sm.font.name = font
            sb = sp.add_run()
            sb.text = s
            sb.font.size = Pt(size - 2)
            sb.font.color.rgb = color
            sb.font.name = font

    return tb


def add_title_bar(slide, title_text):
    """Hacettepe-red title bar across the top with white title text."""
    bar = add_rect(slide, 0, 0, SLIDE_W, Inches(0.85), HACETTEPE_RED)
    tf = bar.text_frame
    tf.margin_left = Inches(0.4)
    tf.margin_right = Inches(0.4)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    run = p.add_run()
    run.text = title_text
    run.font.size = Pt(24)
    run.font.bold = True
    run.font.color.rgb = WHITE
    run.font.name = "Calibri"
    return bar


def add_footer(slide, page_num, total):
    bar = add_rect(slide, 0, Inches(7.1), SLIDE_W, Inches(0.4), HACETTEPE_RED_DARK)
    tf = bar.text_frame
    tf.margin_left = Inches(0.4)
    tf.margin_right = Inches(0.4)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    run = p.add_run()
    run.text = "CMP 674 · Parallel Computing with GPUs · Hacettepe University"
    run.font.size = Pt(11)
    run.font.color.rgb = WHITE
    run.font.name = "Calibri"

    # Page number on right
    tb = slide.shapes.add_textbox(Inches(11.5), Inches(7.1), Inches(1.7), Inches(0.4))
    tf2 = tb.text_frame
    tf2.margin_right = Inches(0.4)
    tf2.vertical_anchor = MSO_ANCHOR.MIDDLE
    p2 = tf2.paragraphs[0]
    p2.alignment = PP_ALIGN.RIGHT
    r2 = p2.add_run()
    r2.text = f"{page_num} / {total}"
    r2.font.size = Pt(11)
    r2.font.color.rgb = WHITE
    r2.font.name = "Calibri"


def add_card(slide, left, top, width, height, header, body_lines,
             header_color=HACETTEPE_RED, body_color=LIGHT_GRAY,
             header_text_color=WHITE, body_text_color=CHARCOAL,
             body_size=14, header_size=14):
    """A two-section card: colored header band with title, body with bullets."""
    header_h = Inches(0.45)
    add_rect(slide, left, top, width, header_h, header_color)
    add_rect(slide, left, top + header_h, width, height - header_h, body_color)

    # header text
    tb = slide.shapes.add_textbox(left, top, width, header_h)
    tf = tb.text_frame
    tf.margin_left = Inches(0.15)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    r = p.add_run()
    r.text = header
    r.font.size = Pt(header_size)
    r.font.bold = True
    r.font.color.rgb = header_text_color
    r.font.name = "Calibri"

    # body bullets
    tb2 = slide.shapes.add_textbox(left + Inches(0.15), top + header_h + Inches(0.05),
                                   width - Inches(0.3), height - header_h - Inches(0.1))
    tf2 = tb2.text_frame
    tf2.word_wrap = True
    for i, line in enumerate(body_lines):
        p = tf2.paragraphs[0] if i == 0 else tf2.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        p.space_after = Pt(2)
        marker = p.add_run()
        marker.text = "▶  "
        marker.font.size = Pt(body_size)
        marker.font.color.rgb = NVIDIA_GREEN
        marker.font.bold = True
        body = p.add_run()
        body.text = line
        body.font.size = Pt(body_size)
        body.font.color.rgb = body_text_color
        body.font.name = "Calibri"


# --- slides ------------------------------------------------------------------
TOTAL = 10  # filled in below


def slide_title():
    slide = prs.slides.add_slide(BLANK)
    # Top red band
    add_rect(slide, 0, 0, SLIDE_W, Inches(1.6), HACETTEPE_RED)
    # Bottom green strip
    add_rect(slide, 0, Inches(6.9), SLIDE_W, Inches(0.6), NVIDIA_GREEN)

    # Course label top-left
    add_text_box(slide, Inches(0.5), Inches(0.45), Inches(7), Inches(0.6),
                 "CMP 674 — Parallel Computing with GPUs",
                 size=18, bold=True, color=WHITE)
    # Hacettepe label top-right
    add_text_box(slide, Inches(8), Inches(0.45), Inches(5), Inches(0.6),
                 "Hacettepe University · Computer Engineering",
                 size=16, color=WHITE, align=PP_ALIGN.RIGHT)

    # Title
    add_text_box(slide, Inches(1), Inches(2.4), Inches(11.3), Inches(1.5),
                 "Attention Performance Comparison Analysis",
                 size=44, bold=True, color=HACETTEPE_RED, align=PP_ALIGN.CENTER)
    add_text_box(slide, Inches(1), Inches(3.5), Inches(11.3), Inches(0.7),
                 "on Blackwell-Architecture GPUs",
                 size=36, bold=True, color=HACETTEPE_RED, align=PP_ALIGN.CENTER)

    # Subtitle
    add_text_box(slide, Inches(1), Inches(4.6), Inches(11.3), Inches(0.6),
                 "A Five-Variant Pareto Replication Study from FP32 to NVFP4 on the RTX 5080",
                 size=20, color=CHARCOAL, align=PP_ALIGN.CENTER)
    add_text_box(slide, Inches(1), Inches(5.2), Inches(11.3), Inches(0.6),
                 "Course project · Spring 2026",
                 size=16, color=NVIDIA_GREEN_DARK, align=PP_ALIGN.CENTER)

    # Repo
    add_text_box(slide, Inches(1), Inches(6.0), Inches(11.3), Inches(0.5),
                 "github.com/canoztas/cmp674-attention-blackwell",
                 size=14, color=CHARCOAL, align=PP_ALIGN.CENTER, font="Consolas")


def slide_introduction():
    slide = prs.slides.add_slide(BLANK)
    add_title_bar(slide, "Introduction · Attention is the LLM Bottleneck")

    add_text_box(slide, Inches(0.5), Inches(1.1), Inches(12.3), Inches(0.6),
                 "Modern Large Language Models are bottlenecked by the attention operation.",
                 size=18, bold=True, color=HACETTEPE_RED)

    add_bullets(slide, Inches(0.5), Inches(1.8), Inches(12.3), Inches(2.5), [
        ("Self-attention computes softmax(QKᵀ/√D)·V — quadratic in sequence length N.",
         ["For each query token, every other token contributes to the softmax — O(N²) work and O(N²) memory.",
          "The N×N score matrix S = QKᵀ is the central object that breaks long-context inference."]),
        ("On every transformer forward pass, attention dominates both compute and memory traffic.",
         ["Two matmuls (QKᵀ and PV) carry the FLOPs.",
          "S and P (attention probabilities) carry the HBM traffic; both are N×N."]),
    ], size=15)

    # Alert block
    add_card(slide, Inches(0.5), Inches(4.7), Inches(12.3), Inches(2.0),
             "Why this matters at the systems level",
             ["At seq_len = 8192, batch = 2, heads = 8, the score matrix alone is ≈ 4 GB of HBM traffic per forward pass.",
              "At seq_len = 16384 it is ≈ 17 GB — does not fit on a 16 GB consumer card at all.",
              "Long-context inference is therefore not an algorithm problem; it is a kernel-and-memory problem."],
             header_color=NVIDIA_GREEN_DARK, body_color=NVIDIA_GREEN_LIGHT, body_size=14)

    add_footer(slide, 2, TOTAL)


def slide_gpu_obstacles():
    slide = prs.slides.add_slide(BLANK)
    add_title_bar(slide, "Memory and Compute Obstacles in LLM Training on GPUs")

    add_text_box(slide, Inches(0.5), Inches(1.1), Inches(12.3), Inches(0.5),
                 "Three concrete walls every attention kernel hits on contemporary GPUs:",
                 size=16, bold=True, color=CHARCOAL)

    # Three cards side by side
    card_w = Inches(4.0)
    card_h = Inches(3.6)
    top = Inches(1.7)

    add_card(slide, Inches(0.5), top, card_w, card_h,
             "1. The Memory Wall",
             ["Materialized N×N score matrix scales O(N²) in HBM.",
              "VRAM ceiling: 16 GB on RTX 5080.",
              "HBM bandwidth peak: ≈ 960 GB/s — moving the score matrix dominates total traffic.",
              "Causes OOMs at long context even before compute matters."],
             body_size=12)

    add_card(slide, Inches(4.65), top, card_w, card_h,
             "2. The Compute Wall",
             ["Two matmuls (QKᵀ and PV) define the FLOP budget.",
              "Tensor Cores (5th-gen on Blackwell) accelerate matmuls only at supported precisions: FP16, FP8, FP4.",
              "FP32 ALU is ≈ 50× slower; using it kills throughput.",
              "Each precision halves the storage but narrows the dynamic range."],
             body_size=12)

    add_card(slide, Inches(8.8), top, card_w, card_h,
             "3. Software-Stack Wall",
             ["Vendor SDPA backend dispatch is build-dependent.",
              "Some PyTorch wheels (e.g. Windows + sm_120) silently fall back to MATH backend — no FlashAttention.",
              "FA3 backend not yet first-classed on consumer Blackwell (vLLM #14452).",
              "TransformerEngine has explicit FP8-attention skip lists (TE #2186)."],
             body_size=12)

    # Bottom takeaway
    add_text_box(slide, Inches(0.5), Inches(5.5), Inches(12.3), Inches(1.5),
                 "Therefore: training and inference performance on consumer Blackwell is determined by "
                 "(a) avoiding HBM materialization, (b) engaging Tensor Cores at the narrowest viable precision, "
                 "and (c) explicitly verifying the vendor backend the framework actually dispatches.",
                 size=14, color=CHARCOAL, anchor=MSO_ANCHOR.TOP)

    add_footer(slide, 3, TOTAL)


def slide_related_work_1():
    slide = prs.slides.add_slide(BLANK)
    add_title_bar(slide, "Related Work (1/2) · Algorithmic & Mixed-Precision Attention")

    # FlashAttention lineage
    add_card(slide, Inches(0.5), Inches(1.1), Inches(6.1), Inches(2.7),
             "FlashAttention lineage",
             ["FlashAttention (Dao et al., 2022): IO-aware tiling + online softmax; eliminates N×N materialization.",
              "FlashAttention-2 (Dao, 2023): improved warp-level work partitioning.",
              "FlashAttention-3 (Shah et al., 2024): Hopper-specific TMA pipelining + FP8 — does NOT transfer to consumer Blackwell (no TMA on sm_120a).",
              "FlexAttention (Dong et al., 2025): compiler-level alternative via torch.compile."],
             body_size=12)

    # Mixed-precision attention
    add_card(slide, Inches(6.75), Inches(1.1), Inches(6.1), Inches(2.7),
             "Mixed-precision attention (positive)",
             ["Micikevicius et al. (2022): defined E4M3 / E5M2 FP8 formats; foundational for FP8 attention.",
              "SageAttention (Zhang et al., ICLR 2025): first systematic FP8 attention; ≈ 2.1× over FA-2 on Hopper.",
              "SageAttention3 (Zhang et al., NeurIPS 2025): NVFP4 attention on RTX 5090 using hardware-microscaled mxf4nvf4.m16n8k64 — closest related work."],
             body_size=12)

    # Failure modes
    add_card(slide, Inches(0.5), Inches(4.0), Inches(12.4), Inches(3.0),
             "Failure-mode literature (where our negative result connects)",
             ["Attn-QAT (Zhang et al., 2026): first FP4 quantization-aware training — naive forward+backward recipe is unstable.",
              "Qiu & Yao (ICLR 2026): mechanistic explanation of low-precision FlashAttention training collapse — low-rank representation + biased rounding accumulation.",
              "SnapMLA (Zhang et al., 2026): FP8 attention sub-types in MLA decoding require per-component precision retention.",
              "Our V4 (FP4 with software microscaling) extends this lineage on the inference axis: a quantitative inference-time cost the training-stability work above does not address."],
             body_size=13)

    add_footer(slide, 4, TOTAL)


def slide_related_work_2():
    slide = prs.slides.add_slide(BLANK)
    add_title_bar(slide, "Related Work (2/2) · Consumer-Blackwell Ecosystem")

    add_text_box(slide, Inches(0.5), Inches(1.1), Inches(12.3), Inches(0.6),
                 "The consumer-Blackwell software stack is genuinely immature — a pattern across multiple projects:",
                 size=15, bold=True, color=HACETTEPE_RED)

    # Issue tracker evidence (left)
    add_card(slide, Inches(0.5), Inches(1.85), Inches(7.5), Inches(3.6),
             "Public issue trackers documenting the immaturity",
             ["vLLM #14452 — FA-3 backend not yet working with Blackwell; specific CUDA 12.8 + container required.",
              "vLLM #29030 — vLLM 0.11.1 + CUDA 13 fails with undefined symbol; SM 12.0 (consumer) is NOT a superset of SM 10.0 (datacenter).",
              "vLLM #30707 — RTX 5080 + NVFP4 model rejected by overly-aggressive 16 GB pre-flight check, despite fitting.",
              "TransformerEngine #2186 — FP8 attention test-suite skip list (THD layout, bias, sliding window, MLA-CP).",
              "FlashInfer/vLLM #35138 — Blackwell FP8 attention correctness regression; recommended fix is backend switch."],
             body_size=11)

    # Adjacent papers and our position (right)
    add_card(slide, Inches(8.15), Inches(1.85), Inches(4.7), Inches(3.6),
             "Adjacent consumer-Blackwell papers",
             ["SlideSparse (Shao et al., 2026) — uses RTX 5080 as a sparse-GEMM evaluation platform, NOT as an attention target.",
              "Knoop & Holtmann (2026) — consumer-Blackwell LLM deployment study; covers RTX 5060 Ti, 5070 Ti, 5090 — explicitly OMITS the 5080.",
              "Triton fused-attention tutorial — Blackwell branch gated on a `major == 10` runtime check; sm_120a not first-classed."],
             body_size=11)

    # Our SDPA-dispatch finding box at bottom
    add_card(slide, Inches(0.5), Inches(5.6), Inches(12.4), Inches(1.5),
             "Our finding adds one data point to this pattern",
             ["PyTorch 2.9.1+cu128 Windows wheel dispatches SDPA exclusively to the MATH backend on sm_120 / FP16 / non-causal — Flash, cuDNN, MemEff are all runtime-disabled. The vendor 'SDPA baseline' is therefore a moving target."],
             header_color=NVIDIA_GREEN_DARK, body_color=NVIDIA_GREEN_LIGHT, body_size=12)

    add_footer(slide, 5, TOTAL)


def slide_bridge():
    slide = prs.slides.add_slide(BLANK)
    add_title_bar(slide, "From Related Work to Our Experiments")

    add_text_box(slide, Inches(0.5), Inches(1.1), Inches(12.3), Inches(0.6),
                 "The gap and our contribution:",
                 size=18, bold=True, color=HACETTEPE_RED)

    add_bullets(slide, Inches(0.5), Inches(1.8), Inches(12.3), Inches(2.4), [
        ("RTX 5080 (mid-tier consumer Blackwell, sm_120a) is essentially absent from kernel-level attention literature.",
         ["Same compute capability as RTX 5090, but half the SMs, narrower memory bus, tighter VRAM.",
          "Adjacent papers either cover other SKUs (5060 Ti, 5070 Ti, 5090) or use 5080 for non-attention workloads (sparse GEMM)."]),
        ("Existing FP4 attention work (SageAttention3, Attn-QAT) presents a single optimized kernel — per-step contributions of fusion / FP8 / FP4 are not separable.",
         []),
    ], size=14)

    add_card(slide, Inches(0.5), Inches(4.4), Inches(12.4), Inches(2.6),
             "Our experimental approach: a controlled, five-point ladder ablation",
             ["Five hand-rolled CUDA kernels share ONE algorithmic skeleton (online softmax, single-kernel fusion, register-resident accumulator from V3 onward).",
              "Each kernel changes EXACTLY ONE design choice against its predecessor — fusion, precision, or microscaling granularity.",
              "Result: each step's contribution to latency / memory / accuracy is measurable in isolation, on the same hardware/software stack.",
              "All five variants ship as a reproducible artifact — 495 passing correctness tests against an FP32 reference."],
             header_color=NVIDIA_GREEN_DARK, body_color=NVIDIA_GREEN_LIGHT, body_size=14)

    add_footer(slide, 6, TOTAL)


def slide_v0_v1_v2():
    slide = prs.slides.add_slide(BLANK)
    add_title_bar(slide, "Pipelines V0 → V1 → V2 · What changed, what we noticed")

    # Three columns
    col_w = Inches(4.05)
    col_h = Inches(5.0)
    top = Inches(1.1)

    add_card(slide, Inches(0.4), top, col_w, col_h,
             "V0 — FP32 naive (oracle)",
             ["Three separate kernel launches: QKᵀ → softmax → PV.",
              "Materializes full N×N score matrix S in HBM.",
              "All FP32 ALU — Tensor Cores not engaged.",
              "Role: correctness reference for V1..V4.",
              "",
              "Numbers (N=2048, D=128):",
              "  Latency: ≈ 4500 ms",
              "  Memory: ≈ 640 MB",
              "  Tensor Core insns: 0",
              "",
              "Lesson: naive attention is unworkable past tiny N — optimization is mandatory."],
             header_color=HACETTEPE_RED, body_size=11)

    add_card(slide, Inches(4.65), top, col_w, col_h,
             "V1 — FP16 tiled, WMMA",
             ["Same three-launch structure as V0, but FP16 inputs.",
              "Both matmuls go through nvcuda::wmma → HMMA.16816.F32 Tensor Cores.",
              "Trick #1: QKᵀ without explicit transpose — load K as col-major operand.",
              "Trick #2: single dynamic SMEM allocation (48 KB cap budget).",
              "Still materializes S, P in HBM.",
              "",
              "Numbers (N=8192, D=128):",
              "  Latency: 77.1 ms (≈ 60× V0)",
              "  Memory: 2176 MB",
              "  Rel L2 vs FP32: 6e-4",
              "",
              "Lesson: Tensor Cores alone give 60× — fusion still missing."],
             header_color=HACETTEPE_RED, body_size=11)

    add_card(slide, Inches(8.9), top, col_w, col_h,
             "V2 — FP16 fused (FlashAttention)",
             ["Single fused kernel + online softmax recurrence.",
              "Streams over K, V tiles — N×N matrix never materialized.",
              "Per-block SMEM = 96 KB (over default cap; needs cudaFuncSetAttribute).",
              "Two correctness traps: first iteration (m=−∞) and fully-masked rows (NaN guards required).",
              "",
              "Numbers (N=8192, D=128):",
              "  Latency: 60.9 ms",
              "  Memory: 128 MB ← 17× drop from V1",
              "  Rel L2 vs FP32: 3e-4",
              "",
              "★ Finding 1: 17× memory reduction with NO precision change."],
             header_color=NVIDIA_GREEN_DARK, body_size=11)

    add_text_box(slide, Inches(0.5), Inches(6.3), Inches(12.3), Inches(0.7),
                 "Largest single memory finding in the sweep is at the V1→V2 step — algorithmic fusion, not precision, breaks the memory wall.",
                 size=14, bold=True, color=NVIDIA_GREEN_DARK, align=PP_ALIGN.CENTER)

    add_footer(slide, 7, TOTAL)


def slide_v3_v4():
    slide = prs.slides.add_slide(BLANK)
    add_title_bar(slide, "Pipelines V3 & V4 · FP8 win, NVFP4 negative result")

    # V3 column
    add_card(slide, Inches(0.4), Inches(1.1), Inches(6.2), Inches(5.0),
             "V3 — FP8 fused with hand-rolled mma.sync (the win)",
             ["Replaces FP16 wmma with FP8 (E4M3) mma.sync.m16n8k32 via inline PTX.",
              "PTX fully specifies per-thread fragment layout → output accumulator can stay in registers across the K/V loop (V2 had to round-trip through SMEM).",
              "Per-tile FP8 scaling for Q, K, V, P; quantize cooperatively at load time.",
              "Why not CUTLASS? `77_blackwell_fmha` reference is gated to sm_100a/sm_103a only — no consumer-Blackwell FMHA reference exists.",
              "",
              "Numbers (N=8192, D=128):",
              "  Latency: 25.8 ms  (2.4× V2)",
              "  Memory: 128 MB  · QMMA insns: 96",
              "  Rel L2 vs FP32: 5.3e-2",
              "",
              "★ Finding 2: V3 closes ≈ half the SDPA latency gap."],
             header_color=NVIDIA_GREEN_DARK, body_size=11)

    # V4 column
    add_card(slide, Inches(6.75), Inches(1.1), Inches(6.2), Inches(5.0),
             "V4 — NVFP4 with software microscaling (negative result)",
             ["Same kernel structure as V3, but FP4 (E2M1) via kind::f8f6f4.m16n8k32.",
              "FP4 dynamic range is too narrow → per-row, per-K-block (block=32) FP32 scales applied at the accumulator before each mma.",
              "Each mma adds ≈ 12 register ops; across 96 mmas/iter ≈ 1100 ops/thread/iter overhead.",
              "Hardware-microscaled mxf4nvf4.m16n8k64 would internalize the cost — but its scattered-fragment layout is incompatible with row-major SMEM (multi-week port).",
              "",
              "Numbers (N=8192, D=128):",
              "  Latency: 33.4 ms  (0.77× V3 — SLOWER)",
              "  Memory: 128 MB  · QMMA insns: 96",
              "  Rel L2 vs FP32: 0.21",
              "",
              "★ Finding 3: software microscaling overhead > FP4 throughput advantage."],
             header_color=HACETTEPE_RED, body_size=11)

    # Bottom call-out: future work
    add_card(slide, Inches(0.5), Inches(6.25), Inches(12.4), Inches(0.85),
             "Where we go next",
             ["Lift V4 onto the hardware-microscaled mxf4nvf4.m16n8k64 path via ldmatrix-style swizzled SMEM and a CUTLASS-mediated load — expected to invert V4 vs V3."],
             header_color=NVIDIA_GREEN, body_color=NVIDIA_GREEN_LIGHT,
             header_text_color=WHITE, body_size=12)

    add_footer(slide, 8, TOTAL)


def slide_references():
    slide = prs.slides.add_slide(BLANK)
    add_title_bar(slide, "References")

    refs = [
        "Dao et al. (NeurIPS 2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. arXiv:2205.14135.",
        "Dao (ICLR 2024). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. arXiv:2307.08691.",
        "Shah et al. (NeurIPS 2024). FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-Precision. arXiv:2407.08608.",
        "Micikevicius et al. (2022). FP8 Formats for Deep Learning. arXiv:2209.05433.",
        "Zhang et al. (ICLR 2025). SageAttention: Accurate 8-Bit Attention for Plug-and-Play Inference Acceleration. arXiv:2410.02367.",
        "Zhang et al. (NeurIPS 2025). SageAttention3: Microscaling FP4 Attention for Inference. arXiv:2505.11594.",
        "Zhang et al. (2026). Attn-QAT: 4-Bit Attention with Quantization-Aware Training. arXiv:2603.00040.",
        "Qiu & Yao (ICLR 2026). Why Low-Precision Transformer Training Fails: An Analysis on Flash Attention. arXiv:2510.04212.",
        "Zhang et al. (2026). SnapMLA: Efficient Long-Context MLA Decoding via Hardware-Aware FP8 Quantized Pipelining. arXiv:2602.10718.",
        "Dong et al. (MLSys 2025). Flex Attention: A Programming Model for Generating Optimized Attention Kernels. arXiv:2412.05496.",
        "Prabhu et al. (ASPLOS 2025). vAttention: Dynamic Memory Management for Serving LLMs. arXiv:2405.04437.",
        "Shao et al. (2026). SlideSparse: Fast and Flexible (2N-2):2N Structured Sparsity. arXiv:2603.05232.",
        "Knoop & Holtmann (2026). Private LLM Inference on Consumer Blackwell GPUs. arXiv:2601.09527.",
        "NVIDIA (2025). RTX Blackwell GPU Architecture Whitepaper.",
        "NVIDIA (2024). CUTLASS: CUDA Templates for Linear Algebra Subroutines, v4.4.2.",
        "NVIDIA (2024). Parallel Thread Execution ISA Version 8.7.",
        "Pineau et al. (JMLR 2021). Improving Reproducibility in Machine Learning Research. arXiv:2003.12206.",
        "Kalibera & Jones (ISMM 2013). Rigorous Benchmarking in Reasonable Time. doi:10.1145/2464157.2464160.",
    ]

    tb = slide.shapes.add_textbox(Inches(0.5), Inches(1.1), Inches(12.3), Inches(5.8))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, r in enumerate(refs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        p.space_after = Pt(2)
        idx = p.add_run()
        idx.text = f"[{i+1}]  "
        idx.font.size = Pt(11)
        idx.font.color.rgb = NVIDIA_GREEN_DARK
        idx.font.bold = True
        body = p.add_run()
        body.text = r
        body.font.size = Pt(11)
        body.font.color.rgb = CHARCOAL

    add_footer(slide, 9, TOTAL)


def slide_thanks():
    slide = prs.slides.add_slide(BLANK)
    # Decorative bands
    add_rect(slide, 0, 0, SLIDE_W, Inches(0.85), HACETTEPE_RED)
    add_rect(slide, 0, Inches(6.9), SLIDE_W, Inches(0.6), NVIDIA_GREEN)

    add_text_box(slide, Inches(1), Inches(2.0), Inches(11.3), Inches(1.5),
                 "Questions?",
                 size=72, bold=True, color=HACETTEPE_RED, align=PP_ALIGN.CENTER)

    add_text_box(slide, Inches(1), Inches(3.6), Inches(11.3), Inches(0.8),
                 "Code, paper, slides:",
                 size=22, color=CHARCOAL, align=PP_ALIGN.CENTER)
    add_text_box(slide, Inches(1), Inches(4.2), Inches(11.3), Inches(0.8),
                 "github.com/canoztas/cmp674-attention-blackwell",
                 size=22, bold=True, color=NVIDIA_GREEN_DARK, align=PP_ALIGN.CENTER, font="Consolas")

    add_text_box(slide, Inches(1), Inches(5.8), Inches(11.3), Inches(0.6),
                 "CMP 674 — Parallel Computing with GPUs — Hacettepe University · Spring 2026",
                 size=14, color=CHARCOAL, align=PP_ALIGN.CENTER)


# --- build the deck ----------------------------------------------------------
slide_title()
slide_introduction()
slide_gpu_obstacles()
slide_related_work_1()
slide_related_work_2()
slide_bridge()
slide_v0_v1_v2()
slide_v3_v4()
slide_references()
slide_thanks()

assert len(prs.slides) == TOTAL, f"expected {TOTAL} slides, got {len(prs.slides)}"

# --- save --------------------------------------------------------------------
out_path = Path(__file__).resolve().parent / "Attention_Performance_Comparison_Analysis.pptx"
prs.save(out_path)
print(f"wrote {out_path} ({out_path.stat().st_size / 1024:.1f} KB, {len(prs.slides)} slides)")
