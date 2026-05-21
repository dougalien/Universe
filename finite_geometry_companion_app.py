import os
import json
import re
from pathlib import Path
from html import unescape

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch

try:
    import requests
except ImportError:
    requests = None


OUTPUT_DIR = Path("model_outputs")
F_VIS = 0.390906
Z_CMB = 1089.92
Z_T = 34.5754
A_T = np.log1p(Z_T)
DEFAULT_G0 = 1.0
DEFAULT_DB = 0.5
DEFAULT_ETA = 1.0
DEFAULT_U = 0.2
DEFAULT_D_N = 0.8
DEFAULT_H0_BASELINE = 70.0
LOCAL_H0 = 73.0
EARLY_H0 = 67.4
DEFAULT_PROJECTION_STRENGTH = 0.627714
DEFAULT_TRANSITION_SHARPNESS = 8.0
C_KM_S = 299792.458
H0_STANDARD = 67.4
OMEGA_M = 0.315
OMEGA_LAMBDA = 0.685
MPC_TO_KM = 3.085677581491367e19
SEC_PER_GYR = 3.15576e16
MPC_TO_GLY = 0.00326156
LIGHT_PRESETS = {
    "Nearby galaxy": 0.1,
    "Distant galaxy": 2.0,
    "JWST-era galaxy": 8.0,
    "Boundary transition": 34.5754,
    "CMB-side view": 1089.92,
    "Custom redshift": None,
}
REGION_COLORS = {
    "Interior": "#dce9f8",
    "Approach": "#e9f2df",
    "Boundary": "#f5e4bf",
    "Compressed far side": "#f7ddd5",
    "CMB-side": "#e2d9ef",
}
PRESET_GUIDANCE = {
    "Nearby galaxy": "Nearby light stays in the interior part of the picture, so the finite-geometry reading stays close to the nearby value.",
    "Distant galaxy": "The light has traveled farther through the finite geometry, but it has not reached the strongest boundary region.",
    "JWST-era galaxy": "This light sits near the region where projection effects start to matter.",
    "Boundary transition": "This is the part of the model where the reading changes fastest.",
    "Compressed far side": "This light is read through the compressed far-side part of the geometry.",
    "CMB-side view": "This light sits at the most compressed end of the display.",
    "Custom redshift": "This custom redshift lets you compare how one observed signal gets two different quantitative readings.",
}
PRESET_READING = {
    "Nearby galaxy": "mostly nearby behavior",
    "Distant galaxy": "more path response",
    "JWST-era galaxy": "projection begins to matter",
    "Boundary transition": "fastest model change",
    "CMB-side view": "strongest compressed view",
    "Custom redshift": "custom mapped reading",
}
BOOK_CONTEXT = """
A Finite Universe presents a finite deformable geometry model in which redshift, distance, gravity, shape, and boundary behavior are treated as linked observational registers.

The central idea is that cosmological redshift can be read as accumulated path response through a finite geometric domain rather than only as universal expansion.

Key model ideas:
- Observed redshift z is kept as the measured quantity.
- A = ln(1 + z) is accumulated path response.
- X = A / A_t places light relative to the transition scale.
- z_t = 34.5754 marks the boundary-transition target.
- P_star(z) = 1 - 0.627714*z**8/(z**8 + 8**8) is the projection law used by the app.
- Nearby/interior light maps near the nearby H0 value.
- CMB-side light maps near the early-universe H0 value.
- The finite model reads far-side light through path response and projection, rather than treating every redshift comparison as one universal expansion scale.
- The companion app shows how the same selected light is read in the expansion picture and in the finite-geometry picture.
- The Prediction Board compares expansion-picture readings with finite-geometry readings for the selected light.
"""

BOOK_CANDIDATE_FILES = [
    "index.html",
    "chapter_1.html",
    "chapter_2.html",
    "chapter_3.html",
    "chapter_4.html",
    "quantitative_appendix.html",
    "measurement_registers.html",
    "note_from_author.html",
    "references.html",
    "guidebook.txt",
]

THEME = {
    "ink": "#182230",
    "muted": "#5e6b7a",
    "paper": "#fbf8f1",
    "panel": "#ffffff",
    "blue": "#09233f",
    "blue2": "#123a63",
    "gold": "#c89b3c",
    "line": "#d9d0bd",
    "soft": "#eef2f6",
}


def standard_baseline(distance, H0):
    """Standard expansion-style apparent H0 baseline."""
    return np.full_like(distance, H0, dtype=float)


def accumulated_response_from_z(z):
    """Conceptual accumulated path response A = ln(1 + z)."""
    return np.log1p(z)


def finite_depth_chi(z):
    return F_VIS * accumulated_response_from_z(z) / np.log1p(Z_CMB)


def transition_coordinate_x(z):
    return accumulated_response_from_z(z) / A_T


def local_response(distance, g0, db):
    return g0 / (1 + distance / db)


def projection_law(z):
    return 1 - 0.627714 * z**8 / (z**8 + 8**8)


def finite_h0_from_projection(z):
    P = projection_law(z)
    P_CMB = projection_law(Z_CMB)
    compression_fraction = np.clip((1 - P) / (1 - P_CMB), 0, 1)
    return LOCAL_H0 - (LOCAL_H0 - EARLY_H0) * compression_fraction


def adjustable_projection(z, projection_strength, transition_sharpness):
    sharpness = max(transition_sharpness, 0.2)
    return 1 - projection_strength * z**sharpness / (z**sharpness + 8**sharpness)


def E_z(z):
    z_arr = np.asarray(z, dtype=float)
    return np.sqrt(OMEGA_M * (1.0 + z_arr) ** 3 + OMEGA_LAMBDA)


def comoving_distance_mpc(z, n_steps=2200):
    z_value = float(max(z, 0.0))
    if z_value == 0.0:
        return 0.0
    grid_steps = int(max(600, min(30000, n_steps)))
    z_grid = np.linspace(0.0, z_value, grid_steps)
    integral = np.trapz(1.0 / E_z(z_grid), z_grid)
    return (C_KM_S / H0_STANDARD) * integral


def luminosity_distance_mpc(z):
    z_value = float(max(z, 0.0))
    return (1.0 + z_value) * comoving_distance_mpc(z_value)


def angular_diameter_distance_mpc(z):
    z_value = float(max(z, 0.0))
    return comoving_distance_mpc(z_value) / (1.0 + z_value)


def lookback_time_gyr(z, n_steps=2200):
    z_value = float(max(z, 0.0))
    if z_value == 0.0:
        return 0.0
    grid_steps = int(max(600, min(30000, n_steps)))
    z_grid = np.linspace(0.0, z_value, grid_steps)
    integrand = 1.0 / ((1.0 + z_grid) * E_z(z_grid))
    integral = np.trapz(integrand, z_grid)
    H0_s = H0_STANDARD / MPC_TO_KM
    return integral / H0_s / SEC_PER_GYR


def kpc_per_arcsec(z):
    arcsec_to_rad = np.pi / 648000.0
    d_a_mpc = angular_diameter_distance_mpc(z)
    return d_a_mpc * 1000.0 * arcsec_to_rad


def physical_size_kpc_from_arcsec(z, arcsec):
    return kpc_per_arcsec(z) * float(arcsec)


def mpc_to_million_light_years(mpc):
    return float(mpc) * MPC_TO_GLY * 1000.0


def mpc_to_billion_light_years(mpc):
    return float(mpc) * MPC_TO_GLY


def format_distance_human(mpc):
    million_ly = mpc_to_million_light_years(mpc)
    if million_ly >= 1000.0:
        return f"{mpc_to_billion_light_years(mpc):,.2f} billion light-years"
    return f"{million_ly:,.1f} million light-years"


def _format_sig(value, sig=3):
    if value == 0:
        return "0"
    return f"{value:.{sig}g}"


def format_distance_scale_mpc(mpc):
    million_ly = mpc_to_million_light_years(mpc)
    if million_ly >= 1000.0:
        return f"{_format_sig(million_ly / 1000.0)} billion light-years"
    return f"{_format_sig(million_ly)} million light-years"


def format_size_kpc(kpc):
    abs_kpc = abs(float(kpc))
    if abs_kpc >= 1000.0:
        return f"{_format_sig(kpc / 1000.0)} Mpc"
    if abs_kpc >= 0.1:
        return f"{_format_sig(kpc)} kpc"
    return f"{_format_sig(kpc * 1000.0)} pc"


def format_time_gyr(gyr):
    abs_gyr = abs(float(gyr))
    if abs_gyr >= 0.1:
        return f"{_format_sig(gyr)} Gyr"
    return f"{_format_sig(gyr * 1000.0)} Myr"


def format_ratio(value):
    return _format_sig(float(value), sig=3)


def response_engine(X, eta, u, D_n):
    """Bounded companion-model placeholder for the finite response engine."""
    X_term = np.tanh(2.5 * X)
    eta_term = eta / (1 + eta)
    u_term = 1 - np.exp(-3 * np.clip(u, 0, 1))
    D_term = D_n / (1 + D_n)
    return 0.40 * X_term + 0.25 * eta_term + 0.20 * u_term + 0.15 * D_term


def full_finite_response(z_grid, g0, db, projection_strength, sharpness, eta, u, D_n):
    A_grid = accumulated_response_from_z(z_grid)
    X_grid = A_grid / A_T
    chi_grid = finite_depth_chi(z_grid)
    G_grid = local_response(chi_grid, g0, db)
    D_grid = A_grid * G_grid
    C_grid = response_engine(X_grid, eta, u, D_n)
    P_grid = adjustable_projection(z_grid, projection_strength, sharpness)
    combined = D_grid * C_grid * P_grid
    return A_grid, X_grid, chi_grid, G_grid, D_grid, C_grid, P_grid, combined


def apparent_H0_curve(distance, H0_baseline, finite_response):
    """Apparent H0 after adding the finite-model response term."""
    return standard_baseline(distance, H0_baseline) + finite_response


def _clean_html_book_text(raw_text):
    without_script = re.sub(
        r"(?is)<(script|style|nav|noscript|header|footer|aside)[^>]*>.*?</\1>",
        " ",
        raw_text,
    )
    without_tags = re.sub(r"(?is)<[^>]+>", " ", without_script)
    text = unescape(without_tags)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _book_search_roots():
    cwd = Path.cwd()
    roots = [cwd / "book_context", cwd]
    parent = cwd.parent
    if parent:
        roots.append(parent / "web-book")
        roots.append(parent / "book_context")
        roots.append(parent)
    return roots


@st.cache_data(show_spinner=False)
def load_full_book_context():
    snippets = []
    seen = set()
    for root in _book_search_roots():
        if not root.exists() or not root.is_dir():
            continue
        for file_name in BOOK_CANDIDATE_FILES:
            candidate = root / file_name
            if not candidate.exists() or not candidate.is_file():
                continue
            resolved = str(candidate.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            try:
                raw = candidate.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            cleaned = (
                _clean_html_book_text(raw)
                if candidate.suffix.lower() in {".html", ".htm"}
                else re.sub(r"\s+", " ", raw).strip()
            )
            if cleaned:
                snippets.append(f"[{candidate.name}] {cleaned}")
    if snippets:
        return "\n\n".join(snippets)
    return BOOK_CONTEXT


def apply_book_theme():
    st.markdown(
        f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&family=Inter:wght@400;500;600;700&display=swap');
:root {{
  --ink: {THEME["ink"]};
  --muted: {THEME["muted"]};
  --paper: {THEME["paper"]};
  --panel: {THEME["panel"]};
  --blue: {THEME["blue"]};
  --blue-2: {THEME["blue2"]};
  --gold: {THEME["gold"]};
  --line: {THEME["line"]};
}}
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {{
  background: var(--paper);
  color: var(--ink);
  font-family: "Source Serif 4", Georgia, serif;
}}
[data-testid="stHeader"] {{
  background: linear-gradient(135deg, rgba(9,35,63,0.96), rgba(18,58,99,0.96));
  border-bottom: 3px solid var(--gold);
}}
[data-testid="collapsedControl"] {{
  z-index: 1002;
  opacity: 1;
  visibility: visible;
}}
h1, h2, h3 {{ color: var(--blue); }}
[data-testid="stSidebar"] {{
  background: #f6f1e6;
  border-right: 1px solid var(--line);
}}
[data-testid="stMetric"] {{
  background: var(--panel);
  border: 1px solid var(--line);
  border-top: 4px solid var(--gold);
  border-radius: 8px;
  padding: 0.42rem 0.58rem;
  min-height: 5.2rem;
}}
[data-testid="stMetricLabel"],
[data-testid="stMetricValue"] {{
  white-space: normal !important;
  overflow-wrap: anywhere;
  word-break: break-word;
  line-height: 1.2;
}}
[data-testid="stMetricLabel"] {{
  font-size: clamp(0.76rem, 0.70rem + 0.24vw, 0.93rem);
}}
[data-testid="stMetricValue"] {{
  font-size: clamp(1.00rem, 0.92rem + 0.42vw, 1.40rem);
}}
[data-testid="stMetricDelta"] {{
  white-space: normal !important;
  overflow-wrap: anywhere;
  font-size: clamp(0.70rem, 0.64rem + 0.22vw, 0.86rem);
}}
.afu-title-band {{
  background: linear-gradient(135deg, rgba(9,35,63,0.98), rgba(18,58,99,0.98));
  border: 1px solid var(--line);
  border-left: 6px solid var(--gold);
  border-radius: 10px;
  padding: 0.95rem 1rem;
  margin: 0.25rem 0 0.8rem 0;
}}
.afu-title-main {{
  color: #f6e8c8;
  font-family: "Source Serif 4", Georgia, serif;
  font-size: clamp(1.35rem, 1.1rem + 0.9vw, 2.0rem);
  line-height: 1.1;
  font-weight: 700;
}}
.afu-title-sub {{
  color: #e4edf6;
  font-family: Inter, system-ui, sans-serif;
  font-size: clamp(0.78rem, 0.72rem + 0.24vw, 0.95rem);
  letter-spacing: 0.02em;
  margin-top: 0.18rem;
}}
[data-testid="stInfo"] {{
  background: #fffaf0;
  border: 1px solid var(--line);
  border-left: 4px solid var(--gold);
  color: #263545;
}}
div[data-testid="stTable"] table {{
  border: 1px solid var(--line);
  background: white;
  border-collapse: collapse;
}}
div[data-testid="stTable"] th {{
  background: #f4ead5;
  color: var(--blue);
}}
div[data-testid="stTable"] th, div[data-testid="stTable"] td {{
  border-bottom: 1px solid var(--line);
}}
.stButton > button, .stDownloadButton > button, button[kind="primary"] {{
  background: #f6e2ae;
  border: 1px solid var(--gold);
  color: var(--blue);
  font-family: Inter, system-ui, sans-serif;
  font-weight: 700;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
  background: #f0d691;
  border-color: #b88926;
}}
[data-testid="stCaptionContainer"] {{ color: var(--muted); }}
</style>
""",
        unsafe_allow_html=True,
    )


def apply_plot_theme():
    plt.rcParams.update(
        {
            "figure.facecolor": THEME["panel"],
            "axes.facecolor": THEME["panel"],
            "axes.edgecolor": THEME["line"],
            "axes.labelcolor": THEME["ink"],
            "axes.titlecolor": THEME["blue"],
            "xtick.color": THEME["muted"],
            "ytick.color": THEME["muted"],
            "grid.color": THEME["line"],
            "font.family": ["Source Serif 4", "DejaVu Serif", "serif"],
        }
    )


def figure_file_name(page_name, figure_name, index):
    safe_page = page_name.lower().replace(" ", "_").replace("/", "_")
    safe_figure = figure_name.lower().replace(" ", "_").replace("/", "_")
    return f"{index:02d}_{safe_page}_{safe_figure}.png"


def export_figures(figures):
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_paths = []
    for index, (page_name, figure_name, fig) in enumerate(figures, start=1):
        output_path = OUTPUT_DIR / figure_file_name(page_name, figure_name, index)
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
        output_paths.append(output_path)
    return output_paths


def add_regime_zones(ax, ymin=None, ymax=None):
    spans = [
        (0.0, 0.6, "interior", REGION_COLORS["Interior"]),
        (0.6, 1.0, "approach", REGION_COLORS["Approach"]),
        (1.0, 1.35, "transition", REGION_COLORS["Boundary"]),
        (1.35, 2.0, "compressed side", REGION_COLORS["Compressed far side"]),
    ]
    for start, end, label, color in spans:
        ax.axvspan(start, end, color=color, alpha=0.08)
        y_text = ymin if ymin is not None else ax.get_ylim()[0]
        if ymax is not None:
            y_text = ymin + 0.06 * (ymax - ymin)
        ax.text((start + end) / 2, y_text, label, ha="center", fontsize=9)


def add_interpretation(text):
    st.info(text)


def regime_label(X):
    if X < 0.6:
        return "Interior"
    if X < 1.0:
        return "Approach"
    if X < 1.35:
        return "Transition"
    return "Compressed side"


def qualitative_response_label(value):
    if value < 0.35:
        return "Low response"
    if value < 1.0:
        return "Moderate response"
    return "Strong response"


def projection_label(P_value):
    compression = 1 - P_value
    if compression < 0.15:
        return "weak"
    if compression < 0.40:
        return "moderate"
    return "strong"


def display_position_from_z(z):
    X = transition_coordinate_x(z)
    x_cmb = transition_coordinate_x(Z_CMB)
    return float(np.clip(X / x_cmb, 0.02, 1.0))


def display_region_from_position(position):
    if position < 0.20:
        return "Interior"
    if position < 0.45:
        return "Approach"
    if position < 0.62:
        return "Boundary"
    if position < 0.88:
        return "Compressed far side"
    return "CMB-side"


def selected_path_point(z):
    t = display_position_from_z(z)
    x = -0.90 + 1.80 * t
    y = 0.18 * np.sin(np.pi * t) - 0.18
    return x, y, t


def build_universe_schematic(z, preset_name):
    point_x, point_y, _ = selected_path_point(z)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_aspect("equal")
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-0.85, 0.85)
    ax.axis("off")

    universe = Circle((0, 0), 1.0, facecolor="#fffdf8", edgecolor=THEME["blue"], linewidth=2)
    interior = Circle((0, 0), 0.52, facecolor=REGION_COLORS["Interior"], alpha=0.40, edgecolor="none")
    approach = Circle((0, 0), 0.72, facecolor=REGION_COLORS["Approach"], alpha=0.28, edgecolor="none")
    boundary = Circle((0, 0), 0.90, facecolor="none", edgecolor=REGION_COLORS["Boundary"], linewidth=16, alpha=0.50)
    compressed = Ellipse((0.66, 0), 0.52, 1.68, facecolor=REGION_COLORS["Compressed far side"], alpha=0.38, edgecolor="none")
    cmb_side = Ellipse((0.90, 0), 0.18, 1.50, facecolor=REGION_COLORS["CMB-side"], alpha=0.50, edgecolor="none")
    for patch in [universe, approach, interior, compressed, cmb_side, boundary]:
        ax.add_patch(patch)

    path_x = np.linspace(-0.90, 0.90, 180)
    t = (path_x + 0.90) / 1.80
    path_y = 0.18 * np.sin(np.pi * t) - 0.18
    ax.plot(path_x, path_y, color=THEME["blue2"], linewidth=3, label="light/path response")
    ax.add_patch(
        FancyArrowPatch(
            (path_x[-12], path_y[-12]),
            (path_x[-1], path_y[-1]),
            arrowstyle="->",
            mutation_scale=18,
            color=THEME["blue2"],
            linewidth=2,
        )
    )

    ax.scatter([-0.62], [-0.18], s=120, color=THEME["blue"], zorder=4)
    ax.text(-0.62, -0.32, "observer", ha="center", fontsize=10)
    ax.scatter([point_x], [point_y], s=160, color=THEME["gold"], zorder=5)
    ax.text(point_x, point_y + 0.10, preset_name, ha="center", fontsize=10)

    ax.text(-0.08, 0.10, "Interior", ha="center", fontsize=12, weight="bold")
    ax.text(-0.08, 0.58, "Approach", ha="center", fontsize=11)
    ax.text(0.08, 0.96, "Boundary", ha="center", fontsize=11)
    ax.text(0.70, 0.42, "Compressed\nfar side", ha="center", fontsize=11)
    ax.text(0.91, -0.50, "CMB-side", ha="center", fontsize=10)
    ax.set_title("Finite universe view")
    return fig


def build_simple_output_curve(X_grid, combined, X_value, combined_value):
    fig, ax = plt.subplots()
    ax.plot(X_grid, combined, linewidth=2.5, color=THEME["blue2"], label="model output")
    ax.scatter([X_value], [combined_value], color=THEME["gold"], s=80, zorder=3)
    ymin, ymax = 0, max(1.05, np.max(combined) * 1.15)
    ax.set_ylim(ymin, ymax)
    ax.axvspan(0.0, 0.6, color=REGION_COLORS["Interior"], alpha=0.20)
    ax.axvspan(0.6, 1.0, color=REGION_COLORS["Approach"], alpha=0.20)
    ax.axvspan(1.0, 1.35, color=REGION_COLORS["Boundary"], alpha=0.22)
    ax.axvspan(1.35, 2.0, color=REGION_COLORS["Compressed far side"], alpha=0.20)
    ax.text(0.30, ymin + 0.08 * (ymax - ymin), "low response", ha="center")
    ax.text(1.05, ymin + 0.08 * (ymax - ymin), "transition", ha="center")
    ax.text(1.65, ymin + 0.08 * (ymax - ymin), "compressed projection", ha="center")
    ax.set_title("What does the model output do?")
    ax.set_xlabel("position in the finite picture")
    ax.set_ylabel("model output")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig


def build_journey_bar(selected_z, preset_name):
    selected_position = display_position_from_z(selected_z)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    zones = [
        (0.00, 0.20, "Interior"),
        (0.20, 0.45, "Approach"),
        (0.45, 0.62, "Boundary"),
        (0.62, 0.88, "Compressed far side"),
        (0.88, 1.00, "CMB-side"),
    ]
    for start, end, label in zones:
        ax.axvspan(start, end, color=REGION_COLORS[label], alpha=0.75)
        ax.text((start + end) / 2, 0.60, label, ha="center", va="center", fontsize=10)
    ax.plot([0, 1], [0.25, 0.25], color=THEME["blue2"], linewidth=5)
    ax.scatter([selected_position], [0.25], color=THEME["gold"], s=150, zorder=4)
    ax.text(selected_position, 0.08, preset_name, ha="center", fontsize=10)
    ax.text(0.00, -0.05, "Observer side", ha="left", fontsize=9)
    ax.text(1.00, -0.05, "Farthest edge", ha="right", fontsize=9)
    ax.set_title("Same journey, flattened into a path")
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.12, 0.78)
    ax.spines[["left", "right", "top", "bottom"]].set_visible(False)
    return fig


def build_hubble_number_line(finite_reading):
    finite_marker = float(np.clip(finite_reading, 65, 75))
    fig, ax = plt.subplots(figsize=(8, 2.5))
    ax.hlines(0, 65, 75, color=THEME["muted"], linewidth=2)
    ax.scatter([EARLY_H0], [0], color=THEME["blue2"], s=130, zorder=4)
    ax.scatter([LOCAL_H0], [0], color=THEME["blue"], s=130, zorder=4)
    ax.scatter([finite_marker], [0.18], color=THEME["gold"], s=130, zorder=4)
    ax.annotate(
        "gap = 5.6",
        xy=(EARLY_H0, -0.12),
        xytext=(LOCAL_H0, -0.12),
        arrowprops={"arrowstyle": "<->", "color": THEME["blue"]},
        ha="center",
        va="center",
    )
    ax.text(EARLY_H0, 0.10, "early-universe value\n67.4", ha="center", color=THEME["blue2"])
    ax.text(LOCAL_H0, 0.10, "nearby value\n73", ha="center", color=THEME["blue"])
    ax.text(
        finite_marker,
        0.31,
        f"finite-geometry mapped reading\nH = {finite_reading:.1f}",
        ha="center",
        color=THEME["gold"],
    )
    ax.text(70, -0.28, "Hubble tension", ha="center", weight="bold")
    ax.set_title("Hubble tension, shown as two numbers")
    ax.set_xlim(65, 75)
    ax.set_ylim(-0.38, 0.45)
    ax.set_yticks([])
    ax.set_xlabel("H0 reading")
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.grid(True, axis="x", alpha=0.20)
    return fig


def build_quantitative_hubble_comparison(selected_z, selected_finite_h0):
    z = np.geomspace(0.01, Z_CMB, 500)
    x = accumulated_response_from_z(z) / A_T
    baseline = np.full_like(z, DEFAULT_H0_BASELINE, dtype=float)
    finite_curve = finite_h0_from_projection(z)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x, baseline, color=THEME["muted"], linewidth=2, label="Expansion-style baseline")
    ax.plot(x, finite_curve, color=THEME["blue2"], linewidth=2.5, label="Finite-geometry mapped reading")
    ax.scatter(
        [transition_coordinate_x(selected_z)],
        [selected_finite_h0],
        color=THEME["gold"],
        s=90,
        zorder=5,
        label="Selected light",
    )
    ax.set_xlabel("X = A / A_t")
    ax.set_ylabel("Hubble/register reading")
    ax.set_title("How the reading shifts across the finite register")
    ax.grid(True, alpha=0.25)
    ax.legend()
    return fig


def standard_distance_curves(z_values):
    z_arr = np.asarray(z_values, dtype=float)
    z_arr = np.clip(z_arr, 0.0, None)
    if z_arr.size == 0:
        return np.array([]), np.array([]), np.array([])
    sort_idx = np.argsort(z_arr)
    z_sorted = z_arr[sort_idx]
    e_vals = E_z(z_sorted)
    integrand = 1.0 / e_vals
    dz = np.diff(z_sorted)
    cumulative = np.zeros_like(z_sorted)
    if z_sorted.size > 1:
        cumulative[1:] = np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * dz)
    d_c_sorted = (C_KM_S / H0_STANDARD) * cumulative
    d_a_sorted = d_c_sorted / (1.0 + z_sorted)
    d_l_sorted = d_c_sorted * (1.0 + z_sorted)

    d_c = np.empty_like(d_c_sorted)
    d_a = np.empty_like(d_a_sorted)
    d_l = np.empty_like(d_l_sorted)
    d_c[sort_idx] = d_c_sorted
    d_a[sort_idx] = d_a_sorted
    d_l[sort_idx] = d_l_sorted
    return d_c, d_a, d_l


def build_distance_scale_comparison_plot(z_max, selected_z):
    z_floor = 0.001
    max_for_plot = max(0.2, float(z_max))
    z_curve = np.geomspace(z_floor, max_for_plot, 650)
    _, d_a_curve, _ = standard_distance_curves(z_curve)
    finite_curve = d_a_curve * np.clip(projection_law(z_curve), 1e-3, None)
    selected_d_a = angular_diameter_distance_mpc(selected_z)
    selected_finite = selected_d_a * np.clip(projection_law(selected_z), 1e-3, None)

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(z_curve, d_a_curve, color=THEME["blue"], linewidth=2.2, label="Standard D_A")
    ax.plot(
        z_curve,
        finite_curve,
        color=THEME["gold"],
        linewidth=2.2,
        label="Finite-geometry mapped apparent-size scale",
    )
    ax.scatter([selected_z], [selected_d_a], color=THEME["blue2"], s=85, zorder=5)
    ax.scatter([selected_z], [selected_finite], color=THEME["gold"], s=85, zorder=5)
    ax.set_xscale("log")
    ax.set_xlabel("Observed redshift z")
    ax.set_ylabel("Scale distance (Mpc)")
    ax.set_title("Distance/scale comparison for the same observed light")
    ax.grid(True, alpha=0.25)
    ax.legend()
    return fig


def build_physical_size_comparison_bar(standard_size_kpc, finite_size_kpc, angular_size_arcsec):
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = ["Standard inferred size", "Finite-geometry mapped size"]
    values = [standard_size_kpc, finite_size_kpc]
    colors = [THEME["blue2"], THEME["gold"]]
    bars = ax.bar(labels, values, color=colors, width=0.55)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value * 1.01 if value >= 0 else value,
            f"{value:.2f} kpc",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylabel("Physical size estimate (kpc)")
    ax.set_title(f"Size from selected angular width ({angular_size_arcsec:.3f} arcsec)")
    ax.grid(True, axis="y", alpha=0.25)
    return fig


def build_human_scale_ruler_plot(finite_inches):
    standard_inches = 12.0
    max_inches = max(standard_inches, float(finite_inches), 0.5) * 1.2
    fig, ax = plt.subplots(figsize=(7.5, 2.7))
    ax.barh(
        [1, 0],
        [standard_inches, float(finite_inches)],
        color=[THEME["blue2"], THEME["gold"]],
        height=0.45,
    )
    ax.text(standard_inches + max_inches * 0.01, 1, "Standard reading: 12 inches", va="center", fontsize=9)
    ax.text(
        float(finite_inches) + max_inches * 0.01,
        0,
        f"Finite-geometry mapped reading: {float(finite_inches):.2f} inches",
        va="center",
        fontsize=9,
    )
    ax.set_xlim(0, max_inches)
    ax.set_yticks([1, 0])
    ax.set_yticklabels(["Standard", "Finite-mapped"])
    ax.set_xlabel("Analogy ruler (inches)")
    ax.set_title("Human-scale version")
    ax.grid(True, axis="x", alpha=0.25)
    return fig


def build_one_page_comparison_figure(selected_z, selected_preset):
    position = display_position_from_z(selected_z)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    source_x = position

    top = axes[0]
    top.set_title("Expansion picture")
    top.set_xlim(0, 1)
    top.set_ylim(0, 1)
    top.axis("off")
    top.hlines(0.50, 0.08, 0.94, color="0.25", linewidth=2)
    top.scatter([0.10], [0.50], s=130, color="black", zorder=4)
    top.text(0.10, 0.34, "observer", ha="center")
    galaxy_positions = [0.28, 0.46, 0.66, source_x]
    galaxy_positions = sorted(set(round(max(0.18, min(0.92, x)), 3) for x in galaxy_positions))
    for x in galaxy_positions:
        is_selected = abs(x - source_x) < 0.015
        top.scatter([x], [0.50], s=150 if is_selected else 75, color=THEME["gold"] if is_selected else "0.55", zorder=4)
        top.arrow(x, 0.58, 0.06, 0, head_width=0.035, head_length=0.025, color="0.45", length_includes_head=True)
    top.text(0.52, 0.78, "Expansion: redshift read as stretching space", ha="center", weight="bold")
    top.text(source_x, 0.18, selected_preset, ha="center", fontsize=10)

    bottom = axes[1]
    bottom.set_title("Finite geometry picture")
    bottom.set_xlim(0, 1)
    bottom.set_ylim(0, 1)
    bottom.axis("off")
    zones = [
        (0.00, 0.20, "Interior"),
        (0.20, 0.45, "Approach"),
        (0.45, 0.62, "Boundary"),
        (0.62, 0.88, "Compressed far side"),
        (0.88, 1.00, "CMB-side"),
    ]
    for start, end, label in zones:
        bottom.axvspan(start, end, ymin=0.18, ymax=0.82, color=REGION_COLORS[label], alpha=0.75)
        bottom.text((start + end) / 2, 0.78, label, ha="center", va="center", fontsize=9)
    path_x = np.linspace(0.06, 0.94, 160)
    path_y = 0.42 + 0.08 * np.sin(np.linspace(0, np.pi, 160))
    bottom.plot(path_x, path_y, color=THEME["blue2"], linewidth=4)
    marker_y = 0.42 + 0.08 * np.sin(np.pi * position)
    bottom.scatter([0.08], [0.42], s=130, color="black", zorder=4)
    bottom.text(0.08, 0.25, "observer", ha="center")
    bottom.scatter([source_x], [marker_y], s=160, color=THEME["gold"], zorder=5)
    bottom.text(source_x, 0.16, selected_preset, ha="center", fontsize=10)
    bottom.text(0.52, 0.92, "Finite geometry: redshift read as path response + projection", ha="center", weight="bold")
    bottom.text(0.50, 0.32, "Light path", ha="center", color=THEME["blue2"])
    return fig


def build_storyboard(selected_z, preset_name):
    pos = display_position_from_z(selected_z)
    source_x = 0.15 + 0.70 * pos
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    left = axes[0]
    left.set_title("Expansion picture")
    left.set_xlim(0, 1)
    left.set_ylim(0, 1)
    left.axis("off")
    left.scatter([0.18], [0.50], s=120, color="black")
    left.text(0.18, 0.37, "observer", ha="center")
    galaxy_x = [0.35, 0.55, source_x]
    galaxy_y = [0.62, 0.38, 0.55]
    left.scatter(galaxy_x, galaxy_y, s=[70, 70, 130], color=["0.5", "0.5", THEME["gold"]])
    for gx, gy in zip(galaxy_x, galaxy_y):
        left.arrow(gx, gy, 0.09, 0.02 * np.sign(gy - 0.5), head_width=0.025, color="0.35")
    left.text(0.50, 0.15, "Redshift is read mainly as space stretching.", ha="center")

    right = axes[1]
    right.set_title("Finite geometry picture")
    right.set_aspect("equal")
    right.set_xlim(-1.15, 1.15)
    right.set_ylim(-0.90, 0.90)
    right.axis("off")
    universe = Circle((0, 0), 1.0, facecolor="#fffdf8", edgecolor=THEME["blue"], linewidth=2)
    boundary = Circle((0, 0), 0.90, facecolor="none", edgecolor=REGION_COLORS["Boundary"], linewidth=12, alpha=0.50)
    compressed = Ellipse((0.66, 0), 0.52, 1.68, facecolor=REGION_COLORS["Compressed far side"], alpha=0.35, edgecolor="none")
    right.add_patch(universe)
    right.add_patch(compressed)
    right.add_patch(boundary)
    path_x = np.linspace(-0.90, 0.90, 180)
    t = (path_x + 0.90) / 1.80
    path_y = 0.18 * np.sin(np.pi * t) - 0.18
    right.plot(path_x, path_y, color=THEME["blue2"], linewidth=3)
    px, py, _ = selected_path_point(selected_z)
    right.scatter([-0.62], [-0.18], s=100, color="black", zorder=4)
    right.scatter([px], [py], s=130, color=THEME["gold"], zorder=5)
    right.text(0.0, -0.72, "Redshift is read as path response through a finite geometry.", ha="center")
    return fig


def read_dotenv_file(path):
    values = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def load_ollama_config():
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        dotenv_values = read_dotenv_file(Path(".env"))
        for key, value in dotenv_values.items():
            os.environ.setdefault(key, value)

    return {
        "api_key": os.getenv("OLLAMA_API_KEY", ""),
        "base_url": os.getenv(
            "OLLAMA_BASE_URL", "http://localhost:11434/api"
        ).rstrip("/"),
        "model": os.getenv("OLLAMA_MODEL", ""),
    }


def resolve_ollama_chat_url(base_url):
    normalized_base_url = base_url.rstrip("/")
    if normalized_base_url.endswith("/api"):
        return f"{normalized_base_url}/chat"
    return f"{normalized_base_url}/api/chat"


def call_ollama_chat(messages, model_context):
    config = load_ollama_config()
    if requests is None:
        return "The Python package 'requests' is not available, so Ollama chat cannot run."
    if not config["model"]:
        return (
            "Ollama chat is not configured. Add OLLAMA_MODEL to .env or the "
            "environment. OLLAMA_BASE_URL defaults to http://localhost:11434/api."
        )

    book_context = model_context.get("book_context", BOOK_CONTEXT)
    app_state = model_context.get("app_state", {})
    explanation_count = model_context.get("explanation_count", 1)
    response_mode = model_context.get("response_mode", "plain visual overview")
    recent_explanations = model_context.get("recent_explanations", [])
    recent_block = "\n".join(
        [f"{idx}. {text}" for idx, text in enumerate(recent_explanations[-4:], start=1)]
    )
    system_content = (
        "You are the explanation helper for a finite-geometry companion app.\n"
        "Explain what the user is seeing in plain language using only the provided book/app context.\n"
        "If the app state has not changed, vary wording and framing from recent explanations.\n"
        "Do not invent equations, constants, citations, or claims outside this context.\n"
        "Keep normal answers under about 180 words.\n\n"
        f"Explanation count: {explanation_count}\n"
        f"Response mode: {response_mode}\n"
        f"Current app state: {app_state}\n"
        f"Recent explanation history (avoid repeating): {recent_block}\n\n"
        f"Loaded book context:\n{book_context}"
    )
    system_message = {
        "role": "system",
        "content": system_content,
    }
    payload = {
        "model": config["model"],
        "messages": [system_message] + messages,
        "stream": False,
    }
    headers = {"Content-Type": "application/json"}
    if config["api_key"]:
        headers["Authorization"] = f"Bearer {config['api_key']}"

    try:
        chat_url = resolve_ollama_chat_url(config["base_url"])
        response = requests.post(
            chat_url,
            json=payload,
            headers=headers,
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        try:
            return data["message"]["content"]
        except KeyError:
            return f"Ollama returned JSON without message.content: {data}"
    except requests.RequestException as exc:
        return f"Ollama chat request failed: {exc}"


def full_model_page(selected_z, selected_preset):
    page_name = "finite_universe_view"
    st.header("Finite Universe View")
    projection_strength = DEFAULT_PROJECTION_STRENGTH
    transition_sharpness = DEFAULT_TRANSITION_SHARPNESS

    A_value = accumulated_response_from_z(selected_z)
    chi_value = finite_depth_chi(selected_z)
    X_value = transition_coordinate_x(selected_z)
    G_value = local_response(chi_value, DEFAULT_G0, DEFAULT_DB)
    D_value = A_value * G_value
    C_value = response_engine(np.array([X_value]), DEFAULT_ETA, DEFAULT_U, DEFAULT_D_N)[0]
    P_value = adjustable_projection(
        np.array([selected_z]), projection_strength, transition_sharpness
    )[0]
    combined_value = D_value * C_value * P_value
    display_position = display_position_from_z(selected_z)
    current_regime = display_region_from_position(display_position)
    selected_shift = float(finite_h0_from_projection(selected_z))
    current_response_label = PRESET_READING[selected_preset]

    st.info(
        "Light from different parts of the universe is read differently in this model. "
        "Nearby light is mostly interior behavior. Far-side light passes through boundary "
        "and projection regions before it is read by the observer."
    )

    result_cols = st.columns(3)
    result_cols[0].metric("What you selected", selected_preset, f"z = {selected_z:g}")
    result_cols[1].metric("Where it sits", current_regime)
    result_cols[2].metric("What the model does", current_response_label)

    fig1 = build_universe_schematic(selected_z, selected_preset)
    st.pyplot(fig1)
    add_interpretation("Light path. Boundary. Projection.")

    fig2 = build_journey_bar(selected_z, selected_preset)
    st.pyplot(fig2)
    add_interpretation("Same colors. Same marker. Same selected light.")

    st.info(f"What this means: {PRESET_GUIDANCE[selected_preset]}")

    with st.expander("Optional numbers"):
        st.markdown(
            f"""
```text
selected = {selected_preset}
z = {selected_z:g}
A = ln(1 + z) = {A_value:.4f}
X = A / A_t = {X_value:.4f}
projection P = {P_value:.4f}
finite response = {combined_value:.4f}
selected shift = {selected_shift:.4f}
```
"""
        )

    st.session_state["model_context"] = {
        "page": "Finite Universe View",
        "preset": selected_preset,
        "z": round(selected_z, 4),
        "A": round(A_value, 4),
        "chi": round(chi_value, 4),
        "X": round(X_value, 4),
        "region": current_regime,
        "projection_strength": round(projection_strength, 4),
        "transition_sharpness": round(transition_sharpness, 4),
        "projection_factor": round(P_value, 4),
        "combined_response": round(combined_value, 4),
    }

    return [
        (page_name, "universe_schematic", fig1),
        (page_name, "journey_bar", fig2),
    ]


def comparison_page(selected_z, selected_preset):
    page_name = "expansion_vs_finite_geometry"
    st.header("Expansion Picture vs Finite Geometry Picture")
    st.subheader(f"Selected view: {selected_preset}  |  z = {selected_z:g}")
    projection_strength = DEFAULT_PROJECTION_STRENGTH
    transition_sharpness = DEFAULT_TRANSITION_SHARPNESS

    z = np.geomspace(0.01, Z_CMB, 900)
    distance = z / np.max(z)
    baseline = standard_baseline(distance, DEFAULT_H0_BASELINE)
    _, _, _, _, _, _, _, combined = full_finite_response(
        z,
        DEFAULT_G0,
        DEFAULT_DB,
        projection_strength,
        transition_sharpness,
        DEFAULT_ETA,
        DEFAULT_U,
        DEFAULT_D_N,
    )
    finite_curve = finite_h0_from_projection(z)
    *_, selected_combined = full_finite_response(
        np.array([selected_z]),
        DEFAULT_G0,
        DEFAULT_DB,
        projection_strength,
        transition_sharpness,
        DEFAULT_ETA,
        DEFAULT_U,
        DEFAULT_D_N,
    )
    selected_finite_h0 = float(finite_h0_from_projection(selected_z))
    selected_shift = float(selected_finite_h0 - DEFAULT_H0_BASELINE)
    A_value = accumulated_response_from_z(selected_z)
    X_value = transition_coordinate_x(selected_z)
    P_value = adjustable_projection(
        np.array([selected_z]), projection_strength, transition_sharpness
    )[0]

    fig1 = build_storyboard(selected_z, selected_preset)
    st.pyplot(fig1)
    add_interpretation("Expansion reading. Finite geometry reading.")

    summary_cols = st.columns(4)
    summary_cols[0].metric("Nearby universe", f"{LOCAL_H0:.1f}")
    summary_cols[1].metric("Early universe", f"{EARLY_H0:.1f}")
    summary_cols[2].metric("Hubble tension", f"{LOCAL_H0 - EARLY_H0:.1f} km/s/Mpc")
    summary_cols[3].metric("Finite shift", f"{selected_shift:.3f}")

    fig2 = build_hubble_number_line(selected_finite_h0)
    st.pyplot(fig2)
    add_interpretation("Expansion picture: two values disagree. Finite geometry picture: the reading can shift through path and projection.")

    card_cols = st.columns(3)
    card_cols[0].info("H0\n\nA number used to compare expansion-rate readings.")
    card_cols[1].info("The tension\n\nNearby and early-universe readings do not match.")
    card_cols[2].info("Finite geometry difference\n\nThe model changes how far-side light is read before comparing it to nearby light.")

    with st.expander("Optional numbers"):
        st.markdown(
            f"""
```text
selected = {selected_preset}
z = {selected_z:g}
A = {A_value:.4f}
X = {X_value:.4f}
projection P = {float(P_value):.4f}
finite response = {float(selected_combined[0]):.4f}
selected shift = {selected_shift:.4f}
```
"""
        )

    st.session_state["model_context"] = {
        "page": "Expansion Model vs Finite Geometry",
        "preset": selected_preset,
        "selected_z": round(selected_z, 4),
        "nearby_H0": LOCAL_H0,
        "early_H0": EARLY_H0,
        "hubble_gap": round(LOCAL_H0 - EARLY_H0, 4),
        "finite_shift": round(selected_shift, 4),
        "projection_strength": round(projection_strength, 4),
        "transition_sharpness": round(transition_sharpness, 4),
    }

    return [
        (page_name, "hubble_tension", fig1),
        (page_name, "model_comparison", fig2),
    ]


def response_mode_from_count(explanation_count):
    if explanation_count <= 1:
        return "plain visual overview"
    if explanation_count == 2:
        return "quantitative comparison"
    if explanation_count == 3:
        return "book/model meaning"
    if explanation_count == 4:
        return "implications/what to notice"
    return "converged synthesis"


def build_state_signature(app_state):
    return json.dumps(app_state, sort_keys=True, ensure_ascii=True)


def render_app_title_band():
    st.markdown(
        """
<div class="afu-title-band">
  <div class="afu-title-main">A Finite Universe</div>
  <div class="afu-title-sub">Visual Companion</div>
</div>
""",
        unsafe_allow_html=True,
    )


def _ensure_explanation_state(app_state):
    if "explanation_count" not in st.session_state:
        st.session_state["explanation_count"] = 0
    if "last_state_signature" not in st.session_state:
        st.session_state["last_state_signature"] = ""
    if "explanation_history" not in st.session_state:
        st.session_state["explanation_history"] = []
    if "latest_explanation" not in st.session_state:
        st.session_state["latest_explanation"] = ""

    current_signature = build_state_signature(app_state)
    if st.session_state["last_state_signature"] != current_signature:
        st.session_state["explanation_count"] = 0
        st.session_state["explanation_history"] = []
        st.session_state["latest_explanation"] = ""
        st.session_state["last_state_signature"] = current_signature


def _render_explanation_helper_controls(app_state, loaded_book_context, key_prefix):
    config = load_ollama_config()
    if not config["model"]:
        st.info(
            "Ollama helper unavailable. Set OLLAMA_MODEL (and optional OLLAMA_BASE_URL / OLLAMA_API_KEY). "
            "The rest of the app remains fully usable."
        )
    else:
        st.caption(f"Ollama model: {config['model']}")

    _ensure_explanation_state(app_state)

    st.caption(
        f"Mode on next press: {response_mode_from_count(st.session_state['explanation_count'] + 1)}"
    )
    explain_pressed = st.button(
        "Tell me what I am seeing",
        use_container_width=True,
        key=f"{key_prefix}_explain_btn",
    )
    clear_pressed = st.button(
        "Clear explanation history",
        use_container_width=True,
        key=f"{key_prefix}_clear_btn",
    )

    if clear_pressed:
        st.session_state["explanation_count"] = 0
        st.session_state["explanation_history"] = []
        st.session_state["latest_explanation"] = ""
        st.rerun()

    if explain_pressed:
        if not config["model"]:
            st.session_state["latest_explanation"] = (
                "Ollama helper is currently unavailable. Configure OLLAMA_MODEL to enable "
                "generated explanations based on this view."
            )
        else:
            st.session_state["explanation_count"] += 1
            mode = response_mode_from_count(st.session_state["explanation_count"])
            helper_prompt = (
                "Tell me what I am seeing in this app state. "
                f"Use response mode: {mode}. "
                "Explain the two-column comparison plainly: expansion reading versus finite-geometry reading. "
                "Explain what the selected redshift means, how the expansion reading infers distance/size, "
                "and how the finite-geometry reading changes inferred scale. "
                "Use wording such as 'standard expansion reading would infer' and "
                "'in this app's finite-geometry mapping'. "
                "Do not repeat wording from recent explanations. "
                "If this is press 1, give a visual overview. "
                "If press 2, emphasize quantitative comparison. "
                "If press 3, connect to book/model meaning. "
                "If press 4, focus on implications and what to notice. "
                "Press 5+ may converge to synthesis."
            )
            model_context = {
                "book_context": loaded_book_context,
                "app_state": app_state,
                "explanation_count": st.session_state["explanation_count"],
                "response_mode": mode,
                "recent_explanations": st.session_state["explanation_history"][-4:],
            }
            with st.spinner("Generating explanation from Ollama..."):
                answer = call_ollama_chat(
                    [{"role": "user", "content": helper_prompt}],
                    model_context,
                )
            st.session_state["latest_explanation"] = answer
            st.session_state["explanation_history"].append(answer)
            st.session_state["explanation_history"] = st.session_state["explanation_history"][-8:]
        st.rerun()

    if st.session_state["latest_explanation"]:
        st.markdown("**Latest explanation**")
        st.write(st.session_state["latest_explanation"])


def explanation_helper_panel(app_state, loaded_book_context):
    with st.sidebar:
        st.header("Explanation Helper")
        _render_explanation_helper_controls(
            app_state=app_state,
            loaded_book_context=loaded_book_context,
            key_prefix="sidebar",
        )


def explanation_helper_fallback(app_state, loaded_book_context):
    with st.expander("Tell me what I am seeing", expanded=False):
        st.caption("Fallback helper access if the sidebar is collapsed.")
        _render_explanation_helper_controls(
            app_state=app_state,
            loaded_book_context=loaded_book_context,
            key_prefix="main_fallback",
        )


def main():
    st.set_page_config(
        page_title="Finite Geometry Companion Model",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_book_theme()
    apply_plot_theme()

    render_app_title_band()
    st.info(
        "Choose a kind of light or enter a redshift. The app compares how the same observation is read in standard expansion cosmology and in the finite deformable geometry model."
    )
    st.caption("The finite-geometry side shows the app's model reading of the same observed light.")

    st.subheader("Same observed light, two readings")
    control_cols = st.columns([2.0, 1.0, 1.0])
    with control_cols[0]:
        selected_preset = st.selectbox(
            "Object/light preset",
            list(LIGHT_PRESETS.keys()),
        )
    with control_cols[1]:
        angular_size_arcsec = st.slider(
            "Angular size (arcsec)",
            min_value=0.01,
            max_value=10.0,
            value=0.2,
            step=0.01,
        )
    with control_cols[2]:
        nickname_choice = st.selectbox(
            "Object nickname",
            [
                "small JWST galaxy candidate",
                "Milky Way-sized galaxy",
                "galaxy cluster",
                "custom nickname",
            ],
        )

    custom_nickname = ""
    if nickname_choice == "custom nickname":
        custom_nickname = st.text_input("Custom object nickname", value="deep-field target")
    object_nickname = custom_nickname.strip() if nickname_choice == "custom nickname" else nickname_choice

    preset_z = LIGHT_PRESETS[selected_preset]
    if preset_z is None:
        selected_z = st.number_input(
            "Custom redshift z",
            min_value=0.001,
            max_value=2000.0,
            value=2.0,
            step=0.001,
            format="%.3f",
        )
    else:
        selected_z = float(preset_z)

    projection_strength = DEFAULT_PROJECTION_STRENGTH
    transition_sharpness = DEFAULT_TRANSITION_SHARPNESS
    A_value = float(accumulated_response_from_z(selected_z))
    chi_value = float(finite_depth_chi(selected_z))
    X_value = float(transition_coordinate_x(selected_z))
    G_value = float(local_response(chi_value, DEFAULT_G0, DEFAULT_DB))
    D_value = float(A_value * G_value)
    C_value = float(response_engine(np.array([X_value]), DEFAULT_ETA, DEFAULT_U, DEFAULT_D_N)[0])
    P_value = float(adjustable_projection(np.array([selected_z]), projection_strength, transition_sharpness)[0])
    P_star_value = float(projection_law(selected_z))
    finite_response = float(D_value * C_value * P_value)
    finite_marker = float(finite_h0_from_projection(selected_z))
    selected_position = display_position_from_z(selected_z)
    selected_region = display_region_from_position(selected_position)

    standard_comoving_mpc = float(comoving_distance_mpc(selected_z))
    standard_luminosity_mpc = float((1.0 + selected_z) * standard_comoving_mpc)
    standard_angular_mpc = float(standard_comoving_mpc / (1.0 + selected_z))
    standard_lookback_gyr = float(lookback_time_gyr(selected_z))
    standard_kpc_arcsec = float(kpc_per_arcsec(selected_z))
    standard_size_kpc = float(physical_size_kpc_from_arcsec(selected_z, angular_size_arcsec))

    finite_size_factor = float(np.clip(P_star_value, 1e-3, None))
    finite_mapped_angular_mpc = float(standard_angular_mpc * finite_size_factor)
    finite_mapped_kpc_arcsec = float(standard_kpc_arcsec * finite_size_factor)
    finite_mapped_size_kpc = float(standard_size_kpc * finite_size_factor)
    size_ratio = float(finite_mapped_size_kpc / max(standard_size_kpc, 1e-9))
    standard_comoving_gly = standard_comoving_mpc * MPC_TO_GLY
    finite_luminosity_scale_mpc = standard_luminosity_mpc * finite_size_factor

    summary_cols = st.columns(4)
    summary_cols[0].metric("Observed redshift z", f"{selected_z:.4g}")
    summary_cols[1].metric("Angular width", f"{angular_size_arcsec:.3g} arcsec")
    summary_cols[2].metric("Expansion distance scale", format_distance_scale_mpc(standard_angular_mpc))
    summary_cols[3].metric("Finite-geometry distance scale", format_distance_scale_mpc(finite_mapped_angular_mpc))

    st.subheader("Expansion reading | Finite-geometry reading")
    comparison_rows = [
        {
            "Quantity": "Observed redshift",
            "Expansion reading": f"z = {selected_z:.4g}",
            "Finite-geometry reading": f"z = {selected_z:.4g}",
        },
        {
            "Quantity": "Distance scale",
            "Expansion reading": format_distance_scale_mpc(standard_angular_mpc),
            "Finite-geometry reading": format_distance_scale_mpc(finite_mapped_angular_mpc),
        },
        {
            "Quantity": "Lookback time / path reading",
            "Expansion reading": format_time_gyr(standard_lookback_gyr),
            "Finite-geometry reading": f"A = ln(1+z) = {A_value:.5f}",
        },
        {
            "Quantity": "Size from angular width",
            "Expansion reading": format_size_kpc(standard_size_kpc),
            "Finite-geometry reading": format_size_kpc(finite_mapped_size_kpc),
        },
        {
            "Quantity": "Brightness/luminosity scale",
            "Expansion reading": format_distance_scale_mpc(standard_luminosity_mpc),
            "Finite-geometry reading": format_distance_scale_mpc(finite_luminosity_scale_mpc),
        },
        {
            "Quantity": "Hubble/register reading",
            "Expansion reading": f"Nearby {LOCAL_H0:.1f}, early {EARLY_H0:.1f}",
            "Finite-geometry reading": f"{finite_marker:.2f}",
        },
        {
            "Quantity": "Region/register",
            "Expansion reading": "single expansion register",
            "Finite-geometry reading": selected_region,
        },
        {
            "Quantity": "Projection factor",
            "Expansion reading": "1.000",
            "Finite-geometry reading": f"{P_star_value:.5f}",
        },
        {
            "Quantity": "Path response A = ln(1+z)",
            "Expansion reading": "used indirectly",
            "Finite-geometry reading": f"{A_value:.5f}",
        },
        {
            "Quantity": "Transition coordinate X = A/A_t",
            "Expansion reading": "not used directly",
            "Finite-geometry reading": f"{X_value:.5f}",
        },
        {
            "Quantity": "Size ratio (finite / expansion)",
            "Expansion reading": "1.00",
            "Finite-geometry reading": f"{format_ratio(size_ratio)}",
        },
    ]
    st.table(comparison_rows)
    st.caption("At this redshift and angular width, the two readings give different inferred physical sizes.")
    st.write(
        f"Expansion size: **{format_size_kpc(standard_size_kpc)}** | "
        f"Finite-geometry size: **{format_size_kpc(finite_mapped_size_kpc)}**"
    )

    fig_hubble = build_hubble_number_line(finite_marker)
    st.pyplot(fig_hubble)
    st.caption(
        "Expansion compares local and early H0 readings; finite mapping shows where this selected light lands on that register axis."
    )

    fig_quant = build_quantitative_hubble_comparison(selected_z, finite_marker)
    with st.expander("Supporting technical context (A, X, projection/register)", expanded=False):
        fig_distance = build_distance_scale_comparison_plot(
            z_max=min(1200.0, max(0.2, selected_z * 1.2 + 0.3)),
            selected_z=selected_z,
        )
        st.pyplot(fig_distance)
        st.pyplot(fig_quant)
        st.markdown(
            f"""
```text
selected = {selected_preset}
nickname = {object_nickname}
z = {selected_z:.6f}
distance_human_expansion = {format_distance_scale_mpc(standard_angular_mpc)}
distance_human_finite = {format_distance_scale_mpc(finite_mapped_angular_mpc)}
A = {A_value:.6f}
chi = {chi_value:.6f}
X = {X_value:.6f}
G(local response) = {G_value:.6f}
D = A*G = {D_value:.6f}
C(response engine) = {C_value:.6f}
P_adjustable = {P_value:.6f}
P_star = {P_star_value:.6f}
finite response = {finite_response:.6f}
finite marker = {finite_marker:.6f}
standard D_C = {standard_comoving_mpc:.6f} Mpc
standard D_L = {standard_luminosity_mpc:.6f} Mpc
standard D_A = {standard_angular_mpc:.6f} Mpc
standard lookback = {standard_lookback_gyr:.6f} Gyr
standard kpc/arcsec = {standard_kpc_arcsec:.6f}
finite mapped kpc/arcsec = {finite_mapped_kpc_arcsec:.6f}
size ratio = {size_ratio:.6f}
```
"""
        )

    fig_story = build_one_page_comparison_figure(selected_z, selected_preset)
    fig_journey = build_journey_bar(selected_z, selected_preset)
    with st.expander("Concept sketch", expanded=False):
        st.pyplot(fig_story)
        st.pyplot(fig_journey)
        st.info(PRESET_GUIDANCE.get(selected_preset, PRESET_GUIDANCE["Custom redshift"]))

    app_state = {
        "selected_preset": selected_preset,
        "object_nickname": object_nickname,
        "selected_z": round(selected_z, 6),
        "angular_size_arcsec": round(float(angular_size_arcsec), 6),
        "standard_comoving_distance_mpc": round(standard_comoving_mpc, 6),
        "standard_luminosity_distance_mpc": round(standard_luminosity_mpc, 6),
        "standard_angular_diameter_distance_mpc": round(standard_angular_mpc, 6),
        "standard_lookback_time_gyr": round(standard_lookback_gyr, 6),
        "standard_kpc_per_arcsec": round(standard_kpc_arcsec, 6),
        "standard_physical_size_kpc": round(standard_size_kpc, 6),
        "finite_mapped_angular_distance_mpc": round(finite_mapped_angular_mpc, 6),
        "finite_mapped_kpc_per_arcsec": round(finite_mapped_kpc_arcsec, 6),
        "finite_mapped_size_kpc": round(finite_mapped_size_kpc, 6),
        "size_ratio": round(size_ratio, 6),
        "expansion_distance_scale_human": format_distance_scale_mpc(standard_angular_mpc),
        "expansion_lookback_time_human": format_time_gyr(standard_lookback_gyr),
        "expansion_physical_size_human": format_size_kpc(standard_size_kpc),
        "finite_geometry_distance_scale_human": format_distance_scale_mpc(finite_mapped_angular_mpc),
        "finite_geometry_physical_size_human": format_size_kpc(finite_mapped_size_kpc),
        "A": round(A_value, 6),
        "X": round(X_value, 6),
        "projection_factor": round(P_star_value, 6),
        "region": selected_region,
        "finite_marker": round(finite_marker, 6),
        "comparison_board": comparison_rows,
    }
    st.session_state["model_context"] = app_state

    if st.button("Export current figures"):
        figures = [
            ("comparison_model", "distance_scale_comparison", fig_distance),
            ("comparison_model", "hubble_number_line", fig_hubble),
            ("comparison_model", "quantitative_hubble_comparison", fig_quant),
            ("comparison_model", "expansion_vs_finite_geometry_schematic", fig_story),
            ("comparison_model", "journey_register_bar", fig_journey),
        ]
        output_paths = export_figures(figures)
        st.success("Exported: " + ", ".join(str(output_path) for output_path in output_paths))

    loaded_book_context = load_full_book_context()
    explanation_helper_panel(app_state, loaded_book_context)
    explanation_helper_fallback(app_state, loaded_book_context)

    for fig in [fig_distance, fig_hubble, fig_quant, fig_story, fig_journey]:
        plt.close(fig)


if __name__ == "__main__":
    main()

