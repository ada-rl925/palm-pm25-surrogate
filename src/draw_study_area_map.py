"""
Draw the nested study-area map (parent domain + Camden child domain).

Standalone helper (NOT part of the training pipeline). Requires internet and
`pip install osmnx contextily geopandas shapely` — it geocodes the Camden/GLA
boundaries and fetches OpenStreetMap basemap tiles. Run once to regenerate the
figures into notebooks/experiments/figs/ (they are already committed and embedded
in notebook 01, so this script is not needed to reproduce the analysis).

    python3 src/draw_study_area_map.py

Reproduces the style of the reference figure: panel (a) shows the parent
domain bounded by the M25 motorway and the Greater London Authority (GLA)
boundary, with a black box marking the child domain; panel (b) zooms into
the child domain on an OpenStreetMap basemap with the Camden borough
boundary highlighted.

Boundary/road geometry is fetched from OpenStreetMap via osmnx and cached
to data/ on first run. Requires: geopandas, contextily, osmnx, shapely,
pyproj, matplotlib.

Usage:
    python draw_study_area_map.py
"""

import os

import contextily as cx
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import osmnx as ox
from pyproj import Transformer

BNG = "EPSG:27700"
HERE = os.path.dirname(os.path.abspath(__file__))          # src/
ROOT = os.path.dirname(HERE)                                # repo root
FIG_DIR  = os.path.join(ROOT, "notebooks", "experiments", "figs")
DATA_DIR = os.path.join(ROOT, "outputs", "_geo_cache")     # cached OSM geojson
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Parent domain (N01) extent, British National Grid metres
# (kept square — 60 km x 60 km — so it sits flush with the square child panel)
PARENT_XMIN, PARENT_XMAX = 500000, 560000
PARENT_YMIN, PARENT_YMAX = 145000, 205000

# Child domain (N02) extent — requested study area
CHILD_XMIN, CHILD_XMAX = 523850, 531850
CHILD_YMIN, CHILD_YMAX = 180050, 188050


def fetch_boundaries():
    """Fetch (or load cached) Camden, GLA and M25 geometry from OSM."""
    camden_path = os.path.join(DATA_DIR, "camden.geojson")
    gla_path = os.path.join(DATA_DIR, "gla.geojson")
    m25_path = os.path.join(DATA_DIR, "m25.geojson")

    if os.path.exists(camden_path):
        camden = gpd.read_file(camden_path)
    else:
        camden = ox.geocode_to_gdf("London Borough of Camden, London, UK")
        camden.to_file(camden_path, driver="GeoJSON")

    if os.path.exists(gla_path):
        gla = gpd.read_file(gla_path)
    else:
        gla = ox.geocode_to_gdf("Greater London, UK")
        gla.to_file(gla_path, driver="GeoJSON")

    if os.path.exists(m25_path):
        m25 = gpd.read_file(m25_path)
    else:
        tf = Transformer.from_crs(BNG, "EPSG:4326", always_xy=True)
        lon_min, lat_min = tf.transform(PARENT_XMIN, PARENT_YMIN)
        lon_max, lat_max = tf.transform(PARENT_XMAX, PARENT_YMAX)
        m25 = ox.features_from_bbox(
            (lon_min, lat_min, lon_max, lat_max), tags={"ref": "M25"}
        )
        m25 = m25[m25.geom_type == "LineString"]
        m25.to_file(m25_path, driver="GeoJSON")

    m25 = m25[m25.geom_type == "LineString"]

    return (
        camden.to_crs(BNG),
        gla.to_crs(BNG),
        m25.to_crs(BNG) if m25.crs is not None else m25,
    )


def draw_parent_panel(ax, camden, gla, m25):
    gla.boundary.plot(ax=ax, color="green", linewidth=1.2, linestyle="--", zorder=2)
    camden.boundary.plot(ax=ax, color="red", linewidth=1.2, zorder=3)

    ax.set_xlim(PARENT_XMIN, PARENT_XMAX)
    ax.set_ylim(PARENT_YMIN, PARENT_YMAX)
    ax.set_aspect("equal")

    cx.add_basemap(ax, crs=BNG, source=cx.providers.CartoDB.Positron, zoom=10)

    ax.xaxis.set_major_locator(mticker.MultipleLocator(10000))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(10000))
    ax.grid(True, which="major", color="black", linewidth=0.5, alpha=0.6)

    ax.set_xlabel("Eastings (m)")
    ax.set_ylabel("Northings (m)")

    # child-domain box
    ax.add_patch(
        plt.Rectangle(
            (CHILD_XMIN, CHILD_YMIN),
            CHILD_XMAX - CHILD_XMIN,
            CHILD_YMAX - CHILD_YMIN,
            fill=False,
            edgecolor="black",
            linewidth=1.2,
            zorder=4,
        )
    )
    ax.text(CHILD_XMIN - 3000, CHILD_YMAX + 1500, "Study Area", fontsize=11, fontweight="bold")
    ax.text(
        CHILD_XMAX + 400,
        (CHILD_YMIN + CHILD_YMAX) / 2,
        "Camden",
        color="red",
        fontsize=10,
        fontweight="bold",
        va="center",
        zorder=5,
    )
    ax.text(
        PARENT_XMIN + 8000,
        PARENT_YMAX - 12500,
        "Greater London Authority",
        color="green",
        fontsize=10,
    )

    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.2)


# Model grid resolution: the 8 km x 8 km domain is resolved on an 800x800
# grid (see report/rl925-project-plan/report.tex), i.e. 10 m per grid unit.
GRID_RES_M = 10  # metres per local grid unit
GRID_SIZE = (CHILD_XMAX - CHILD_XMIN) // GRID_RES_M  # 800

# Highlighted "transfer tile" boxes — local grid coords (xmin, xmax, ymin, ymax)
TRANSFER_TILES = [
    ("transfer tile 1", (200, 400, 200, 400)),
    ("transfer tile 2", (400, 600, 0, 200)),
]


def draw_child_panel(ax, camden):
    camden.boundary.plot(ax=ax, color="red", linewidth=1.8, zorder=3)

    ax.set_xlim(CHILD_XMIN, CHILD_XMAX)
    ax.set_ylim(CHILD_YMIN, CHILD_YMAX)
    ax.set_aspect("equal")

    cx.add_basemap(ax, crs=BNG, source=cx.providers.OpenStreetMap.Mapnik, zoom=14)

    # 4x4 grid (16 equal tiles) — gridlines every 200 local units (2000 m),
    # anchored to the domain origin (CHILD_XMIN/CHILD_YMIN = local 0) rather
    # than absolute BNG multiples, so labels come out as 0/200/400/600/800.
    tile_step_m = 200 * GRID_RES_M
    xticks = [CHILD_XMIN + i * tile_step_m for i in range(5)]
    yticks = [CHILD_YMIN + i * tile_step_m for i in range(5)]
    ax.xaxis.set_major_locator(mticker.FixedLocator(xticks))
    ax.yaxis.set_major_locator(mticker.FixedLocator(yticks))
    ax.grid(True, which="major", color="black", linewidth=1.2, alpha=0.9)

    # Tick labels in local grid units (0-800) instead of BNG metres
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, pos: f"{int(round((x - CHILD_XMIN) / GRID_RES_M))}")
    )
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda y, pos: f"{int(round((y - CHILD_YMIN) / GRID_RES_M))}")
    )

    ax.text(
        CHILD_XMIN + 700,
        CHILD_YMAX - 900,
        "Camden",
        color="red",
        fontsize=12,
        fontweight="bold",
        zorder=4,
    )
    ax.text(CHILD_XMIN + 300, CHILD_YMAX - 300, "Study Area", fontsize=11, fontweight="bold")

    # transfer tile boxes, in BNG metres derived from local grid coords
    for label, (txmin, txmax, tymin, tymax) in TRANSFER_TILES:
        bx0 = CHILD_XMIN + txmin * GRID_RES_M
        bx1 = CHILD_XMIN + txmax * GRID_RES_M
        by0 = CHILD_YMIN + tymin * GRID_RES_M
        by1 = CHILD_YMIN + tymax * GRID_RES_M
        ax.add_patch(
            plt.Rectangle(
                (bx0, by0),
                bx1 - bx0,
                by1 - by0,
                fill=False,
                edgecolor="blue",
                linewidth=2.5,
                zorder=6,
            )
        )
        ax.text(
            bx1 + 150,
            (by0 + by1) / 2,
            label,
            color="blue",
            fontsize=11,
            fontweight="bold",
            va="center",
            zorder=6,
        )

    # no outer frame — just the 4x4 grid lines
    for spine in ax.spines.values():
        spine.set_visible(False)


FIGSIZE = (8, 8)  # inches
DPI = 300  # -> 2400 x 2400 px per panel (>= full-page width)

# Fixed, identical margins for the two standalone panels (fraction of figure).
# Using the same subplots_adjust() for both — instead of each figure's own
# tight_layout() — guarantees the map rectangle sits at the exact same pixel
# position/size in both PNGs, so top/bottom whitespace matches when the two
# images are placed side by side (e.g. as separate \includegraphics in LaTeX).
PANEL_MARGINS = dict(left=0.115, right=0.97, top=0.975, bottom=0.08)


# ── Transfer-experiment layouts (train vs test tiles) ────────────────────────
# 200-m tiles on the 4x4 grid, in LOCAL grid units (0-800). Test tile fixed per
# transfer block; training tiles grow 1 -> 3 -> 8 -> 15 (15 = all others).
_TILE = 200
_ALL_TILES = [(y, x) for y in range(0, 800, _TILE) for x in range(0, 800, _TILE)]
EXPERIMENTS = {
    "transfer1": {"test": (200, 200), "ratios": {
        1:  [(200, 0)],
        3:  [(200, 0), (400, 0), (400, 200)],
        8:  [(200, 0), (200, 400), (400, 0), (400, 200), (400, 400),
             (600, 0), (600, 200), (600, 400)],
        15: "ALL"}},
    "transfer2": {"test": (0, 400), "ratios": {
        1:  [(0, 200)],
        3:  [(0, 200), (200, 200), (200, 400)],
        8:  [(0, 200), (0, 600), (200, 200), (200, 400), (200, 600),
             (400, 200), (400, 400), (400, 600)],
        15: "ALL"}},
}


def _tile_rect(ty, tx):
    """Local tile origin (ty, tx) -> (x0, y0, w, h) in BNG metres."""
    x0 = CHILD_XMIN + tx * GRID_RES_M
    y0 = CHILD_YMIN + ty * GRID_RES_M
    s  = _TILE * GRID_RES_M
    return x0, y0, s, s


def draw_experiment_layouts():
    """2x4 grid of transfer experiments on the Camden OSM basemap:
    training tiles (blue) vs the held-out test tile (red)."""
    from matplotlib.patches import Patch
    fig, axes = plt.subplots(2, 4, figsize=(22, 12))
    row_labels = {"transfer1": "Spatial Transfer 1", "transfer2": "Spatial Transfer 2"}
    col_labels = {1: "1 training tile", 3: "3 training tiles",
                  8: "8 training tiles", 15: "15 training tiles"}
    for r, tname in enumerate(("transfer1", "transfer2")):
        spec = EXPERIMENTS[tname]; test = spec["test"]
        for c, k in enumerate((1, 3, 8, 15)):
            ax = axes[r, c]
            ax.set_xlim(CHILD_XMIN, CHILD_XMAX); ax.set_ylim(CHILD_YMIN, CHILD_YMAX)
            ax.set_aspect("equal")
            cx.add_basemap(ax, crs=BNG, source=cx.providers.OpenStreetMap.Mapnik, zoom=13)
            train = spec["ratios"][k]
            if train == "ALL":
                train = [t for t in _ALL_TILES if t != test]
            for ty, tx in train:
                x0, y0, w, h = _tile_rect(ty, tx)
                ax.add_patch(plt.Rectangle((x0, y0), w, h, facecolor="tab:blue",
                                           alpha=0.35, edgecolor="tab:blue", linewidth=1.2))
            x0, y0, w, h = _tile_rect(*test)
            ax.add_patch(plt.Rectangle((x0, y0), w, h, facecolor="tab:red",
                                       alpha=0.45, edgecolor="red", linewidth=2.0))
            for i in range(5):                    # 4x4 grid lines
                ax.axvline(CHILD_XMIN + i * _TILE * GRID_RES_M, color="k", lw=0.4, alpha=0.4)
                ax.axhline(CHILD_YMIN + i * _TILE * GRID_RES_M, color="k", lw=0.4, alpha=0.4)
            if r == 0:
                ax.set_title(col_labels[k], fontsize=20, fontweight="bold", pad=12)
            if c == 0:
                ax.set_ylabel(row_labels[tname], fontsize=20, fontweight="bold", labelpad=14)
            ax.set_xticks([]); ax.set_yticks([])
    fig.legend(handles=[Patch(facecolor="tab:blue", alpha=0.35, label="training tile"),
                        Patch(facecolor="tab:red",  alpha=0.45, label="test (held-out) tile")],
               loc="lower center", ncol=2, fontsize=18, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.subplots_adjust(wspace=0.18, hspace=0.12)
    fig.savefig(os.path.join(FIG_DIR, "transfer_layouts.png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("Saved transfer_layouts.png")


def main():
    camden, gla, m25 = fetch_boundaries()

    # Panel (a) — parent domain
    fig_a, ax_a = plt.subplots(figsize=FIGSIZE)
    draw_parent_panel(ax_a, camden, gla, m25)
    fig_a.subplots_adjust(**PANEL_MARGINS)
    fig_a.savefig(os.path.join(FIG_DIR, "study_area_parent_domain.png"), dpi=DPI)
    plt.close(fig_a)

    # Panel (b) — child domain (requested study area)
    fig_b, ax_b = plt.subplots(figsize=FIGSIZE)
    draw_child_panel(ax_b, camden)
    fig_b.subplots_adjust(**PANEL_MARGINS)
    fig_b.savefig(os.path.join(FIG_DIR, "study_area_child_domain.png"), dpi=DPI)
    plt.close(fig_b)

    # Combined side-by-side figure, matching the reference layout
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIGSIZE[0] * 2, FIGSIZE[1]))
    draw_parent_panel(ax1, camden, gla, m25)
    draw_child_panel(ax2, camden)
    ax1.set_title("(a) Nested domain")
    ax2.set_title("(b) Child domain (Camden)")
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(FIG_DIR, "study_area_combined.png"), dpi=DPI)
    plt.close(fig)

    # Transfer-experiment layouts (train vs test tiles) on the same OSM basemap
    draw_experiment_layouts()

    print("Saved figures to", FIG_DIR)
    print(f"Panel size: {FIGSIZE[0]*DPI} x {FIGSIZE[1]*DPI} px each "
          f"(figsize={FIGSIZE}in @ {DPI} dpi)")


if __name__ == "__main__":
    main()
