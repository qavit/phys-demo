from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from physics import (
    complex_field,
    field_views,
    m_range_for_levels,
    path_difference_level,
    wavefront_radii,
)

st.set_page_config(
    page_title="雙點波源干涉｜Kakau",
    page_icon="〽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

matplotlib.rcParams["font.sans-serif"] = [
    "Taipei Sans TC",
    "PingFang TC",
    "Microsoft JhengHei",
    "Noto Sans CJK TC",
    "sans-serif",
]
matplotlib.rcParams["axes.unicode_minus"] = False

KAKAU_TEAL = "#0f8f8a"
KAKAU_AMBER = "#d8951d"


def draw_level_family(
    ax,
    x_grid,
    y_grid,
    path_difference,
    wavelength,
    phase_difference,
    m_values,
    offset,
    source1,
    source2,
    source_distance,
    color,
    x_limits,
):
    """Draw phase-condition hyperbolas, including their degenerate rays."""
    tolerance = 1e-3 * wavelength
    for m in m_values:
        level = path_difference_level(m, wavelength, phase_difference, offset)

        if abs(level - source_distance) <= tolerance:
            ax.plot(
                [x_limits[0], source1[0]],
                [source1[1], source1[1]],
                color=color,
                linewidth=1.25,
                zorder=3,
            )
            continue
        if abs(level + source_distance) <= tolerance:
            ax.plot(
                [source2[0], x_limits[1]],
                [source2[1], source2[1]],
                color=color,
                linewidth=1.25,
                zorder=3,
            )
            continue

        ax.contour(
            x_grid,
            y_grid,
            path_difference - level,
            levels=[0.0],
            colors=[color],
            linewidths=1.25,
            linestyles="-",
            zorder=3,
        )


def draw_wavefronts(ax, source, wavelength, phase, maximum_radius):
    """Draw crest and trough circles using the same cos convention as the field."""
    for radius in wavefront_radii(wavelength, phase, maximum_radius):
        ax.add_patch(
            plt.Circle(
                source,
                radius,
                fill=False,
                color="white",
                linewidth=0.6,
                linestyle="-",
                zorder=4,
            )
        )
    for radius in wavefront_radii(wavelength, phase, maximum_radius, trough=True):
        ax.add_patch(
            plt.Circle(
                source,
                radius,
                fill=False,
                color="white",
                linewidth=0.6,
                linestyle="--",
                zorder=4,
            )
        )


st.title("雙點波源干涉")
st.caption("Kakau Interactive Lab｜把路徑差、相位差與雙曲線放在同一張圖上")

view = st.sidebar.radio(
    "底圖",
    ("時間平均強度", "合成振幅", "合成相位"),
    index=0,
    help="強度與振幅來自複數相量合成；相位是合成場的 arg(C)。",
)
show_antinode = st.sidebar.checkbox("顯示相長條件線", value=True)
show_nodal = st.sidebar.checkbox("顯示相消條件線", value=True)
show_wavefront = st.sidebar.checkbox("顯示兩源波前", value=False)

wavelength = st.sidebar.slider("波長 λ（m）", 0.05, 2.0, 0.5, step=0.05)
source_distance = st.sidebar.slider("波源間距 d（m）", 0.1, 5.0, 1.0, step=0.1)

st.sidebar.markdown("### 波源 1")
amplitude1 = st.sidebar.slider("振幅 A₁", 0.0, 2.0, 1.0, step=0.1)
phase1_degrees = st.sidebar.slider("初相位 φ₁（度）", 0, 360, 0, step=5)

st.sidebar.markdown("### 波源 2")
amplitude2 = st.sidebar.slider("振幅 A₂", 0.0, 2.0, 1.0, step=0.1)
phase2_degrees = st.sidebar.slider("初相位 φ₂（度）", 0, 360, 0, step=5)

with st.sidebar.expander("進階設定"):
    grid_size = st.slider("觀測區域邊長（m）", 2.0, 12.0, 6.0, step=0.5)
    resolution = st.slider("解析度（點）", 200, 800, 450, step=50)
    attenuate = st.checkbox(
        "啟用 1/√r 距離衰減",
        value=False,
        help="採二維圓柱波的遠場近似，並在波源附近使用有限核心半徑避免發散。",
    )

phase1 = np.deg2rad(phase1_degrees)
phase2 = np.deg2rad(phase2_degrees)
phase_difference = phase2 - phase1

coordinates = np.linspace(-grid_size / 2.0, grid_size / 2.0, resolution)
x_grid, y_grid = np.meshgrid(coordinates, coordinates)
source1 = np.array([-source_distance / 2.0, 0.0])
source2 = np.array([source_distance / 2.0, 0.0])
r1 = np.hypot(x_grid - source1[0], y_grid - source1[1])
r2 = np.hypot(x_grid - source2[0], y_grid - source2[1])
path_difference = r2 - r1

field = complex_field(
    r1,
    r2,
    wavelength,
    amplitude1,
    amplitude2,
    phase1,
    phase2,
    attenuate,
)
amplitude, intensity, phase = field_views(field)

figure, axis = plt.subplots(figsize=(7.2, 7.2), constrained_layout=True)
extent = (
    -grid_size / 2.0,
    grid_size / 2.0,
    -grid_size / 2.0,
    grid_size / 2.0,
)

if view == "合成相位":
    phase_floor = max(float(amplitude.max()) * 1e-8, 1e-12)
    display_data = np.ma.masked_where(amplitude < phase_floor, phase)
    image = axis.imshow(
        display_data,
        extent=extent,
        cmap="twilight",
        origin="lower",
        vmin=-np.pi,
        vmax=np.pi,
        zorder=1,
    )
    colorbar_label = "Phase (rad)"
elif view == "合成振幅":
    display_max = max(float(np.percentile(amplitude, 99.5)), 1e-12)
    image = axis.imshow(
        amplitude,
        extent=extent,
        cmap="magma",
        origin="lower",
        vmin=0,
        vmax=display_max,
        zorder=1,
    )
    colorbar_label = "Resultant amplitude (relative)"
else:
    display_max = max(float(np.percentile(intensity, 99.5)), 1e-12)
    image = axis.imshow(
        intensity,
        extent=extent,
        cmap="gray",
        origin="lower",
        vmin=0,
        vmax=display_max,
        zorder=1,
    )
    colorbar_label = "Mean intensity ∝ |C|²"

colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
colorbar.set_label(colorbar_label)

visible_minimum, visible_maximum = -source_distance, source_distance
if show_antinode:
    antinode_m = m_range_for_levels(
        visible_minimum,
        visible_maximum,
        wavelength,
        phase_difference,
        offset=0.0,
    )
    draw_level_family(
        axis,
        x_grid,
        y_grid,
        path_difference,
        wavelength,
        phase_difference,
        antinode_m,
        0.0,
        source1,
        source2,
        source_distance,
        KAKAU_AMBER,
        extent[:2],
    )

if show_nodal:
    nodal_m = m_range_for_levels(
        visible_minimum,
        visible_maximum,
        wavelength,
        phase_difference,
        offset=0.5,
    )
    draw_level_family(
        axis,
        x_grid,
        y_grid,
        path_difference,
        wavelength,
        phase_difference,
        nodal_m,
        0.5,
        source1,
        source2,
        source_distance,
        KAKAU_TEAL,
        extent[:2],
    )

if show_wavefront:
    maximum_radius = np.sqrt(2.0) * grid_size
    draw_wavefronts(axis, source1, wavelength, phase1, maximum_radius)
    draw_wavefronts(axis, source2, wavelength, phase2, maximum_radius)

axis.scatter(
    [source1[0], source2[0]],
    [source1[1], source2[1]],
    c=[KAKAU_AMBER, KAKAU_AMBER],
    edgecolors="white",
    linewidths=0.8,
    s=42,
    zorder=5,
)
axis.text(source1[0], source1[1] + 0.12, "S₁", color="white", ha="center", zorder=6)
axis.text(source2[0], source2[1] + 0.12, "S₂", color="white", ha="center", zorder=6)
axis.set_xlabel("x (m)")
axis.set_ylabel("y (m)")
axis.set_aspect("equal", adjustable="box")
axis.set_xlim(extent[0], extent[1])
axis.set_ylim(extent[2], extent[3])
axis.set_autoscale_on(False)

st.pyplot(figure, width="stretch")
plt.close(figure)

legend_items = []
if show_antinode:
    legend_items.append("🟠 相長條件線")
if show_nodal:
    legend_items.append("🟢 相消條件線")
if show_wavefront:
    legend_items.append("白實線／虛線：單一波源的波峰／波谷")
if legend_items:
    st.caption("　｜　".join(legend_items))

if amplitude1 != amplitude2 or attenuate:
    st.info(
        "相消條件線表示兩列波相位相反；當抵達該點的振幅不同時，合成振幅不會完全等於零。"
    )

with st.expander("這張圖算的是什麼？"):
    st.markdown(
        r"""
令兩列波的複數相量為

$$
C=A_1e^{i(kr_1+\phi_1)}+A_2e^{i(kr_2+\phi_2)}.
$$

合成振幅是 $|C|$，時間平均強度正比於 $|C|^2$，合成相位是 $\arg C$。
若開啟距離衰減，振幅採二維圓柱波遠場的 $1/\sqrt r$ 近似；波源附近以有限核心半徑取代不合理的奇點。
"""
    )

st.markdown("[返回 Kakau 互動實驗室](https://kakau.tw/lab/interference)")
