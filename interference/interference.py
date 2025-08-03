import numpy as np
import streamlit as st
import matplotlib
import matplotlib.pyplot as plt

# 中文字體
matplotlib.rcParams["font.sans-serif"] = [
    "Taipei Sans TC Beta", "PingFang TC", "Microsoft JhengHei", "sans-serif"
]
matplotlib.rcParams["axes.unicode_minus"] = False


def m_range_for_levels(d_min, d_max, lam, delta_phi, offset):
    """
    決定 m 的整數範圍，使得 c_m = (m + offset - delta_phi/(2π)) * lam
    會落在 [d_min, d_max] 之內。
    offset = 0   → 波腹線 (constructive)
    offset = 0.5 → 波節線 (destructive)
    """
    shift = delta_phi / (2.0 * np.pi)  # (φ2 - φ1) / 2π
    lower = int(np.ceil(d_min / lam - offset + shift))
    upper = int(np.floor(d_max / lam - offset + shift))
    if upper < lower:
        return []
    return list(range(lower, upper + 1))


def draw_hyperbola_family(ax, X, Y, rdiff, lam, delta_phi, m_vals, offset, color,
                          linewidth=1.0, linestyle="-", zorder=3):
    """
    以數學方程繪製一族雙曲線：D(x,y) - c_m = 0 的等高線（level=0）。
    rdiff = r2 - r1；c_m = (m + offset - delta_phi/(2π)) * lam
    """
    shift = delta_phi / (2.0 * np.pi)
    for m in m_vals:
        c_m = (m + offset - shift) * lam
        ax.contour(X, Y, rdiff - c_m, levels=[0.0], colors=[color],
                   linewidths=linewidth, linestyles=linestyle, zorder=zorder)


def draw_wavefronts(ax, source, lam, phi, r_max, crest_style, trough_style,
                    linewidth=0.6, zorder=4):
    """
    以數學方程繪製單一波源的波前（同心圓）：
    crest: r = (n - phi/(2π)) * lam（白實線）
    trough: r = (n + 1/2 - phi/(2π)) * lam（白虛線）
    """
    phi_shift = phi / (2.0 * np.pi)

    # 波峰（白實線）
    n_min_peak = int(np.ceil(phi_shift))               # r >= 0
    n_max_peak = int(np.floor(r_max / lam + phi_shift))
    for n in range(n_min_peak, n_max_peak + 1):
        r = (n - phi_shift) * lam
        if r <= 0:
            continue
        circle = plt.Circle(source, r, fill=False, **crest_style,
                            linewidth=linewidth, zorder=zorder)
        ax.add_patch(circle)

    # 波谷（白虛線）
    n_min_trough = int(np.ceil(phi_shift - 0.5))       # r >= 0
    n_max_trough = int(np.floor(r_max / lam + phi_shift - 0.5))
    for n in range(n_min_trough, n_max_trough + 1):
        r = (n + 0.5 - phi_shift) * lam
        if r <= 0:
            continue
        circle = plt.Circle(source, r, fill=False, **trough_style,
                            linewidth=linewidth, zorder=zorder)
        ax.add_patch(circle)


st.title("雙點波源干涉模擬")

# === 側欄參數 ===
use_atten = st.sidebar.checkbox("啟用距離衰減", value=False)
show_phase_map = st.sidebar.checkbox("顯示相位圖（彩色）", value=False)
show_nodal = st.sidebar.checkbox("顯示波節線（cyan 實線）", value=False)
show_antinode = st.sidebar.checkbox("顯示波腹線（magenta 實線）", value=False)
show_wavefronts = st.sidebar.checkbox("顯示兩源波前（白：峰/谷）", value=False)

lam = st.sidebar.slider("波長 λ（m）", 0.01, 2.0, 0.5, step=0.01)
d_src = st.sidebar.slider("波源間距 d（m）", 0.1, 5.0, 1.0, step=0.1)
grid = st.sidebar.slider("觀測區域邊長（m）", 2.0, 10.0, 5.0, step=0.5)
res = st.sidebar.slider("解析度（點）", 200, 1200, 500, step=50)

st.sidebar.markdown("### 波源 1")
A1 = st.sidebar.slider("振幅 A₁", 0.0, 2.0, 1.0, step=0.1)
phi1_deg = st.sidebar.slider("初相位 φ₁（度）", 0, 360, 0, step=5)

st.sidebar.markdown("### 波源 2")
A2 = st.sidebar.slider("振幅 A₂", 0.0, 2.0, 1.0, step=0.1)
phi2_deg = st.sidebar.slider("初相位 φ₂（度）", 0, 360, 0, step=5)

# === 網格 ===
x = np.linspace(-grid / 2.0, grid / 2.0, res)
y = np.linspace(-grid / 2.0, grid / 2.0, res)
X, Y = np.meshgrid(x, y)

# 兩源位置（置於 x 軸）
s1 = np.array([-d_src / 2.0, 0.0])
s2 = np.array([+d_src / 2.0, 0.0])

# 距離
r1 = np.hypot(X - s1[0], Y - s1[1])
r2 = np.hypot(X - s2[0], Y - s2[1])
rdiff = r2 - r1

# 相位
phi1 = np.deg2rad(phi1_deg)
phi2 = np.deg2rad(phi2_deg)
delta_phi = phi2 - phi1
k = 2.0 * np.pi / lam

# 波形（強度/相位底圖用）
if use_atten:
    eps = 1e-6
    w1 = (A1 / (r1 + eps)) * np.sin(k * r1 + phi1)
    w2 = (A2 / (r2 + eps)) * np.sin(k * r2 + phi2)
else:
    w1 = A1 * np.sin(k * r1 + phi1)
    w2 = A2 * np.sin(k * r2 + phi2)

psi = w1 + w2

# === 底圖 ===
fig, ax = plt.subplots(figsize=(6.8, 6.8))
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")

extent = (-grid / 2.0, grid / 2.0, -grid / 2.0, grid / 2.0)
if show_phase_map:
    phase_map = np.angle(np.exp(1j * psi))
    ax.imshow(phase_map, extent=extent, cmap="twilight", origin="lower", zorder=1)
    ax.set_title("相位圖（twilight colormap）")
else:
    intensity = psi ** 2
    ax.imshow(intensity, extent=extent, cmap="gray", origin="lower", zorder=1)
    ax.set_title("強度圖（灰階）")

# === 決定 m 範圍（僅繪出觀測區內會出現的雙曲線） ===
d_min = float(np.min(rdiff))
d_max = float(np.max(rdiff))

# 波腹線（offset=0）
if show_antinode:
    m_antinode = m_range_for_levels(d_min, d_max, lam, delta_phi, offset=0.0)
    draw_hyperbola_family(
        ax, X, Y, rdiff, lam, delta_phi, m_antinode,
        offset=0.0, color="magenta", linewidth=1.1, linestyle="-", zorder=3
    )

# 波節線（offset=0.5）
if show_nodal:
    m_nodal = m_range_for_levels(d_min, d_max, lam, delta_phi, offset=0.5)
    draw_hyperbola_family(
        ax, X, Y, rdiff, lam, delta_phi, m_nodal,
        offset=0.5, color="cyan", linewidth=1.1, linestyle="-", zorder=3
    )

# === 兩源波前（白實：峰；白虛：谷） ===
ax.set_aspect("equal", adjustable="box")
ax.set_xlim(-grid / 2.0, grid / 2.0)
ax.set_ylim(-grid / 2.0, grid / 2.0)
ax.set_autoscale_on(False)  # 防止 circle 影響座標範圍

if show_wavefronts:
    r_max = np.sqrt(2.0) * (grid / 2.0)  # 畫到角落即可
    crest_style = dict(color="white", linestyle="-")
    trough_style = dict(color="white", linestyle="--")
    draw_wavefronts(ax, s1, lam, phi1, r_max, crest_style, trough_style)
    draw_wavefronts(ax, s2, lam, phi2, r_max, crest_style, trough_style)

st.pyplot(fig)
