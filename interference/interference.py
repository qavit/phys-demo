import numpy as np
import streamlit as st
import matplotlib
import matplotlib.pyplot as plt

# 中文字型（避免缺字）
matplotlib.rcParams["font.sans-serif"] = [
    "Taipei Sans TC", "PingFang TC", "Microsoft JhengHei", "sans-serif"
]
matplotlib.rcParams["axes.unicode_minus"] = False


def m_range_for_levels(d_min, d_max, lam, delta_phi, offset):
    """
    取得使 c_m = (m + offset - delta_phi/(2π)) * lam ∈ [d_min, d_max] 的整數 m 範圍。
    offset=0   → 波腹線 (constructive)
    offset=0.5 → 波節線 (destructive)
    """
    shift = delta_phi / (2.0 * np.pi)
    lower = int(np.ceil(d_min / lam - offset + shift))
    upper = int(np.floor(d_max / lam - offset + shift))
    if upper < lower:
        return []
    return list(range(lower, upper + 1))


def draw_antinode_family(ax, X, Y, rdiff, lam, delta_phi, m_vals, d, s1, s2,
                         color="magenta", linewidth=1.1, zorder=3, tol=None, xlim=None):
    """
    以數學方程繪製「波腹線」族：
      - 一般情況：r2 - r1 = c_m 的等高線（雙曲線）
      - 退化情況：c_m ≈ +d → 畫出 S1 外側的左半線；c_m ≈ -d → 畫出 S2 外側的右半線
    """
    # 容差：用 λ 的小比例，避免浮點誤差導致退化線漏畫
    if tol is None:
        tol = 1e-3 * lam

    shift = delta_phi / (2.0 * np.pi)
    for m in m_vals:
        c_m = (m + 0.0 - shift) * lam  # offset=0 for antinode

        # 退化：r2 - r1 = +d（左半線，從 S1 向左延伸）
        if abs(c_m - d) <= tol:
            if xlim is None:
                x_left = np.min(X)
            else:
                x_left = xlim[0]
            ax.plot([x_left, s1[0]], [s1[1], s1[1]], color=color,
                    linestyle='-', linewidth=linewidth, zorder=zorder)
            continue

        # 退化：r2 - r1 = -d（右半線，從 S2 向右延伸）
        if abs(c_m + d) <= tol:
            if xlim is None:
                x_right = np.max(X)
            else:
                x_right = xlim[1]
            ax.plot([s2[0], x_right], [s2[1], s2[1]], color=color,
                    linestyle='-', linewidth=linewidth, zorder=zorder)
            continue

        # 一般：用等值線畫 r2 - r1 = c_m
        ax.contour(X, Y, rdiff - c_m, levels=[0.0], colors=[color],
                   linewidths=linewidth, linestyles='-', zorder=zorder)


def draw_hyperbola_family(ax, X, Y, rdiff, lam, delta_phi, m_vals, offset, color,
                          linewidth=1.1, linestyle="-", zorder=3):
    """
    以數學方程繪製一族雙曲線（節線或腹線的通用器，但這裡保留給「節線」使用）：
    rdiff - c_m = 0 with c_m = (m + offset - delta_phi/(2π)) * lam
    """
    shift = delta_phi / (2.0 * np.pi)
    for m in m_vals:
        c_m = (m + offset - shift) * lam
        ax.contour(X, Y, rdiff - c_m, levels=[0.0], colors=[color],
                   linewidths=linewidth, linestyles=linestyle, zorder=zorder)


def draw_wavefronts(ax, source, lam, phi, r_max,
                    crest_style, trough_style, linewidth=0.6, zorder=4):
    """
    單一波源的波前（同心圓）：
      - 波峰（白實線）：r = (n - phi/(2π)) * lam
      - 波谷（白虛線）：r = (n + 1/2 - phi/(2π)) * lam
    """
    phi_shift = phi / (2.0 * np.pi)

    # 波峰
    n_min_peak = int(np.ceil(phi_shift))
    n_max_peak = int(np.floor(r_max / lam + phi_shift))
    for n in range(n_min_peak, n_max_peak + 1):
        r = (n - phi_shift) * lam
        if r <= 0:
            continue
        circle = plt.Circle(source, r, fill=False, **crest_style,
                            linewidth=linewidth, zorder=zorder)
        ax.add_patch(circle)

    # 波谷
    n_min_trough = int(np.ceil(phi_shift - 0.5))
    n_max_trough = int(np.floor(r_max / lam + phi_shift - 0.5))
    for n in range(n_min_trough, n_max_trough + 1):
        r = (n + 0.5 - phi_shift) * lam
        if r <= 0:
            continue
        circle = plt.Circle(source, r, fill=False, **trough_style,
                            linewidth=linewidth, zorder=zorder)
        ax.add_patch(circle)


st.title("雙點波源干涉模擬")

# === 側欄 ===
use_atten = st.sidebar.checkbox("啟用距離衰減", value=False)
show_phase_map = st.sidebar.checkbox("顯示相位圖（彩色）", value=False)
show_nodal = st.sidebar.checkbox("顯示波節線", value=False)
show_antinode = st.sidebar.checkbox("顯示波腹線", value=False)
show_wavefronts = st.sidebar.checkbox("顯示兩源波前", value=False)

lam = st.sidebar.slider("波長 λ（m）", 0.01, 2.0, 0.5, step=0.01)
d_src = st.sidebar.slider("波源間距 d（m）", 0.1, 5.0, 1.0, step=0.1)
grid = st.sidebar.slider("觀測區域邊長（m）", 2.0, 12.0, 6.0, step=0.5)
res = st.sidebar.slider("解析度（點）", 200, 1200, 500, step=50)

st.sidebar.markdown("### 波源 1")
A1 = st.sidebar.slider("振幅 A₁", 0.0, 2.0, 1.0, step=0.1)
phi1_deg = st.sidebar.slider("初相位 φ₁（度）", 0, 360, 0, step=5)

st.sidebar.markdown("### 波源 2")
A2 = st.sidebar.slider("振幅 A₂", 0.0, 2.0, 1.0, step=0.1)
phi2_deg = st.sidebar.slider("初相位 φ₂（度）", 0, 360, 0, step=5)

# === 網格與幾何 ===
x = np.linspace(-grid / 2.0, grid / 2.0, res)
y = np.linspace(-grid / 2.0, grid / 2.0, res)
X, Y = np.meshgrid(x, y)

s1 = np.array([-d_src / 2.0, 0.0])
s2 = np.array([+d_src / 2.0, 0.0])
d = float(d_src)  # 兩焦點距離

r1 = np.hypot(X - s1[0], Y - s1[1])
r2 = np.hypot(X - s2[0], Y - s2[1])
rdiff = r2 - r1

phi1 = np.deg2rad(phi1_deg)
phi2 = np.deg2rad(phi2_deg)
delta_phi = phi2 - phi1
k = 2.0 * np.pi / lam

# === 底圖 ===
if use_atten:
    eps = 1e-6
    w1 = (A1 / (r1 + eps)) * np.sin(k * r1 + phi1)
    w2 = (A2 / (r2 + eps)) * np.sin(k * r2 + phi2)
else:
    w1 = A1 * np.sin(k * r1 + phi1)
    w2 = A2 * np.sin(k * r2 + phi2)

psi = w1 + w2

fig, ax = plt.subplots(figsize=(7.2, 7.2))
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")

extent = (-grid / 2.0, grid / 2.0, -grid / 2.0, grid / 2.0)
if show_phase_map:
    phase_map = np.angle(np.exp(1j * psi))
    ax.imshow(phase_map, extent=extent, cmap="twilight", origin="lower", zorder=1)
    ax.set_title("Phase map（twilight）")
else:
    intensity = psi ** 2
    ax.imshow(intensity, extent=extent, cmap="gray", origin="lower", zorder=1)
    ax.set_title("Intensity map (greyscale)")

# 固定視窗，避免波前圓改變範圍
ax.set_aspect("equal", adjustable="box")
ax.set_xlim(-grid / 2.0, grid / 2.0)
ax.set_ylim(-grid / 2.0, grid / 2.0)
ax.set_autoscale_on(False)

# === 計算可見範圍的 m 範圍 ===
# 性質：對任意點，|r2 - r1| ≤ d，所以只需在 [-d, d] 搜索
d_min, d_max = -d, d

# 波腹線（含退化半線）
if show_antinode:
    m_antinode = m_range_for_levels(d_min, d_max, lam, delta_phi, offset=0.0)
    draw_antinode_family(
        ax, X, Y, rdiff, lam, delta_phi, m_antinode, d, s1, s2,
        color="magenta", linewidth=1.2, zorder=3, tol=1e-3 * lam, xlim=(-grid / 2.0, grid / 2.0)
    )

# 波節線（純雙曲線）
if show_nodal:
    m_nodal = m_range_for_levels(d_min, d_max, lam, delta_phi, offset=0.5)
    draw_hyperbola_family(
        ax, X, Y, rdiff, lam, delta_phi, m_nodal,
        offset=0.5, color="cyan", linewidth=1.2, linestyle="-", zorder=3
    )

# 兩源波前（白：峰/谷）
if show_wavefronts:
    r_max = np.sqrt(2.0) * (grid / 2.0)
    crest_style = dict(color="white", linestyle="-")
    trough_style = dict(color="white", linestyle="--")
    draw_wavefronts(ax, s1, lam, phi1, r_max, crest_style, trough_style)
    draw_wavefronts(ax, s2, lam, phi2, r_max, crest_style, trough_style)

st.pyplot(fig)
