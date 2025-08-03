import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 中文字體設定
matplotlib.rcParams['font.sans-serif'] = ['Taipei Sans TC Beta', 'Microsoft JhengHei', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

st.title("雙點波源干涉模擬")

# === Toggle 按鈕 ===
use_attenuation = st.sidebar.checkbox("啟用距離衰減", value=True)
show_phase_map = st.sidebar.checkbox("顯示相位圖（彩色）", value=False)

# === 參數區 ===
wavelength = st.sidebar.slider("波長 λ（m）", 0.01, 2.0, 0.5, step=0.01)
source_distance = st.sidebar.slider("波源間距 d（m）", 0.1, 5.0, 1.0, step=0.1)
grid_size = st.sidebar.slider("觀測區域大小（m）", 2.0, 10.0, 5.0, step=0.5)
resolution = st.sidebar.slider("解析度（點數）", 200, 1000, 400, step=50)

st.sidebar.markdown("### 波源1")
amp1 = st.sidebar.slider("振幅 A₁", 0.0, 2.0, 1.0, step=0.1)
phase1 = st.sidebar.slider("初相位 φ₁（度）", 0, 360, 0, step=5)

st.sidebar.markdown("### 波源2")
amp2 = st.sidebar.slider("振幅 A₂", 0.0, 2.0, 1.0, step=0.1)
phase2 = st.sidebar.slider("初相位 φ₂（度）", 0, 360, 0, step=5)

# === 空間網格 ===
x = np.linspace(-grid_size / 2, grid_size / 2, resolution)
y = np.linspace(-grid_size / 2, grid_size / 2, resolution)
X, Y = np.meshgrid(x, y)

s1 = np.array([-source_distance / 2, 0])
s2 = np.array([ source_distance / 2, 0])
r1 = np.sqrt((X - s1[0])**2 + (Y - s1[1])**2)
r2 = np.sqrt((X - s2[0])**2 + (Y - s2[1])**2)

φ1 = np.deg2rad(phase1)
φ2 = np.deg2rad(phase2)
k = 2 * np.pi / wavelength

# === 波函數 ===
if use_attenuation:
    wave1 = (amp1 / (r1 + 1e-6)) * np.sin(k * r1 + φ1)
    wave2 = (amp2 / (r2 + 1e-6)) * np.sin(k * r2 + φ2)
else:
    wave1 = amp1 * np.sin(k * r1 + φ1)
    wave2 = amp2 * np.sin(k * r2 + φ2)

ψ = wave1 + wave2

# === 相位圖 or 強度圖 ===
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")

if show_phase_map:
    phase = np.angle(np.exp(1j * ψ))  # 抽象出相位
    im = ax.imshow(phase, extent=(-grid_size/2, grid_size/2, -grid_size/2, grid_size/2),
                   cmap='twilight', origin='lower')
    ax.set_title("干涉圖樣（相位彩圖）")
else:
    intensity = ψ**2
    im = ax.imshow(intensity, extent=(-grid_size/2, grid_size/2, -grid_size/2, grid_size/2),
                   cmap='gray', origin='lower')
    ax.set_title("干涉圖樣（強度）")

st.pyplot(fig)
