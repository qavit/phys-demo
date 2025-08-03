import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("雙點波源干涉模擬（可調振幅與相位）")

# 側邊欄：主要參數
wavelength = st.sidebar.slider("波長 λ（m）", 0.01, 2.0, 0.5, step=0.01)
source_distance = st.sidebar.slider("波源間距 d（m）", 0.1, 5.0, 1.0, step=0.1)
grid_size = st.sidebar.slider("觀測區域大小（m）", 2.0, 10.0, 5.0, step=0.5)
resolution = st.sidebar.slider("解析度（點數）", 200, 1000, 400, step=50)

# 側邊欄：兩源參數設定
st.sidebar.markdown("### 波源1")
amp1 = st.sidebar.slider("振幅 A₁", 0.0, 2.0, 1.0, step=0.1)
phase1 = st.sidebar.slider("初相位 φ₁（度）", 0, 360, 0, step=5)

st.sidebar.markdown("### 波源2")
amp2 = st.sidebar.slider("振幅 A₂", 0.0, 2.0, 1.0, step=0.1)
phase2 = st.sidebar.slider("初相位 φ₂（度）", 0, 360, 0, step=5)

# 空間網格
x = np.linspace(-grid_size / 2, grid_size / 2, resolution)
y = np.linspace(-grid_size / 2, grid_size / 2, resolution)
X, Y = np.meshgrid(x, y)

# 波源位置
s1 = np.array([-source_distance / 2, 0])
s2 = np.array([ source_distance / 2, 0])

# 距離與波數
r1 = np.sqrt((X - s1[0])**2 + (Y - s1[1])**2)
r2 = np.sqrt((X - s2[0])**2 + (Y - s2[1])**2)
k = 2 * np.pi / wavelength

# 將度轉為弧度
φ1 = np.deg2rad(phase1)
φ2 = np.deg2rad(phase2)

# 波疊加：加入振幅與相位偏移
wave1 = amp1 * np.sin(k * r1 + φ1)
wave2 = amp2 * np.sin(k * r2 + φ2)
interference = wave1 + wave2
intensity = interference**2

# 顯示圖形
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_title("干涉圖樣")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
im = ax.imshow(intensity, extent=(-grid_size/2, grid_size/2, -grid_size/2, grid_size/2),
               cmap='gray', origin='lower')
st.pyplot(fig)
