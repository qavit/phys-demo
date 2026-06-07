"""Transient circuit lab with Streamlit, Schemdraw, and Matplotlib.

Install:
    pip install streamlit numpy matplotlib schemdraw

Run:
    streamlit run app.py
"""

from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass
from enum import Enum

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import schemdraw
import schemdraw.elements as elm
import streamlit as st
import streamlit.components.v1 as components


EPSILON = 1e-12


class CircuitMode(str, Enum):
    """Available circuit modes."""

    AUTO = "自動判斷"
    RC = "RC 充電"
    RL = "RL 通電"
    LC = "LC 自由振盪"
    RLC = "RLC 自由響應"


class PlotType(str, Enum):
    """Available waveform plots."""

    CURRENT = "電流"
    VOLTAGE = "電壓"
    ENERGY = "能量"


@dataclass
class CircuitParams:
    """Circuit parameters."""

    resistance: float
    capacitance: float
    inductance: float
    voltage: float
    initial_charge: float
    time_max: float
    points: int


@dataclass
class CircuitState:
    """Instantaneous state."""

    time: float
    current: float
    voltage_r: float
    voltage_c: float
    voltage_l: float
    energy_c: float
    energy_l: float


def configure_matplotlib_fonts() -> None:
    """Configure Matplotlib fonts for common Traditional Chinese systems."""
    preferred_fonts = [
        "PingFang TC",
        "Heiti TC",
        "Microsoft JhengHei",
        "Noto Sans CJK TC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    available_fonts = {font.name for font in fm.fontManager.ttflist}

    for font_name in preferred_fonts:
        if font_name in available_fonts:
            plt.rcParams["font.family"] = font_name
            break

    plt.rcParams["axes.unicode_minus"] = False


def is_positive(value: float) -> bool:
    """Return whether a component value is effectively positive."""
    return value > EPSILON


def format_si(value: float, unit: str) -> str:
    """Format a number with engineering prefixes."""
    abs_value = abs(value)

    if abs_value < EPSILON:
        return f"0 {unit}"

    prefixes = [
        (1e9, "G"),
        (1e6, "M"),
        (1e3, "k"),
        (1.0, ""),
        (1e-3, "m"),
        (1e-6, "μ"),
        (1e-9, "n"),
        (1e-12, "p"),
    ]

    for scale, prefix in prefixes:
        scaled = value / scale
        if abs(scaled) >= 1:
            return f"{scaled:.3g} {prefix}{unit}"

    return f"{value:.3g} {unit}"


def rc_response(time_array: np.ndarray, params: CircuitParams) -> dict:
    """Return RC charging response."""
    tau = params.resistance * params.capacitance
    voltage_c = params.voltage * (1.0 - np.exp(-time_array / tau))
    current = (params.voltage / params.resistance) * np.exp(
        -time_array / tau
    )
    voltage_r = current * params.resistance
    voltage_l = np.zeros_like(time_array)
    energy_c = 0.5 * params.capacitance * voltage_c**2
    energy_l = np.zeros_like(time_array)

    return {
        "time": time_array,
        "current": current,
        "voltage_r": voltage_r,
        "voltage_c": voltage_c,
        "voltage_l": voltage_l,
        "energy_c": energy_c,
        "energy_l": energy_l,
        "summary": f"τ = RC = {format_si(tau, 's')}",
        "title": rf"RC charging: $\tau=RC={tau:.4g}\ \mathrm{{s}}$",
    }


def rl_response(time_array: np.ndarray, params: CircuitParams) -> dict:
    """Return RL switch-on response."""
    tau = params.inductance / params.resistance
    current = (params.voltage / params.resistance) * (
        1.0 - np.exp(-time_array / tau)
    )
    voltage_r = current * params.resistance
    voltage_l = params.voltage * np.exp(-time_array / tau)
    voltage_c = np.zeros_like(time_array)
    energy_c = np.zeros_like(time_array)
    energy_l = 0.5 * params.inductance * current**2

    return {
        "time": time_array,
        "current": current,
        "voltage_r": voltage_r,
        "voltage_c": voltage_c,
        "voltage_l": voltage_l,
        "energy_c": energy_c,
        "energy_l": energy_l,
        "summary": f"τ = L/R = {format_si(tau, 's')}",
        "title": rf"RL switch-on: $\tau=L/R={tau:.4g}\ \mathrm{{s}}$",
    }


def lc_response(time_array: np.ndarray, params: CircuitParams) -> dict:
    """Return ideal LC free oscillation response."""
    omega = 1.0 / np.sqrt(params.inductance * params.capacitance)
    charge = params.initial_charge * np.cos(omega * time_array)
    current = params.initial_charge * omega * np.sin(omega * time_array)
    voltage_c = charge / params.capacitance
    voltage_l = -voltage_c
    voltage_r = np.zeros_like(time_array)
    energy_c = charge**2 / (2.0 * params.capacitance)
    energy_l = 0.5 * params.inductance * current**2

    return {
        "time": time_array,
        "current": current,
        "charge": charge,
        "voltage_r": voltage_r,
        "voltage_c": voltage_c,
        "voltage_l": voltage_l,
        "energy_c": energy_c,
        "energy_l": energy_l,
        "summary": f"ω₀ = {omega:.3g} rad/s",
        "title": (
            rf"LC oscillation: $\omega_0=1/\sqrt{{LC}}"
            rf"={omega:.4g}\ \mathrm{{rad/s}}$"
        ),
    }


def rlc_response(time_array: np.ndarray, params: CircuitParams) -> dict:
    """Return free response for a series RLC circuit.

    Equation:
        L q'' + R q' + q / C = 0

    Initial conditions:
        q(0) = Q0
        i(0) = 0
        i = -dq/dt
    """
    alpha = params.resistance / (2.0 * params.inductance)
    omega_0 = 1.0 / np.sqrt(params.inductance * params.capacitance)

    if abs(alpha - omega_0) < 1e-9:
        charge = params.initial_charge * np.exp(-alpha * time_array) * (
            1.0 + alpha * time_array
        )
        damping_type = "臨界阻尼"

    elif alpha < omega_0:
        omega_d = np.sqrt(omega_0**2 - alpha**2)
        charge = params.initial_charge * np.exp(-alpha * time_array) * (
            np.cos(omega_d * time_array)
            + alpha / omega_d * np.sin(omega_d * time_array)
        )
        damping_type = "欠阻尼"

    else:
        beta = np.sqrt(alpha**2 - omega_0**2)
        s1 = -alpha + beta
        s2 = -alpha - beta
        denominator = s1 - s2
        coef_a = -params.initial_charge * s2 / denominator
        coef_b = params.initial_charge * s1 / denominator
        charge = coef_a * np.exp(s1 * time_array)
        charge += coef_b * np.exp(s2 * time_array)
        damping_type = "過阻尼"

    current = -np.gradient(charge, time_array)
    voltage_c = charge / params.capacitance
    voltage_r = current * params.resistance
    voltage_l = -(voltage_r + voltage_c)
    energy_c = charge**2 / (2.0 * params.capacitance)
    energy_l = 0.5 * params.inductance * current**2

    return {
        "time": time_array,
        "current": current,
        "charge": charge,
        "voltage_r": voltage_r,
        "voltage_c": voltage_c,
        "voltage_l": voltage_l,
        "energy_c": energy_c,
        "energy_l": energy_l,
        "alpha": alpha,
        "omega_0": omega_0,
        "damping_type": damping_type,
        "summary": (
            f"{damping_type}｜α = {alpha:.3g}，"
            f"ω₀ = {omega_0:.3g} rad/s"
        ),
        "title": (
            rf"RLC free response: $\alpha={alpha:.4g}$, "
            rf"$\omega_0={omega_0:.4g}\ \mathrm{{rad/s}}$"
        ),
    }


def infer_mode(selected_mode: CircuitMode, params: CircuitParams) -> CircuitMode:
    """Infer circuit mode when AUTO is selected."""
    if selected_mode != CircuitMode.AUTO:
        return selected_mode

    has_r = is_positive(params.resistance)
    has_l = is_positive(params.inductance)
    has_c = is_positive(params.capacitance)

    if has_r and has_l and has_c:
        return CircuitMode.RLC
    if has_l and has_c:
        return CircuitMode.LC
    if has_r and has_l:
        return CircuitMode.RL
    if has_r and has_c:
        return CircuitMode.RC

    return CircuitMode.AUTO


def get_active_kinds(mode: CircuitMode) -> list[str]:
    """Return component kinds used by the selected mode."""
    if mode == CircuitMode.RC:
        return ["R", "C"]
    if mode == CircuitMode.RL:
        return ["R", "L"]
    if mode == CircuitMode.LC:
        return ["L", "C"]
    if mode == CircuitMode.RLC:
        return ["R", "L", "C"]

    return []


def validate_mode(mode: CircuitMode, params: CircuitParams) -> str | None:
    """Validate parameters for a circuit mode."""
    if mode == CircuitMode.RC:
        if not is_positive(params.resistance):
            return "RC 充電需要 R > 0。"
        if not is_positive(params.capacitance):
            return "RC 充電需要 C > 0。"

    elif mode == CircuitMode.RL:
        if not is_positive(params.resistance):
            return "RL 通電需要 R > 0。"
        if not is_positive(params.inductance):
            return "RL 通電需要 L > 0。"

    elif mode == CircuitMode.LC:
        if not is_positive(params.inductance):
            return "LC 自由振盪需要 L > 0。"
        if not is_positive(params.capacitance):
            return "LC 自由振盪需要 C > 0。"
        if not is_positive(params.initial_charge):
            return "LC 自由振盪需要 Q₀ > 0。"

    elif mode == CircuitMode.RLC:
        if not is_positive(params.resistance):
            return "RLC 自由響應需要 R > 0。"
        if not is_positive(params.inductance):
            return "RLC 自由響應需要 L > 0。"
        if not is_positive(params.capacitance):
            return "RLC 自由響應需要 C > 0。"
        if not is_positive(params.initial_charge):
            return "RLC 自由響應需要 Q₀ > 0。"

    else:
        return "目前有效元件不足，請設定 RC、RL、LC 或 RLC。"

    return None


def get_response(
    mode: CircuitMode,
    time_array: np.ndarray,
    params: CircuitParams,
) -> tuple[dict | None, str | None]:
    """Return circuit response or error message."""
    error = validate_mode(mode, params)

    if error is not None:
        return None, error

    if mode == CircuitMode.RC:
        return rc_response(time_array, params), None
    if mode == CircuitMode.RL:
        return rl_response(time_array, params), None
    if mode == CircuitMode.LC:
        return lc_response(time_array, params), None
    if mode == CircuitMode.RLC:
        return rlc_response(time_array, params), None

    return None, "目前有效元件不足。"


def get_state(response: dict, frame_index: int) -> CircuitState:
    """Get instantaneous state from a response dictionary."""
    index = min(frame_index, len(response["time"]) - 1)

    return CircuitState(
        time=float(response["time"][index]),
        current=float(response["current"][index]),
        voltage_r=float(response["voltage_r"][index]),
        voltage_c=float(response["voltage_c"][index]),
        voltage_l=float(response["voltage_l"][index]),
        energy_c=float(response["energy_c"][index]),
        energy_l=float(response["energy_l"][index]),
    )


def component_label(
    kind: str,
    params: CircuitParams,
    state: CircuitState | None,
) -> str:
    """Return schematic label for a component."""
    if kind == "R":
        label = f"R = {format_si(params.resistance, 'Ω')}"
        if state is not None:
            label += f"\nV_R = {format_si(state.voltage_r, 'V')}"
        return label

    if kind == "L":
        label = f"L = {format_si(params.inductance, 'H')}"
        if state is not None:
            label += f"\nV_L = {format_si(state.voltage_l, 'V')}"
        return label

    if kind == "C":
        label = f"C = {format_si(params.capacitance, 'F')}"
        if state is not None:
            label += f"\nV_C = {format_si(state.voltage_c, 'V')}"
        return label

    return ""


def add_active_components(
    drawing: schemdraw.Drawing,
    mode: CircuitMode,
    params: CircuitParams,
    state: CircuitState | None,
) -> None:
    """Add active nonzero components to a Schemdraw drawing."""
    active_kinds = get_active_kinds(mode)

    for kind in active_kinds:
        if kind == "R" and not is_positive(params.resistance):
            continue
        if kind == "L" and not is_positive(params.inductance):
            continue
        if kind == "C" and not is_positive(params.capacitance):
            continue

        label = component_label(kind, params, state)

        if kind == "R":
            drawing += elm.Resistor().right().label(label, loc="top")
        elif kind == "L":
            drawing += elm.Inductor().right().label(label, loc="top")
        elif kind == "C":
            drawing += elm.Capacitor().right().label(label, loc="top")


def save_drawing_to_svg_string(drawing: schemdraw.Drawing) -> str:
    """Save a Schemdraw drawing to an SVG XML string."""
    with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as file:
        svg_path = file.name

    try:
        drawing.save(svg_path)
        with open(svg_path, "r", encoding="utf-8") as file:
            return file.read()
    finally:
        if os.path.exists(svg_path):
            os.remove(svg_path)


def draw_circuit_svg(
    mode: CircuitMode,
    params: CircuitParams,
    state: CircuitState | None,
) -> str:
    """Draw the current circuit as SVG XML string."""
    show_source = mode in {CircuitMode.RC, CircuitMode.RL}

    drawing = schemdraw.Drawing(show=False)
    drawing.config(unit=2.7, fontsize=12)

    if show_source:
        source_label = f"E = {format_si(params.voltage, 'V')}"
        drawing += elm.SourceV().up().label(source_label, loc="left")
    else:
        drawing += elm.Line().up().label("free response", loc="left")

    drawing += elm.Line().right().length(0.8)
    add_active_components(drawing, mode, params, state)

    drawing += elm.Line().right().length(0.8)
    drawing += elm.Line().down()
    drawing += elm.Line().left().tox(0)

    return save_drawing_to_svg_string(drawing)


def make_empty_circuit_svg(message: str) -> str:
    """Draw an empty schematic placeholder."""
    drawing = schemdraw.Drawing(show=False)
    drawing.config(unit=2.5, fontsize=13)
    drawing += elm.Line().right().length(5).label(message, loc="top")
    return save_drawing_to_svg_string(drawing)


def show_svg(container, svg_string: str, height: int = 260) -> None:
    """Render SVG string safely in Streamlit through HTML component."""
    container.empty()

    with container.container():
        components.html(
            f"""
            <div style="
                width: 100%;
                display: flex;
                justify-content: center;
                align-items: center;
                overflow-x: auto;
                padding: 0.25rem 0;
            ">
                {svg_string}
            </div>
            """,
            height=height,
            scrolling=False,
        )


def plot_waveform(
    response: dict,
    state: CircuitState,
    plot_type: PlotType,
) -> plt.Figure:
    """Plot one selected waveform."""
    time_array = response["time"]

    fig, axis = plt.subplots(figsize=(9.5, 4.2), constrained_layout=True)

    if plot_type == PlotType.CURRENT:
        axis.plot(time_array, response["current"], label=r"$I(t)$")
        axis.scatter([state.time], [state.current], s=55, zorder=5)
        axis.set_ylabel(r"$I\;(\mathrm{A})$")
        axis.set_title(response["title"])
        axis.legend(loc="best")

    elif plot_type == PlotType.VOLTAGE:
        axis.plot(time_array, response["voltage_r"], label=r"$V_R$")
        axis.plot(time_array, response["voltage_c"], label=r"$V_C$")
        axis.plot(time_array, response["voltage_l"], label=r"$V_L$")
        axis.scatter([state.time], [state.voltage_c], s=55, zorder=5)
        axis.set_ylabel(r"$V\;(\mathrm{V})$")
        axis.set_title("Voltages")
        axis.legend(loc="best")

    else:
        total_energy = response["energy_c"] + response["energy_l"]
        axis.plot(time_array, response["energy_c"], label=r"$U_C$")
        axis.plot(time_array, response["energy_l"], label=r"$U_L$")
        axis.plot(time_array, total_energy, label=r"$U_C+U_L$")
        axis.scatter([state.time], [state.energy_c], s=55, zorder=5)
        axis.set_ylabel(r"$U\;(\mathrm{J})$")
        axis.set_title("Energy")
        axis.legend(loc="best")

    axis.axvline(state.time, linestyle="--", alpha=0.35)
    axis.set_xlabel(r"$t\;(\mathrm{s})$")
    axis.grid(True, alpha=0.25)

    return fig


def setup_page() -> None:
    """Set up Streamlit page."""
    st.set_page_config(
        page_title="暫態電路實驗室",
        page_icon="⚡",
        layout="wide",
    )

    st.title("⚡ 暫態電路實驗室")
    st.caption(
        "Schemdraw 負責電路圖，Matplotlib 負責波形。"
        "R、L、C 設為 0 時，對應元件會從電路圖消失。"
    )


def make_sidebar() -> tuple[
    CircuitParams,
    CircuitMode,
    PlotType,
    bool,
    int,
    float,
    float,
]:
    """Create sidebar controls."""
    st.sidebar.header("控制面板")

    selected_mode = st.sidebar.radio(
        "電路模式",
        [
            CircuitMode.AUTO,
            CircuitMode.RC,
            CircuitMode.RL,
            CircuitMode.LC,
            CircuitMode.RLC,
        ],
        index=0,
        format_func=lambda mode: mode.value,
    )

    plot_type = st.sidebar.radio(
        "顯示波形",
        [PlotType.CURRENT, PlotType.VOLTAGE, PlotType.ENERGY],
        index=0,
        horizontal=True,
        format_func=lambda item: item.value,
    )

    st.sidebar.divider()

    resistance = st.sidebar.slider(
        "電阻 R (Ω)",
        min_value=0.0,
        max_value=100.0,
        value=25.0,
        step=1.0,
    )

    inductance_milli = st.sidebar.slider(
        "電感 L (mH)",
        min_value=0.0,
        max_value=1000.0,
        value=100.0,
        step=1.0,
    )

    capacitance_micro = st.sidebar.slider(
        "電容 C (μF)",
        min_value=0.0,
        max_value=1000.0,
        value=100.0,
        step=1.0,
    )

    st.sidebar.divider()

    voltage = st.sidebar.slider(
        "電池電壓 E (V)",
        min_value=0.0,
        max_value=24.0,
        value=12.0,
        step=1.0,
    )

    initial_charge_micro = st.sidebar.slider(
        "LC / RLC 初始電荷 Q₀ (μC)",
        min_value=0.0,
        max_value=1000.0,
        value=100.0,
        step=1.0,
    )

    st.sidebar.divider()

    time_max = st.sidebar.slider(
        "觀察時間上限 (s)",
        min_value=0.001,
        max_value=2.0,
        value=0.2,
        step=0.001,
        format="%.3f",
    )

    points = st.sidebar.slider(
        "時間取樣點數",
        min_value=300,
        max_value=3000,
        value=1200,
        step=100,
    )

    time_position = st.sidebar.slider(
        "目前時間比例",
        min_value=0.0,
        max_value=1.0,
        value=1.0,
        step=0.01,
    )

    animate = st.sidebar.checkbox("播放動畫", value=False)

    frame_count = st.sidebar.slider(
        "動畫幀數",
        min_value=20,
        max_value=200,
        value=80,
        step=10,
    )

    frame_delay = st.sidebar.slider(
        "每幀間隔 (秒)",
        min_value=0.01,
        max_value=0.20,
        value=0.03,
        step=0.01,
    )

    params = CircuitParams(
        resistance=resistance,
        capacitance=capacitance_micro * 1e-6,
        inductance=inductance_milli * 1e-3,
        voltage=voltage,
        initial_charge=initial_charge_micro * 1e-6,
        time_max=time_max,
        points=points,
    )

    return (
        params,
        selected_mode,
        plot_type,
        animate,
        frame_count,
        frame_delay,
        time_position,
    )


def show_metric_panel(
    mode: CircuitMode,
    response: dict | None,
    state: CircuitState | None,
    params: CircuitParams,
) -> None:
    """Show compact state metrics."""
    st.subheader("目前狀態")

    st.metric("模式", mode.value)

    if response is not None:
        st.metric("特徵量", response["summary"])

    col_1, col_2 = st.columns(2)
    col_1.metric("R", format_si(params.resistance, "Ω"))
    col_2.metric("L", format_si(params.inductance, "H"))

    col_3, col_4 = st.columns(2)
    col_3.metric("C", format_si(params.capacitance, "F"))
    col_4.metric("E", format_si(params.voltage, "V"))

    if state is None:
        return

    st.divider()

    col_5, col_6 = st.columns(2)
    col_5.metric("t", format_si(state.time, "s"))
    col_6.metric("i(t)", format_si(state.current, "A"))

    col_7, col_8 = st.columns(2)
    col_7.metric("V_C", format_si(state.voltage_c, "V"))
    col_8.metric("V_L", format_si(state.voltage_l, "V"))

    col_9, col_10 = st.columns(2)
    col_9.metric("U_C", format_si(state.energy_c, "J"))
    col_10.metric("U_L", format_si(state.energy_l, "J"))


def show_formula_expander(mode: CircuitMode, response: dict | None) -> None:
    """Show formulas in a collapsed expander."""
    with st.expander("公式與教學提示", expanded=False):
        if mode == CircuitMode.RC:
            st.markdown(
                r"""
                ### RC 充電

                $$
                RC\frac{dV_C}{dt}+V_C=E
                $$

                $$
                V_C(t)=E(1-e^{-t/RC})
                $$

                $$
                I(t)=\frac{E}{R}e^{-t/RC}
                $$
                """
            )

        elif mode == CircuitMode.RL:
            st.markdown(
                r"""
                ### RL 通電

                $$
                L\frac{dI}{dt}+RI=E
                $$

                $$
                I(t)=\frac{E}{R}(1-e^{-t/(L/R)})
                $$

                $$
                V_L(t)=Ee^{-t/(L/R)}
                $$
                """
            )

        elif mode == CircuitMode.LC:
            st.markdown(
                r"""
                ### LC 自由振盪

                $$
                \frac{d^2q}{dt^2}+\frac{1}{LC}q=0
                $$

                $$
                \omega_0=\frac{1}{\sqrt{LC}}
                $$

                $$
                U_C=\frac{q^2}{2C},\quad U_L=\frac{1}{2}LI^2
                $$
                """
            )

        elif mode == CircuitMode.RLC:
            damping_text = ""

            if response is not None:
                damping_text = f"\n\n目前阻尼型態：**{response['damping_type']}**"

            st.markdown(
                r"""
                ### RLC 自由響應

                $$
                L\frac{d^2q}{dt^2}
                +R\frac{dq}{dt}
                +\frac{1}{C}q=0
                $$

                $$
                \alpha=\frac{R}{2L},\quad
                \omega_0=\frac{1}{\sqrt{LC}}
                $$

                判斷：

                - $\alpha<\omega_0$：欠阻尼
                - $\alpha=\omega_0$：臨界阻尼
                - $\alpha>\omega_0$：過阻尼
                """
                + damping_text
            )

        st.markdown(
            """
            ### 觀察重點

            - RC：電容電壓上升，電流下降。
            - RL：電流上升，電感電壓下降。
            - LC：電場能與磁場能來回交換。
            - RLC：電阻把能量耗散掉，總能量下降。
            """
        )


def render_frame(
    circuit_placeholder,
    plot_placeholder,
    response: dict,
    frame_index: int,
    mode: CircuitMode,
    params: CircuitParams,
    plot_type: PlotType,
) -> CircuitState:
    """Render one animation/static frame."""
    state = get_state(response, frame_index)

    svg_string = draw_circuit_svg(mode, params, state)
    show_svg(circuit_placeholder, svg_string)

    fig = plot_waveform(response, state, plot_type)
    plot_placeholder.pyplot(fig, use_container_width=True)
    plt.close(fig)

    return state


def main() -> None:
    """Run the Streamlit app."""
    configure_matplotlib_fonts()
    setup_page()

    (
        params,
        selected_mode,
        plot_type,
        animate,
        frame_count,
        frame_delay,
        time_position,
    ) = make_sidebar()

    mode = infer_mode(selected_mode, params)
    time_array = np.linspace(0.0, params.time_max, params.points)
    response, error = get_response(mode, time_array, params)

    top_left, top_right = st.columns([2.4, 1.0], vertical_alignment="top")

    with top_left:
        st.subheader("電路圖")
        circuit_placeholder = st.empty()

    with top_right:
        metric_placeholder = st.container()

    st.subheader(plot_type.value)
    plot_placeholder = st.empty()

    if error is not None:
        svg_string = make_empty_circuit_svg(error)
        show_svg(circuit_placeholder, svg_string)
        st.warning(error)

        with metric_placeholder:
            show_metric_panel(mode, None, None, params)

        show_formula_expander(mode, None)
        return

    if animate:
        frame_indices = np.linspace(
            0,
            len(time_array) - 1,
            frame_count,
            dtype=int,
        )

        latest_state = None

        for frame_index in frame_indices:
            latest_state = render_frame(
                circuit_placeholder,
                plot_placeholder,
                response,
                frame_index,
                mode,
                params,
                plot_type,
            )
            time.sleep(frame_delay)

        with metric_placeholder:
            show_metric_panel(mode, response, latest_state, params)

    else:
        frame_index = int(time_position * (len(time_array) - 1))
        state = render_frame(
            circuit_placeholder,
            plot_placeholder,
            response,
            frame_index,
            mode,
            params,
            plot_type,
        )

        with metric_placeholder:
            show_metric_panel(mode, response, state, params)

    show_formula_expander(mode, response)


if __name__ == "__main__":
    main()