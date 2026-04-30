"""
基于 Python + Streamlit + Plotly 的相对论动力学 Web 交互式仿真平台

重点修正版：
1. 网页主图全部使用 Plotly，不再使用 Matplotlib 作为主显示图，解决云端中文字体乱码问题。
2. Plotly 图由浏览器渲染，中文在 Windows/Edge/Chrome 中通常可正常显示。
3. 单粒子仿真提供真正的播放动画。
4. 图右上角相机按钮可导出超高清 PNG。代码中设置为 7200×4800 像素，约等效于 6×4 英寸 1200dpi。

运行方式：
    pip install streamlit numpy pandas plotly
    streamlit run app.py

requirements.txt 建议内容：
    streamlit
    numpy
    pandas
    plotly

说明：
- 默认采用无量纲单位 c=1, m=1, q=1。
- 如果你要在论文里使用图片，不要截图；请点击 Plotly 图右上角的“相机”按钮导出高清 PNG。
"""

import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ============================================================
# 1. 页面与全局设置
# ============================================================

st.set_page_config(
    page_title="相对论动力学 Web 仿真平台",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

COL_REL = "#1f77b4"
COL_CLS = "#d62728"
COL_GREEN = "#2ca02c"
COL_ORANGE = "#ff7f0e"
COL_PURPLE = "#9467bd"
COL_DARK = "#222222"
COL_GRAY = "#888888"

# 关键：Plotly 由浏览器渲染字体。这里列出常见中文字体，浏览器会自动选择可用字体。
PLOTLY_FONT = "Microsoft YaHei, SimHei, Noto Sans CJK SC, WenQuanYi Micro Hei, Arial Unicode MS, sans-serif"

# 关键：导出高清图。5400×3600 像素约等效于 6×4 英寸 900dpi。
PLOTLY_CONFIG = {
    "displaylogo": False,
    "scrollZoom": True,
    "toImageButtonOptions": {
        "format": "png",
        "filename": "relativity_simulation_900dpi_equivalent",
        "width": 5400,
        "height": 3600,
        "scale": 1,
    },
}

# 网页绘图加速参数：计算数据可以多，但传给浏览器显示的数据必须适当抽样。
MAX_PLOT_POINTS = 2500
MAX_ANIMATION_FRAMES = 60


# ============================================================
# 2. 相对论动力学基础函数
# ============================================================


def gamma_from_speed(speed: float, c: float) -> float:
    beta2 = (float(speed) / float(c)) ** 2
    beta2 = min(beta2, 1.0 - 1e-14)
    return 1.0 / math.sqrt(1.0 - beta2)


def gamma_from_momentum(p: np.ndarray, m: float, c: float) -> float:
    p = np.asarray(p, dtype=float)
    p2 = float(np.dot(p, p))
    return math.sqrt(1.0 + p2 / (m * m * c * c))


def velocity_from_momentum(p: np.ndarray, m: float, c: float) -> np.ndarray:
    gamma = gamma_from_momentum(p, m, c)
    return np.asarray(p, dtype=float) / (gamma * m)


def relativistic_momentum_from_velocity(v: np.ndarray, m: float, c: float) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    speed = float(np.linalg.norm(v))
    gamma = gamma_from_speed(speed, c)
    return gamma * m * v


def kinetic_energy_rel(gamma: float, m: float, c: float) -> float:
    return (gamma - 1.0) * m * c * c


def kinetic_energy_cls(speed: float, m: float) -> float:
    return 0.5 * m * speed * speed


def lorentz_force(q: float, e_field: np.ndarray, v: np.ndarray, b_field: np.ndarray) -> np.ndarray:
    return q * (np.asarray(e_field, dtype=float) + np.cross(v, b_field))


# ============================================================
# 3. 数值算法
# ============================================================


def check_params(params: Dict[str, float]) -> None:
    if params["m"] <= 0:
        raise ValueError("质量 m 必须大于 0。")
    if params["c"] <= 0:
        raise ValueError("光速 c 必须大于 0。")
    if params["dt"] <= 0:
        raise ValueError("时间步长 dt 必须大于 0。")
    if params["total_time"] <= 0:
        raise ValueError("总时间 T 必须大于 0。")
    steps = int(params["total_time"] / params["dt"]) + 1
    if steps > 120000:
        raise ValueError("计算步数过多。请增大 dt 或减小总时间 T。")


def relativistic_boris_push(
    p: np.ndarray,
    e_field: np.ndarray,
    b_field: np.ndarray,
    dt: float,
    q: float,
    m: float,
    c: float,
) -> np.ndarray:
    """相对论 Boris 推进器。"""
    p = np.asarray(p, dtype=float)
    e = np.asarray(e_field, dtype=float)
    b = np.asarray(b_field, dtype=float)

    p_minus = p + q * e * dt / 2.0
    gamma_minus = gamma_from_momentum(p_minus, m, c)

    t_vec = q * b * dt / (2.0 * m * gamma_minus)
    t2 = float(np.dot(t_vec, t_vec))
    s_vec = 2.0 * t_vec / (1.0 + t2)

    p_prime = p_minus + np.cross(p_minus, t_vec)
    p_plus = p_minus + np.cross(p_prime, s_vec)
    p_new = p_plus + q * e * dt / 2.0
    return p_new


def params_to_tuple(params: Dict[str, float]) -> Tuple[Tuple[str, float], ...]:
    return tuple(sorted((k, float(v)) for k, v in params.items()))


@st.cache_data(show_spinner=False)
def simulate_cached(model: str, params_tuple: Tuple[Tuple[str, float], ...]) -> Dict[str, np.ndarray]:
    params = dict(params_tuple)
    return simulate(model, params)


def simulate(model: str, params: Dict[str, float]) -> Dict[str, np.ndarray]:
    check_params(params)
    if model == "相对论模型":
        return simulate_relativistic(params)
    if model == "经典模型":
        return simulate_classical(params)
    raise ValueError("未知模型：%s" % model)


def simulate_relativistic(params: Dict[str, float]) -> Dict[str, np.ndarray]:
    c = params["c"]
    m = params["m"]
    q = params["q"]
    dt = params["dt"]
    total_time = params["total_time"]

    e_field = np.array([params["Ex"], params["Ey"], params["Ez"]], dtype=float)
    b_field = np.array([params["Bx"], params["By"], params["Bz"]], dtype=float)
    r0 = np.array([params["x0"], params["y0"], params["z0"]], dtype=float)
    v0 = np.array([params["vx0"], params["vy0"], params["vz0"]], dtype=float)

    speed0 = float(np.linalg.norm(v0))
    if speed0 >= c:
        raise ValueError("相对论模型要求初速度小于光速 c。")

    n = int(total_time / dt) + 1
    t = np.linspace(0.0, total_time, n)
    r = np.zeros((n, 3))
    v = np.zeros((n, 3))
    p = np.zeros((n, 3))
    force = np.zeros((n, 3))
    speed = np.zeros(n)
    gamma = np.zeros(n)
    kinetic = np.zeros(n)

    r[0] = r0
    v[0] = v0
    p[0] = relativistic_momentum_from_velocity(v0, m, c)
    speed[0] = speed0
    gamma[0] = gamma_from_speed(speed0, c)
    kinetic[0] = kinetic_energy_rel(gamma[0], m, c)
    force[0] = lorentz_force(q, e_field, v[0], b_field)

    for i in range(1, n):
        p[i] = relativistic_boris_push(p[i - 1], e_field, b_field, dt, q, m, c)
        v[i] = velocity_from_momentum(p[i], m, c)
        r[i] = r[i - 1] + v[i] * dt
        speed[i] = float(np.linalg.norm(v[i]))
        gamma[i] = gamma_from_speed(speed[i], c)
        kinetic[i] = kinetic_energy_rel(gamma[i], m, c)
        force[i] = lorentz_force(q, e_field, v[i], b_field)

    return build_result("相对论模型", params, t, r, v, p, force, speed, gamma, kinetic)


def simulate_classical(params: Dict[str, float]) -> Dict[str, np.ndarray]:
    m = params["m"]
    q = params["q"]
    dt = params["dt"]
    total_time = params["total_time"]

    e_field = np.array([params["Ex"], params["Ey"], params["Ez"]], dtype=float)
    b_field = np.array([params["Bx"], params["By"], params["Bz"]], dtype=float)
    r0 = np.array([params["x0"], params["y0"], params["z0"]], dtype=float)
    v0 = np.array([params["vx0"], params["vy0"], params["vz0"]], dtype=float)

    n = int(total_time / dt) + 1
    t = np.linspace(0.0, total_time, n)
    r = np.zeros((n, 3))
    v = np.zeros((n, 3))
    p = np.zeros((n, 3))
    force = np.zeros((n, 3))
    speed = np.zeros(n)
    gamma = np.ones(n)
    kinetic = np.zeros(n)

    r[0] = r0
    v[0] = v0
    p[0] = m * v0
    speed[0] = float(np.linalg.norm(v0))
    kinetic[0] = kinetic_energy_cls(speed[0], m)
    force[0] = lorentz_force(q, e_field, v[0], b_field)

    for i in range(1, n):
        force[i - 1] = lorentz_force(q, e_field, v[i - 1], b_field)
        acceleration = force[i - 1] / m
        v[i] = v[i - 1] + acceleration * dt
        r[i] = r[i - 1] + v[i] * dt
        p[i] = m * v[i]
        speed[i] = float(np.linalg.norm(v[i]))
        kinetic[i] = kinetic_energy_cls(speed[i], m)
    force[-1] = lorentz_force(q, e_field, v[-1], b_field)

    return build_result("经典模型", params, t, r, v, p, force, speed, gamma, kinetic)


def build_result(model, params, t, r, v, p, force, speed, gamma, kinetic):
    return {
        "model": model,
        "params": params.copy(),
        "t": t,
        "r": r,
        "v": v,
        "p": p,
        "force": force,
        "speed": speed,
        "gamma": gamma,
        "kinetic": kinetic,
    }


# ============================================================
# 4. 场景预设与参数输入
# ============================================================

PRESETS = {
    "一维电场加速": {
        "c": 1.0, "m": 1.0, "q": 1.0,
        "Ex": 0.05, "Ey": 0.0, "Ez": 0.0,
        "Bx": 0.0, "By": 0.0, "Bz": 0.0,
        "x0": 0.0, "y0": 0.0, "z0": 0.0,
        "vx0": 0.0, "vy0": 0.0, "vz0": 0.0,
        "dt": 0.02, "total_time": 120.0,
    },
    "纯磁场圆周运动": {
        "c": 1.0, "m": 1.0, "q": 1.0,
        "Ex": 0.0, "Ey": 0.0, "Ez": 0.0,
        "Bx": 0.0, "By": 0.0, "Bz": 1.0,
        "x0": 0.0, "y0": 0.0, "z0": 0.0,
        "vx0": 0.8, "vy0": 0.0, "vz0": 0.0,
        "dt": 0.005, "total_time": 50.0,
    },
    "复合电磁场运动": {
        "c": 1.0, "m": 1.0, "q": 1.0,
        "Ex": 0.04, "Ey": 0.0, "Ez": 0.0,
        "Bx": 0.0, "By": 0.0, "Bz": 1.0,
        "x0": 0.0, "y0": 0.0, "z0": 0.0,
        "vx0": 0.25, "vy0": 0.0, "vz0": 0.0,
        "dt": 0.01, "total_time": 80.0,
    },
    "三维螺旋运动": {
        "c": 1.0, "m": 1.0, "q": 1.0,
        "Ex": 0.0, "Ey": 0.0, "Ez": 0.0,
        "Bx": 0.0, "By": 0.0, "Bz": 1.0,
        "x0": 0.0, "y0": 0.0, "z0": 0.0,
        "vx0": 0.55, "vy0": 0.0, "vz0": 0.35,
        "dt": 0.005, "total_time": 80.0,
    },
    "经典模型失效演示": {
        "c": 1.0, "m": 1.0, "q": 1.0,
        "Ex": 0.06, "Ey": 0.0, "Ez": 0.0,
        "Bx": 0.0, "By": 0.0, "Bz": 0.0,
        "x0": 0.0, "y0": 0.0, "z0": 0.0,
        "vx0": 0.0, "vy0": 0.0, "vz0": 0.0,
        "dt": 0.02, "total_time": 100.0,
    },
    "低速近似演示": {
        "c": 1.0, "m": 1.0, "q": 1.0,
        "Ex": 0.002, "Ey": 0.0, "Ez": 0.0,
        "Bx": 0.0, "By": 0.0, "Bz": 1.0,
        "x0": 0.0, "y0": 0.0, "z0": 0.0,
        "vx0": 0.08, "vy0": 0.0, "vz0": 0.0,
        "dt": 0.01, "total_time": 50.0,
    },
}


def sidebar_params() -> Tuple[str, str, Dict[str, float]]:
    st.sidebar.title("参数控制区")
    st.sidebar.caption("推荐流程：选择场景 → 调整参数 → 运行对应功能。")

    preset_name = st.sidebar.selectbox("场景预设", list(PRESETS.keys()), index=2)
    base = PRESETS[preset_name]

    model = st.sidebar.radio("模型选择", ["相对论模型", "经典模型"], index=0)

    with st.sidebar.expander("基本常量", expanded=True):
        c = st.number_input("光速 c", value=float(base["c"]), min_value=1e-12, format="%.6f")
        m = st.number_input("质量 m", value=float(base["m"]), min_value=1e-12, format="%.6f")
        q = st.number_input("电荷 q", value=float(base["q"]), format="%.6f")

    with st.sidebar.expander("电场 E", expanded=True):
        Ex = st.number_input("Ex", value=float(base["Ex"]), format="%.6f")
        Ey = st.number_input("Ey", value=float(base["Ey"]), format="%.6f")
        Ez = st.number_input("Ez", value=float(base["Ez"]), format="%.6f")

    with st.sidebar.expander("磁场 B", expanded=True):
        Bx = st.number_input("Bx", value=float(base["Bx"]), format="%.6f")
        By = st.number_input("By", value=float(base["By"]), format="%.6f")
        Bz = st.number_input("Bz", value=float(base["Bz"]), format="%.6f")

    with st.sidebar.expander("初始位置 r0", expanded=False):
        x0 = st.number_input("x0", value=float(base["x0"]), format="%.6f")
        y0 = st.number_input("y0", value=float(base["y0"]), format="%.6f")
        z0 = st.number_input("z0", value=float(base["z0"]), format="%.6f")

    with st.sidebar.expander("初始速度 v0", expanded=True):
        vx0 = st.number_input("vx0", value=float(base["vx0"]), format="%.6f")
        vy0 = st.number_input("vy0", value=float(base["vy0"]), format="%.6f")
        vz0 = st.number_input("vz0", value=float(base["vz0"]), format="%.6f")
        speed0 = math.sqrt(vx0 * vx0 + vy0 * vy0 + vz0 * vz0)
        st.caption("当前初速度大小 |v0|/c = %.4f" % (speed0 / c))

    with st.sidebar.expander("时间设置", expanded=True):
        dt = st.number_input("时间步长 dt", value=float(base["dt"]), min_value=1e-6, format="%.6f")
        total_time = st.number_input("总时间 T", value=float(base["total_time"]), min_value=1e-6, format="%.6f")
        steps = int(total_time / dt) + 1
        st.caption("预计计算步数：%d" % steps)

    params = {
        "c": c, "m": m, "q": q,
        "Ex": Ex, "Ey": Ey, "Ez": Ez,
        "Bx": Bx, "By": By, "Bz": Bz,
        "x0": x0, "y0": y0, "z0": z0,
        "vx0": vx0, "vy0": vy0, "vz0": vz0,
        "dt": dt, "total_time": total_time,
    }
    return preset_name, model, params


# ============================================================
# 5. 数据工具
# ============================================================


def result_to_dataframe(result: Dict[str, np.ndarray], label: str = "") -> pd.DataFrame:
    t = result["t"]
    r = result["r"]
    v = result["v"]
    p = result["p"]
    force = result["force"]
    data = {
        "label": label or result["model"],
        "model": result["model"],
        "t": t,
        "x": r[:, 0], "y": r[:, 1], "z": r[:, 2],
        "vx": v[:, 0], "vy": v[:, 1], "vz": v[:, 2],
        "px": p[:, 0], "py": p[:, 1], "pz": p[:, 2],
        "Fx": force[:, 0], "Fy": force[:, 1], "Fz": force[:, 2],
        "speed": result["speed"],
        "gamma": result["gamma"],
        "kinetic": result["kinetic"],
    }
    return pd.DataFrame(data)


def download_dataframe(df: pd.DataFrame, filename: str) -> None:
    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="下载 CSV 数据",
        data=csv,
        file_name=filename,
        mime="text/csv",
    )


def downsample_indices(n: int, max_points: int = 1400) -> np.ndarray:
    if n <= max_points:
        return np.arange(n)
    return np.linspace(0, n - 1, max_points).astype(int)


# ============================================================
# 6. Plotly 绘图函数
# ============================================================


def common_layout(fig: go.Figure, title: str, height: int = 850) -> go.Figure:
    fig.update_layout(
        title=title,
        height=height,
        font=dict(family=PLOTLY_FONT, size=15),
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1),
        margin=dict(l=35, r=35, t=90, b=45),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e9ecef", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#e9ecef", zeroline=False)
    return fig


def plot_static_dashboard(result: Dict[str, np.ndarray], frame_ratio: float = 1.0, show_arrows: bool = True) -> go.Figure:
    """单粒子仿真静态交互图。"""
    t = result["t"]
    r = result["r"]
    v = result["v"]
    force = result["force"]
    speed = result["speed"]
    gamma = result["gamma"]
    kinetic = result["kinetic"]
    c = result["params"]["c"]
    n = len(t)
    idx = max(1, min(n - 1, int(frame_ratio * (n - 1))))

    # 抽样后再传给浏览器，避免 Plotly 图像卡顿。
    pidx = downsample_indices(idx + 1, MAX_PLOT_POINTS)

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("x-y 平面轨迹", "速度随时间变化", "洛仑兹因子变化", "动能随时间变化"),
        horizontal_spacing=0.10,
        vertical_spacing=0.15,
    )

    fig.add_trace(go.Scatter(x=r[pidx, 0], y=r[pidx, 1], mode="lines", name="轨迹", line=dict(color=COL_REL, width=3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=[r[idx, 0]], y=[r[idx, 1]], mode="markers", name="当前位置", marker=dict(color=COL_CLS, size=12)), row=1, col=1)

    if show_arrows:
        vxy = v[idx, :2]
        fxy = force[idx, :2]
        v_norm = np.linalg.norm(vxy)
        f_norm = np.linalg.norm(fxy)
        if v_norm > 1e-12:
            fig.add_annotation(x=r[idx, 0] + vxy[0] / v_norm * 0.28, y=r[idx, 1] + vxy[1] / v_norm * 0.28,
                               ax=r[idx, 0], ay=r[idx, 1], xref="x", yref="y", axref="x", ayref="y",
                               showarrow=True, arrowhead=3, arrowwidth=3, arrowcolor=COL_REL, text="v")
        if f_norm > 1e-12:
            fig.add_annotation(x=r[idx, 0] + fxy[0] / f_norm * 0.28, y=r[idx, 1] + fxy[1] / f_norm * 0.28,
                               ax=r[idx, 0], ay=r[idx, 1], xref="x", yref="y", axref="x", ayref="y",
                               showarrow=True, arrowhead=3, arrowwidth=3, arrowcolor=COL_CLS, text="F")

    fig.add_trace(go.Scatter(x=t[pidx], y=speed[pidx] / c, mode="lines", name="速度 v/c", line=dict(color=COL_REL, width=3)), row=1, col=2)
    fig.add_trace(go.Scatter(x=t, y=np.ones_like(t), mode="lines", name="光速 c", line=dict(color=COL_DARK, width=2, dash="dot")), row=1, col=2)
    fig.add_trace(go.Scatter(x=t[pidx], y=gamma[pidx], mode="lines", name="γ", line=dict(color=COL_PURPLE, width=3)), row=2, col=1)
    fig.add_trace(go.Scatter(x=t[pidx], y=kinetic[pidx], mode="lines", name="动能 K", line=dict(color=COL_GREEN, width=3)), row=2, col=2)

    fig.update_xaxes(title_text="x", row=1, col=1)
    fig.update_yaxes(title_text="y", row=1, col=1, scaleanchor="x", scaleratio=1)
    fig.update_xaxes(title_text="时间 t", row=1, col=2)
    fig.update_yaxes(title_text="速度 v/c", row=1, col=2)
    fig.update_xaxes(title_text="时间 t", row=2, col=1)
    fig.update_yaxes(title_text="γ", row=2, col=1)
    fig.update_xaxes(title_text="时间 t", row=2, col=2)
    fig.update_yaxes(title_text="动能 K", row=2, col=2)

    return common_layout(fig, "%s：粒子运动仿真结果" % result["model"])


def plot_animation_dashboard(result: Dict[str, np.ndarray], frame_count: int = 60) -> go.Figure:
    """单粒子仿真动画图。

    加速策略：
    - 完整轨迹和曲线只画一次作为背景；
    - 动画帧只更新粒子位置和时间指示线，不再每帧传输累计曲线；
    - 这样网页播放会比累计轨迹动画快很多。
    """
    t = result["t"]
    r = result["r"]
    speed = result["speed"]
    gamma = result["gamma"]
    kinetic = result["kinetic"]
    c = result["params"]["c"]
    n = len(t)
    frame_count = min(frame_count, MAX_ANIMATION_FRAMES, n - 1)
    frame_indices = np.linspace(1, n - 1, frame_count).astype(int)
    didx = downsample_indices(n, MAX_PLOT_POINTS)

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("x-y 平面轨迹", "速度随时间变化", "洛仑兹因子变化", "动能随时间变化"),
        horizontal_spacing=0.10,
        vertical_spacing=0.15,
    )

    # 背景轨迹和背景曲线：抽样显示，提升网页性能。
    fig.add_trace(go.Scatter(x=r[didx, 0], y=r[didx, 1], mode="lines", name="完整轨迹", line=dict(color="#c9d6e8", width=2)), row=1, col=1)
    start = frame_indices[0]
    fig.add_trace(go.Scatter(x=[r[start, 0]], y=[r[start, 1]], mode="markers", name="粒子", marker=dict(color=COL_CLS, size=13)), row=1, col=1)
    fig.add_trace(go.Scatter(x=t[didx], y=speed[didx] / c, mode="lines", name="速度 v/c", line=dict(color=COL_REL, width=3)), row=1, col=2)
    fig.add_trace(go.Scatter(x=t[didx], y=np.ones_like(t[didx]), mode="lines", name="光速 c", line=dict(color=COL_DARK, width=2, dash="dot")), row=1, col=2)
    fig.add_trace(go.Scatter(x=t[didx], y=gamma[didx], mode="lines", name="γ", line=dict(color=COL_PURPLE, width=3)), row=2, col=1)
    fig.add_trace(go.Scatter(x=t[didx], y=kinetic[didx], mode="lines", name="动能 K", line=dict(color=COL_GREEN, width=3)), row=2, col=2)
    # 三条动态时间指示线：比重画曲线更轻量。
    fig.add_trace(go.Scatter(x=[t[start], t[start]], y=[0, max(1.05, np.max(speed / c) * 1.10)], mode="lines", name="当前时刻", line=dict(color=COL_ORANGE, width=2, dash="dash")), row=1, col=2)
    fig.add_trace(go.Scatter(x=[t[start], t[start]], y=[0, max(1.2, np.max(gamma) * 1.10)], mode="lines", showlegend=False, line=dict(color=COL_ORANGE, width=2, dash="dash")), row=2, col=1)
    fig.add_trace(go.Scatter(x=[t[start], t[start]], y=[0, max(0.1, np.max(kinetic) * 1.10)], mode="lines", showlegend=False, line=dict(color=COL_ORANGE, width=2, dash="dash")), row=2, col=2)

    frames = []
    y_speed_max = max(1.05, np.max(speed / c) * 1.10)
    y_gamma_max = max(1.2, np.max(gamma) * 1.10)
    y_energy_max = max(0.1, np.max(kinetic) * 1.10)
    for k in frame_indices:
        frames.append(
            go.Frame(
                data=[
                    go.Scatter(x=[r[k, 0]], y=[r[k, 1]]),
                    go.Scatter(x=[t[k], t[k]], y=[0, y_speed_max]),
                    go.Scatter(x=[t[k], t[k]], y=[0, y_gamma_max]),
                    go.Scatter(x=[t[k], t[k]], y=[0, y_energy_max]),
                ],
                traces=[1, 6, 7, 8],
                name=str(k),
            )
        )
    fig.frames = frames

    fig.update_layout(
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.08,
                "y": -0.08,
                "buttons": [
                    {"label": "▶ 播放", "method": "animate", "args": [None, {"frame": {"duration": 45, "redraw": True}, "fromcurrent": True}]},
                    {"label": "⏸ 暂停", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]},
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "x": 0.25,
                "y": -0.08,
                "len": 0.70,
                "currentvalue": {"prefix": "帧："},
                "steps": [
                    {"label": str(i), "method": "animate", "args": [[str(k)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}]}
                    for i, k in enumerate(frame_indices)
                ],
            }
        ],
    )

    fig.update_xaxes(title_text="x", row=1, col=1)
    fig.update_yaxes(title_text="y", row=1, col=1, scaleanchor="x", scaleratio=1)
    fig.update_xaxes(title_text="时间 t", row=1, col=2)
    fig.update_yaxes(title_text="速度 v/c", row=1, col=2, range=[0, max(1.05, np.max(speed / c) * 1.10)])
    fig.update_xaxes(title_text="时间 t", row=2, col=1)
    fig.update_yaxes(title_text="γ", row=2, col=1, range=[0, max(1.2, np.max(gamma) * 1.10)])
    fig.update_xaxes(title_text="时间 t", row=2, col=2)
    fig.update_yaxes(title_text="动能 K", row=2, col=2, range=[0, max(0.1, np.max(kinetic) * 1.10)])

    return common_layout(fig, "%s：粒子运动动态仿真" % result["model"], height=900)


def plot_compare(rel: Dict[str, np.ndarray], cls: Dict[str, np.ndarray]) -> go.Figure:
    c = rel["params"]["c"]
    n = min(len(rel["t"]), len(cls["t"]))
    diff = np.linalg.norm(rel["r"][:n] - cls["r"][:n], axis=1)

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("轨迹对比", "速度对比", "动能对比", "两种模型的位置差异"),
        horizontal_spacing=0.10,
        vertical_spacing=0.15,
    )
    fig.add_trace(go.Scatter(x=rel["r"][:, 0], y=rel["r"][:, 1], mode="lines", name="相对论轨迹", line=dict(color=COL_REL, width=3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=cls["r"][:, 0], y=cls["r"][:, 1], mode="lines", name="经典轨迹", line=dict(color=COL_CLS, width=3, dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=rel["t"], y=rel["speed"] / c, mode="lines", name="相对论速度", line=dict(color=COL_REL, width=3)), row=1, col=2)
    fig.add_trace(go.Scatter(x=cls["t"], y=cls["speed"] / c, mode="lines", name="经典速度", line=dict(color=COL_CLS, width=3, dash="dash")), row=1, col=2)
    fig.add_trace(go.Scatter(x=rel["t"], y=np.ones_like(rel["t"]), mode="lines", name="光速 c", line=dict(color=COL_DARK, width=2, dash="dot")), row=1, col=2)
    fig.add_trace(go.Scatter(x=rel["t"], y=rel["kinetic"], mode="lines", name="相对论动能", line=dict(color=COL_REL, width=3)), row=2, col=1)
    fig.add_trace(go.Scatter(x=cls["t"], y=cls["kinetic"], mode="lines", name="经典动能", line=dict(color=COL_CLS, width=3, dash="dash")), row=2, col=1)
    fig.add_trace(go.Scatter(x=rel["t"][:n], y=diff, mode="lines", name="位置差", line=dict(color=COL_ORANGE, width=3)), row=2, col=2)

    fig.update_xaxes(title_text="x", row=1, col=1)
    fig.update_yaxes(title_text="y", row=1, col=1, scaleanchor="x", scaleratio=1)
    fig.update_xaxes(title_text="时间 t", row=1, col=2)
    fig.update_yaxes(title_text="速度 v/c", row=1, col=2)
    fig.update_xaxes(title_text="时间 t", row=2, col=1)
    fig.update_yaxes(title_text="动能 K", row=2, col=1)
    fig.update_xaxes(title_text="时间 t", row=2, col=2)
    fig.update_yaxes(title_text="位置差", row=2, col=2)

    return common_layout(fig, "经典力学模型与相对论动力学模型对比")


def plot_scan(results: List[Dict[str, np.ndarray]], scan_name: str, c: float) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("不同参数下的轨迹对比", "不同参数下的速度对比", "参数对最大速度的影响", "参数对 γ 和动能的影响"),
        horizontal_spacing=0.10,
        vertical_spacing=0.15,
    )
    scan_values = []
    max_speed = []
    max_gamma = []
    final_energy = []

    for result in results:
        label = result["label"]
        scan_values.append(result["scan_value"])
        max_speed.append(float(np.max(result["speed"]) / c))
        max_gamma.append(float(np.max(result["gamma"])))
        final_energy.append(float(result["kinetic"][-1]))
        fig.add_trace(go.Scatter(x=result["r"][:, 0], y=result["r"][:, 1], mode="lines", name=label, line=dict(width=3)), row=1, col=1)
        fig.add_trace(go.Scatter(x=result["t"], y=result["speed"] / c, mode="lines", name=label, showlegend=False, line=dict(width=3)), row=1, col=2)

    fig.add_trace(go.Scatter(x=results[0]["t"], y=np.ones_like(results[0]["t"]), mode="lines", name="光速 c", line=dict(color=COL_DARK, width=2, dash="dot")), row=1, col=2)
    fig.add_trace(go.Scatter(x=scan_values, y=max_speed, mode="lines+markers", name="最大速度/c", line=dict(color=COL_REL, width=3)), row=2, col=1)
    fig.add_trace(go.Scatter(x=scan_values, y=max_gamma, mode="lines+markers", name="最大 γ", line=dict(color=COL_PURPLE, width=3)), row=2, col=2)
    fig.add_trace(go.Scatter(x=scan_values, y=final_energy, mode="lines+markers", name="末态动能", line=dict(color=COL_GREEN, width=3)), row=2, col=2)

    fig.update_xaxes(title_text="x", row=1, col=1)
    fig.update_yaxes(title_text="y", row=1, col=1, scaleanchor="x", scaleratio=1)
    fig.update_xaxes(title_text="时间 t", row=1, col=2)
    fig.update_yaxes(title_text="速度 v/c", row=1, col=2)
    fig.update_xaxes(title_text=scan_name, row=2, col=1)
    fig.update_yaxes(title_text="最大速度/c", row=2, col=1)
    fig.update_xaxes(title_text=scan_name, row=2, col=2)
    fig.update_yaxes(title_text="数值", row=2, col=2)

    return common_layout(fig, "参数扫描：%s" % scan_name)


def estimate_radius_xy(result: Dict[str, np.ndarray]) -> float:
    x = result["r"][:, 0]
    y = result["r"][:, 1]
    cx = 0.5 * (np.max(x) + np.min(x))
    cy = 0.5 * (np.max(y) + np.min(y))
    return float(np.mean(np.sqrt((x - cx) ** 2 + (y - cy) ** 2)))


def theoretical_radius_rel(params: Dict[str, float]) -> float:
    v0 = np.array([params["vx0"], params["vy0"], params["vz0"]], dtype=float)
    b = np.array([params["Bx"], params["By"], params["Bz"]], dtype=float)
    speed = float(np.linalg.norm(v0))
    b_abs = float(np.linalg.norm(b))
    if b_abs <= 1e-14 or abs(params["q"]) <= 1e-14:
        return float("nan")
    gamma = gamma_from_speed(speed, params["c"])
    return gamma * params["m"] * speed / (abs(params["q"]) * b_abs)


def plot_diagnostics(result: Dict[str, np.ndarray]) -> Tuple[go.Figure, pd.DataFrame]:
    params = result["params"]
    t = result["t"]
    k = result["kinetic"]
    k0 = max(abs(k[0]), 1e-12)
    energy_error = np.abs(k - k[0]) / k0
    max_speed_c = float(np.max(result["speed"]) / params["c"])
    max_gamma = float(np.max(result["gamma"]))
    max_energy_error = float(np.max(energy_error))
    r_num = estimate_radius_xy(result)
    r_theory = theoretical_radius_rel(params)
    radius_error = float("nan") if math.isnan(r_theory) else abs(r_num - r_theory) / max(abs(r_theory), 1e-12)

    table = pd.DataFrame({
        "指标": ["最大速度/c", "最大γ", "最大动能相对变化", "数值轨道半径", "理论轨道半径", "轨道半径相对误差"],
        "数值": [max_speed_c, max_gamma, max_energy_error, r_num, r_theory, radius_error],
    })
    table_display = table.copy()
    table_display["数值"] = table_display["数值"].apply(lambda x: "不适用" if pd.isna(x) else "%.6g" % x)

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{}, {}], [{}, {"type": "table"}]],
        subplot_titles=("动能变化", "动能相对变化 / 数值误差", "仿真诊断指标", "诊断结果表"),
        horizontal_spacing=0.10,
        vertical_spacing=0.15,
    )
    fig.add_trace(go.Scatter(x=t, y=k, mode="lines", name="动能 K", line=dict(color=COL_GREEN, width=3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=energy_error, mode="lines", name="动能相对变化", line=dict(color=COL_ORANGE, width=3)), row=1, col=2)
    fig.add_trace(go.Bar(x=["最大速度/c", "最大γ", "最大动能变化"], y=[max_speed_c, max_gamma, max_energy_error], name="诊断指标", marker_color=[COL_REL, COL_PURPLE, COL_ORANGE]), row=2, col=1)
    fig.add_trace(
        go.Table(
            header=dict(values=list(table_display.columns), fill_color="#f0f2f6", align="center", font=dict(size=15)),
            cells=dict(values=[table_display["指标"], table_display["数值"]], align="center", font=dict(size=14)),
        ),
        row=2,
        col=2,
    )
    fig.update_xaxes(title_text="时间 t", row=1, col=1)
    fig.update_yaxes(title_text="动能 K", row=1, col=1)
    fig.update_xaxes(title_text="时间 t", row=1, col=2)
    fig.update_yaxes(title_text="相对变化", row=1, col=2)
    fig.update_yaxes(title_text="数值", row=2, col=1)

    return common_layout(fig, "数值仿真误差与物理诊断"), table


def plot_3d_trajectory(result: Dict[str, np.ndarray]) -> go.Figure:
    r = result["r"]
    speed_c = result["speed"] / result["params"]["c"]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=r[:, 0], y=r[:, 1], z=r[:, 2], mode="lines",
            line=dict(color=speed_c, colorscale="Viridis", width=6, colorbar=dict(title="速度 v/c")),
            name="三维轨迹",
        )
    )
    fig.add_trace(go.Scatter3d(x=[r[0, 0]], y=[r[0, 1]], z=[r[0, 2]], mode="markers", marker=dict(size=6, color=COL_GREEN), name="起点"))
    fig.add_trace(go.Scatter3d(x=[r[-1, 0]], y=[r[-1, 1]], z=[r[-1, 2]], mode="markers", marker=dict(size=6, color=COL_CLS), name="终点"))
    fig.update_layout(
        title="三维粒子轨迹：可旋转、缩放和查看速度颜色",
        height=820,
        font=dict(family=PLOTLY_FONT, size=15),
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z", aspectmode="data"),
        margin=dict(l=0, r=0, t=70, b=0),
        paper_bgcolor="white",
    )
    return fig


def plot_lorentz_contraction(beta: float) -> go.Figure:
    beta = max(0.0, min(0.999999, beta))
    gamma = gamma_from_speed(beta, 1.0)
    length = 1.0 / gamma
    fig = go.Figure()
    fig.add_shape(type="rect", x0=0.1, y0=0.72, x1=1.1, y1=0.84, fillcolor=COL_REL, opacity=0.75, line=dict(width=0))
    fig.add_shape(type="rect", x0=0.1, y0=0.36, x1=0.1 + length, y1=0.48, fillcolor=COL_ORANGE, opacity=0.85, line=dict(width=0))
    fig.add_annotation(x=0.1, y=0.93, text="静止长度 L0 = 1", showarrow=False, xanchor="left", font=dict(size=18))
    fig.add_annotation(x=0.1, y=0.57, text="运动长度 L = L0/γ = %.3f" % length, showarrow=False, xanchor="left", font=dict(size=18))
    fig.add_annotation(x=0.1 + length + 0.18, y=0.42, text="运动方向", showarrow=True, ax=-55, ay=0, font=dict(size=16), arrowcolor=COL_CLS)
    fig.add_annotation(x=0.1, y=0.16, text="v/c = %.3f    γ = %.3f    L/L0 = %.3f" % (beta, gamma, length), showarrow=False, xanchor="left", font=dict(size=20), bgcolor="#f8f9fa", bordercolor="#bbbbbb", borderpad=8)
    fig.update_xaxes(range=[-0.05, 1.25], visible=False)
    fig.update_yaxes(range=[0.0, 1.1], visible=False)
    fig.update_layout(
        title="洛仑兹收缩演示：运动方向长度变短",
        height=540,
        font=dict(family=PLOTLY_FONT, size=15),
        margin=dict(l=20, r=20, t=70, b=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


# ============================================================
# 7. 页面功能
# ============================================================


def page_home() -> None:
    st.title("基于 Python 的相对论动力学 Web 交互式仿真平台")
    st.markdown(
        """
        本平台用于演示带电粒子在电场、磁场和复合电磁场中的运动规律，支持相对论动力学模型和经典力学模型。

        **使用流程：**
        1. 在左侧选择一个场景预设；
        2. 根据需要修改质量、电荷、电场、磁场、初始速度和时间步长；
        3. 进入“单粒子仿真”“模型对比”“参数扫描”等页面运行分析；
        4. 下载 CSV 数据或点击图右上角相机按钮导出高清图片。
        """
    )
    col1, col2, col3 = st.columns(3)
    col1.metric("核心方程", "dp/dt = F")
    col2.metric("相对论动量", "p = γmv")
    col3.metric("洛仑兹因子", "γ = 1/sqrt(1-v²/c²)")
    st.info("网页主图使用 Plotly 浏览器渲染。若要导出论文图片，请使用图右上角相机按钮，不要直接截图。")


def page_single_simulation(model: str, params: Dict[str, float]) -> None:
    st.header("单粒子仿真")
    st.write("本页用于观察单个粒子在当前参数下的轨迹、速度、洛仑兹因子和动能变化。")

    try:
        result = simulate_cached(model, params_to_tuple(params))
    except Exception as exc:
        st.error(str(exc))
        return

    display_mode = st.radio("显示方式", ["动态播放", "静态交互图"], horizontal=True)
    if display_mode == "动态播放":
        frame_count = st.slider("动画帧数", 20, 60, 40, 10, help="帧数越少，网页播放越流畅。论文展示建议 40 帧左右。")
        fig = plot_animation_dashboard(result, frame_count=frame_count)
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
        st.caption("点击图下方的 ▶ 播放按钮观察粒子动态运动。")
    else:
        frame = st.slider("帧位置", 0.0, 1.0, 1.0, 0.01)
        show_arrows = st.checkbox("显示速度/受力方向箭头", value=True)
        fig = plot_static_dashboard(result, frame_ratio=frame, show_arrows=show_arrows)
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    c = params["c"]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("最大速度 / c", "%.4f" % (np.max(result["speed"]) / c))
    col2.metric("最大 γ", "%.4f" % np.max(result["gamma"]))
    col3.metric("末态动能", "%.4f" % result["kinetic"][-1])
    col4.metric("计算步数", len(result["t"]))

    df = result_to_dataframe(result)
    with st.expander("查看仿真数据表"):
        st.dataframe(df.head(200), use_container_width=True)
    download_dataframe(df, "single_simulation.csv")


def page_model_compare(params: Dict[str, float]) -> None:
    st.header("经典模型与相对论模型对比")
    st.write("本页在同一组参数下同时计算经典模型和相对论模型，用于观察高速情况下经典力学的失效。")

    try:
        rel = simulate_cached("相对论模型", params_to_tuple(params))
        cls = simulate_cached("经典模型", params_to_tuple(params))
    except Exception as exc:
        st.error(str(exc))
        return

    fig = plot_compare(rel, cls)
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    c = params["c"]
    col1, col2, col3 = st.columns(3)
    col1.metric("相对论最大速度/c", "%.4f" % (np.max(rel["speed"]) / c))
    col2.metric("经典最大速度/c", "%.4f" % (np.max(cls["speed"]) / c))
    col3.metric("最大位置差", "%.4f" % np.max(np.linalg.norm(rel["r"] - cls["r"], axis=1)))

    df = pd.concat([
        result_to_dataframe(rel, label="相对论模型"),
        result_to_dataframe(cls, label="经典模型"),
    ], ignore_index=True)
    download_dataframe(df, "model_compare.csv")


def parse_scan_values(text: str) -> List[float]:
    values = []
    for item in text.split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    if len(values) < 2:
        raise ValueError("至少需要输入两个扫描值。")
    return values


def apply_scan_value(params: Dict[str, float], scan_name: str, value: float) -> Tuple[Dict[str, float], str]:
    p = params.copy()
    if scan_name == "初速度 vx0/c":
        p["vx0"] = value * p["c"]
        return p, "vx0=%.3gc" % value
    if scan_name == "电场 Ex":
        p["Ex"] = value
        return p, "Ex=%.3g" % value
    if scan_name == "磁场 Bz":
        p["Bz"] = value
        return p, "Bz=%.3g" % value
    if scan_name == "质量 m":
        p["m"] = value
        return p, "m=%.3g" % value
    if scan_name == "电荷 q":
        p["q"] = value
        return p, "q=%.3g" % value
    if scan_name == "时间步长 dt":
        p["dt"] = value
        return p, "dt=%.3g" % value
    raise ValueError("未知扫描参数。")


def page_parameter_scan(model: str, params: Dict[str, float]) -> None:
    st.header("参数扫描")
    st.write("本页用于批量改变一个参数，观察不同参数条件下的轨迹、速度、γ 和动能变化。")

    col1, col2 = st.columns([1, 2])
    with col1:
        scan_name = st.selectbox("扫描参数", ["初速度 vx0/c", "电场 Ex", "磁场 Bz", "质量 m", "电荷 q", "时间步长 dt"])
    with col2:
        values_text = st.text_input("扫描值，用英文逗号分隔", value="0.2,0.4,0.6,0.8,0.95")

    if not st.button("运行参数扫描", type="primary"):
        st.info("设置扫描参数后，点击“运行参数扫描”。")
        return

    try:
        values = parse_scan_values(values_text)
        results = []
        frames = []
        for value in values:
            p, label = apply_scan_value(params, scan_name, value)
            result = simulate_cached(model, params_to_tuple(p))
            result = result.copy()
            result["label"] = label
            result["scan_value"] = value
            results.append(result)
            frames.append(result_to_dataframe(result, label=label))
    except Exception as exc:
        st.error(str(exc))
        return

    fig = plot_scan(results, scan_name, params["c"])
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    summary = pd.DataFrame({
        scan_name: [r["scan_value"] for r in results],
        "最大速度/c": [np.max(r["speed"]) / params["c"] for r in results],
        "最大γ": [np.max(r["gamma"]) for r in results],
        "末态动能": [r["kinetic"][-1] for r in results],
    })
    st.subheader("扫描结果摘要")
    st.dataframe(summary, use_container_width=True)

    df = pd.concat(frames, ignore_index=True)
    download_dataframe(df, "parameter_scan.csv")


def page_diagnostics(model: str, params: Dict[str, float]) -> None:
    st.header("误差诊断与物理检验")
    st.write("本页用于检查仿真结果是否符合物理规律。纯磁场中磁场不做功，因此动能应近似守恒。")

    try:
        result = simulate_cached(model, params_to_tuple(params))
    except Exception as exc:
        st.error(str(exc))
        return

    fig, table = plot_diagnostics(result)
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
    st.subheader("诊断指标表")
    st.dataframe(table, use_container_width=True)
    download_dataframe(result_to_dataframe(result), "diagnostics_data.csv")


def page_3d(model: str, params: Dict[str, float]) -> None:
    st.header("三维轨迹")
    st.write("本页用于观察粒子在三维空间中的运动，适合展示磁场中的螺旋运动。")

    try:
        result = simulate_cached(model, params_to_tuple(params))
    except Exception as exc:
        st.error(str(exc))
        return

    fig = plot_3d_trajectory(result)
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
    download_dataframe(result_to_dataframe(result), "trajectory_3d.csv")


def page_contraction() -> None:
    st.header("洛仑兹收缩演示")
    st.write("拖动滑块观察速度越接近光速时，洛仑兹因子增大、运动方向长度缩短的现象。")

    beta = st.slider("速度 v/c", 0.0, 0.99, 0.80, 0.01)
    gamma = gamma_from_speed(beta, 1.0)
    length = 1.0 / gamma

    col1, col2, col3 = st.columns(3)
    col1.metric("v/c", "%.3f" % beta)
    col2.metric("γ", "%.3f" % gamma)
    col3.metric("L/L0", "%.3f" % length)

    fig = plot_lorentz_contraction(beta)
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)


# ============================================================
# 8. 主程序
# ============================================================


def main() -> None:
    preset_name, model, params = sidebar_params()

    st.sidebar.divider()
    page = st.sidebar.radio(
        "功能页面",
        ["首页", "单粒子仿真", "模型对比", "参数扫描", "误差诊断", "三维轨迹", "洛仑兹收缩"],
        index=0,
    )

    st.sidebar.divider()
    st.sidebar.write("当前场景：**%s**" % preset_name)
    st.sidebar.write("当前模型：**%s**" % model)

    if page == "首页":
        page_home()
    elif page == "单粒子仿真":
        page_single_simulation(model, params)
    elif page == "模型对比":
        page_model_compare(params)
    elif page == "参数扫描":
        page_parameter_scan(model, params)
    elif page == "误差诊断":
        page_diagnostics(model, params)
    elif page == "三维轨迹":
        page_3d(model, params)
    elif page == "洛仑兹收缩":
        page_contraction()


if __name__ == "__main__":
    main()
