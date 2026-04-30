"""
基于 Python + Streamlit 的相对论动力学 Web 交互式仿真平台

建议论文题目：
    基于 Python 的相对论动力学 Web 交互式仿真平台设计与实现

运行方式：
    pip install streamlit numpy pandas matplotlib
    streamlit run app.py

主要功能：
1. 通过网页输入质量、电荷、电场、磁场、初始速度和时间步长等参数。
2. 支持典型场景预设：一维电场加速、纯磁场圆周运动、复合电磁场、三维螺旋运动、经典模型失效演示。
3. 支持相对论模型和经典力学模型。
4. 支持经典模型与相对论模型对比。
5. 支持参数扫描，例如不同初速度、不同电场、不同磁场、不同质量、不同电荷。
6. 支持误差诊断：纯磁场中能量守恒、轨道半径误差等。
7. 支持洛仑兹收缩演示。
8. 支持 CSV 数据下载。

说明：
- 默认采用无量纲单位：c=1, m=1, q=1。
- 初学者建议保持 c=1，不要直接改成 3e8，否则时间步长、电场、磁场都需要重新缩放。
"""

import io
import math
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ============================================================
# 1. 页面设置
# ============================================================

st.set_page_config(
    page_title="相对论动力学 Web 仿真平台",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 300


COL_REL = "#1f77b4"
COL_CLS = "#d62728"
COL_GREEN = "#2ca02c"
COL_ORANGE = "#ff7f0e"
COL_PURPLE = "#9467bd"
COL_DARK = "#222222"


# ============================================================
# 2. 相对论动力学基础函数
# ============================================================


def gamma_from_speed(speed: float, c: float) -> float:
    """由速率计算洛仑兹因子 gamma。"""
    beta2 = (float(speed) / float(c)) ** 2
    beta2 = min(beta2, 1.0 - 1e-14)
    return 1.0 / math.sqrt(1.0 - beta2)


def gamma_from_momentum(p: np.ndarray, m: float, c: float) -> float:
    """由动量向量计算洛仑兹因子 gamma。"""
    p = np.asarray(p, dtype=float)
    p2 = float(np.dot(p, p))
    return math.sqrt(1.0 + p2 / (m * m * c * c))


def velocity_from_momentum(p: np.ndarray, m: float, c: float) -> np.ndarray:
    """由相对论动量反推出速度。"""
    gamma = gamma_from_momentum(p, m, c)
    return np.asarray(p, dtype=float) / (gamma * m)


def relativistic_momentum_from_velocity(v: np.ndarray, m: float, c: float) -> np.ndarray:
    """由速度计算相对论动量 p = gamma m v。"""
    v = np.asarray(v, dtype=float)
    speed = float(np.linalg.norm(v))
    gamma = gamma_from_speed(speed, c)
    return gamma * m * v


def kinetic_energy_rel(gamma: float, m: float, c: float) -> float:
    """相对论动能 K = (gamma - 1)mc^2。"""
    return (gamma - 1.0) * m * c * c


def kinetic_energy_cls(speed: float, m: float) -> float:
    """经典动能 K = 1/2 mv^2。"""
    return 0.5 * m * speed * speed


def lorentz_force(q: float, e_field: np.ndarray, v: np.ndarray, b_field: np.ndarray) -> np.ndarray:
    """洛伦兹力 F = q(E + v × B)。"""
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
    """
    相对论 Boris 推进器。

    该算法适用于带电粒子在电磁场中的相对论运动。
    对纯磁场运动，它比普通欧拉法更稳定，更适合做轨迹仿真。
    """
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
    """将参数字典转为可缓存的 tuple。"""
    return tuple(sorted((k, float(v)) for k, v in params.items()))


@st.cache_data(show_spinner=False)
def simulate_cached(model: str, params_tuple: Tuple[Tuple[str, float], ...]) -> Dict[str, np.ndarray]:
    params = dict(params_tuple)
    return simulate(model, params)


def simulate(model: str, params: Dict[str, float]) -> Dict[str, np.ndarray]:
    """统一仿真入口。"""
    check_params(params)
    if model == "相对论模型":
        return simulate_relativistic(params)
    if model == "经典模型":
        return simulate_classical(params)
    raise ValueError("未知模型：%s" % model)


def simulate_relativistic(params: Dict[str, float]) -> Dict[str, np.ndarray]:
    """相对论模型仿真。"""
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
    """经典模型仿真。"""
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
# 5. 数据与绘图工具
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


def style_axis(ax, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold", pad=10)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_result_dashboard(result: Dict[str, np.ndarray], frame_ratio: float = 1.0, show_arrows: bool = True) -> plt.Figure:
    """单模型仿真仪表盘。"""
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

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.2))
    ax1, ax2, ax3, ax4 = axes.ravel()

    ax1.plot(r[:idx + 1, 0], r[:idx + 1, 1], color=COL_REL, linewidth=2.2, label="轨迹")
    ax1.scatter(r[idx, 0], r[idx, 1], color=COL_CLS, s=45, label="当前位置")
    if show_arrows:
        vxy = v[idx, :2]
        fxy = force[idx, :2]
        v_norm = np.linalg.norm(vxy)
        f_norm = np.linalg.norm(fxy)
        if v_norm > 1e-12:
            ax1.arrow(r[idx, 0], r[idx, 1], vxy[0] / v_norm * 0.25, vxy[1] / v_norm * 0.25,
                      color=COL_REL, width=0.01, length_includes_head=True)
        if f_norm > 1e-12:
            ax1.arrow(r[idx, 0], r[idx, 1], fxy[0] / f_norm * 0.25, fxy[1] / f_norm * 0.25,
                      color=COL_CLS, width=0.01, length_includes_head=True)
    ax1.set_aspect("equal", adjustable="box")
    style_axis(ax1, "x", "y", "x-y 平面轨迹")
    ax1.legend(loc="best")

    ax2.plot(t[:idx + 1], speed[:idx + 1] / c, color=COL_REL, linewidth=2.0)
    ax2.axhline(1.0, color=COL_DARK, linestyle=":", label="光速 c")
    ax2.set_xlim(t[0], t[-1])
    ax2.set_ylim(0, max(1.05, np.max(speed / c) * 1.1))
    style_axis(ax2, "时间 t", "速度 v/c", "速度随时间变化")
    ax2.legend(loc="best")

    ax3.plot(t[:idx + 1], gamma[:idx + 1], color=COL_PURPLE, linewidth=2.0)
    ax3.set_xlim(t[0], t[-1])
    ax3.set_ylim(0, max(1.2, np.max(gamma) * 1.1))
    style_axis(ax3, "时间 t", "γ", "洛仑兹因子变化")

    ax4.plot(t[:idx + 1], kinetic[:idx + 1], color=COL_GREEN, linewidth=2.0)
    ax4.set_xlim(t[0], t[-1])
    ax4.set_ylim(0, max(0.1, np.max(kinetic) * 1.1))
    style_axis(ax4, "时间 t", "动能 K", "动能随时间变化")

    fig.suptitle("%s：粒子运动仿真结果" % result["model"], fontsize=15, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_compare(rel: Dict[str, np.ndarray], cls: Dict[str, np.ndarray]) -> plt.Figure:
    """经典模型与相对论模型对比图。"""
    c = rel["params"]["c"]
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.2))
    ax1, ax2, ax3, ax4 = axes.ravel()

    ax1.plot(rel["r"][:, 0], rel["r"][:, 1], color=COL_REL, label="相对论模型")
    ax1.plot(cls["r"][:, 0], cls["r"][:, 1], "--", color=COL_CLS, label="经典模型")
    ax1.set_aspect("equal", adjustable="box")
    style_axis(ax1, "x", "y", "轨迹对比")
    ax1.legend(loc="best")

    ax2.plot(rel["t"], rel["speed"] / c, color=COL_REL, label="相对论模型")
    ax2.plot(cls["t"], cls["speed"] / c, "--", color=COL_CLS, label="经典模型")
    ax2.axhline(1.0, color=COL_DARK, linestyle=":", label="光速 c")
    style_axis(ax2, "时间 t", "速度 v/c", "速度对比")
    ax2.legend(loc="best")

    ax3.plot(rel["t"], rel["kinetic"], color=COL_REL, label="相对论动能")
    ax3.plot(cls["t"], cls["kinetic"], "--", color=COL_CLS, label="经典动能")
    style_axis(ax3, "时间 t", "动能 K", "动能对比")
    ax3.legend(loc="best")

    n = min(len(rel["t"]), len(cls["t"]))
    diff = np.linalg.norm(rel["r"][:n] - cls["r"][:n], axis=1)
    ax4.plot(rel["t"][:n], diff, color=COL_ORANGE)
    style_axis(ax4, "时间 t", "位置差 |r_rel-r_cls|", "两种模型的位置差异")

    fig.suptitle("经典力学模型与相对论动力学模型对比", fontsize=15, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_scan(results: List[Dict[str, np.ndarray]], scan_name: str, c: float) -> plt.Figure:
    """参数扫描结果图。"""
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.2))
    ax1, ax2, ax3, ax4 = axes.ravel()

    scan_values = []
    max_speed = []
    max_gamma = []
    final_energy = []

    for result in results:
        label = result["label"]
        scan_values.append(result["scan_value"])
        max_speed.append(np.max(result["speed"]) / c)
        max_gamma.append(np.max(result["gamma"]))
        final_energy.append(result["kinetic"][-1])

        ax1.plot(result["r"][:, 0], result["r"][:, 1], label=label, linewidth=2.0)
        ax2.plot(result["t"], result["speed"] / c, label=label, linewidth=2.0)

    ax1.set_aspect("equal", adjustable="box")
    style_axis(ax1, "x", "y", "不同参数下的轨迹对比")
    ax1.legend(loc="best")

    ax2.axhline(1.0, color=COL_DARK, linestyle=":", label="光速 c")
    style_axis(ax2, "时间 t", "速度 v/c", "不同参数下的速度对比")
    ax2.legend(loc="best")

    ax3.plot(scan_values, max_speed, marker="o", color=COL_REL)
    style_axis(ax3, scan_name, "最大速度 / c", "参数对最大速度的影响")
    ax3.set_ylim(0, max(1.05, max(max_speed) * 1.1))

    ax4.plot(scan_values, max_gamma, marker="o", color=COL_PURPLE, label="最大 γ")
    ax4.plot(scan_values, final_energy, marker="s", color=COL_GREEN, label="末态动能")
    style_axis(ax4, scan_name, "数值", "参数对 γ 和动能的影响")
    ax4.legend(loc="best")

    fig.suptitle("参数扫描：%s" % scan_name, fontsize=15, fontweight="bold")
    fig.tight_layout()
    return fig


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


def plot_diagnostics(result: Dict[str, np.ndarray]) -> Tuple[plt.Figure, pd.DataFrame]:
    params = result["params"]
    t = result["t"]
    k = result["kinetic"]
    k0 = max(abs(k[0]), 1e-12)
    energy_error = np.abs(k - k[0]) / k0
    max_speed_c = np.max(result["speed"]) / params["c"]
    max_gamma = np.max(result["gamma"])
    max_energy_error = np.max(energy_error)
    r_num = estimate_radius_xy(result)
    r_theory = theoretical_radius_rel(params)
    if math.isnan(r_theory):
        radius_error = float("nan")
    else:
        radius_error = abs(r_num - r_theory) / max(abs(r_theory), 1e-12)

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.2))
    ax1, ax2, ax3, ax4 = axes.ravel()

    ax1.plot(t, k, color=COL_GREEN)
    style_axis(ax1, "时间 t", "动能 K", "动能变化")

    ax2.plot(t, energy_error, color=COL_ORANGE)
    style_axis(ax2, "时间 t", "相对变化", "动能相对变化 / 数值误差")

    labels = ["最大速度/c", "最大γ", "最大动能变化"]
    values = [max_speed_c, max_gamma, max_energy_error]
    ax3.bar(labels, values, color=[COL_REL, COL_PURPLE, COL_ORANGE])
    style_axis(ax3, "指标", "数值", "仿真诊断指标")
    for i, value in enumerate(values):
        ax3.text(i, value, "%.3g" % value, ha="center", va="bottom")

    ax4.axis("off")
    text = (
        "诊断说明\n\n"
        "最大速度/c：%.6g\n"
        "最大洛仑兹因子 γ：%.6g\n"
        "最大动能相对变化：%.6e\n"
        "数值轨道半径：%.6g\n"
        "理论轨道半径：%s\n"
        "轨道半径相对误差：%s\n\n"
        "纯磁场中磁场不做功，动能应近似守恒。\n"
        "若存在电场，动能变化属于正常现象。"
    ) % (
        max_speed_c,
        max_gamma,
        max_energy_error,
        r_num,
        "不适用" if math.isnan(r_theory) else "%.6g" % r_theory,
        "不适用" if math.isnan(radius_error) else "%.6e" % radius_error,
    )
    ax4.text(0.02, 0.98, text, va="top", fontsize=10,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#bbbbbb"))

    fig.suptitle("数值仿真误差与物理诊断", fontsize=15, fontweight="bold")
    fig.tight_layout()

    table = pd.DataFrame({
        "指标": ["最大速度/c", "最大γ", "最大动能相对变化", "数值轨道半径", "理论轨道半径", "轨道半径相对误差"],
        "数值": [max_speed_c, max_gamma, max_energy_error, r_num, r_theory, radius_error],
    })
    return fig, table


def plot_3d_trajectory(result: Dict[str, np.ndarray]) -> plt.Figure:
    r = result["r"]
    speed = result["speed"]
    c = result["params"]["c"]

    fig = plt.figure(figsize=(10.8, 7.6))
    ax = fig.add_subplot(111, projection="3d")

    n = len(r)
    segments = 120
    step = max(1, n // segments)
    cmap = plt.get_cmap("viridis")
    vmax = max(np.max(speed / c), 1e-12)
    for i in range(0, n - step, step):
        color = cmap((speed[i] / c) / vmax)
        ax.plot(r[i:i + step + 1, 0], r[i:i + step + 1, 1], r[i:i + step + 1, 2], color=color, linewidth=2.0)

    ax.scatter(r[0, 0], r[0, 1], r[0, 2], color=COL_GREEN, s=60, label="起点")
    ax.scatter(r[-1, 0], r[-1, 1], r[-1, 2], color=COL_CLS, s=60, label="终点")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("三维粒子轨迹", fontweight="bold")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def plot_lorentz_contraction(beta: float) -> plt.Figure:
    beta = max(0.0, min(0.999999, beta))
    gamma = gamma_from_speed(beta, 1.0)
    length = 1.0 / gamma

    fig, ax = plt.subplots(figsize=(10.8, 5.6))
    ax.set_xlim(-0.15, 1.2)
    ax.set_ylim(-0.15, 1.15)
    ax.axis("off")
    ax.set_title("洛仑兹收缩演示：运动方向长度变短", fontsize=15, fontweight="bold")

    ax.add_patch(plt.Rectangle((0.1, 0.72), 1.0, 0.12, color=COL_REL, alpha=0.75))
    ax.text(0.1, 0.90, "静止长度 L0 = 1", fontsize=12)

    ax.add_patch(plt.Rectangle((0.1, 0.36), length, 0.12, color=COL_ORANGE, alpha=0.85))
    ax.text(0.1, 0.54, "运动长度 L = L0/γ = %.3f" % length, fontsize=12)
    ax.arrow(0.1 + length + 0.03, 0.42, 0.15, 0, width=0.01, color=COL_CLS, length_includes_head=True)
    ax.text(0.1 + length + 0.20, 0.39, "运动方向", fontsize=11)

    explanation = "v/c = %.3f    γ = %.3f    L/L0 = %.3f" % (beta, gamma, length)
    ax.text(0.1, 0.12, explanation, fontsize=14, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#f8f9fa", edgecolor="#bbbbbb"))
    fig.tight_layout()
    return fig


# ============================================================
# 6. 页面功能
# ============================================================


def page_home() -> None:
    st.title("基于 Python 的相对论动力学 Web 交互式仿真平台")
    st.markdown(
        """
        本平台用于演示带电粒子在电场、磁场和复合电磁场中的运动规律，支持相对论动力学模型和经典力学模型。

        **推荐使用流程：**
        1. 在左侧选择一个场景预设；
        2. 根据需要修改质量、电荷、电场、磁场、初始速度和时间步长；
        3. 进入“单粒子仿真”“模型对比”“参数扫描”等页面运行分析；
        4. 下载 CSV 数据或保存图像用于论文。
        """
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("核心方程", "dp/dt = F")
    col2.metric("相对论动量", "p = γmv")
    col3.metric("洛仑兹因子", "γ = 1/sqrt(1-v²/c²)")

    st.info("左侧参数区默认采用无量纲单位 c=1, m=1, q=1。该处理便于观察相对论效应，也避免真实 SI 单位下数值过大。")

    st.subheader("平台功能")
    st.markdown(
        """
        - **单粒子仿真：** 显示轨迹、速度、洛仑兹因子和动能变化。
        - **模型对比：** 同一组参数下对比经典模型与相对论模型。
        - **参数扫描：** 批量改变某个参数，观察轨迹和物理量变化。
        - **误差诊断：** 检查纯磁场中动能守恒和轨道半径误差。
        - **三维轨迹：** 显示三维空间中的粒子轨迹，适合观察螺旋运动。
        - **洛仑兹收缩：** 通过滑块直观观察长度收缩效应。
        """
    )


def page_single_simulation(model: str, params: Dict[str, float]) -> None:
    st.header("单粒子仿真")
    st.write("本页用于观察单个粒子在当前参数下的轨迹、速度、洛仑兹因子和动能变化。")

    try:
        result = simulate_cached(model, params_to_tuple(params))
    except Exception as exc:
        st.error(str(exc))
        return

    frame = st.slider("动画帧位置", 0.0, 1.0, 1.0, 0.01, help="拖动滑块可以观察粒子从初始时刻到末时刻的运动过程。")
    show_arrows = st.checkbox("显示速度/受力方向箭头", value=True)
    fig = plot_result_dashboard(result, frame_ratio=frame, show_arrows=show_arrows)
    st.pyplot(fig, use_container_width=True)

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
    st.pyplot(fig, use_container_width=True)

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
    st.pyplot(fig, use_container_width=True)

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
    st.pyplot(fig, use_container_width=True)
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
    st.pyplot(fig, use_container_width=True)
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
    st.pyplot(fig, use_container_width=True)


# ============================================================
# 7. 主程序
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
