import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle, Circle
import streamlit as st

# 页面基础配置
st.set_page_config(page_title="菲涅尔偏振动画演示", layout="wide")
plt.rcParams["font.sans-serif"] = ["SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 固定动画参数
FPS = 20
ANGLE_STEP = 0.5
HOLD_SECONDS = 1.5


# ===================== 核心光学计算函数 =====================
def calculate_fresnel_data(N1, N2):
    BREWSTER_ANGLE = math.degrees(math.atan(N2 / N1))
    CRITICAL_ANGLE = math.degrees(math.asin(N2 / N1)) if N1 > N2 else 90.0

    uniform_angles = np.arange(0.0, 90.0 + ANGLE_STEP / 2, ANGLE_STEP)
    uniform_angles = np.unique(np.append(uniform_angles, [BREWSTER_ANGLE, CRITICAL_ANGLE]))
    data = []

    for angle in uniform_angles:
        row = {"incident": float(angle)}
        if angle >= CRITICAL_ANGLE:
            row.update({
                "Rs": 1.0, "Rp": 1.0, "Rn": 1.0,
                "Ts": 0.0, "Tp": 0.0, "Tn": 0.0,
                "refracted": None
            })
        else:
            sin_theta_t = N1 / N2 * math.sin(math.radians(angle))
            cos_i = math.cos(math.radians(angle))
            cos_t = math.cos(math.asin(sin_theta_t))

            rs = (N1 * cos_i - N2 * cos_t) / (N1 * cos_i + N2 * cos_t)
            Rs = abs(rs) ** 2
            Ts = (4 * N1 * N2 * cos_i * cos_t) / (N1 * cos_i + N2 * cos_t) ** 2

            rp = (N2 * cos_i - N1 * cos_t) / (N2 * cos_i + N1 * cos_t)
            Rp = abs(rp) ** 2
            Tp = (4 * N1 * N2 * cos_i * cos_t) / (N2 * cos_i + N1 * cos_t) ** 2

            Rn = (Rs + Rp) / 2
            Tn = (Ts + Tp) / 2

            row.update({
                "Rs": Rs, "Rp": Rp, "Rn": Rn,
                "Ts": Ts, "Tp": Tp, "Tn": Tn,
                "refracted": math.degrees(math.asin(sin_theta_t))
            })
        data.append(row)
    return data, BREWSTER_ANGLE, CRITICAL_ANGLE


def strength_style(intensity):
    intensity = max(0.0, min(1.0, float(intensity)))
    alpha = 0.03 + 0.97 * intensity
    linewidth = 0.5 + 6.5 * math.sqrt(intensity)
    return intensity, alpha, linewidth


def add_beam(ax, start, end, intensity, color="#d7191c", zorder=5):
    intensity, alpha, linewidth = strength_style(intensity)
    if intensity <= 0.0005:
        return False
    ax.plot([start[0], end[0]], [start[1], end[1]],
            color="#ffd1d1", linewidth=linewidth + 4.0,
            alpha=0.20 * alpha, solid_capstyle="round", zorder=zorder)
    ax.plot([start[0], end[0]], [start[1], end[1]],
            color=color, linewidth=linewidth,
            alpha=alpha, solid_capstyle="round", zorder=zorder + 1)
    return True


def add_direction_arrow(ax, start, end, intensity, color="#d7191c", zorder=16):
    intensity, alpha, linewidth = strength_style(intensity)
    if intensity <= 0.0005:
        return
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    direction = end - start
    norm = np.linalg.norm(direction)
    if norm == 0:
        return
    direction /= norm
    normal = np.array([-direction[1], direction[0]])
    tip = end
    arrow_length = 0.16
    arrow_half_width = 0.075
    base_center = tip - direction * arrow_length
    wing_a = base_center + normal * arrow_half_width
    wing_b = base_center - normal * arrow_half_width
    arrow_linewidth = max(1.6, linewidth * 0.45)
    arrow_alpha = max(0.45, alpha)
    ax.plot([wing_a[0], tip[0], wing_b[0]], [wing_a[1], tip[1], wing_b[1]],
            color=color, linewidth=arrow_linewidth, alpha=arrow_alpha,
            solid_capstyle="round", solid_joinstyle="round", zorder=zorder)


def add_brewster_right_angle(ax, reflected_end, transmitted_end, color="#0057d9", zorder=18):
    origin = np.array([0.0, 0.0])
    reflected_dir = np.asarray(reflected_end, dtype=float) - origin
    transmitted_dir = np.asarray(transmitted_end, dtype=float) - origin
    reflected_norm = np.linalg.norm(reflected_dir)
    transmitted_norm = np.linalg.norm(transmitted_dir)
    if reflected_norm == 0 or transmitted_norm == 0:
        return
    reflected_dir /= reflected_norm
    transmitted_dir /= transmitted_norm
    size = 0.22
    p_reflected = origin + reflected_dir * size
    p_corner = origin + (reflected_dir + transmitted_dir) * size
    p_transmitted = origin + transmitted_dir * size
    ax.plot([p_reflected[0], p_corner[0], p_transmitted[0]],
            [p_reflected[1], p_corner[1], p_transmitted[1]],
            color=color, linewidth=2.0, solid_capstyle="round",
            solid_joinstyle="round", zorder=zorder)


def add_polarization_marks(ax, start, end, kind, count=8, color="black", intensity=1.0, zorder=10):
    intensity, alpha, _ = strength_style(intensity)
    if intensity <= 0.0005:
        return
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    direction = end - start
    norm = np.linalg.norm(direction)
    if norm == 0:
        return
    direction /= norm
    normal = np.array([-direction[1], direction[0]])
    positions = np.linspace(0.16, 0.86, count)
    for idx, t in enumerate(positions):
        pos = start * (1 - t) + end * t
        draw_dot = kind == "s" or (kind == "natural" and idx % 2 == 0)
        draw_line = kind == "p" or (kind == "natural" and idx % 2 == 1)
        if draw_dot:
            ax.add_patch(Circle(pos, 0.035, facecolor=color, edgecolor=color,
                                linewidth=0, alpha=alpha, zorder=zorder))
        if draw_line:
            half_len = 0.095
            a = pos - normal * half_len
            b = pos + normal * half_len
            ax.plot([a[0], b[0]], [a[1], b[1]], color=color,
                    linewidth=1.6, alpha=alpha, zorder=zorder)


def setup_axis(ax, title, N1, N2):
    ax.set_aspect("equal")
    ax.set_xlim(-2.35, 2.35)
    ax.set_ylim(-1.85, 1.85)
    ax.axis("off")
    ax.add_patch(Rectangle((-2.35, 0), 4.70, 1.85,
                           facecolor="#f1f9ff", edgecolor="none", zorder=0))
    ax.add_patch(Rectangle((-2.35, -1.85), 4.70, 1.85,
                           facecolor="#dff1ff", edgecolor="none", zorder=0))
    ax.plot([-2.20, 2.20], [0, 0], color="#ff00cc", linewidth=2.6, zorder=2)
    ax.plot([0, 0], [-1.65, 1.65], linestyle="--", color="#bf00ff",
            linewidth=1.1, alpha=0.75, zorder=1)
    ax.text(-2.18, 1.55, f"上层介质  n2={N2:.4f}",
            fontsize=12, color="#155a8a", ha="left", va="center")
    ax.text(-2.18, -0.88, f"下层介质  n1={N1:.4f}",
            fontsize=12, color="#0f5c99", ha="left", va="center")
    ax.text(0.05, 0.07, "介质界面", fontsize=10, color="#aa0088", ha="left", va="bottom")
    ax.text(0.0, 1.76, title, fontsize=16, fontweight="bold",
            ha="center", va="top", color="#111111")


def draw_angle_arc(ax, incident_angle, refracted_angle):
    theta = np.linspace(0, math.radians(incident_angle), 120)
    radius = 0.45
    x = -radius * np.sin(theta)
    y = -radius * np.cos(theta)
    ax.plot(x, y, color="#333333", linewidth=1.1, alpha=0.75, zorder=12)
    mid = math.radians(max(incident_angle, 2.0) * 0.55)
    ax.text(-0.62 * math.sin(mid), -0.62 * math.cos(mid),
            f"i1={incident_angle:.1f}°",
            fontsize=11, ha="center", va="center", color="#333333", zorder=13)
    if refracted_angle is not None:
        theta_t = np.linspace(0, math.radians(refracted_angle), 120)
        x_t = radius * np.sin(theta_t)
        y_t = radius * np.cos(theta_t)
        ax.plot(x_t, y_t, color="#333333", linewidth=1.1, alpha=0.75, zorder=12)
        mid_t = math.radians(max(refracted_angle, 2.0) * 0.42)
        ax.text(0.60 * math.sin(mid_t), 0.60 * math.cos(mid_t),
                f"i2={refracted_angle:.1f}°",
                fontsize=11, ha="center", va="center", color="#333333", zorder=13)


def draw_ray_scene(ax, row, config, N1, N2, BREWSTER_ANGLE, CRITICAL_ANGLE):
    setup_axis(ax, config["title"], N1, N2)
    incident_angle = row["incident"]
    refracted_angle = row["refracted"]
    R = row[config["reflect_key"]]
    T = row[config["transmit_key"]]
    kind = config["kind"]
    origin = np.array([0.0, 0.0])
    ray_length = 2.05

    incident_start = origin + np.array(
        [-math.sin(math.radians(incident_angle)), -math.cos(math.radians(incident_angle))]
    ) * ray_length
    reflected_end = origin + np.array(
        [math.sin(math.radians(incident_angle)), -math.cos(math.radians(incident_angle))]
    ) * ray_length

    add_beam(ax, incident_start, origin, 1.0, zorder=5)
    add_direction_arrow(ax, incident_start, origin, 1.0, zorder=11)
    add_polarization_marks(ax, incident_start, origin, kind, count=7, intensity=1.0, zorder=12)
    ax.text(incident_start[0] - 0.05, incident_start[1] - 0.12, "入射光",
            fontsize=11, color="#8b0000", ha="center", va="top", zorder=14)

    reflected_visible = add_beam(ax, origin, reflected_end, R, zorder=5)
    if reflected_visible:
        add_direction_arrow(ax, origin, reflected_end, R, zorder=11)
        reflection_kind = "s" if kind == "natural" and abs(
            incident_angle - BREWSTER_ANGLE) <= ANGLE_STEP / 2 else kind
        add_polarization_marks(ax, origin, reflected_end, reflection_kind,
                               count=7, intensity=R, zorder=12)
        ax.text(reflected_end[0] + 0.05, reflected_end[1] - 0.12, "反射光",
                fontsize=11, color="#8b0000", ha="center", va="top",
                alpha=max(0.10, R), zorder=14)

    if refracted_angle is None or T <= 0.001:
        ax.text(0.78, 0.78, "全反射\n透射光消失", fontsize=13, fontweight="bold",
                color="#c00000", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.30", facecolor="white",
                          edgecolor="#ffb0b0", alpha=0.90), zorder=15)
    else:
        transmitted_end = origin + np.array(
            [math.sin(math.radians(refracted_angle)), math.cos(math.radians(refracted_angle))]
        ) * ray_length
        transmitted_visible = add_beam(ax, origin, transmitted_end, T, zorder=5)
        if transmitted_visible:
            add_direction_arrow(ax, origin, transmitted_end, T, zorder=11)
            add_polarization_marks(ax, origin, transmitted_end, kind,
                                   count=7, intensity=T, zorder=12)
            if reflected_visible and abs(incident_angle - BREWSTER_ANGLE) <= ANGLE_STEP / 2:
                add_brewster_right_angle(ax, reflected_end, transmitted_end)
            ax.text(transmitted_end[0] + 0.05, transmitted_end[1] + 0.10, "透射光",
                    fontsize=11, color="#8b0000", ha="center", va="bottom",
                    alpha=max(0.10, T), zorder=14)

    draw_angle_arc(ax, incident_angle, refracted_angle)
    ax.scatter([0], [0], s=34, color="black", zorder=15)

    if refracted_angle is None:
        gamma_text = "i2：全反射"
    else:
        gamma_text = f"i2 = {refracted_angle:.1f}°"
    ax.text(-2.18, -1.18,
            f"入射角 i1 = {incident_angle:.1f}°\n折射角 {gamma_text}\n反射率 R = {R:.3f}\n透射率 T = {T:.3f}",
            fontsize=12, color="#222222", ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor="#dddddd", alpha=0.90), zorder=20)
    ax.text(2.18, -1.50,
            f"布儒斯特角：{BREWSTER_ANGLE:.2f}°\n临界角：{CRITICAL_ANGLE:.2f}°",
            fontsize=11, color="#444444", ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.30", facecolor="white",
                      edgecolor="#dddddd", alpha=0.80), zorder=20)


def get_hold_index(rows, target_angle):
    return min(range(len(rows)), key=lambda idx: abs(rows[idx]["incident"] - target_angle))


def make_frame_sequence(rows, BREWSTER_ANGLE, CRITICAL_ANGLE):
    frame_indices = list(range(len(rows)))
    hold_frame_count = int(round(HOLD_SECONDS * FPS))
    hold_indices = sorted(
        (get_hold_index(rows, BREWSTER_ANGLE), get_hold_index(rows, CRITICAL_ANGLE)),
        reverse=True,
    )
    for key_idx in hold_indices:
        frame_indices[key_idx + 1:key_idx + 1] = [key_idx] * hold_frame_count
    return frame_indices


# ===================== Streamlit 网页界面逻辑 =====================
st.title("🔍 菲涅尔偏振反射透射动画演示")

# 侧边栏参数输入
with st.sidebar:
    st.header("参数设置")
    n1 = st.number_input("下层介质折射率 n1", min_value=1.0, max_value=5.0, value=1.3328, step=0.01)
    n2 = st.number_input("上层介质折射率 n2", min_value=1.0, max_value=5.0, value=1.0, step=0.01)
    pol_type = st.radio("偏振类型", ["s偏振", "p偏振", "自然光"], index=2)
    generate_btn = st.button("生成动画", type="primary", use_container_width=True)

# 配置映射
config_map = {
    "s偏振": {"title": "s 偏振反射与透射：点表示 s 偏振",
             "kind": "s", "reflect_key": "Rs", "transmit_key": "Ts"},
    "p偏振": {"title": "p 偏振反射与透射：横线表示 p 偏振",
             "kind": "p", "reflect_key": "Rp", "transmit_key": "Tp"},
    "自然光": {"title": "自然光反射与透射：点 + 横线表示两种偏振分量",
              "kind": "natural", "reflect_key": "Rn", "transmit_key": "Tn"},
}

# 点击按钮生成动画
if generate_btn:
    if n1 <= 0 or n2 <= 0:
        st.error("折射率必须大于0！")
    else:
        with st.spinner("正在计算并生成动画，请稍候..."):
            # 计算菲涅尔数据
            rows, brewster, critical = calculate_fresnel_data(n1, n2)
            config = config_map[pol_type]
            frame_seq = make_frame_sequence(rows, brewster, critical)

            # 生成动画
            fig, ax = plt.subplots(figsize=(9, 6.5))
            fig.subplots_adjust(left=0.03, right=0.97, bottom=0.04, top=0.96)

            def update_frame(frame_idx):
                ax.clear()
                draw_ray_scene(ax, rows[frame_idx], config, n1, n2, brewster, critical)
                return []

            anim = animation.FuncAnimation(
                fig, update_frame,
                frames=frame_seq,
                interval=1000 / FPS,
                blit=False,
                repeat=True
            )

            # 转成HTML格式嵌入网页播放
            anim_html = anim.to_jshtml()
            plt.close(fig)

            # 显示动画
            st.components.v1.html(anim_html, height=700, scrolling=False)

            # 显示参数校验
            st.success(f"✅ 动画生成完成\n\n布儒斯特角：{brewster:.4f}°\n临界角：{critical:.4f}°")
else:
    st.info("👈 在左侧输入折射率、选择偏振类型，点击「生成动画」即可播放")