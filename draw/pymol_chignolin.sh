reinitialize

load lower_representative.pdb, prot

# =========================
# 基础显示
# =========================

hide everything
bg_color white

# 去掉氢原子
remove prot and elem H

# =========================
# 整体蛋白：cartoon tube
# =========================

show cartoon, prot
cartoon tube, prot

set cartoon_tube_radius, 0.22
set cartoon_sampling, 30

set cartoon_fancy_helices, 1
set cartoon_smooth_loops, 0

# =========================
# 深蓝 -> 灰 -> 深红
# =========================

set_color c1,  [0.03, 0.05, 0.45]
set_color c2,  [0.10, 0.14, 0.58]
set_color c3,  [0.28, 0.31, 0.60]
set_color c4,  [0.45, 0.47, 0.58]
set_color c5,  [0.62, 0.62, 0.62]
set_color c6,  [0.66, 0.57, 0.57]
set_color c7,  [0.68, 0.43, 0.43]
set_color c8,  [0.68, 0.28, 0.28]
set_color c9,  [0.58, 0.12, 0.12]
set_color c10, [0.45, 0.03, 0.03]

color c1,  prot and resi 1
color c2,  prot and resi 2
color c3,  prot and resi 3
color c4,  prot and resi 4
color c5,  prot and resi 5
color c6,  prot and resi 6
color c7,  prot and resi 7
color c8,  prot and resi 8
color c9,  prot and resi 9
color c10, prot and resi 10

# =========================
# 只显示四个关键残基 sticks
# =========================

select keyres, prot and resi 3+5+7+8

show sticks, keyres

set stick_radius, 0.22
set stick_quality, 24
set stick_h_scale, 0.6
set valence, 0

# =========================
# 元素颜色
# =========================

color blue, keyres and elem N
color red,  keyres and elem O
color yellow, keyres and elem S

# =========================
# 抗锯齿
# =========================

set antialias, 4

# =========================
# 光照
# =========================

set ambient, 0.35
set direct, 0.7
set specular, 0.25
set shininess, 10

# =========================
# 阴影 / ray tracing
# =========================

set ray_shadows, off
set ray_trace_mode, 0

# =========================
# GPU shader
# =========================

set use_shaders, on

# =========================
# 去雾
# =========================

set depth_cue, 0
set ray_trace_fog, 0

# =========================
# 投影
# =========================

set orthoscopic, on

# =========================
# 聚焦局部结构
# =========================

select local, prot and resi 3-8

orient local
zoom local, 3

# =========================
# 可选：最终高分辨率渲染
# =========================

# ray 1800, 1800
# png upper_structure.png, dpi=300
