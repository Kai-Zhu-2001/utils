load frame_0.pdb
load frame_1.pdb
load frame_2.pdb
load frame_3.pdb
load frame_4.pdb
load frame_5.pdb
load frame_6.pdb
load frame_7.pdb
load frame_8.pdb
load frame_9.pdb

alter all, ss='L'
rebuild

# 抗锯齿
set antialias, 4


# 光照
set ambient, 0.35
set direct, 0.7
set specular, 0.25
set shininess, 10

# 阴影
set ray_shadow, off
set ray_trace_mode, 0

# 使用GPU shader
set use_shaders, on

# 边缘轮廓（非常重要）
set depth_cue, 0
set ray_trace_fog, 0

# 颜色更清晰
set cartoon_fancy_helices, 1
set cartoon_smooth_loops, 1

spectrum resi, blue_grey90_red,

set cartoon_transparency, 0.6, not frame_0
