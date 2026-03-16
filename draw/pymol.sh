distance hbonds, protein, lig, mode=2
distance hbonds, lig, water, mode=2
distance hbonds, protein, water, mode=2
show spheres, water
set sphere_scale, 0.4, water
set dash_width, 4
import center_of_mass
com sele, object=p1
com sele, object=p2
pseudoatom V4, pos=[49.843,  37.073,  42.591]
show spheres, V4
set sphere_scale, 2, V4
set cartoon_transparency, 0.3
set sphere_transparency, 0.5, sele

# stick外观
set stick_radius, 0.28
set stick_h_scale, 0.6
set valence, 0

# 抗锯齿
set antialias, 4

# 光照
set ambient, 0.35
set direct, 0.7
set specular, 0.25
set shininess, 10

# 阴影
set ray_shadow, off
set ray_trace_mode, 1

# 使用GPU shader
set use_shaders, on

# 边缘轮廓（非常重要）
set depth_cue, 0
set ray_trace_fog, 0

# 颜色更清晰
set cartoon_fancy_helices, 1
set cartoon_smooth_loops, 1

