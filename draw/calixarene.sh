select my_atoms, (resn MOL and name C1+C2+C3+C4+C5+C6+C8) or (resn OCB and name C8+C18+C26+C32+C97+C98+C99+C46+C52+C58+C64)
select water_5A, resn WAT and (my_atoms around 5)
remove resn WAT and not water_5A
show spheres, water_5A
set sphere_scale, 0.4, water_5A
set stick_transparency, 0.75, not my_atoms
color yellow, resn MOL
color gray70, resn OCB
remove elem Na+
remove elem H

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
