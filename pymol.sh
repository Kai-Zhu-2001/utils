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
set stick_radius, 0.3
set stick_h_scale, 0.5
set valence, 0
# 设置渲染模式
#set ray_trace_mode, 0
set antialias, 2
set ray_shadow, off
# 添加灯光效果
set ambient, 0.4
set direct, 0.5
# 使用高级渲染模式
set use_shaders, on
