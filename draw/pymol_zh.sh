Remove hydrogen #去除氢
remove solvent#移除溶剂
bg_color white#设置背景为白色
set valence, 0#不显示双键

“editing”模式下，选中两个目标原子
unbond pk1,pk2  #解除键
bond pk1,pk2,2  #生成双键

“Setting”—“Edit all”界面修改以下参数：
Cartoon_transparency,0.6
Cartoon_color, white
Ambient, 0.4
Ray_trace_mode, 1
Specular,2#光泽值

在pymol自带颜色中选择喜欢的颜色即可；
也可以输入以下指令使用已知16进制号码的颜色：
color 0xf7d8b7,obj03 #16进制颜色前面直接加“0x”即可，“obj0”3是你的目标对象
如果选色后氮氧原子颜色消失，可以输入如下指令恢复：
color atomic, (not elem C) #除了碳原子，其他根据元素显示颜色

“Wizard”-“measurement”-选择两个原子测量距离
所生成的距离线条可以编辑粗细、颜色等

ray 1800,1200 #加载会有些慢 
File-Export image as-Save image as，保存即可
