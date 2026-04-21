import mujoco

# 1. 加载 URDF 文件
# 如果你的 URDF 引用了外部 mesh 文件，请确保路径正确
model = mujoco.MjModel.from_xml_path("panda_description/urdf/panda_mujoco.urdf")

# 2. 将模型保存为 MJCF (XML) 格式
mujoco.mj_saveLastXML("my_robot_converted.xml", model)

print("转换完成！已生成 my_robot_converted.xml")