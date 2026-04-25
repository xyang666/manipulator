import mujoco
import numpy as np
import time
import mujoco.viewer
from env.kinematics import ManipulatorKinematics

# 加载模型
model = mujoco.MjModel.from_xml_path("models/panda_scene.xml")
data = mujoco.MjData(model)
viewer = mujoco.viewer.launch_passive(model, data)
k = ManipulatorKinematics(urdf_path="panda_description/urdf/panda_mujoco.urdf")

# 获取末端 site id
site_name = "ee_site"
site_id = model.site(site_name).id

# 控制参数
Kp = np.diag([200, 200, 200])
Kd = np.diag([20, 20, 20])

# 仿真步长
dt = model.opt.timestep


# 轨迹函数（圆轨迹）
def desired_trajectory(t):
    center = np.array([0.5, 0.0, 0.5])
    radius = 0.1
    omega = 1.0

    x_d = center + np.array([radius * np.cos(omega * t), radius * np.sin(omega * t), 0])

    v_d = np.array(
        [-radius * omega * np.sin(omega * t), radius * omega * np.cos(omega * t), 0]
    )

    return x_d, v_d

x_start, _ = desired_trajectory(0.0)

q = k.inverse_kinematics(x_start)
data.qpos[:] = np.hstack((q, np.zeros(2)))
data.qvel[:] = np.zeros(data.qvel.shape)
x1, r1 = k.forward_kinematics(q)
mujoco.mj_step(model, data)
viewer.sync()

# 主循环
for step in range(10000):
    t = step * dt

    # 当前末端位置 & 速度
    x = data.site_xpos[site_id].copy()

    # 计算雅可比
    J_pos = np.zeros((3, model.nv))
    J_rot = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, J_pos, J_rot, site_id)

    v = J_pos @ data.qvel

    # 期望轨迹
    x_d, v_d = desired_trajectory(0)

    # PD 控制
    # F = Kp @ (x_d - x) + Kd @ (v_d - v)
    F = Kp @ (x_d - x)

    # 映射到关节空间
    tau = J_pos.T @ F
    
    # 循环内
    Kp_joint = 10.0  # 刚度
    Kd_joint = 5.0   # 阻尼

    q_target = k.inverse_kinematics(x_d)
    # 关节空间 PD 控制
    error = q_target - data.qpos[:7]
    d_error = -data.qvel[:7]
    tau = Kp_joint * error + Kd_joint * d_error + data.qfrc_bias[:7]

    # 控制输入
    data.ctrl[:] = tau[:7]
    data.qpos[7:9] = [0.02, 0.02] 
    data.qvel[7:9] = 0
    
    # 前进一步
    mujoco.mj_step(model, data)

    scene = viewer.user_scn
    scene.ngeom = 0
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([0.004, 0., 0.]),
        x_d,
        np.eye(3).flatten(),
        np.array([1.0, 0.0, 0.0, 1.0]),  # Red
    )
    scene.ngeom += 1

    # 可视化（如果你用 viewer）
    viewer.sync()

    time.sleep(dt)
