import robosuite as suite
import numpy as np
import time


# ==========================
# 创建环境
# ==========================

env = suite.make(
    env_name="Lift",
    robots="Panda",

    # 使用robosuite默认Panda控制器
    has_renderer=True,
    has_offscreen_renderer=False,

    # 不使用相机
    use_camera_obs=False,

    # 控制频率
    control_freq=20,
)


# reset
obs = env.reset()


print("==============================")
print("Environment loaded")
print("Action dimension:", env.action_dim)
print("==============================")


# 打印观察空间
print("\nObservation keys:")
for key in obs.keys():
    print(key)


# ==========================
# 控制循环
# ==========================

step = 0

while True:

    # 创建动作
    # Panda默认:
    # 前6维 控制机械臂
    # 最后一维 控制夹爪

    action = np.zeros(env.action_dim)


    # --------------------------
    # 1. 保持机械臂
    # --------------------------

    action[:] = 0


    # --------------------------
    # 2. 让夹爪闭合
    # --------------------------

    if step > 100:
        action[-1] = -1


    # --------------------------
    # 3. 让末端稍微移动
    # --------------------------

    if 200 < step < 300:
        action[0] = 0.2


    # 执行动作

    obs, reward, done, info = env.step(action)


    # 渲染

    env.render()


    # 打印末端位置

    if step % 50 == 0:

        if "robot0_eef_pos" in obs:

            print(
                "EEF position:",
                obs["robot0_eef_pos"]
            )

        print(
            "Reward:",
            reward
        )


    # episode结束重新开始

    if done:
        print("Reset environment")
        obs = env.reset()
        step = 0


    step += 1


    time.sleep(0.02)