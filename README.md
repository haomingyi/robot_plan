# Wheeled Dual-Arm Pick-and-Place

这是一个面向简历和后续研究扩展的 MuJoCo 轮式双臂移动操作项目。项目名定为 **Wheeled Dual-Arm Pick-and-Place**，中文可以写作“轮式双臂移动抓取与放置仿真”。

## Final Effect

当前最终效果是一段可在官方 MuJoCo Viewer 中调试的轮式双臂协作任务：

- 机器人通过轮式底盘移动到桌前。
- 左臂抓取工件。
- 右臂抓取托盘。
- 双臂协同抬起并把工件放入托盘目标位置。

项目重点不是单纯“能播放动画”，而是逐步形成一个可调试、可评估、可扩展的移动双臂操作仿真基准。

## Project Structure

- `models/wheeled_dual_arm/`: 轮式双臂机器人、桌面、托盘、工件和必要 STL 网格。
- `scripts/dual_arm_pick_place.py`: 当前主线演示，包含底盘、双臂、夹爪和物体状态的阶段式控制逻辑。
- `run_dual_arm.sh`: 一键启动 MuJoCo Viewer，适合日常调试。
- `manipulation_benchmark.py`: 旧的 robosuite 单臂抓取 baseline，保留用于对照学习 viewer、日志和评估流程。
- `scripts/evaluate_policy.py`: 单臂 baseline 的重复评估脚本。
- `scripts/plot_run.py`: 单臂 baseline 的 CSV 诊断图生成脚本。

## Environment

进入项目后先激活环境：

```bash
cd /path/to/wheeled-dual-arm-pick-place
source /home/hzm/anaconda3/etc/profile.d/conda.sh
conda activate mujoco
```

`source /home/hzm/anaconda3/etc/profile.d/conda.sh` 的作用是把 conda 的 shell 函数加载进当前终端。加载之后，`conda activate mujoco` 才能在这个终端里切换到 MuJoCo 环境。

## Run The Main Demo

打开官方 MuJoCo Viewer：

```bash
./run_dual_arm.sh
```

不打开窗口，只检查模型和控制序列能否正常跑完：

```bash
python scripts/dual_arm_pick_place.py --headless
```

调整播放速度：

```bash
./run_dual_arm.sh --realtime 0.5
./run_dual_arm.sh --realtime 2.0
```

当前 viewer 会循环播放整段动作，不会在一轮结束后自动关闭。你想停下来调试时，可以在 Viewer 里暂停；想结束时，直接关闭 Viewer 窗口。

## What To Learn First

1. 先运行 `./run_dual_arm.sh`，观察机器人完整动作：底盘前进、双臂靠近、夹爪闭合、抬起、对准、释放。
2. 打开 `scripts/dual_arm_pick_place.py`，重点看 `STAGES`。每一行就是一个动作阶段，包含底盘位置、左右臂关节目标、夹爪开合、工件是否被“附着”。
3. 看 `sample_stage()`，理解两个阶段之间如何用 `smoothstep()` 平滑插值。
4. 看 `set_base_pose()`，理解轮式底盘的位置和轮子转角如何被设置。
5. 看 `set_joint_pose()`，理解左右 7 自由度机械臂和双夹爪的 actuator 控制。
6. 看 `set_workpiece_pose()` 和 `set_tray_pose()`，理解当前版本如何用脚本方式稳定演示双臂协作流程。
7. 下一步再把“脚本搬物体”逐渐替换为更真实的接触、抓取、传感器和控制器。

## MuJoCo Viewer Usage

左侧面板主要用于控制仿真和显示选项，例如暂停、单步、速度、渲染模式、接触点、坐标系、相机等。

右侧面板主要用于查看模型结构和调参数，例如 body、joint、actuator、sensor、camera、option 等。调试机器人时常用它确认关节名、执行器名、坐标系和接触设置。

如果坐标圆柱体太大，通常在 Viewer 的可视化设置里找 frame/site 相关显示项，或者在 XML 里调小对应 `site` 的 `size`。本项目后续可以把关键调试 site 的尺寸统一整理成更舒服的默认值。

## Current Baseline

主线双臂 demo 目前是阶段式 scripted controller，适合学习机器人模型、viewer 调试、轨迹流程和任务定义。它还不是完整的动力学抓取规划器。

旧的单臂 baseline 仍可运行：

```bash
./run_single_arm_baseline.sh
python manipulation_benchmark.py --no-render --steps 400 --policy pick --log-file logs/pick.csv
python scripts/evaluate_policy.py --runs 5 --steps 400 --policy pick
python scripts/plot_run.py logs/pick.csv --output outputs/pick_diagnostics.png
```

## Roadmap

1. 整理轮式双臂模型命名、关节表、执行器表和坐标系说明。
2. 给双臂 demo 增加 CSV 日志：阶段、底盘位置、左右夹爪位置、托盘位置、工件位置。
3. 增加双臂任务评估脚本：是否到达桌前、是否抓起、是否放入托盘、最终误差。
4. 增加诊断图：底盘轨迹、夹爪轨迹、工件高度、阶段时间线。
5. 把当前“脚本附着物体”替换为更真实的接触抓取和约束策略。
6. 加入相机、传感器和数据集导出，为模仿学习或强化学习做准备。
7. 把项目 README、演示 GIF、指标结果整理成简历项目描述。
