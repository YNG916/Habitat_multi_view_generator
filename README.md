# Habitat-Sim 多机器人世界想象数据集生成器

这是 MRI Dataset v1 的正式生成代码。唯一状态真值是可序列化的 `WorldState`；Level 1 factual state 和 Level 2 intervention after-state 使用同一套 Habitat-Sim 0.3.3 渲染、NavMesh、Bullet 和坐标约定。

## 正式版规格

正式配置是 `configs/collector.json`：

- 数据限制：ReplicaCAD 的全部 scene 都来自同一个 FRL apartment 建筑壳体；它不提供跨住宅/跨房型多样性。
- 正式 scene instances：公开的 `v3_sc0`–`v3_sc3` 四个 macro furniture layouts，每个使用 `_00`、`_01` 两个 micro rearrangements。
- macro-layout 隔离划分：train=`sc0,sc1`，val=`sc2`，test=`sc3`；同一 macro family 不会跨 split。
- 该划分衡量“同一公寓中的未见家具宏布局”泛化，不能写成“未见房间/未见住宅”泛化。
- 机器人视角：`2048 × 2048` RGB、metric Z-depth、OBJECT_ID、controlled semantic。
- 机器人本体：同一套 CC BY 4.0 成品扫地机器人 mesh，红/绿/蓝材质区分 agent；机身直径约 0.46 m、高约 0.107 m；相机固定在前保险杠上方 0.15 m、前向 0.235 m。
- BEV：目标分辨率 `0.00625 m/pixel`，实际宽高由每个场景的 visual AABB 决定。
- factual states：train 4000、val 250、test 250，共 4500。
- Level 2：每个 train state 生成 2 个 ID edit；val/test 分别生成 2 个 ID 和 2 个 OOD edit，共 10000 个 intervention pairs。
- 总渲染状态：4500 factual + 10000 after = 14500。
- 数值数组使用压缩 `.npz`，正式模式不为每个样本保存调试 contact sheet。

ID 参数域：

- robot translation：0.5、1.0 m；
- robot rotation：±30°、±60°；
- object translation/relative placement：0.5、1.0 m。

OOD 参数域：

- robot translation：0.75、1.5 m；
- robot rotation：±45°、±90°；
- object translation/relative placement：0.75、1.5 m。

动作类型、目标、方向、reference robot 和距离由稳定 SHA-256 seed 确定性采样。目标必须至少被一个机器人看到 64 个像素；非法路径、碰撞、错误地面支撑和重复 intervention 会被拒绝并确定性重采样。

## 为什么 apt_0–apt_5 看起来是同一个房间

这不是采样器重复加载场景，而是 ReplicaCAD 数据本身的定义：`apt_0`–`apt_5`
都实例化同一个 `frl_apartment_stage`，主要改变家具和对象配置。仓库现在按实际 stage
family 做强制校验；把同一个 family 放进多个 split 会在启动时直接报错。

本地只安装了 ReplicaCAD，因此当前无法提供真正不同建筑/房型。若论文目标包含
architecture-level generalization，需要另行安装并接入 HM3D、Gibson 或 MP3D，且按
house/scene ID 隔离；不要把 ReplicaCAD macro layouts 当作不同住宅。

## 正式高清 smoke（先运行这个）

唯一的正式 smoke 配置是 `configs/collector_formal_smoke.json`。它不是缩略图模式，
而是直接使用与完整数据集相同的图像规格：

- 每个机器人 RGB-D/instance/semantic：`2048×2048`；
- BEV：约 `1163×2078`、约 `0.00625 m/pixel`；
- 四个 macro families 各取一个代表 scene；
- 每个 scene 生成 1 个 factual state；
- train 生成 ID intervention，val/test 同时生成 ID 和 OOD intervention；
- 保存 annotated BEV、深度可视化和 `contact_sheet.png`；
- 最后执行 full Bullet/geometry validation。

直接运行完整高清 smoke：

```bash
conda run -n habitat python scripts/generate_dataset.py \
  --config configs/collector_formal_smoke.json
```

默认输出：`outputs/mri_dataset_formal_smoke_hd_v1_4`。当前实测生成 4 个 factual、
6 个 intervention/after-states，共 10 个渲染状态，约 159 MB，full validation
通过；30 次贴地检查的最大绝对误差约 0.000091 mm，自身可见像素为 0。v1.4 强制检查成品 robot proxy 的视觉底面、物理地面、base 原点和前保险杠
相机安装位姿。旧 v1.1/v1.2/v1.3 输出包含程序化高杆或中心相机方案，不应继续用于正式
数据集。
成功后再运行完整正式数据集：

```bash
conda run -n habitat python scripts/generate_dataset.py \
  --config configs/collector.json
```

完整正式输出默认是 `outputs/mri_dataset_v1_4`。两个配置的传感器分辨率和几何协议
相同，区别仅在场景实例数、每场景状态数和是否额外保存人工检查图。相同命令可安全
重复执行；已完成样本会跳过，配置指纹不一致时会拒绝混合输出。

建议完整生成前检查高清 smoke 的实际体积：

```bash
du -sh outputs/mri_dataset_formal_smoke_hd_v1_4
```

## 分阶段运行

只生成或补齐 Level 1：

```bash
conda run -n habitat python scripts/generate_dataset.py \
  --config configs/collector.json --stage level1 --validation metadata
```

只生成或补齐 Level 2：

```bash
conda run -n habitat python scripts/generate_dataset.py \
  --config configs/collector.json --stage level2 --validation metadata
```

重新执行最终 full validation：

```bash
conda run -n habitat python scripts/generate_dataset.py \
  --config configs/collector.json --stage validate --validation full
```

调试时可只运行一个配置内场景或覆盖样本目标数：

```bash
conda run -n habitat python scripts/generate_dataset.py \
  --config configs/collector_formal_smoke.json \
  --scene v3_sc0_staging_00 --num-states 1 --num-edits-per-state 1
```

`num-edits-per-state` 是“每个 factual state、每个 regime”的数量。例如 val 的 `id,ood` 且值为 2 时，每个 factual state 会得到 4 个 after-states。

## 输出结构

```text
mri_dataset_v1/
├── dataset.json
├── categories.json
├── calibration_report.json
├── generation_report.json
├── validation_report.json
├── scenes/<scene>/
│   ├── scene.json
│   ├── level1_collection_status.json
│   └── states/<state>/
│       ├── state.json
│       ├── objects.json
│       ├── bev/{rgb.png,height.npz,occupancy.npz,instance.npz,semantic.npz}
│       └── robots/robot_XX/{rgb.png,depth.npz,instance.npz,semantic.npz}
├── interventions/<scene>/
│   ├── level2_collection_status.json
│   └── edit_*.json
└── splits/
    ├── level1_{train,val,test}.json
    ├── level2_{train,val,test}_{id,ood}.json
    └── {train,val,test}.json
```

`states` 只表示 factual task inputs；`after_states` 和 `interventions` 单独索引，避免 intervention target 泄漏到 Level 1 training inputs。

## 几何和质量门禁

冻结坐标系：Habitat 右手世界坐标，`+Y` 向上，地面为 XZ，零 yaw 朝 `-Z`；BEV 右方为 `+X`，下方为 `+Z`。metadata 同时保存 Habitat/OpenCV 相机变换、xyzw quaternion、内参和实际 BEV 像素尺度。

正式生成会强制执行：

- Habitat 0.3.3 orthographic reciprocal pseudo-depth 专用 inverse + metric linearization；
- 0.2/0.5/1.0/1.5 m 共用 render/collision slab 的真实 Habitat + Bullet 标定；
- 3×3 非中心像素 pinhole Z-depth 与 Bullet stage 射线标定；
- BEV OBJECT_ID mask centroid 注册，容差 `max(3 px, 2 cm)`；
- robot forward exact XZ 距离、`try_step_no_sliding` 路径和目标 physical floor；
- robot/object 对场景家具、墙体和受控实体的 Bullet collision；
- 物体目标位置重新支撑到局部 physical floor；
- factual state 至少存在一个 benchmark-visible robot target 和 object target；
- train/val/test ReplicaCAD macro-family 防泄漏、ID/OOD 参数域和 manifest 完整性。

## Robot mesh 与相机高度

正式版不再程序化拼装机器人。三台 agent 使用同一套成品扫地机器人拓扑（直径约
0.46 m、高约 0.107 m），仅用红、绿、蓝 albedo 区分身份。源 FBX、作者、许可证、
哈希和变换记录见 `assets/robot_proxies/ATTRIBUTION.md`。

相机不再使用可变高杆：固定高度 0.15 m，并沿机器人局部 forward 安装在前缘
0.235 m 处，符合前保险杠 RGB-D/避障传感器布局，也避免第一视角看到自己的机身。
重新准备资产：

```bash
conda run -n habitat python scripts/prepare_robot_proxy_assets.py
```

该脚本只对作者成品 mesh 做 Y-up、尺度、中心和底面规范化，并生成三色材质；不会
创建或拼装几何。

## Instance 与 semantic

`instance.npz` 使用 Habitat `SemanticSensorTarget.OBJECT_ID`，每个 state 在 metadata 中记录 runtime object-ID 映射。

ReplicaCAD 安装包没有完整原生 room/furniture semantic scene，因此 `semantic.npz` 明确定义为“生成器受控实体类别”：0=未标注场景，1=robot，10=cup，11=bowl，12=book。它由 OBJECT_ID 精确派生，不伪造墙、桌子或沙发的类别标签。

## 测试

```bash
PYTHONPATH=src conda run -n habitat python -m unittest discover -s tests -v
```

测试包含纯 Python 协议测试，以及真实 Habitat/ReplicaCAD/Bullet 的 orthographic、多高度、OBJECT_ID/semantic、碰撞、路径和地面支撑测试。

## 仍需正确解释的标签

`difficulty_overlap` 是不考虑墙体和遮挡的 geometric FOV wedge overlap，不应在论文中称为真实 visual overlap。RGB BEV 是每个 scene 显式配置高度的 interior cutaway；扩展到新的 apartment 时必须增加 scene override 并重新通过正式 smoke/full validation。
