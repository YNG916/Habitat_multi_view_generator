# Habitat-Sim 多机器人世界想象数据集生成器

这是 MRI Dataset v1 的正式生成代码。唯一状态真值是可序列化的 `WorldState`；Level 1 factual state 和 Level 2 intervention after-state 使用同一套 Habitat-Sim 0.3.3 渲染、NavMesh、Bullet 和坐标约定。

## 正式版规格

正式配置是 `configs/collector.json`：

- 场景：`apt_0`–`apt_5`。
- 场景隔离划分：train=`apt_0..3`，val=`apt_4`，test=`apt_5`。
- 机器人视角：`2048 × 2048` RGB、metric Z-depth、OBJECT_ID、controlled semantic。
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

## 一键生成

先运行六场景正式 smoke test：

```bash
conda run -n habitat python scripts/generate_dataset.py \
  --config configs/collector_formal_smoke.json
```

它会生成 6 个 factual states、8 个 ID/OOD edits，执行四高度标定和 full Bullet validation。通过后再启动完整数据集：

```bash
conda run -n habitat python scripts/generate_dataset.py \
  --config configs/collector.json
```

正式输出目录默认是 `outputs/mri_dataset_v1`。同一命令可以安全重复执行：完整 state/edit 会跳过，未完成的 after-state transaction 会恢复，不会覆盖已发布样本。配置指纹不一致时会拒绝把两种协议混入同一个 root。

建议长任务先确认磁盘空间。当前 apt_1 单条 2048 factual state 的实测压缩体积约
12 MB，线性估算 14500 个状态约 174 GB；不同场景、after-state 和 PNG/NPZ
压缩率会使最终体积波动，因此仍建议预留 200–300 GB 并用 smoke 实测：

```bash
du -sh outputs/mri_dataset_formal_smoke
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
  --scene apt_1 --num-states 1 --num-edits-per-state 1
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
- train/val/test apartment-family 防泄漏、ID/OOD 参数域和 manifest 完整性。

## Robot mesh 与相机高度

机器人是低多边形扫地机器人，加一根细伸缩桅杆和小型相机头。正式版只采样 0.4、0.6、0.9、1.2、1.4 m 五个高度，并选择完全对应高度的 mesh；第三方视角中的相机头中心与该机器人的第一视角 optical center 一致。重新生成资产：

```bash
conda run -n habitat python scripts/generate_robot_proxy_meshes.py
```

## Instance 与 semantic

`instance.npz` 使用 Habitat `SemanticSensorTarget.OBJECT_ID`，每个 state 在 metadata 中记录 runtime object-ID 映射。

ReplicaCAD 安装包没有完整原生 room/furniture semantic scene，因此 `semantic.npz` 明确定义为“生成器受控实体类别”：0=未标注场景，1=robot，10=cup，11=bowl，12=book。它由 OBJECT_ID 精确派生，不伪造墙、桌子或沙发的类别标签。

## 测试

```bash
conda run -n habitat python -m unittest discover -s tests -v
```

测试包含纯 Python 协议测试，以及真实 Habitat/ReplicaCAD/Bullet 的 orthographic、多高度、OBJECT_ID/semantic、碰撞、路径和地面支撑测试。

## 仍需正确解释的标签

`difficulty_overlap` 是不考虑墙体和遮挡的 geometric FOV wedge overlap，不应在论文中称为真实 visual overlap。RGB BEV 是每个 scene 显式配置高度的 interior cutaway；扩展到新的 apartment 时必须增加 scene override 并重新通过正式 smoke/full validation。
