# HSSD 多机器人区域级数据集生成器

本仓库只支持标准 `hssd-hab`。正式协议是 HSSD Scene → Floor → Semantic Region → WorldState；不会回退到 ReplicaCAD、uncluttered、articulated 或手写 apartment 列表。

## 当前数据源

`data/scene_datasets/hssd-hab` 是符号链接，实际指向同一仓库内的 `data/versioned_data/hssd-hab`。它们是同一份事先安装的数据，不是两份副本，生成器读取前者配置后解析到后者。不要重复下载。

标准配置必须是：

```text
data/scene_datasets/hssd-hab/hssd-hab.scene_dataset_config.json
data/scene_datasets/hssd-hab/scene_splits.yaml
data/versioned_data/hssd-hab/semantics/scenes/<scene>.semantic_config.json
```

Habitat-Sim 0.3.3 在标准 HSSD 场景中没有填充 active `SemanticScene`，因此区域协议直接读取 HSSD 官方 `region_annotations/poly_loop` 文件，并用官方多边形做严格点归属；没有自定义房间分割。

房间边界以官方语义区域为准，不按墙体或 RGB 重新猜测。开放式布局可能出现 U 形或带凹口的区域：例如 dining room 会绕开被官方单独标成 kitchen 的中央区域。预处理五联图依次显示原始 RGB、RGB+官方边界叠加、原始 semantic mask、NavMesh（绿色为 region 内可导航部分）、metric height，便于区分“语义房间范围”和“机器人实际可活动范围”。

## 正式协议

- 三台机器人必须在同一个 `RegionSpec`、同一物理楼层和允许的 NavMesh island 中。
- 机器人都是同一款完成 mesh 的简洁扫地机器人（直径 0.46 m、高 0.107 m），仅用红/绿/蓝区分；代理原点和视觉底部严格贴地。
- 相机高度固定为地面上方 0.15 m，位于机器人前缘；不会随机器人或房间随机改变。
- 正式 RGB/height BEV 由 semantic region 的局部 footprint 加 0.75 m context 得到，不再覆盖整层。
- `region_mask`、二值 occupancy、`navigable_region_mask = region_mask ∩ occupancy`、controlled semantic、OBJECT_ID instance 注册到同一 BEV frame。
- 每个状态确定性采样 2–4 个受控物体；类别无放回、同类 asset variant 随机，因此一个状态内不会出现重复类别；8 类、每类 4 个经批准的 whole-object HSSD asset。
- Level‑2 只移动受控机器人/物体，before/after 保持同一 `region_id` 和完全相同的 BEV bounds。
- split 单位始终是 HSSD scene：官方 train → formal train/internal val，官方 val → novel-scene test；同一 scene 的 region 不会跨 split。

Semantic 输出只标生成器控制的实体：robot=1，cup/bowl/book/bottle/box/bag/basket/toy=10–19 中的已配置 ID；原生 HSSD 背景为 0。场景房间监督由单独的 `region_mask` 提供。

## 配置

仓库只保留四个必要配置：

- `configs/collector_hssd_smoke.json`：与正式数据同规格的 2048×2048 Pilot/Smoke，每个 region 2 个状态、每 scene 最多 4 个 factual 状态。
- `configs/collector_hssd.json`：正式 2048×2048、0.00625 m/px；每 region 配额，train 每 scene 最多 500 个 factual 状态。
- `configs/hssd_controlled_objects.json`：显式批准的 canonical 相对 asset ID 和 SHA‑256。
- `configs/hssd_preprocess_overrides.json`：人工审核例外；当前只显式拒绝一个会触发 Habitat PBR `SIGABRT` 的损坏场景。

Smoke 使用 3 场景 `scene_registry_smoke.json`；正式配置使用全量 `scene_registry.json`。

输出名称固定，不再通过添加 `v2`、`v3`、`hd`、`final` 或 `correctness` 后缀创建并行版本：

- `outputs/mri_hssd_smoke`：当前高清 Smoke，可断点续跑；
- `outputs/mri_hssd`：正式数据集；
- `outputs/hssd_region_previews`：当前区域预处理审阅图；
- `outputs/hssd_object_candidates`：候选对象审阅图；
- `outputs/hssd_object_review` 与 `outputs/hssd_object_preflight.json`：正式对象池审阅结果。

协议指纹不兼容时，应明确归档或清空原固定目录后重新生成，不再创造带新后缀的 output root。

## 1. 构建候选物体清单（不会自动批准）

```bash
conda run --no-capture-output -n habitat \
  python scripts/build_hssd_object_registry.py \
  --config configs/collector_hssd.json
```

默认排除 `/decomposed/`，输出 `data/hssd_processed/object_candidates.json` 和每类候选 contact sheet。正式生成只读取人工维护的 `configs/hssd_controlled_objects.json`。

## 2. 正式对象池预检

```bash
conda run --no-capture-output -n habitat   python scripts/validate_hssd_controlled_objects.py   --config configs/collector_hssd.json
```

该命令对 approved 8×4 全池检查安全 canonical path、文件/hash、semantic ID、唯一 runtime handle、identity 朝向、visual/collision AABB、物理地面支撑和 OBJECT_ID 渲染。报告写到 `outputs/hssd_object_preflight.json`，每类正式审阅图写到 `outputs/hssd_object_review/`。所有正式 Level‑1/2 入口会在采集前自动运行同一预检；任一资源失败即停止。

HSSD rigid object 的默认 4 cm Bullet margin 只用于一般碰撞扩张，不代表 mesh 底面。approved 小物体在 runtime 统一使用 0 margin，再按 collision AABB 支撑到地面；原始 HSSD 配置文件不会被修改。

## 3. 预处理

三场景 Pilot：

```bash
conda run --no-capture-output -n habitat \
  python scripts/preprocess_hssd.py \
  --config configs/collector_hssd_smoke.json \
  --scene 102344022 --scene 102344307 --scene 102344094 \
  --registry-path data/hssd_processed/scene_registry_smoke.json \
  --split-manifest-path data/hssd_processed/split_manifest_smoke.json \
  --preview-root outputs/hssd_region_previews
```

全量 168 场景只生成 registry/NavMesh/预览，不生成正式图像数据：

```bash
conda run --no-capture-output -n habitat \
  python scripts/preprocess_hssd_parallel.py \
  --config configs/collector_hssd.json --workers 4 \
  --preview-root outputs/hssd_region_previews
```

缓存复用同时校验 NavMesh 参数、NavMesh SHA‑256、预处理 schema、所有区域/楼层/BEV 阈值、scene instance SHA‑256 和 semantic region 文件 SHA‑256。

不带 `--scene/--scene-file/--limit` 是显式全量重建；带任一筛选参数则是安全的 subset update：只替换选中 scene，保留其余 registry 条目，按 `scene_id` 排序后一次原子发布。旧 registry 的 schema、HSSD 配置来源或预处理配置不一致时会硬错误，不会静默混合。

## 4. 高清 Pilot/Smoke

```bash
conda run --no-capture-output -n habitat \
  python scripts/generate_dataset.py \
  --config configs/collector_hssd_smoke.json \
  --stage all --validation full
```

输出在 `outputs/mri_hssd_smoke`。可重复执行，完整 state/edit 会断点跳过；中断在 after-state 与 edit JSON 之间时会确定性恢复。

重建高清审阅图（不重新渲染）：

```bash
conda run --no-capture-output -n habitat \
  python scripts/rebuild_contact_sheets.py \
  --root outputs/mri_hssd_smoke
```

固定单个 region 调试：

```bash
conda run --no-capture-output -n habitat \
  python scripts/debug_world_state.py \
  --config configs/collector_hssd_smoke.json \
  --scene 102344022 --floor floor_00 \
  --region region_004_other_room --seed 123
```

## 5. 统计与正式生成

先生成全量统计和存储投影：

```bash
conda run --no-capture-output -n habitat \
  python scripts/report_hssd_regions.py \
  --config configs/collector_hssd.json
```

审核 `outputs/hssd_protocol_report.json` 后才启动正式数据：

```bash
conda run --no-capture-output -n habitat \
  python scripts/generate_dataset.py \
  --config configs/collector_hssd.json \
  --stage all --validation full
```

也可分别运行 `collect_level1.py`、`collect_level2.py` 和 `validate_dataset.py`。所有入口都支持 `--scene/--floor/--region`。

## 输出结构

```text
dataset_root/
  dataset.json
  categories.json
  calibration_report.json
  approved_object_preflight_report.json
  generation_report.json
  validation_report.json
  scenes/<scene_id>/
    scene.json
    floors/<floor_id>/
      floor.json
      regions/<region_id>/
        region.json
        states/<state_id>/
          state.json
          objects.json
          bev/{rgb,height,occupancy,region_mask,navigable_region_mask,semantic,instance,...}
          robots/robot_0{1,2,3}/{rgb,depth,semantic,instance,...}
          contact_sheet.png
  interventions/<scene_id>/<floor_id>/<region_id>/
    edit_*.json
    edit_*_contact_sheet.png
  splits/
```

每个 state 还保存相机内外参、两两欧氏/测地距离、真实 asset identifier、bbox、可见像素/图像占比、机器人贴地报告、pinhole/orthographic depth 校验和 Bullet 碰撞/高度验证。

`floor.json` 只保存 `FloorSpec` 的楼层级范围/统计；每个房间的 polygon、navigable/visual BEV bounds、BEV 高度和预处理统计只写入对应 `region.json`。对象的 `asset_identifier` 是跨机器重放的持久身份；`asset_handle` 仅保留为采集时 runtime 调试信息，重放不会依赖它。

## 测试

```bash
PYTHONPATH=src conda run --no-capture-output -n habitat \
  python -m unittest discover -s tests -v
```

测试覆盖区域 schema/点归属/同区域采样、mask 与 occupancy 注册、缓存失效、对象类别无放回确定性与 variant 多样性、approved registry 8×4 一致性、canonical 唯一解析、decomposed 排除、Level‑2 region/BEV 一致性、可见性阈值、机器人/对象贴地、真实 HSSD 渲染和 split 防泄漏。
