# HSSD-only Multi-Robot Dataset Generator

本仓库的正式流程只支持标准 HSSD 文件
hssd-hab.scene_dataset_config.json。不接受 ReplicaCAD、HSSD-uncluttered、
HSSD-articulated、手写 apt 列表或任何场景 fallback。

## 协议要点

- split 单位是 hssd_scene_id：官方 train 确定性划分为 90% train 和 10%
  internal val；官方 val 完整作为 test。
- 预处理会发现每个场景的楼层。采样、occupancy 和 BEV 都受 scene_id、floor_id
  和 allowed_island_ids 约束。
- NavMesh 按真实机器人重算：radius 0.28 m、height 0.20 m、max climb
  0.05 m、max slope 20 度，并包含静态物体。
- 三个机器人使用同一个真实简洁扫地机器人 mesh，直径 0.46 m、高 0.107 m，
  只通过红、绿、蓝材质区分。
- 机器人相机固定在物理地板上方 0.15 m、身体前缘 0.235 m，不随机变高。
- BEV 相机与机器人相机无关；高度按每个楼层的天花板射线结果确定并保存到 registry。
- 每个机器人视角和 BEV 都保存 RGB、depth、OBJECT_ID instance 和受控实体 semantic。
  semantic 中 1=robot，10=cup，11=bowl，12=book，13=bottle，14=box；
  0 是未标注的原生 HSSD 背景。
- Level 2 支持 robot/object translate、robot rotate、relative placement 和 remove。

## 1. 下载标准 HSSD

HSSD 使用 CC BY-NC 4.0，请先确认许可。运行 Habitat-Sim 官方下载器：

    conda run --no-capture-output -n habitat       python -m habitat_sim.utils.datasets_download       --uids hssd-hab       --data-path "$PWD/data/"       --no-replace

必须存在：

    data/scene_datasets/hssd-hab/hssd-hab.scene_dataset_config.json
    data/scene_datasets/hssd-hab/scene_splits.yaml
    data/scene_datasets/hssd-hab/scenes/
    data/scene_datasets/hssd-hab/stages/
    data/scene_datasets/hssd-hab/objects/

## 2. 一次性预处理

先选择真实 HSSD 受控物体的精确 template handle：

    conda run --no-capture-output -n habitat       python scripts/build_hssd_object_registry.py       --config configs/collector_hssd.json

再处理全部官方场景：

    conda run --no-capture-output -n habitat       python scripts/preprocess_hssd.py       --config configs/collector_hssd.json

预处理可安全重跑：NavMesh 设置和 SHA256 相同时复用缓存。产物为：

    data/hssd_processed/scene_registry.json
    data/hssd_processed/split_manifest.json
    data/hssd_processed/navmeshes/<scene_id>.navmesh
    outputs/hssd_preprocess_previews/<scene_id>_<floor_id>_bev.png

单场景调试可增加 --scene，例如：

    conda run --no-capture-output -n habitat       python scripts/preprocess_hssd.py       --config configs/collector_hssd_smoke.json       --scene 102344022

## 3. 高清 smoke

configs/collector_hssd_smoke.json 是正式高清 smoke 配置：

- train、val、test 各最多一个真实 HSSD 场景；
- 三个机器人视角均为 768x768；
- BEV 为 0.02 m/px，实际像素大小随房屋尺寸变化；
- 每个 scene/floor 一个 factual state 和一个允许 regime 下的 intervention。

运行：

    conda run --no-capture-output -n habitat       python scripts/generate_dataset.py       --config configs/collector_hssd_smoke.json       --stage all       --validation full

固定场景、楼层和 seed 调试：

    conda run --no-capture-output -n habitat       python scripts/debug_world_state.py       --config configs/collector_hssd_smoke.json       --scene 102344022       --floor floor_00       --seed 123

检查 registry 和 BEV 尺寸：

    conda run --no-capture-output -n habitat       python scripts/inspect_hssd_scene.py       --config configs/collector_hssd_smoke.json       --scene 102344022       --floor floor_00

## 4. 生成完整正式数据集

确认 HSSD LFS 下载、物体 registry 和全部场景预处理完成后：

    conda run --no-capture-output -n habitat       python scripts/generate_dataset.py       --config configs/collector_hssd.json       --stage all       --validation full

正式配置中机器人视角为 2048x2048，BEV 为 0.00625 m/px。生成器会遍历
registry 里的全部合格 scene/floor，并支持断点续跑。全量运行前先检查磁盘空间、
registry 拒绝统计和三场景 smoke。

也可分阶段执行：

    conda run --no-capture-output -n habitat       python scripts/collect_level1.py --config configs/collector_hssd.json

    conda run --no-capture-output -n habitat       python scripts/collect_level2.py --config configs/collector_hssd.json

    conda run --no-capture-output -n habitat       python scripts/validate_dataset.py       --root outputs/mri_hssd_formal_v2       --config configs/collector_hssd.json

## 输出结构

    dataset_root/
      dataset.json
      categories.json
      calibration_report.json
      generation_report.json
      validation_report.json
      scenes/<scene_id>/
        scene.json
        floors/<floor_id>/
          floor.json
          level1_collection_status.json
          states/<state_id>/
      interventions/<scene_id>/<floor_id>/
        edit_*.json
        level2_collection_status.json
      splits/

每个 state 保存三组视角、楼层局部 BEV、相机内外参、真实 mesh handle、物体 bbox、
机器人贴地报告，以及 render/Bullet 深度和接触校验。

## 测试

    PYTHONPATH=src conda run --no-capture-output -n habitat       python -m unittest discover -s tests -p "test_*.py"

测试覆盖 split 防泄漏、scene/floor registry、机器人 NavMesh 设置、正交深度、
HSSD 真场景加载、楼层限定采样、semantic/instance、机器人贴地和数据集索引。
