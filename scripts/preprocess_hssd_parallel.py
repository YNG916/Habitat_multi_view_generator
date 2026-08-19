#!/usr/bin/env python3
import argparse
import subprocess
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401
from mri_dataset.config import load_config
from mri_dataset.hssd_preprocess import discover_installed_hssd_scenes
from mri_dataset.scene_registry import SceneRegistry
from mri_dataset.serialization import write_json


def main():
    parser=argparse.ArgumentParser(description="Parallel all-scene HSSD preprocessing")
    parser.add_argument("--config",default="configs/collector_hssd.json")
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument(
        "--preview-root", default="outputs/hssd_region_previews"
    )
    args=parser.parse_args()
    if not 1<=args.workers<=8: raise ValueError("workers must be in [1,8]")
    config=load_config(args.config,require_preprocessed_registry=False)
    scenes=discover_installed_hssd_scenes(config.dataset_config_path)
    if not scenes: raise RuntimeError("No installed standard HSSD scenes")
    current={}
    if config.scene_registry_path.is_file():
        current={item.scene_id:item for item in SceneRegistry.load(config.scene_registry_path).scenes}
    work_root=config.scene_registry_path.parent/"parallel"
    log_root=work_root/"logs"
    log_root.mkdir(parents=True,exist_ok=True)
    processes=[]
    logs=[]
    for index in range(args.workers):
        part=scenes[index::args.workers]
        scene_file=work_root/f"scenes_{index:02d}.txt"
        scene_file.write_text("\n".join(part)+"\n",encoding="utf-8")
        registry_path=work_root/f"registry_{index:02d}.json"
        split_path=work_root/f"split_{index:02d}.json"
        cached=[current[scene] for scene in part if scene in current]
        SceneRegistry(
            dataset_source="hssd",
            dataset_config_path=config.scene_dataset_config,
            official_splits_path=config.official_scene_splits,
            scenes=cached,
        ).save(registry_path)
        log_path=log_root/f"worker_{index:02d}.log"
        log=log_path.open("w",encoding="utf-8")
        command=[
            sys.executable,str(Path(__file__).with_name("preprocess_hssd.py")),
            "--config",args.config,"--scene-file",str(scene_file),
            "--registry-path",str(registry_path),
            "--split-manifest-path",str(split_path),
            "--preview-root",args.preview_root,
        ]
        processes.append(subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT))
        logs.append(log)
    try:
        while True:
            complete=sum(process.poll() is not None for process in processes)
            counts=[]
            for index in range(args.workers):
                path=work_root/f"registry_{index:02d}.json"
                try: counts.append(len(SceneRegistry.load(path).scenes))
                except Exception: counts.append(0)
            print(
                f"parallel HSSD progress: {sum(counts)}/{len(scenes)} "
                f"workers_complete={complete}/{args.workers}",
                flush=True,
            )
            if complete==args.workers: break
            time.sleep(15)
    finally:
        for log in logs: log.close()
    failed=[(index,p.returncode) for index,p in enumerate(processes) if p.returncode]
    if failed:
        for index,code in failed:
            path=log_root/f"worker_{index:02d}.log"
            tail=path.read_text(encoding="utf-8",errors="replace").splitlines()[-30:]
            print(f"worker {index} failed ({code}):\n"+"\n".join(tail))
        raise SystemExit(1)
    merged=[]
    preprocessing_config={}
    for index in range(args.workers):
        registry=SceneRegistry.load(work_root/f"registry_{index:02d}.json")
        merged.extend(registry.scenes)
        preprocessing_config.update(registry.preprocessing_config)
    registry=SceneRegistry(
        dataset_source="hssd",
        dataset_config_path=config.scene_dataset_config,
        official_splits_path=config.official_scene_splits,
        scenes=sorted(merged,key=lambda item:item.scene_id),
        preprocessing_config=preprocessing_config,
    )
    registry.save(config.scene_registry_path)
    split=__import__("json").loads(
        (work_root/"split_00.json").read_text(encoding="utf-8")
    )
    write_json(config.split_manifest_path,split)
    print(registry.statistics)


if __name__=="__main__": main()
