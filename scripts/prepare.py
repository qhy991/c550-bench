#!/usr/bin/env python3
"""Materialize selected official problems locally; raw dataset stays out of Git."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def ensure_source(lock):
    target = ROOT / ".deps/sol-execbench"
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", lock["url"], str(target)], check=True)
        if subprocess.check_output(["git", "-C", str(target), "rev-parse", "HEAD"], text=True).strip() != lock["revision"]:
            subprocess.run(["git", "-C", str(target), "fetch", "--depth", "1", "origin", lock["revision"]], check=True)
            subprocess.run(["git", "-C", str(target), "checkout", "--detach", lock["revision"]], check=True)
    observed = subprocess.check_output(["git", "-C", str(target), "rev-parse", "HEAD"], text=True).strip()
    if observed != lock["revision"]:
        raise RuntimeError("Existing SOL-ExecBench revision differs; preserve it and provide the pinned source separately")
    return target


def download(url, destination):
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as out:
        while chunk := response.read(1024 * 1024):
            out.write(chunk)
    temporary.replace(destination)


def write_or_verify(path, text):
    if path.exists():
        if path.read_text() != text:
            raise RuntimeError("Refusing to replace changed data: " + str(path))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, help="Directory already containing L1.parquet and L2.parquet")
    args = parser.parse_args()
    lock = json.loads((ROOT / "sources.lock.json").read_text())
    suite = json.loads((ROOT / "suite.json").read_text())
    ensure_source(lock["sol_execbench"])
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError("Preparation needs pyarrow; install it in a separate preparation environment, not the MetaX PyTorch environment") from exc
    dataset = lock["dataset"]
    cache = args.cache_dir or ROOT / ".data/cache" / dataset["revision"]
    base = "https://huggingface.co/datasets/{}/resolve/{}".format(dataset["id"], dataset["revision"])
    cache.mkdir(parents=True, exist_ok=True)
    download(base + "/LICENSE", ROOT / ".data/DATASET-LICENSE")
    materialized = []
    for level in ("L1", "L2"):
        filename = cache / (level + ".parquet")
        download(base + "/data/" + filename.name, filename)
        rows = {row["name"]: row for row in parquet.read_table(filename).to_pylist()}
        for task in suite["tasks"]:
            category, name = task["id"].split("/", 1)
            if category != level:
                continue
            row = rows[name]
            # Same projection as upstream scripts/download_solexecbench.py.
            definition = {
                "name": row["name"], "hf_id": row.get("hf_id"), "description": row["description"],
                "axes": json.loads(row["axes"]), "custom_inputs_entrypoint": row.get("custom_inputs_entrypoint"),
                "inputs": json.loads(row["inputs"]), "outputs": json.loads(row["outputs"]), "reference": row["reference"],
            }
            workloads = json.loads(row["workloads"])
            if len(workloads) != 16 or len({w["uuid"] for w in workloads}) != 16:
                raise RuntimeError("Expected 16 distinct original workloads: " + task["id"])
            if task["smoke_workload_uuid"] not in {w["uuid"] for w in workloads}:
                raise RuntimeError("Frozen smoke workload is absent: " + task["id"])
            target = ROOT / ".data/benchmark" / task["id"]
            write_or_verify(target / "definition.json", json.dumps(definition, indent=4) + "\n")
            write_or_verify(target / "reference.py", row["reference"])
            write_or_verify(target / "workload.jsonl", "".join(json.dumps(w) + "\n" for w in workloads))
            materialized.append({"task": task["id"], "workloads": len(workloads)})
    receipt = {"sources": lock, "tasks": materialized, "raw_dataset_in_git": False}
    version = suite.get("version", 1)
    receipt_name = "materialization.json" if version == 1 else f"materialization-suite-v{version}.json"
    write_or_verify(ROOT / ".data" / receipt_name, json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
