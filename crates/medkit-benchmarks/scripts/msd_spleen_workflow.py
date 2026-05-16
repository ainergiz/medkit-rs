"""Real-data MSD Task09 Spleen workflow for medkit-vs-MONAI benchmarking."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path
from typing import Any


MSD_SPLEEN_URL = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar"
MSD_SPLEEN_MD5 = "410d4a301da4e5b2f6f86ec3ddba524e"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download MSD Task09_Spleen and run medkit + MONAI data-loading benchmarks."
    )
    parser.add_argument("--work-dir", type=Path, default=Path("data/msd-spleen"))
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--url", default=MSD_SPLEEN_URL)
    parser.add_argument("--cases", type=int, default=4, help="training cases to extract; 0 means all")
    parser.add_argument("--patch", default="96,96,96")
    parser.add_argument("--cache-shape", default="160,160,160")
    parser.add_argument("--chunk", help="training-cache chunk shape; defaults to --patch")
    parser.add_argument("--spacing", default="1.0,1.0,1.0")
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--monai-workers", type=int, default=0)
    parser.add_argument("--medkit-torch-workers", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--medkit-torch-backend",
        choices=["map", "ffi-batch", "native-batch", "native-chunk-batch", "view-batch"],
        default="map",
    )
    parser.add_argument("--medkit-bin", type=Path, default=Path("target/release/medkit"))
    parser.add_argument(
        "--python",
        type=Path,
        default=Path("target/monai-baseline-venv/bin/python"),
        help="Python with monai, torch, and nibabel installed",
    )
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-monai", action="store_true")
    parser.add_argument("--skip-medkit-torch", action="store_true")
    parser.add_argument("--verify-md5", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    report = run(args)
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text)
    return 0


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.cases < 0:
        raise ValueError("--cases must be >= 0")
    if args.samples <= 0:
        raise ValueError("--samples must be > 0")
    if args.workers <= 0:
        raise ValueError("--workers must be > 0")
    if args.monai_workers < 0:
        raise ValueError("--monai-workers must be >= 0")
    medkit_torch_workers = args.monai_workers if args.medkit_torch_workers is None else args.medkit_torch_workers
    if medkit_torch_workers < 0:
        raise ValueError("--medkit-torch-workers must be >= 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    patch = parse_int3(args.patch, "--patch")
    cache_shape = parse_int3(args.cache_shape, "--cache-shape")
    chunk_shape = parse_int3(args.chunk or args.patch, "--chunk")
    spacing = parse_float3(args.spacing, "--spacing")

    args.work_dir.mkdir(parents=True, exist_ok=True)
    archive = args.archive or args.work_dir / "Task09_Spleen.tar"
    if not args.skip_download and not archive.exists():
        download(args.url, archive)
    if not archive.exists():
        raise FileNotFoundError(f"missing archive: {archive}")
    if args.verify_md5:
        verify_md5(archive, MSD_SPLEEN_MD5)

    dataset_root = args.work_dir / (
        "Task09_Spleen" if args.cases == 0 else f"Task09_Spleen_subset_{args.cases}"
    )
    if not (dataset_root / "imagesTr").exists() or not (dataset_root / "labelsTr").exists():
        extract_dataset(archive, dataset_root, args.cases)

    plan_path = dataset_root / "ct-spleen-medkit.toml"
    write_medkit_plan(plan_path, cache_shape, spacing)
    cache_dir = dataset_root / ".medkit" / "cache"
    manifest_path = dataset_root / "manifest.json"
    report_path = dataset_root / "report.txt"
    patches_path = dataset_root / "patches.jsonl"
    medkit_report_path = dataset_root / "medkit-workflow.json"
    monai_report_path = dataset_root / "monai-baseline.json"
    medkit_torch_report_path = dataset_root / "medkit-torch-dataloader.json"

    medkit_stages = [
        run_stage(
            "medkit_validate",
            [
                str(args.medkit_bin),
                "dataset",
                "validate",
                str(dataset_root),
                "--out",
                str(manifest_path),
                "--report",
                str(report_path),
            ],
        ),
        run_stage(
            "medkit_prepare",
            [
                str(args.medkit_bin),
                "prepare",
                str(dataset_root),
                "--manifest",
                str(manifest_path),
                "--plan",
                str(plan_path),
                "--cache",
                str(cache_dir),
                "--chunk",
                format_int3(chunk_shape),
            ],
        ),
        run_stage(
            "medkit_sample",
            [
                str(args.medkit_bin),
                "sample",
                str(cache_dir),
                "--patch",
                args.patch,
                "--strategy",
                "foreground-balanced",
                "--count",
                str(args.samples),
                "--out",
                str(patches_path),
            ],
        ),
        run_stage(
            "medkit_bench",
            [
                str(args.medkit_bin),
                "bench",
                str(cache_dir),
                "--patch",
                args.patch,
                "--workers",
                str(args.workers),
                "--samples",
                str(args.samples),
            ],
        ),
        run_stage(
            "medkit_bench_plan",
            [
                str(args.medkit_bin),
                "bench-plan",
                str(cache_dir),
                "--patches",
                str(patches_path),
                "--workers",
                str(args.workers),
                "--samples",
                str(args.samples),
            ],
        ),
    ]
    medkit_summary = summarize_medkit(medkit_stages)
    medkit_report_path.write_text(json.dumps({"stages": medkit_stages, "summary": medkit_summary}, indent=2))

    monai_report = None
    if not args.skip_monai:
        baseline_script = Path(__file__).with_name("monai_baseline.py")
        run_stage(
            "monai_baseline",
            [
                str(args.python),
                str(baseline_script),
                "--data-root",
                str(dataset_root),
                "--patch",
                args.patch,
                "--samples",
                str(args.samples),
                "--workers",
                str(args.monai_workers),
                "--batch-size",
                str(args.batch_size),
                "--spacing",
                args.spacing,
                "--out",
                str(monai_report_path),
            ],
        )
        monai_report = json.loads(monai_report_path.read_text())

    medkit_torch_report = None
    if not args.skip_medkit_torch:
        medkit_torch_script = Path(__file__).with_name("medkit_torch_dataset_baseline.py")
        run_stage(
            "medkit_torch_dataloader",
            [
                str(args.python),
                str(medkit_torch_script),
                "--cache",
                str(cache_dir),
                "--patches",
                str(patches_path),
                "--samples",
                str(args.samples),
                "--workers",
                str(medkit_torch_workers),
                "--batch-size",
                str(args.batch_size),
                "--backend",
                args.medkit_torch_backend,
                "--out",
                str(medkit_torch_report_path),
            ],
        )
        medkit_torch_report = json.loads(medkit_torch_report_path.read_text())

    return {
        "dataset": "MSD Task09_Spleen",
        "source_url": args.url,
        "dataset_root": str(dataset_root),
        "cases_requested": args.cases,
        "cases": count_cases(dataset_root),
        "patch": list(patch),
        "cache_shape": list(cache_shape),
        "chunk_shape": list(chunk_shape),
        "spacing": list(spacing),
        "samples": args.samples,
        "workers": args.workers,
        "monai_workers": args.monai_workers,
        "medkit_torch_workers": medkit_torch_workers,
        "batch_size": args.batch_size,
        "medkit_torch_backend": args.medkit_torch_backend,
        "medkit": medkit_summary,
        "monai": monai_report,
        "medkit_torch": medkit_torch_report,
        "reports": {
            "medkit": str(medkit_report_path),
            "monai": str(monai_report_path) if monai_report else None,
            "medkit_torch": str(medkit_torch_report_path) if medkit_torch_report else None,
            "manifest": str(manifest_path),
            "cache": str(cache_dir),
        },
        "comparison": compare(medkit_summary, monai_report, medkit_torch_report),
    }


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    print(f"downloading {url} -> {path}", file=sys.stderr)
    with urllib.request.urlopen(url) as response, tmp.open("wb") as out:
        shutil.copyfileobj(response, out, length=1024 * 1024)
    tmp.replace(path)


def verify_md5(path: Path, expected: str) -> None:
    digest = hashlib.md5()  # noqa: S324 - dataset integrity check, not security.
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    actual = digest.hexdigest()
    if actual != expected:
        raise ValueError(f"MD5 mismatch for {path}: expected {expected}, got {actual}")


def extract_dataset(archive: Path, dataset_root: Path, cases: int) -> None:
    print(f"extracting {archive} -> {dataset_root}", file=sys.stderr)
    if dataset_root.exists():
        shutil.rmtree(dataset_root)
    dataset_root.mkdir(parents=True)
    with tarfile.open(archive) as tar:
        members = [member for member in tar.getmembers() if member.isfile()]
        case_ids = select_case_ids(members, cases)
        wanted = []
        for member in members:
            relative = task_relative_path(member.name)
            if not relative or is_hidden_archive_member(relative):
                continue
            if relative == "dataset.json":
                wanted.append((member, relative))
                continue
            parts = Path(relative).parts
            if len(parts) != 2 or parts[0] not in {"imagesTr", "labelsTr"}:
                continue
            if case_id_from_name(parts[1]) in case_ids:
                wanted.append((member, relative))
        for member, relative in wanted:
            out_path = dataset_root / relative
            out_path.parent.mkdir(parents=True, exist_ok=True)
            source = tar.extractfile(member)
            if source is None:
                continue
            with source, out_path.open("wb") as out:
                shutil.copyfileobj(source, out, length=1024 * 1024)


def select_case_ids(members: list[tarfile.TarInfo], cases: int) -> set[str]:
    labels = []
    for member in members:
        relative = task_relative_path(member.name)
        if not relative or is_hidden_archive_member(relative):
            continue
        parts = Path(relative).parts
        if len(parts) == 2 and parts[0] == "labelsTr" and parts[1].endswith((".nii", ".nii.gz")):
            labels.append(case_id_from_name(parts[1]))
    labels = sorted(set(labels))
    if cases == 0:
        return set(labels)
    return set(labels[:cases])


def task_relative_path(name: str) -> str | None:
    path = Path(name)
    parts = path.parts
    if not parts:
        return None
    if parts[0] == "Task09_Spleen":
        return str(Path(*parts[1:])) if len(parts) > 1 else None
    return None


def is_hidden_archive_member(relative: str) -> bool:
    return any(part.startswith(".") or part == "__MACOSX" for part in Path(relative).parts)


def write_medkit_plan(path: Path, cache_shape: tuple[int, int, int], spacing: tuple[float, float, float]) -> None:
    path.write_text(
        f'''name = "msd-spleen-real-workflow"
image_interpolation = "linear"
label_interpolation = "nearest"

[[operations]]
op = "ct_window"
min = -57.0
max = 164.0

[[operations]]
op = "normalize"
mean = 0.0
std = 1.0

[[operations]]
op = "crop_foreground"
margin = 4

[[operations]]
op = "pad_crop"
size = [{cache_shape[0]}, {cache_shape[1]}, {cache_shape[2]}]

[[operations]]
op = "resample"
spacing = [{spacing[0]}, {spacing[1]}, {spacing[2]}]
'''
    )


def run_stage(name: str, command: list[str]) -> dict[str, Any]:
    start = time.perf_counter()
    output = subprocess.run(command, text=True, capture_output=True, check=False)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    stage = {
        "name": name,
        "command": command,
        "elapsed_ms": elapsed_ms,
        "exit_code": output.returncode,
        "stdout": output.stdout,
        "stderr": output.stderr,
    }
    if output.returncode != 0:
        raise RuntimeError(
            f"{name} failed with exit code {output.returncode}\nstdout:\n{output.stdout}\nstderr:\n{output.stderr}"
        )
    return stage


def summarize_medkit(stages: list[dict[str, Any]]) -> dict[str, Any]:
    by_name = {stage["name"]: stage for stage in stages}
    bench = parse_medkit_bench_stdout(by_name["medkit_bench"]["stdout"])
    plan_bench = parse_medkit_plan_bench_stdout(by_name["medkit_bench_plan"]["stdout"])
    sample_records = parse_first_int(by_name["medkit_sample"]["stdout"], r"Samples: (\d+)")
    sample_ms = by_name["medkit_sample"]["elapsed_ms"]
    return {
        "validate_ms": by_name["medkit_validate"]["elapsed_ms"],
        "prepare_ms": by_name["medkit_prepare"]["elapsed_ms"],
        "sample_ms": sample_ms,
        "sample_records_per_second": sample_records / max(sample_ms / 1000.0, sys.float_info.epsilon),
        "bench_elapsed_ms": by_name["medkit_bench"]["elapsed_ms"],
        "bench": bench,
        "plan_bench_elapsed_ms": by_name["medkit_bench_plan"]["elapsed_ms"],
        "plan_bench": plan_bench,
    }


def parse_medkit_bench_stdout(stdout: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    out["samples"] = parse_first_int(stdout, r"Samples: (\d+)")
    for line in stdout.splitlines():
        if line.startswith("Cold:") or line.startswith("Warm:"):
            key = line.split(":", 1)[0].lower()
            nums = [float(value) for value in re.findall(r"[0-9]+(?:\.[0-9]+)?", line)]
            out[key] = {
                "samples_per_second": nums[0],
                "mb_per_second": nums[1],
                "elapsed_ms": nums[2],
            }
    return out


def parse_medkit_plan_bench_stdout(stdout: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    out["records"] = parse_first_int(stdout, r"Records: (\d+)")
    out["samples"] = parse_first_int(stdout, r"Samples: (\d+)")
    for line in stdout.splitlines():
        if line.startswith("Plan cold:") or line.startswith("Plan warm:"):
            key = line.split(":", 1)[0].removeprefix("Plan ").lower()
            nums = [float(value) for value in re.findall(r"[0-9]+(?:\.[0-9]+)?", line)]
            out[key] = {
                "samples_per_second": nums[0],
                "mb_per_second": nums[1],
                "elapsed_ms": nums[2],
            }
    return out


def parse_first_int(text: str, pattern: str) -> int:
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f"could not parse {pattern!r} from text:\n{text}")
    return int(match.group(1))


def compare(
    medkit: dict[str, Any],
    monai: dict[str, Any] | None,
    medkit_torch: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if monai is None:
        return None
    monai_sps = monai["samples_per_second"]
    comparison = {
        "prepare_vs_monai_cache": monai["cache_build_ms"] / medkit["prepare_ms"],
        "sample_stage_vs_monai": medkit["sample_records_per_second"] / monai_sps,
        "cold_extraction_vs_monai": medkit["bench"]["cold"]["samples_per_second"] / monai_sps,
        "warm_extraction_vs_monai": medkit["bench"]["warm"]["samples_per_second"] / monai_sps,
        "plan_cold_extraction_vs_monai": medkit["plan_bench"]["cold"]["samples_per_second"] / monai_sps,
        "plan_warm_extraction_vs_monai": medkit["plan_bench"]["warm"]["samples_per_second"] / monai_sps,
    }
    if medkit_torch is not None:
        comparison["medkit_torch_dataloader_vs_monai"] = medkit_torch["samples_per_second"] / monai_sps
    return comparison


def count_cases(root: Path) -> int:
    return len(list((root / "labelsTr").glob("*.nii*")))


def case_id_from_name(name: str) -> str:
    if name.endswith(".nii.gz"):
        stem = name[:-7]
    elif name.endswith(".nii"):
        stem = name[:-4]
    else:
        stem = Path(name).stem
    if stem.endswith("_0000"):
        stem = stem[:-5]
    return stem


def parse_int3(value: str, flag: str) -> tuple[int, int, int]:
    parts = value.split(",")
    if len(parts) != 3:
        raise ValueError(f"{flag} must be formatted as x,y,z, got {value}")
    out = tuple(int(part) for part in parts)
    if any(part <= 0 for part in out):
        raise ValueError(f"{flag} values must be positive, got {value}")
    return out  # type: ignore[return-value]


def format_int3(value: tuple[int, int, int]) -> str:
    return f"{value[0]},{value[1]},{value[2]}"


def parse_float3(value: str, flag: str) -> tuple[float, float, float]:
    parts = value.split(",")
    if len(parts) != 3:
        raise ValueError(f"{flag} must be formatted as x,y,z, got {value}")
    out = tuple(float(part) for part in parts)
    if any(part <= 0.0 for part in out):
        raise ValueError(f"{flag} values must be positive, got {value}")
    return out  # type: ignore[return-value]


if __name__ == "__main__":
    raise SystemExit(main())
