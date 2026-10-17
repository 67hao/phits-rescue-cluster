"""
run_phits_swarm.py - High-Performance Swarm Worker for PHITS (Paper 10.1 - 10^8 Primaries)
Runs across 13 GitHub Accounts x 12 Runners = 156 Runners.
Streams 1-to-1 Flat CSV outputs directly to Google Drive (02_Flux_Data).
Batches .out files to 03_Simulation_Outputs.
Uploads logs to 05_Execution_Logs_and_Benchmarks.
"""
import os
import sys
import time
import zipfile
import argparse
import subprocess
from pathlib import Path
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import gdrive_helper
from generate_inputs import (
    _parse_thickness_list,
    _fmt_thickness,
    SAMPLE_TEMPLATE,
    BLANK_TEMPLATE,
    COMPOSITION,
    DENSITY,
    format_material
)

ACCOUNTS_ORDER = [
    "haoidlemystic1",
    "vuicho",
    "haoidle4zk",
    "67hao",
    "idlemystich",
    "CarwynDuc",
    "laolaolaoma09",
    "Bethwl",
    "coredaohaojack",
    "htvnhe",
    "haohue27",
    "fewhourtorelax",
    "zotactothesun",
]

DEFAULT_DRIVE_FOLDER = "1KZcZy2j3Bb6Ha7YA2to4yGYqWBWSjtWT"


def parse_args():
    parser = argparse.ArgumentParser(description="PHITS Swarm Worker for Paper 10.1 (10^8 particles)")
    parser.add_argument("--runner-id", type=int, required=True, help="Runner ID on this account (1..12)")
    parser.add_argument("--account-name", type=str, default=None, help="GitHub account username")
    parser.add_argument("--runners-per-acc", type=int, default=12, help="Number of runners per account")
    parser.add_argument("--gdrive-folder", type=str, default=DEFAULT_DRIVE_FOLDER, help="Root Google Drive folder ID")
    parser.add_argument("--maxcas", type=int, default=100000000, help="Number of particles per job (10^8)")
    parser.add_argument("--omp", type=int, default=2, help="Number of OpenMP threads")
    parser.add_argument("--excel-table", type=str, default="thickness_table_all9.xlsx", help="Excel thickness matrix")
    parser.add_argument("--package-path", type=str, default="phits_pack/phits_package.tar.gz", help="Path to phits_package.tar.gz")
    parser.add_argument("--timeout-hours", type=float, default=5.5, help="Safe timeout limit")
    return parser.parse_args()


def parse_track_flux(filepath: Path) -> tuple:
    """Read region 104 flux and relative error from T-Track .out file."""
    lines = filepath.read_text(encoding="utf-8", errors="ignore").splitlines()
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5 and parts[0] == "1" and parts[1] == "104":
            try:
                flux = float(parts[-2])
                err = float(parts[-1])
                return flux, err
            except ValueError:
                pass
    raise ValueError(f"Khong tim thay du lieu flux vung reg 104 trong {filepath}")


def setup_phits_environment(tar_path: Path, target_dir: Path) -> tuple:
    target_dir.mkdir(parents=True, exist_ok=True)
    phits_bin = None
    matches = list(target_dir.rglob("phits"))
    for m in matches:
        if m.is_file() and not m.name.endswith(".tar.gz") and not m.name.endswith(".out"):
            phits_bin = m
            break

    if not phits_bin or not phits_bin.exists():
        print(f">>> [SETUP] Dang giai nen goi PHITS tu {tar_path}...")
        subprocess.run(["tar", "-zxf", str(tar_path.resolve()), "-C", str(target_dir.resolve())], check=True)
        matches = list(target_dir.rglob("phits"))
        for m in matches:
            if m.is_file() and not m.name.endswith(".tar.gz") and not m.name.endswith(".out"):
                phits_bin = m
                break

    if not phits_bin:
        raise FileNotFoundError(f"Khong tim thay binary 'phits' sau khi giai nen {tar_path}")

    subprocess.run(["chmod", "+x", str(phits_bin)], check=True)
    phits_root = phits_bin.parent
    print(f">>> [SETUP] PHITS binary san sang tai: {phits_bin} (root: {phits_root})")
    return phits_bin.resolve(), phits_root.resolve()


def build_all_1711_jobs(excel_path: str, output_base: Path, omp: int, maxcas: int) -> list:
    df = pd.read_excel(excel_path)
    output_base.mkdir(parents=True, exist_ok=True)

    all_materials = [col for col in df.columns if col.lower() != "energy"]
    all_jobs = []
    blank_generated = set()

    for _, row in df.iterrows():
        energy_kev = int(round(row["Energy"]))
        e0_mev = energy_kev / 1000.0
        emax_mev = e0_mev * 1.05
        emin_peak = e0_mev * 0.990
        emax_peak = e0_mev * 1.010

        # 1. Blank job
        if energy_kev not in blank_generated:
            th_blank = 0.5
            z_sample_back = 50.0 + th_blank
            blank_name = f"blank_{energy_kev}keV"
            blank_dir = output_base / blank_name
            blank_dir.mkdir(parents=True, exist_ok=True)
            blank_inp = blank_dir / f"{blank_name}.inp"

            content = BLANK_TEMPLATE.format(
                energy_kev=energy_kev,
                e0_mev=e0_mev,
                emax_mev=emax_mev,
                emin_peak=emin_peak,
                emax_peak=emax_peak,
                omp=omp,
                maxcas=maxcas,
                thickness_str=_fmt_thickness(th_blank),
                z_sample_back=z_sample_back,
            )
            blank_inp.write_text(content, encoding="utf-8")
            blank_generated.add(energy_kev)

            all_jobs.append({
                "name": blank_name,
                "dir": blank_dir,
                "inp": blank_inp,
                "is_blank": True,
                "energy_kev": energy_kev,
                "energy_mev": e0_mev,
                "sample": "BLANK",
                "thickness_cm": th_blank,
                "flat_csv_name": f"BLANK_{_fmt_thickness(th_blank)}cm_{energy_kev}keV.csv"
            })

        # 2. Sample jobs (S1 to S8)
        for mat in all_materials:
            mat_key = mat.lower()
            if mat_key not in COMPOSITION:
                continue

            th_list = _parse_thickness_list(row[mat])
            for th in th_list:
                th_str = _fmt_thickness(th)
                job_name = f"{mat_key}_{energy_kev}keV_th{th_str}cm"
                job_dir = output_base / job_name
                job_dir.mkdir(parents=True, exist_ok=True)
                job_inp = job_dir / f"{job_name}.inp"

                z_sample_back = 50.0 + th
                sample_content = SAMPLE_TEMPLATE.format(
                    job_name=job_name,
                    mat_upper=mat_key.upper(),
                    energy_kev=energy_kev,
                    thickness_str=th_str,
                    e0_mev=e0_mev,
                    emax_mev=emax_mev,
                    emin_peak=emin_peak,
                    emax_peak=emax_peak,
                    omp=omp,
                    maxcas=maxcas,
                    mat_comp=format_material(COMPOSITION[mat_key]),
                    density=DENSITY[mat_key],
                    z_sample_back=z_sample_back,
                )
                job_inp.write_text(sample_content, encoding="utf-8")

                all_jobs.append({
                    "name": job_name,
                    "dir": job_dir,
                    "inp": job_inp,
                    "is_blank": False,
                    "energy_kev": energy_kev,
                    "energy_mev": e0_mev,
                    "sample": mat_key.upper(),
                    "thickness_cm": th,
                    "flat_csv_name": f"{mat_key.upper()}_{th_str}cm_{energy_kev}keV.csv"
                })

    return all_jobs


def main():
    args = parse_args()
    acc_name = args.account_name or os.environ.get("GITHUB_REPOSITORY_OWNER") or "haoidlemystic1"

    if acc_name in ACCOUNTS_ORDER:
        acc_idx = ACCOUNTS_ORDER.index(acc_name)
    else:
        acc_idx = 0

    runners_per_acc = args.runners_per_acc
    total_runners = len(ACCOUNTS_ORDER) * runners_per_acc
    global_runner_id = acc_idx * runners_per_acc + (args.runner_id - 1)

    print("=" * 80)
    print(f"🚀 STARTING PHITS SWARM WORKER: Runner #{args.runner_id} on Account: {acc_name}")
    print(f"Global Runner ID: {global_runner_id + 1} / {total_runners}")
    print(f"Project Folder ID: {args.gdrive_folder}")
    print(f"Primaries per Job: {args.maxcas:,} (10^8) | OMP Threads: {args.omp}")
    print("=" * 80)

    work_dir = Path("/tmp/phits_work").resolve() if os.name != "nt" else Path(r"C:\tmp\phits_work").resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    # 1. Setup PHITS binary
    phits_bin, phits_root = setup_phits_environment(Path(args.package_path), work_dir / "bin")

    # 2. Connect Google Drive
    drive_service = None
    folder_ids = {}
    existing_drive_files = set()
    try:
        drive_service = gdrive_helper.get_drive_service()
        folder_ids = gdrive_helper.get_or_create_subfolders(drive_service, parent_id=args.gdrive_folder)
        print(">>> [DRIVE] Ket noi Google Drive thanh cong!")
        for k, v in folder_ids.items():
            print(f"  • {k}: {v}")

        # Scan 02_Flux_Data for Auto-Resume
        flux_folder_id = folder_ids.get("02_Flux_Data")
        if flux_folder_id:
            page_token = None
            while True:
                q = f"'{flux_folder_id}' in parents and trashed = false"
                res = drive_service.files().list(
                    q=q, supportsAllDrives=True, includeItemsFromAllDrives=True,
                    fields="nextPageToken, files(name)", pageSize=1000, pageToken=page_token
                ).execute()
                for item in res.get("files", []):
                    existing_drive_files.add(item["name"].lower().strip())
                page_token = res.get("nextPageToken")
                if not page_token:
                    break
            print(f">>> [RESUME] Tim thay {len(existing_drive_files)} file da co tren 02_Flux_Data.")
    except Exception as e:
        print(f">>> [CANH BAO] Loi ket noi Google Drive: {e}. Runner tiep tuc xu ly local.")

    # 3. Build all 1,711 jobs
    inputs_base = work_dir / "inputs"
    all_jobs = build_all_1711_jobs(args.excel_table, inputs_base, args.omp, args.maxcas)
    print(f">>> [PLAN] Tong so job toan bo nghien cuu: {len(all_jobs)}")

    # 4. Partition disjoint slice for this runner
    my_jobs = [j for i, j in enumerate(all_jobs) if i % total_runners == global_runner_id]
    todo_jobs = [j for j in my_jobs if j["flat_csv_name"].lower() not in existing_drive_files]

    print(f">>> [RUNNER #{args.runner_id}] Duoc phan bo: {len(my_jobs)} jobs | Da xong: {len(my_jobs) - len(todo_jobs)} | Can chay: {len(todo_jobs)} jobs")

    if not todo_jobs:
        print(f"🎉 Toan bo {len(my_jobs)} jobs cua Runner #{args.runner_id} da hoan tat 100%! Thoat som.")
        return

    # 5. Execute simulation loop
    start_total = time.time()
    batch_outs = []
    batch_idx = 1
    completed_in_run = 0
    timing_records = []

    for idx, job in enumerate(todo_jobs, 1):
        elapsed_overall = time.time() - start_total
        if elapsed_overall > (args.timeout_hours * 3600 - 600):
            print(f">>> [TIMEOUT SAFEGUARD] Sap vuot qua {args.timeout_hours}h. Thoat de dong bo an toan.")
            break

        jname = job["name"]
        jdir = job["dir"]
        jinp = job["inp"]
        flat_csv = job["flat_csv_name"]

        print(f"\n--- [{idx}/{len(todo_jobs)}] Runner #{args.runner_id} ({acc_name}) dang chay: {jname} (10^8 hat) ---")
        t0 = time.time()

        env = os.environ.copy()
        env["PHITSPATH"] = str(phits_root)
        env["OMP_NUM_THREADS"] = str(args.omp)

        cmd = f'"{phits_bin}" < "{jinp.name}"'
        res = subprocess.run(cmd, shell=True, cwd=str(jdir), env=env, capture_output=True, text=True)
        job_elapsed = time.time() - t0

        peak_out = jdir / f"peak_{jname}.out"
        track_out = jdir / f"track_{jname}.out"

        p_val, p_err = (0.0, 0.0)
        t_val, t_err = (0.0, 0.0)

        if peak_out.exists() and track_out.exists():
            try:
                p_val, p_err = parse_track_flux(peak_out)
                t_val, t_err = parse_track_flux(track_out)
                print(f"    -> OK ({job_elapsed:.1f}s) | Peak: {p_val:.6e} (err: {p_err:.4f}) | Tot: {t_val:.6e}")
            except Exception as e:
                print(f"    ⚠️ Warning khi parse flux: {e}")
        else:
            tail_msg = (res.stderr or res.stdout)[-300:]
            print(f"    ❌ LOI THIEU FILE OUT: {tail_msg}")

        # Ghi 1-to-1 CSV
        csv_path = work_dir / flat_csv
        df_csv = pd.DataFrame([{
            "job_name": jname,
            "energy_mev": job["energy_mev"],
            "sample": job["sample"],
            "thickness_cm": job["thickness_cm"],
            "peak_flux": p_val,
            "peak_err": p_err,
            "total_flux": t_val,
            "total_err": t_err,
            "elapsed_s": round(job_elapsed, 2),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }])
        df_csv.to_csv(csv_path, index=False)

        # Upload ngay lap tuc len 02_Flux_Data
        if drive_service and "02_Flux_Data" in folder_ids:
            try:
                gdrive_helper.upload_file_to_folder(drive_service, str(csv_path), folder_ids["02_Flux_Data"])
                print(f"    ☁️ Streamed {flat_csv} -> Google Drive (02_Flux_Data)")
            except Exception as e:
                print(f"    ⚠️ Upload CSV failed: {e}")

        completed_in_run += 1
        timing_records.append(f"{jname}: {job_elapsed:.2f}s | Peak: {p_val:.6e}")

        if peak_out.exists():
            batch_outs.append(peak_out)
        if track_out.exists():
            batch_outs.append(track_out)

        # Dong bo Simulation_Outputs theo batch
        if len(batch_outs) >= 6:
            if drive_service and "03_Simulation_Outputs" in folder_ids:
                try:
                    zip_path = work_dir / f"outputs_{acc_name}_r{args.runner_id:02d}_batch_{batch_idx:03d}.zip"
                    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
                        for out_f in batch_outs:
                            z.write(out_f, arcname=out_f.name)
                    gdrive_helper.upload_file_to_folder(drive_service, str(zip_path), folder_ids["03_Simulation_Outputs"])
                    print(f"    📦 Batch #{batch_idx} .out files synced to Drive (03_Simulation_Outputs)")
                    batch_outs = []
                    batch_idx += 1
                except Exception as e:
                    print(f"    ⚠️ Sync outputs failed: {e}")

    # Dong bo phan du con lai cua Simulation_Outputs
    if batch_outs and drive_service and "03_Simulation_Outputs" in folder_ids:
        try:
            zip_path = work_dir / f"outputs_{acc_name}_r{args.runner_id:02d}_batch_{batch_idx:03d}.zip"
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
                for out_f in batch_outs:
                    z.write(out_f, arcname=out_f.name)
            gdrive_helper.upload_file_to_folder(drive_service, str(zip_path), folder_ids["03_Simulation_Outputs"])
        except Exception as e:
            pass

    # Ghi log benchmark va upload len 05_Execution_Logs_and_Benchmarks
    total_time = time.time() - start_total
    log_path = work_dir / f"benchmark_timing_{acc_name}_runner_{args.runner_id:02d}.txt"
    log_content = f"""================================================================================
PHITS RUNNER TIMING BENCHMARK (Paper 10.1 - 10^8 Primaries)
================================================================================
Account: {acc_name} | Runner ID: {args.runner_id} / {runners_per_acc} (Global #{global_runner_id + 1})
Completed Jobs: {completed_in_run} / {len(todo_jobs)}
Total Duration: {total_time:.1f}s ({total_time / 60:.1f} minutes)
Average per Job: {total_time / max(completed_in_run, 1):.2f}s
Timestamp: {time.strftime("%Y-%m-%d %H:%M:%S")}
--------------------------------------------------------------------------------
JOB BREAKDOWN:
""" + "\n".join(timing_records) + "\n================================================================================\n"

    log_path.write_text(log_content, encoding="utf-8")
    if drive_service and "05_Execution_Logs_and_Benchmarks" in folder_ids:
        try:
            gdrive_helper.upload_file_to_folder(drive_service, str(log_path), folder_ids["05_Execution_Logs_and_Benchmarks"])
            print(f">>> [BENCHMARK] Timing log uploaded to Google Drive (05_Execution_Logs_and_Benchmarks)")
        except Exception as e:
            pass

    print(f"\n🎉 Runner #{args.runner_id} on {acc_name} COMPLETED {completed_in_run} jobs in {total_time:.1f}s!")


if __name__ == "__main__":
    main()
