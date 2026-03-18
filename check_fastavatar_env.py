# check_fastavatar_env.py
# 用法:
#   python check_fastavatar_env.py --project_root D:/Repo/FastAvatar --sample_id 128

import argparse
import importlib
import importlib.util
import os
import platform
import struct
import sys
import traceback
from pathlib import Path

def safe_print(title, value):
    print(f"[CHECK] {title}: {value}")

def find_site_package_version(pkg_name):
    try:
        from importlib.metadata import version
        return version(pkg_name)
    except Exception:
        return "unknown"

def check_pycolmap():
    result = {}
    try:
        import pycolmap
        result["import_ok"] = True
        result["module_file"] = getattr(pycolmap, "__file__", "unknown")
        result["version"] = find_site_package_version("pycolmap")
        result["module_dir"] = str(Path(result["module_file"]).parent)
    except Exception as e:
        result["import_ok"] = False
        result["error"] = repr(e)
    return result

def scan_scene_manager(pycolmap_module_file):
    out = {}
    p = Path(pycolmap_module_file).parent / "scene_manager.py"
    out["path"] = str(p)
    out["exists"] = p.exists()
    if not p.exists():
        out["pattern_unpack_L_read8"] = False
        out["pattern_unpack_Q_read8"] = False
        out["hint"] = "scene_manager.py not found"
        return out

    txt = p.read_text(encoding="utf-8", errors="ignore")
    out["pattern_unpack_L_read8"] = "unpack('L', f.read(8))" in txt or 'unpack("L", f.read(8))' in txt
    out["pattern_unpack_Q_read8"] = "unpack('Q', f.read(8))" in txt or 'unpack("Q", f.read(8))' in txt
    out["first_match_index_L"] = txt.find("unpack('L', f.read(8))")
    return out

def check_sparse_files(project_root, sample_id):
    base = Path(project_root) / "data" / str(sample_id) / "sparse" / "0"
    out = {"sparse_dir": str(base), "exists": base.exists(), "files": {}}
    names = ["cameras.bin", "images.bin", "points3D.bin"]
    for n in names:
        p = base / n
        info = {"exists": p.exists(), "size": p.stat().st_size if p.exists() else -1}
        if p.exists():
            with p.open("rb") as f:
                head = f.read(16)
            info["head16_hex"] = head.hex()
        out["files"][n] = info
    return out

def try_scene_manager_load(project_root, sample_id):
    out = {"ok": False}
    sparse_dir = Path(project_root) / "data" / str(sample_id) / "sparse" / "0"
    try:
        from pycolmap import SceneManager
        sm = SceneManager(str(sparse_dir))
        sm.load_cameras()
        n_cam = len(getattr(sm, "cameras", {}))
        sm.load_images()
        n_img = len(getattr(sm, "images", {}))
        sm.load_points3D()
        n_pts = len(getattr(sm, "points3D", []))
        out["ok"] = True
        out["n_cameras"] = n_cam
        out["n_images"] = n_img
        out["n_points3D"] = n_pts
    except Exception as e:
        out["ok"] = False
        out["error"] = repr(e)
        out["traceback"] = traceback.format_exc()
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", type=str, default=".")
    parser.add_argument("--sample_id", type=int, default=128)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()

    print("=" * 80)
    print("FastAvatar Environment Diagnostic")
    print("=" * 80)
    safe_print("python", sys.version.replace("\n", " "))
    safe_print("platform", platform.platform())
    safe_print("cwd", os.getcwd())
    safe_print("project_root", str(project_root))
    safe_print("struct.calcsize('L')", struct.calcsize("L"))
    safe_print("struct.calcsize('Q')", struct.calcsize("Q"))

    print("\n--- pycolmap ---")
    pyc = check_pycolmap()
    for k, v in pyc.items():
        safe_print(k, v)

    if pyc.get("import_ok"):
        print("\n--- scene_manager.py pattern scan ---")
        sm_scan = scan_scene_manager(pyc["module_file"])
        for k, v in sm_scan.items():
            safe_print(k, v)
    else:
        sm_scan = {}

    print("\n--- sparse files ---")
    sparse = check_sparse_files(project_root, args.sample_id)
    safe_print("sparse_dir", sparse["sparse_dir"])
    safe_print("sparse_exists", sparse["exists"])
    for fn, meta in sparse["files"].items():
        safe_print(f"{fn}.exists", meta["exists"])
        safe_print(f"{fn}.size", meta["size"])
        if meta["exists"]:
            safe_print(f"{fn}.head16_hex", meta["head16_hex"])

    print("\n--- SceneManager load test ---")
    load_res = try_scene_manager_load(project_root, args.sample_id)
    safe_print("load_ok", load_res["ok"])
    if load_res["ok"]:
        safe_print("n_cameras", load_res["n_cameras"])
        safe_print("n_images", load_res["n_images"])
        safe_print("n_points3D", load_res["n_points3D"])
    else:
        safe_print("error", load_res.get("error", "unknown"))
        print("\n[TRACEBACK]")
        print(load_res.get("traceback", ""))

    print("\n--- quick diagnosis ---")
    if not pyc.get("import_ok"):
        print("1) pycolmap 导入失败，先修复 pycolmap 安装。")
    else:
        L_size = struct.calcsize("L")
        bad_pattern = sm_scan.get("pattern_unpack_L_read8", False)
        if L_size == 4 and bad_pattern:
            print("1) 命中高概率根因: Windows 上 L=4 字节，但代码按 read(8) 读取。")
            print("2) 这会导致你看到的 struct.error: unpack requires a buffer of 4 bytes。")
            print("3) 建议更换 pycolmap 版本或在 Linux/WSL2 运行。")
        elif load_res["ok"]:
            print("1) SceneManager 可正常读取，问题可能在后续数据路径或图像匹配阶段。")
        else:
            print("1) SceneManager 读取仍失败，但不是经典 L/read(8) 模式，需看 traceback 进一步定位。")

    print("=" * 80)

if __name__ == "__main__":
    main()
    print("\n诊断完成。")