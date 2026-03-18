"""
Unified launcher for FastAvatar pipelines.

Why this file exists:
1) Provide a single team entrypoint for running all pipelines.
2) Apply Windows compatibility patches (pycolmap + gsplat) automatically.
3) Keep business scripts clean and avoid manual environment edits.
"""

import argparse
import importlib
import importlib.util
import platform
import re
import struct
import sys
from collections import OrderedDict
from pathlib import Path
import array

import numpy as np


def launcher_log(message, verbose=True):
    """Print launcher-prefixed logs when verbose mode is enabled."""
    if verbose:
        print(f"[launcher] {message}")


def is_windows():
    """Return True only on Windows platforms."""
    return platform.system().lower() == "windows"


def should_apply_windows_patch(patch_name, verbose=True):
    """Shared gate for all Windows-only compatibility patches."""
    if not is_windows():
        launcher_log(f"non-windows platform, skip {patch_name} patch", verbose)
        return False
    return True


def apply_pycolmap_windows_patch(verbose=True):
    """
    Apply a runtime patch to pycolmap SceneManager binary loaders on Windows.

    Scope and safety:
    - Only affects current Python process.
    - Does NOT modify any files in site-packages.
    - Patch is applied only once per process.
    """
    if not should_apply_windows_patch("pycolmap", verbose):
        return

    try:
        import pycolmap.scene_manager as sm
    except Exception as e:
        launcher_log(f"pycolmap import failed, skip patch: {e}", verbose)
        return

    # Guard against double patching.
    if getattr(sm.SceneManager, "_fastavatar_patched", False):
        launcher_log("pycolmap already patched", verbose)
        return

    def _load_cameras_bin_fixed(self, input_file):
        """
        Fixed camera loader:
        - Uses explicit little-endian fixed-width formats.
        - COLMAP binary header count is 8-byte unsigned integer (<Q).
        """
        self.cameras = OrderedDict()
        with open(input_file, "rb") as f:
            num_cameras = struct.unpack("<Q", f.read(8))[0]
            for _ in range(num_cameras):
                camera_id, camera_type, w, h = struct.unpack("<IiQQ", f.read(24))
                num_params = sm.Camera.GetNumParams(camera_type)
                params = struct.unpack("d" * num_params, f.read(8 * num_params))
                self.cameras[camera_id] = sm.Camera(camera_type, w, h, params)
                self.last_camera_id = max(self.last_camera_id, camera_id)

    def _load_images_bin_fixed(self, input_file):
        """
        Fixed image loader:
        - Uses <Q for image count and point2D count.
        - Keeps original data layout and conversion behavior.
        """
        self.images = OrderedDict()

        with open(input_file, "rb") as f:
            num_images = struct.unpack("<Q", f.read(8))[0]
            image_struct = struct.Struct("<I 4d 3d I")

            for _ in range(num_images):
                data = image_struct.unpack(f.read(image_struct.size))
                image_id = data[0]
                q = sm.Quaternion(np.array(data[1:5]))
                t = np.array(data[5:8])
                camera_id = data[8]
                name = b"".join(c for c in iter(lambda: f.read(1), b"\x00")).decode()

                image = sm.Image(name, camera_id, q, t)
                num_points2D = struct.unpack("<Q", f.read(8))[0]

                # Read all 2D points in one pass for speed.
                points_array = array.array("d")
                points_array.fromfile(f, 3 * num_points2D)
                points_elements = np.array(points_array).reshape((num_points2D, 3))
                image.points2D = points_elements[:, :2]

                # Third column is point3D ids serialized as doubles in this implementation.
                ids_array = array.array("Q")
                ids_array.frombytes(points_elements[:, 2].tobytes())
                image.point3D_ids = np.array(ids_array, dtype=np.uint64).reshape((num_points2D,))

                self.images[image_id] = image
                self.name_to_image_id[image.name] = image_id
                self.last_image_id = max(self.last_image_id, image_id)

    def _load_points3D_bin_fixed(self, input_file):
        """
        Fixed points3D loader:
        - Uses <Q for point count and preserves original point/track parsing logic.
        """
        with open(input_file, "rb") as f:
            num_points3D = struct.unpack("<Q", f.read(8))[0]

            self.points3D = np.empty((num_points3D, 3))
            self.point3D_ids = np.empty(num_points3D, dtype=np.uint64)
            self.point3D_colors = np.empty((num_points3D, 3), dtype=np.uint8)
            self.point3D_id_to_point3D_idx = {}
            self.point3D_id_to_images = {}
            self.point3D_errors = np.empty(num_points3D)

            data_struct = struct.Struct("<Q 3d 3B d Q")

            for i in range(num_points3D):
                data = data_struct.unpack(f.read(data_struct.size))
                self.point3D_ids[i] = data[0]
                self.points3D[i] = data[1:4]
                self.point3D_colors[i] = data[4:7]
                self.point3D_errors[i] = data[7]
                track_len = data[8]

                self.point3D_id_to_point3D_idx[self.point3D_ids[i]] = i

                data = struct.unpack(f"{2 * track_len}I", f.read(2 * track_len * 4))
                self.point3D_id_to_images[self.point3D_ids[i]] = np.array(
                    data, dtype=np.uint32
                ).reshape(track_len, 2)

    # Monkey patch methods at runtime.
    sm.SceneManager._load_cameras_bin = _load_cameras_bin_fixed
    sm.SceneManager._load_images_bin = _load_images_bin_fixed
    sm.SceneManager._load_points3D_bin = _load_points3D_bin_fixed
    sm.SceneManager._fastavatar_patched = True

    launcher_log("pycolmap Windows patch applied", verbose)


def apply_gsplat_windows_patch(verbose=True):
    """
    Auto-fix gsplat Windows compile flags in site-packages when needed.

    Why:
    - Some gsplat versions pass GCC-style '-Wno-attributes' into MSVC,
      which causes: cl D8021 invalid numeric argument '/Wno-attributes'.

    Behavior:
    - Windows only.
    - If target line already patched, no-op.
    - If gsplat is not installed, safely skip.
    """
    if not should_apply_windows_patch("gsplat", verbose):
        return

    spec = importlib.util.find_spec("gsplat.cuda._backend")
    if spec is None or not spec.origin:
        launcher_log("gsplat not found, skip patch", verbose)
        return

    backend_file = Path(spec.origin)
    if not backend_file.exists():
        launcher_log(f"gsplat backend file not found: {backend_file}", verbose)
        return

    try:
        original = backend_file.read_text(encoding="utf-8")
    except Exception as e:
        launcher_log(f"failed to read gsplat backend, skip patch: {e}", verbose)
        return

    # Already patched by launcher or user.
    if "if os.name == \"nt\" else [opt_level, \"-Wno-attributes\"]" in original:
        launcher_log("gsplat already patched", verbose)
        return

    patched = re.sub(
        r"extra_cflags\s*=\s*\[\s*opt_level\s*,\s*\"-Wno-attributes\"\s*\]",
        'extra_cflags = [opt_level] if os.name == "nt" else [opt_level, "-Wno-attributes"]',
        original,
        count=1,
    )

    if patched == original:
        launcher_log("gsplat patch pattern not found, skip patch", verbose)
        return

    try:
        backend_file.write_text(patched, encoding="utf-8")
    except Exception as e:
        launcher_log(f"failed to write gsplat backend patch: {e}", verbose)
        return

    launcher_log(f"gsplat Windows patch applied: {backend_file}", verbose)


def build_parser():
    """
    Build launcher CLI parser.

    Usage convention:
    python launcher.py --mode <mode> -- <target_args>

    The standalone '--' is an argument separator:
    - before '--': parsed by launcher
    - after '--': forwarded to target script
    """
    parser = argparse.ArgumentParser(description="FastAvatar unified launcher")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["full_guidance", "no_guidance", "train_decoder", "train_encoder"],
        help="Pipeline to run",
    )
    parser.add_argument(
        "--no-patch",
        action="store_true",
        help="Disable pycolmap monkey patch (debug only)",
    )
    parser.add_argument(
        "rest",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to target script (put -- before them)",
    )
    return parser


def resolve_target(mode):
    """
    Map launcher mode to target module name under scripts.
    """
    mapping = {
        "full_guidance": "inference_feedforward_full_guidance",
        "no_guidance": "inference_feedforward_no_guidance",
        "train_decoder": "train_decoder",
        "train_encoder": "train_encoder",
    }
    return mapping[mode]


def mode_needs_patch(mode):
    """
    Decide whether a mode needs the pycolmap patch.

    Modes that parse COLMAP binaries through dataset Parser need patch.
    no_guidance generally uses single-image path without COLMAP binary parsing.
    """
    return mode in {"full_guidance", "train_decoder", "train_encoder"}


def apply_compatibility_patches(mode, no_patch=False, verbose=True):
    """
    Apply runtime compatibility patches in a stable order.

    Order:
    1) gsplat patch (site-packages text fix, needed before first JIT compile)
    2) pycolmap patch (runtime monkey patch, mode-dependent)
    """
    apply_gsplat_windows_patch(verbose=verbose)

    if mode_needs_patch(mode) and not no_patch:
        apply_pycolmap_windows_patch(verbose=verbose)
        return

    launcher_log("pycolmap patch skipped", verbose)


def main():
    """
    Launcher execution flow:
    1) Parse launcher args
    2) Add scripts directory to sys.path
    3) Apply compatibility patches
    4) Import target module by mode
    5) Rewrite sys.argv for target argparse
    6) Call target main()
    """
    parser = build_parser()
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    scripts_dir = repo_root / "scripts"

    # Ensure target modules under scripts can be imported.
    sys.path.insert(0, str(scripts_dir))

    # Keep all compatibility handling in one place.
    apply_compatibility_patches(args.mode, no_patch=args.no_patch, verbose=True)

    module_name = resolve_target(args.mode)
    module = importlib.import_module(module_name)

    # Forward remaining args to target script.
    passthrough = args.rest
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    sys.argv = [module_name + ".py"] + passthrough
    launcher_log(f"dispatch -> {module_name} {' '.join(passthrough)}", verbose=True)

    if not hasattr(module, "main"):
        raise RuntimeError(f"{module_name}.main not found")
    module.main()


if __name__ == "__main__":
    main()