"""
Unified launcher for FastAvatar pipelines.

Why this file exists:
1) Provide a single team entrypoint for running all pipelines.
2) Apply a Windows-only runtime monkey patch for pycolmap binary parsing.
3) Avoid editing site-packages and keep business scripts clean.
"""

import argparse
import importlib
import platform
import struct
import sys
from collections import OrderedDict
from pathlib import Path
import array

import numpy as np


def apply_pycolmap_windows_patch(verbose=True):
    """
    Apply a runtime patch to pycolmap SceneManager binary loaders on Windows.

    Scope and safety:
    - Only affects current Python process.
    - Does NOT modify any files in site-packages.
    - Patch is applied only once per process.
    """
    # [ATTENTION!]This compatibility issue is Windows-specific.
    if platform.system().lower() != "windows":
        if verbose:
            print("[launcher] non-windows platform, skip pycolmap patch")
        return

    try:
        import pycolmap.scene_manager as sm
    except Exception as e:
        if verbose:
            print(f"[launcher] pycolmap import failed, skip patch: {e}")
        return

    # Guard against double patching.
    if getattr(sm.SceneManager, "_fastavatar_patched", False):
        if verbose:
            print("[launcher] pycolmap already patched")
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

    if verbose:
        print("[launcher] pycolmap Windows patch applied")


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
    Decide whether a mode needs pycolmap patch.

    Modes that parse COLMAP binaries through dataset Parser need patch.
    no_guidance generally uses single-image path without COLMAP binary parsing.
    """
    return mode in {"full_guidance", "train_decoder", "train_encoder"}


def main():
    """
    Launcher execution flow:
    1) Parse launcher args
    2) Add scripts directory to sys.path
    3) Conditionally apply pycolmap patch
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

    # Apply compatibility patch only when needed.
    if mode_needs_patch(args.mode) and not args.no_patch:
        apply_pycolmap_windows_patch(verbose=True)
    else:
        print("[launcher] patch skipped")

    module_name = resolve_target(args.mode)
    module = importlib.import_module(module_name)

    # Forward remaining args to target script.
    passthrough = args.rest
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    sys.argv = [module_name + ".py"] + passthrough
    print(f"[launcher] dispatch -> {module_name} {' '.join(passthrough)}")

    if not hasattr(module, "main"):
        raise RuntimeError(f"{module_name}.main not found")
    module.main()


if __name__ == "__main__":
    main()