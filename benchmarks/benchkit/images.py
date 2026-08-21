"""Read side of the shared image store.

Images are written by ``generate_images.jl`` and only ever read here: one
generator means every tool solves the same geometry, which is the only thing
that makes a difference in the reported tortuosity attributable to the solver.
"""

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


@dataclass(frozen=True)
class ImageEntry:
    """One manifest row: what an image is, and what it should hash to."""

    case_id: str
    size: int
    blobiness: float
    porosity_target: float
    porosity: float
    nnodes: int
    sha256: str


def manifest_path(cfg):
    return cfg.imagedir / "manifest.csv"


def image_path(cfg, case):
    return cfg.imagedir / f"{case.id}.h5"


def read_manifest(cfg):
    """The image manifest indexed by case id, or empty when there is none."""
    path = manifest_path(cfg)
    if not path.is_file():
        return {}
    entries = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            entries[row["case_id"]] = ImageEntry(
                row["case_id"],
                int(row["size"]),
                float(row["blobiness"]),
                float(row["porosity_target"]),
                float(row["porosity"]),
                int(row["nnodes"]),
                row["sha256"],
            )
    return entries


def load_image(cfg, case, dtype=np.int32, verify=True):
    """Load one cached image, verifying it against the manifest.

    Returns a contiguous array indexed ``[x, y, z]``, matching how the image was
    generated. HDF5 hands back the reversed shape because Julia writes column
    major and h5py reads row major, so the transpose here restores the original
    axis order rather than changing it.

    Verification is cheap next to any solve and catches the failure that is
    otherwise invisible: a store regenerated under different package versions,
    or copied in part, yields images that look right and results that cannot be
    compared with anything measured before.
    """
    path = image_path(cfg, case)
    if not path.is_file():
        raise FileNotFoundError(f"no image for {case.id} at {path} — run generate_images.jl first")
    with h5py.File(path, "r") as f:
        raw = np.array(f["image"], dtype=np.uint8)

    if verify:
        entry = read_manifest(cfg).get(case.id)
        if entry is None:
            print(f"warning: {case.id} is not in the manifest — cannot verify integrity")
        else:
            # Hashed before the transpose: the bytes of the array as HDF5 stores
            # them are the same sequence the Julia writer hashed, so the two
            # digests are comparable without either side knowing the other's
            # memory layout.
            digest = hashlib.sha256(np.ascontiguousarray(raw).tobytes()).hexdigest()
            if digest != entry.sha256:
                raise ValueError(
                    f"{case.id} does not match the manifest (got {digest}, "
                    f"expected {entry.sha256}). The image store and the manifest describe "
                    "different geometry; regenerate it or restore the store."
                )

    return np.ascontiguousarray(raw.T.astype(dtype))
