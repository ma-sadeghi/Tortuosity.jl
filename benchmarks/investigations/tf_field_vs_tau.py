"""Measure whether taufactor's tortuosity converges before its concentration field does.

For each image: run taufactor's own SOR to a near-exact fixed point (float64,
residual-monitored) to get a reference field and reference tau, then re-run and
record, at every convergence check, the flux-spread criterion taufactor tests, the
tau it would report, and the error in the field it holds at that moment.
"""

import contextlib
import io
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from taufactor.taufactor import SORSolver, Solver

BENCH = Path(r"C:\Users\sadegmo\.julia\dev\Tortuosity\benchmarks")
OUT = Path(__file__).parent


class F64Solver(Solver):
    """taufactor's binary through-transport solver, forced to float64.

    Solver does not pass ``precision`` through to SORSolver, so the reference would
    otherwise be limited by float32 rounding at roughly the size of the errors under
    test. Nothing else about the scheme changes.
    """

    def __init__(self, img, omega=None, D_0=1, device="cuda", precision=torch.float64):
        self._check_binary_labels(img)
        self.conductive_labels = [1]
        self.top_bc, self.bot_bc = (0.0, 1.0)
        SORSolver.__init__(self, img, omega=omega, precision=precision, device=device)
        for cb in self.cb:
            cb[0, :, :] = 0
            cb[-1, :, :] = 0
        self.D_0 = D_0
        self.D_mean = np.mean(self.vol_x, axis=1)


def load(case):
    with h5py.File(BENCH / "data" / "images" / f"{case}.h5", "r") as f:
        raw = np.array(f["image"], dtype=np.uint8)
    return np.ascontiguousarray(raw.T.astype(np.int32))


def run(solver, upto):
    """Advance the real solve loop to iteration ``upto`` (no early exit)."""
    with contextlib.redirect_stdout(io.StringIO()):
        solver.solve(iter_limit=upto, verbose=False, conv_crit=0.0)


def interior(solver):
    return solver.field[:, 1:-1, 1:-1, 1:-1]


def residual_inf(solver, pore):
    """max |A c - b| over interior pore voxels, in the units of the field.

    The SOR fixed point satisfies c_i = (sum of conductive neighbours)/N_i, so the
    increment before relaxation is exactly the residual of the discrete system.
    """
    r = solver.sum_weighted_neighbours() / solver.factor - interior(solver)
    return float(torch.max(torch.abs(r[pore])).item())


def measure(case, device="cuda", ladder_max=40000, ref_max=400000, ref_tol=1e-13):
    img = load(case)
    nx = img.shape[0]
    pore_np = img.astype(bool)
    pore = torch.tensor(pore_np, device=device)[None, ...]
    # Boundary slices are pinned, so they carry no error and no residual.
    pore_int = pore.clone()
    pore_int[:, 0, :, :] = False
    pore_int[:, -1, :, :] = False

    # --- reference: taufactor's own fixed point ------------------------------
    ref = F64Solver(img, device=device)
    t0 = time.time()
    it = 0
    while it < ref_max:
        it += 2000
        run(ref, it)
        res = residual_inf(ref, pore_int)
        if res < ref_tol:
            break
    ref_res, ref_iter, ref_time = res, it, time.time() - t0
    c_ref = interior(ref).clone()
    tau_ref = float(ref.tau[0])
    _, rel_ref = ref.compute_metrics()
    norm_ref = float(torch.linalg.vector_norm(c_ref[pore]).item())
    del ref
    torch.cuda.empty_cache()

    # --- instrumented replay --------------------------------------------------
    s = F64Solver(img, device=device)
    rows = []
    for k in range(100, ladder_max + 1, 100):
        run(s, k)
        tau, rel = s.compute_metrics()
        d = interior(s) - c_ref
        rows.append(
            dict(
                iter=k,
                tau=float(tau[0]),
                flux_spread=float(rel[0]),
                tau_relerr=abs(float(tau[0]) - tau_ref) / tau_ref,
                field_l2=float(torch.linalg.vector_norm(d[pore]).item()) / norm_ref,
                field_linf=float(torch.max(torch.abs(d[pore])).item()),
                residual_inf=residual_inf(s, pore_int),
            )
        )
        if rows[-1]["field_l2"] < 1e-12:
            break
    del s
    torch.cuda.empty_cache()

    return dict(
        case=case,
        nx=nx,
        porosity=float(pore_np.mean()),
        tau_ref=tau_ref,
        ref_iter=ref_iter,
        ref_residual=ref_res,
        ref_flux_spread=float(rel_ref[0]),
        ref_time_s=ref_time,
        rows=rows,
    )


def declared(rows, conv_crit, tau_gate):
    """First check at which taufactor's two-part rule fires.

    Part one is the flux-spread test against ``conv_crit``; part two is the change
    in tau since the previous check, against ``tau_gate`` (hard-coded 2e-3 upstream,
    ``conv_crit`` on the fork). old_tau is only updated on a failed check, matching
    check_convergence.
    """
    old = 0.0
    for r in rows:
        if r["flux_spread"] < conv_crit and abs(r["tau"] - old) < tau_gate:
            return r
        old = r["tau"]
    return None


SETTINGS = [
    ("upstream@1e-2", 1e-2, 2e-3),
    ("fork@1e-2", 1e-2, 1e-2),
    ("fork@1e-3", 1e-3, 1e-3),
    ("fork@1e-4", 1e-4, 1e-4),
]

if __name__ == "__main__":
    cases = sys.argv[1:] or [
        f"n100_b{b}_p{p}"
        for b in ("050", "100", "200")
        for p in ("020", "040", "060", "080", "095")
    ]
    out = []
    lines = ["case,porosity,tau_ref,setting,iter,flux_spread,tau_relerr,field_l2,field_linf,residual_inf"]
    for c in cases:
        r = measure(c)
        print(
            f"{c}: por={r['porosity']:.3f} tau_ref={r['tau_ref']:.6f} "
            f"ref_iter={r['ref_iter']} res={r['ref_residual']:.2e} "
            f"({r['ref_time_s']:.1f}s) rows={len(r['rows'])}",
            flush=True,
        )
        for name, crit, gate in SETTINGS:
            d = declared(r["rows"], crit, gate)
            if d is None:
                print(f"    {name}: never fired within the ladder")
                continue
            print(
                f"    {name}: iter={d['iter']} spread={d['flux_spread']:.3e} "
                f"tau_relerr={d['tau_relerr']:.3e} field_l2={d['field_l2']:.3e} "
                f"field_linf={d['field_linf']:.3e} resid={d['residual_inf']:.3e} "
                f"ratio={d['field_l2'] / max(d['tau_relerr'], 1e-30):.1f}"
            )
            lines.append(
                f"{c},{r['porosity']:.4f},{r['tau_ref']:.6f},{name},{d['iter']},"
                f"{d['flux_spread']:.4e},{d['tau_relerr']:.4e},{d['field_l2']:.4e},"
                f"{d['field_linf']:.4e},{d['residual_inf']:.4e}"
            )
        out.append(r)
    (OUT / "tf_field_vs_tau.json").write_text(json.dumps(out, indent=1))
    (OUT / "tf_field_vs_tau.csv").write_text("\n".join(lines) + "\n")
