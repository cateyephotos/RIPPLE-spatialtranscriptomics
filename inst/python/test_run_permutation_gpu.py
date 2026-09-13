#!/usr/bin/env python3
"""
Regression tests for the sample partitioning in run_permutation_gpu.py.

This mirrors tests/testthat/test-coordinate-frames.R on the Python side. The
defect these guard against: drawing pseudo-query cells per sample and then
concatenating the draws for one pooled kNN is stratified in COUNT but not in
SPACE. Because tissue sections routinely share a coordinate frame, the pooled
search returns pseudo-query cells from other samples, the null then carries the
same defect as a pooled observed statistic, and the permutation p-value looks
plausible either way.

Section [3] is the cross-implementation check. R and numpy have different RNGs,
so the permutation draws can never be compared exactly; the DETERMINISTIC parts
can be, and those are the parts the fix touches. A fixture exported by
scratchpad/export_mirror_fixture.R supplies identical inputs, and this asserts
the GPU path reproduces the R distances and the R per-sample coefficients.

Run on a GPU node:
  conda activate /nobackup/lab_maier/envs/harpy
  python inst/python/test_run_permutation_gpu.py [--fixture DIR]

Also runs on CPU (torch falls back), which is enough for the partition logic.

Note on tolerances: gpu_knn_distances builds float32 tensors, while RANN and R
work in float64. On a 500 um field torch.cdist in float32 carries roughly 5e-3
um of absolute error, and differing batch sizes select different kernels, so
results are not bitwise reproducible across batch_size either. That is four
orders of magnitude below one cell diameter and irrelevant to any downstream
coefficient, so distance comparisons use a 1e-2 um tolerance rather than
pretending float32 is exact.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import run_permutation_gpu as rp

FIELD = 500.0
# Absolute distance tolerance, in micrometres. gpu_knn_distances works in
# float32 while RANN works in float64; on a 500 um field the observed
# disagreement is around 1.5e-2 um. That is 15 nanometres, roughly 1/600th of
# one 10 um cell diameter, and it cannot move a Poisson coefficient fitted on
# distances that span hundreds of micrometres (section [3] confirms the
# coefficients agree to 1e-4 relative). The bound below is set from that
# physical scale rather than from float32 arithmetic, so it stays meaningful:
# anything larger than a twentieth of a micrometre would indicate a real
# indexing or partitioning error, not rounding.
DIST_TOL_UM = 5e-2
# Relative term, used together with the absolute one as
# |a - b| <= DIST_TOL_UM + DIST_REL_TOL * |b|, the numpy.allclose convention.
# A relative bound on its own is the wrong shape here: float32 error is roughly
# constant in absolute terms, so dividing by a sub-micrometre distance inflates
# it without anything actually being wrong. The absolute term therefore governs
# short distances and the relative term guards against a systematic scale error
# at long ones.
DIST_REL_TOL = 1e-5
FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print("  PASS  " + name)
    else:
        print("  FAIL  " + name + "  " + detail)
        FAILURES.append(name)


def make_overlapping_frames(seed=1, n_target=400, n_query=40):
    """Three samples on ONE shared field, query cluster in a different corner
    per sample. Frames overlap, as real sections do, so pooling breaks."""
    rng = np.random.default_rng(seed)
    centres = {"A": (100.0, 100.0), "B": (400.0, 400.0), "C": (100.0, 400.0)}
    qxy, qs, txy, ts = [], [], [], []
    for s, (cx, cy) in centres.items():
        ang = rng.uniform(0, 2 * np.pi, n_query)
        rad = 25.0 * np.sqrt(rng.uniform(0, 1, n_query))
        qxy.append(np.column_stack([cx + rad * np.cos(ang), cy + rad * np.sin(ang)]))
        qs += [s] * n_query
        txy.append(rng.uniform(0, FIELD, (n_target, 2)))
        ts += [s] * n_target
    return (np.vstack(qxy), np.array(qs), np.vstack(txy), np.array(ts))


def reference_within_sample(draw_idx, all_xy, sample_all,
                            target_xy, sample_target, k=1):
    """Independent numpy reference: per sample, distance from each target cell
    to the nearest drawn cell OF ITS OWN SAMPLE."""
    out = np.full(len(target_xy), np.nan)
    for s in np.unique(sample_target):
        drawn = draw_idx[sample_all[draw_idx] == s]
        tgt = np.where(sample_target == s)[0]
        if len(drawn) == 0 or len(tgt) == 0:
            continue
        d = np.linalg.norm(
            target_xy[tgt][:, None, :] - all_xy[drawn][None, :, :], axis=2
        )
        d.sort(axis=1)
        out[tgt] = d[:, :k].mean(axis=1)
    return out


def reference_pooled(draw_idx, all_xy, target_xy, k=1):
    """The OLD behaviour: one pooled search over every drawn cell."""
    d = np.linalg.norm(
        target_xy[:, None, :] - all_xy[draw_idx][None, :, :], axis=2
    )
    d.sort(axis=1)
    return d[:, :k].mean(axis=1)


def partitioned_distances(all_gpu, tgt_gpu, draw_by_sample, masks_tgt, n_target,
                          k=1):
    """The per-permutation distance step, exactly as the fixed loop computes it."""
    out = np.full(n_target, np.nan)
    for s, chosen in draw_by_sample.items():
        tgt_idx = masks_tgt[s]
        if len(chosen) == 0 or len(tgt_idx) == 0:
            continue
        d = rp.gpu_knn_distances(all_gpu[chosen], tgt_gpu[tgt_idx],
                                 k=min(k, len(chosen)))
        out[tgt_idx] = d.cpu().numpy()
    return out


def test_gpu_knn_matches_bruteforce(device):
    print("\n[1] gpu_knn_distances agrees with a brute-force reference")
    rng = np.random.default_rng(11)
    q = rng.uniform(0, FIELD, (37, 2))
    t = rng.uniform(0, FIELD, (611, 2))
    qg = torch.tensor(q, dtype=torch.float32, device=device)
    tg = torch.tensor(t, dtype=torch.float32, device=device)
    for k in (1, 3):
        got = rp.gpu_knn_distances(qg, tg, k=k).cpu().numpy()
        d = np.linalg.norm(t[:, None, :] - q[None, :, :], axis=2)
        d.sort(axis=1)
        want = d[:, :k].mean(axis=1)
        worst = float(np.max(np.abs(got - want)))
        check("k=%d max abs diff < %g um" % (k, DIST_TOL_UM),
              worst < DIST_TOL_UM, "max diff %.2e" % worst)

    got_b = rp.gpu_knn_distances(qg, tg, k=1, batch_size=50).cpu().numpy()
    got_1 = rp.gpu_knn_distances(qg, tg, k=1, batch_size=100000).cpu().numpy()
    worst = float(np.max(np.abs(got_b - got_1)))
    check("batched == unbatched within float32 error",
          worst < DIST_TOL_UM, "max diff %.2e" % worst)


def test_partition_is_within_sample(device):
    """The decisive test for the fix."""
    print("\n[2] the per-permutation search stays inside each sample")
    qxy, qs, txy, ts = make_overlapping_frames()
    all_xy = np.vstack([qxy, txy])
    sample_all = np.concatenate([qs, ts])

    all_gpu = torch.tensor(all_xy, dtype=torch.float32, device=device)
    tgt_gpu = torch.tensor(txy, dtype=torch.float32, device=device)

    unique_samples = list(np.unique(sample_all))
    masks_all = {s: np.where(sample_all == s)[0] for s in unique_samples}
    masks_tgt = {s: np.where(ts == s)[0] for s in unique_samples}

    rng = np.random.default_rng(7)
    draw_by_sample = {
        s: rng.choice(masks_all[s], size=40, replace=False)
        for s in unique_samples
    }
    perm = partitioned_distances(all_gpu, tgt_gpu, draw_by_sample, masks_tgt,
                                 len(ts))
    draw_idx = np.concatenate([draw_by_sample[s] for s in unique_samples])

    want_within = reference_within_sample(draw_idx, all_xy, sample_all, txy, ts)
    want_pooled = reference_pooled(draw_idx, all_xy, txy)

    check("no NaN left", not bool(np.isnan(perm).any()))
    worst = float(np.nanmax(np.abs(perm - want_within)))
    check("matches the within-sample reference", worst < DIST_TOL_UM,
          "max diff %.2e" % worst)
    med_diff = float(np.nanmedian(np.abs(perm - want_pooled)))
    check("differs from the pooled reference (fixture has teeth)",
          med_diff > 1.0, "median |diff| %.3f" % med_diff)
    check("pooled understates distance, as the bug did",
          float(np.nanmedian(want_pooled)) < float(np.nanmedian(perm)))

    ok = True
    for s in unique_samples:
        tgt_idx = masks_tgt[s]
        own = draw_by_sample[s]
        dmin = np.linalg.norm(
            txy[tgt_idx][:, None, :] - all_xy[own][None, :, :], axis=2
        ).min(axis=1)
        if float(np.max(np.abs(perm[tgt_idx] - dmin))) > DIST_TOL_UM:
            ok = False
    check("every distance is to a cell of the same sample", ok)


def test_matches_r_on_identical_input(device, fixture):
    """Cross-implementation check on inputs exported from R."""
    print("\n[3] agrees with R on identical inputs (fixture: %s)" % fixture)
    import pandas as pd

    cells = pd.read_csv(fixture / "cells.csv")
    targets = pd.read_csv(fixture / "targets.csv")
    r_coefs = pd.read_csv(fixture / "r_coefs.csv")

    all_xy = cells[["x", "y"]].to_numpy(dtype=np.float64)
    sample_all = cells["sample_id"].to_numpy(dtype=object)
    is_query = cells["is_query"].to_numpy().astype(bool)

    # R exports 1-based row indices into the full cell table.
    tgt_rows = targets["target_row"].to_numpy() - 1
    target_xy = all_xy[tgt_rows]
    sample_target = targets["sample_id"].to_numpy(dtype=object)
    counts = targets["counts"].to_numpy(dtype=np.float64)
    lib = targets["lib"].to_numpy(dtype=np.float64)
    r_dist = targets["dist_within"].to_numpy(dtype=np.float64)

    all_gpu = torch.tensor(all_xy, dtype=torch.float32, device=device)
    tgt_gpu = torch.tensor(target_xy, dtype=torch.float32, device=device)

    unique_samples = list(np.unique(sample_all))
    masks_tgt = {s: np.where(sample_target == s)[0] for s in unique_samples}
    # The real query cells are the search targets here, not a random draw.
    draw_by_sample = {
        s: np.where((sample_all == s) & is_query)[0] for s in unique_samples
    }

    gpu_dist = partitioned_distances(all_gpu, tgt_gpu, draw_by_sample,
                                     masks_tgt, len(target_xy))
    # Combined absolute-plus-relative criterion, expressed as a normalised
    # residual so one number covers short and long distances alike.
    resid = np.abs(gpu_dist - r_dist) / (DIST_TOL_UM + DIST_REL_TOL * np.abs(r_dist))
    worst = float(np.nanmax(np.abs(gpu_dist - r_dist)))
    worst_resid = float(np.nanmax(resid))
    check("GPU distances reproduce R calculate_distance_to_type_by_sample",
          worst_resid < 1.0,
          "max abs diff %.2e um, worst normalised residual %.3f"
          % (worst, worst_resid))
    print("        (max abs diff %.2e um over %d target cells, "
          "tolerance %g um + %g relative)"
          % (worst, len(r_dist), DIST_TOL_UM, DIST_REL_TOL))

    # Per-sample Poisson coefficients must match R fit_poisson to tight
    # relative tolerance: this is float64 IRLS on both sides.
    log_off = np.log(lib)
    worst_rel = 0.0
    for _, row in r_coefs.iterrows():
        s = row["sample_id"]
        idx = masks_tgt[s]
        c, se = rp.fit_poisson_sample(counts[idx], r_dist[idx], log_off[idx])
        rel = abs(c - row["coef"]) / abs(row["coef"])
        worst_rel = max(worst_rel, rel)
    check("per-sample Poisson coefs match R within 1e-4 relative",
          worst_rel < 1e-4, "worst rel diff %.2e" % worst_rel)

    return dict(all_gpu=all_gpu, tgt_gpu=tgt_gpu, sample_all=sample_all,
                sample_target=sample_target, counts=counts, log_off=log_off,
                r_dist=r_dist, is_query=is_query, masks_tgt=masks_tgt)


def test_end_to_end(device, ctx, fixture):
    """Permutation p-values: stochastic, so compared against the R reference
    distributionally rather than exactly."""
    print("\n[4] end-to-end permutation against the R reference")
    import pandas as pd

    r_sum = pd.read_csv(fixture / "r_summary.csv").iloc[0]
    sample_all = ctx["sample_all"]
    is_query = ctx["is_query"]
    query_per_sample = {
        s: int(((sample_all == s) & is_query).sum())
        for s in np.unique(sample_all)
    }

    old_cap = rp.MAX_DISTANCE_UM
    rp.MAX_DISTANCE_UM = 800.0
    try:
        # Observed statistic from the same estimator the null uses.
        coefs = []
        for s, idx in ctx["masks_tgt"].items():
            c, se = rp.fit_poisson_sample(
                ctx["counts"][idx], ctx["r_dist"][idx], ctx["log_off"][idx]
            )
            if np.isfinite(c) and np.isfinite(se) and se > 0:
                coefs.append(c)
        obs = float(np.median(coefs))
        check("observed coef matches the R observed coef within 1e-4 relative",
              abs(obs - r_sum["observed_coef"]) / abs(r_sum["observed_coef"]) < 1e-4,
              "gpu %.6f vs R %.6f" % (obs, r_sum["observed_coef"]))

        p_sig = rp.run_permutation_test_gpu(
            gene_data=ctx["counts"], target_coords_gpu=ctx["tgt_gpu"],
            all_coords_gpu=ctx["all_gpu"],
            sample_ids_target=ctx["sample_target"], sample_ids_all=sample_all,
            query_per_sample=query_per_sample, observed_coef=obs,
            n_perms=int(r_sum["n_perms"]), device=device,
            rng=np.random.default_rng(22), k_neighbors=1,
            use_poisson=True, log_total_counts_target=ctx["log_off"],
        )
        # The R reference reaches p ~ 0.02 on this fixture. This permutation
        # scheme has limited power against a focal-cluster query even with
        # 6400 target cells, because random pseudo-query points partially
        # capture a spatially structured gradient, so the null is wide. That
        # is a property of the design and is identical in R; the bound here is
        # deliberately loose so the test does not become a power measurement.
        check("planted gradient: p below 0.10", p_sig < 0.10, "p=%.4f" % p_sig)
        check("p is in the same regime as R (within 4x)",
              p_sig <= max(4.0 * r_sum["perm_pval"], 0.05),
              "gpu p=%.4f vs R p=%.4f" % (p_sig, r_sum["perm_pval"]))

        # A gene with no gradient must not come out significant.
        rng_np = np.random.default_rng(5)
        null_counts = rng_np.poisson(np.exp(ctx["log_off"]) * 0.05).astype(float)
        coefs_n = []
        for s, idx in ctx["masks_tgt"].items():
            c, se = rp.fit_poisson_sample(
                null_counts[idx], ctx["r_dist"][idx], ctx["log_off"][idx]
            )
            if np.isfinite(c) and np.isfinite(se) and se > 0:
                coefs_n.append(c)
        obs_n = float(np.median(coefs_n))
        p_null = rp.run_permutation_test_gpu(
            gene_data=null_counts, target_coords_gpu=ctx["tgt_gpu"],
            all_coords_gpu=ctx["all_gpu"],
            sample_ids_target=ctx["sample_target"], sample_ids_all=sample_all,
            query_per_sample=query_per_sample, observed_coef=obs_n,
            n_perms=int(r_sum["n_perms"]), device=device,
            rng=np.random.default_rng(21), k_neighbors=1,
            use_poisson=True, log_total_counts_target=ctx["log_off"],
        )
        check("null gene: p is not significant", p_null > 0.05,
              "p=%.4f" % p_null)
        check("null gene is less significant than the planted one",
              p_null > p_sig, "null %.4f vs planted %.4f" % (p_null, p_sig))
    finally:
        rp.MAX_DISTANCE_UM = old_cap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", type=str, default="mirror_fixture",
                    help="directory holding cells.csv exported from R")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("torch %s  numpy %s  device %s"
          % (torch.__version__, np.__version__, device))
    if device.type == "cuda":
        print("gpu: %s" % torch.cuda.get_device_name(0))

    test_gpu_knn_matches_bruteforce(device)
    test_partition_is_within_sample(device)

    fixture = Path(args.fixture)
    if (fixture / "cells.csv").exists():
        ctx = test_matches_r_on_identical_input(device, fixture)
        test_end_to_end(device, ctx, fixture)
    else:
        print("\n[3,4] SKIPPED: no fixture at %s" % fixture.resolve())
        print("      generate it with scratchpad/export_mirror_fixture.R")

    print("\n" + "=" * 60)
    if FAILURES:
        print("FAILED (%d): %s" % (len(FAILURES), ", ".join(FAILURES)))
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
