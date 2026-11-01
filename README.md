# fsroot2parquet

Converts AmpTools-ready FSRoot skims (signal and sideband data per beam orientation,
accepted and thrown phase-space MC) into partitioned Parquet that a JAX likelihood reads
with `pyarrow` → NumPy → `jnp.asarray`. Built for unbinned SDME and amplitude fits of
γp → K\*⁺Λ with K\*⁺ → K⁺π⁰ and Λ → pπ⁻.

The decay angles reproduce `KStarHyperonExtended::calcUserVars` (halld_sim
`AMPTOOLS_AMPS`, branch `twops_hao`). That way a likelihood built on these columns sees
the same events, frames and P_γ as the AmpTools fit.

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest -q
cp config/example.yaml my.yaml                         # set paths, histogram names, polarization angles
.venv/bin/fsroot2parquet --config my.yaml --out OUT [--mass 'Mass_*'] [--tbin 't_0.10*'] [--samples data,accmc]
sbatch -A ACCOUNT -p PARTITION --array=0-5 scripts/slurm_array.sh my.yaml OUT   # one task per (mass, t) bin; PY=... picks python
```

## Input layout

```
<source_root>/Mass_<lo>_<hi>/t_<lo>_<hi>/
    <prefix>_<ORIENTATION>_SIGNAL_SKIM.root             data, ORIENTATION in PARA_0 PERP_45 PERP_90 PARA_135 AMO
    <prefix>_<ORIENTATION>_SIDEBANDS_SKIM.root(.weight) background and its weight friend tree
    <prefix>_PSMC_SIGNAL_SKIM.root                      accepted phase-space MC
    tree_thrown_PSMCGEN_SKIM.root                       thrown phase-space MC
```

The file patterns, tree name, FSRoot particle labels and polarization inputs are all set
in the config ([`config/example.yaml`](config/example.yaml)).

## Output layout

```
OUT/polarization.parquet                                 orientation, pol_angle_deg, E_lo, E_hi, P_gamma
OUT/manifests/<mass>__<tbin>.json                        source files (size, mtime), TTree cycles, rows, weight sums, bin checks
OUT/sample=data/mass=<lo_hi>/tbin=<lo_hi>/part-0.parquet signal + sideband rows of every orientation
OUT/sample=accmc/mass=<lo_hi>/tbin=<lo_hi>/part-0.parquet accepted MC, once (no orientation)
OUT/sample=genmc/mass=<lo_hi>/tbin=<lo_hi>/part-0.parquet thrown MC, derived float32 columns only
```

The files are partitioned by (mass, t) bin, which is the unit a fit loads. Orientation
and role are columns, not directories, because a bin's likelihood needs all of them
together.

## Columns (`sample=data`)

| column | meaning |
|---|---|
| `role` | `data` (signal skim) or `bkgnd` (sideband skim) |
| `orientation` | `PARA_0`, `PERP_45`, `PERP_90`, `PARA_135`, `AMO` |
| `run`, `event`, `combo` | `combo` counts rows sharing (run, event); rows are combos, not events |
| `weight` | file weight: 1 for signal, the `.weight` friend for sidebands |
| `lik_weight` | +weight for signal, −weight for sidebands (AmpTools: ln L = Σ_data w ln I − Σ_bkgnd w ln I) |
| `E_beam`, `t`, `t_hyperon`, `M_Kpi`, `M_ppi` | `t = (beam − K*)²`, `t_hyperon = (target − Λ)²` |
| `theta_V`, `phi_V` | K⁺ in the K\* helicity frame |
| `n_x`, `n_y`, `n_z`, `theta_Y`, `phi_Y` | proton direction in the hyperon frame (AmpTools stores the cosines n) |
| `Phi_lab` | AmpTools `userVars[kBigPhi]` |
| `pol_angle_deg`, `Phi`, `P_gamma` | `Phi = Phi_lab + polAngle`; `P_gamma` from the TH1 at `E_beam` (under/overflow → 0); NaN / Phi_lab / 0 for AMO |
| `{beam,kp,pi0,proton,pim}_{E,px,py,pz}` | kinematic-fit lab four-vectors |

`accmc` has the same kinematic columns, plus `mc_*` angles from the thrown branches of
the accepted events. It has no `Phi` or `P_gamma`, because those depend on the
orientation: `load.accmc_by_orientation` stacks one copy per orientation for the
normalization integrals Z_o.

The frame definitions are listed in [`fsroot2parquet/kinematics.py`](fsroot2parquet/kinematics.py)
and tagged `convention_version` in every file's metadata.

## Validation

1. `pytest`: boosts, invariants, rotation about the beam, TH1 bin rules, and an end-to-end conversion of a synthetic skim directory.
2. Against the C++ (`validate/calc_user_vars_ref.C` runs the `calcUserVars` body verbatim on a tree):
   ```bash
   root -l -b -q 'validate/calc_user_vars_ref.C("TREE.root","ntFSGlueX_100000000_100001","","POL.root","hPol0",2000,"ref.csv")'
   python validate/compare_reference.py --ref ref.csv --root TREE.root --pol-file POL.root --pol-hist hPol0
   ```
3. Read `bin_containment_signal` in each manifest. It should be ≈ 1 for `M_Kpi` and for whichever −t definition the skims were cut on.

## Notes

- `polAngle` is taken from the config, not from the tree's `PolarizationAngle` branch, so the conversion uses the same angles as the fit.
- When a file holds several cycles of the tree, the highest cycle is read, and every cycle's entry count goes into the manifest.
- Reading the thrown MC dominates the run time. Use `--samples data,accmc` when efficiencies aren't needed.
