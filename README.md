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

## From unpartitioned trees: reduce once, select many times

The two commands below start from the full FSRoot trees, before any skim into bins, and need no ROOT
macro. They work for any reaction of the form γ + target → X + B with one or two particles at each vertex.

```bash
cp config/reduce_example.yaml reduce.yaml              # trees, particle labels, reaction, envelope
cp config/select_example.yaml select.yaml              # cuts, sidebands, bins, polarization
.venv/bin/fsroot2parquet-reduce --config reduce.yaml --out STORE
.venv/bin/fsroot2parquet-select --config select.yaml   # `store` and `out` are in the config
```

**Stage 1, `fsroot2parquet-reduce`,** reads each tree once and keeps the rows inside an *envelope*: the
loosest value of every cut that any later study will use. For those rows it writes the identifiers, the
cut variables, derived masses, the kinematic variables in the helicity and Gottfried–Jackson frames, and the
four-vectors. Nothing is cut at its analysis value, weighted or binned. Rows outside the envelope are
gone until this stage is rerun, so choose it generously.

**Stage 2, `fsroot2parquet-select`,** applies the analysis cuts, the signal and sideband regions with
their weights, the beam polarization and the bin edges, all from one config, and writes the layout
described under "Output layout" (plus the cut variables and a `period` column). A systematic
variation is another config on the same store.

- Cuts are expressions on column names: `(E_beam > 8.2) & (E_beam < 8.8)`, `abs(M_Lambda - 1.115) < 0.010`.
- Sideband weights follow FSRoot's `CUTSBWT`: `scale` for a row in one sideband, `-scale_1 scale_2` in two,
  and so on; such rows are `bkgnd` rows with `lik_weight = -weight`.
- Bins are `lo < x < hi`, as in an FSRoot skim.
- The kinematics are `kinematics.two_vertex_variables(beam, upper, lower)`; `names` in the config sets
  the column names (`mass: M_Kpi`, `theta: theta_V`, ...). `kstar_hyperon_angles` is this function under
  the K\*Λ names.

Timing for trees of typical size (4 CPU threads, network disk); each run writes its own to `reduce.json`
and `selection.json`:

| step | input | time |
|---|---|---|
| reduce, data | 26M entries, 10 GB | 34 s |
| reduce, accepted MC | 2M entries, 1 GB | 6 s |
| reduce, thrown MC | 4M entries, 1 GB | 11 s |
| select, six bins | a 1 GB store | 6–10 s |

The result was compared row by row with the Parquet converted from FSRoot skims of the same trees: the
same rows in every bin for data, accepted and thrown MC, with identical angles, orientations and
polarization. Weights of rows in the RF sideband differ by 3 × 10⁻⁹ because FSRoot rounds the scale 1/6.

## From FSRoot skims (one directory per bin)

The original converter, for trees already skimmed into mass and t bins:

```bash
.venv/bin/fsroot2parquet --config my.yaml --out OUT    # config/example.yaml
```

## Rebinning

Convert once with the t bins of each mass bin merged, then cut that into any binning inside
its range. No ROOT file is read again:

```bash
.venv/bin/fsroot2parquet --config my.yaml --out MERGED --merge-tbins            # tbin=<first lo>_<last hi>
.venv/bin/fsroot2parquet-rebin --src MERGED --out T12 --t-edges 0.1,0.2,0.3,...,2.5 [--mass-edges 0.84,0.89,0.94]
```

Rows go to `lo <= -t_hyperon < hi` and `lo <= M_Kpi < hi`. For data and accepted MC
`t = t_hyperon` (kinematic fit); thrown MC is binned on its own `t_hyperon`, which is the
variable thrown skims are cut on (its `t = (beam − K*)²` is not). The merge is exact only if
the skims' combo selection does not depend on the t bin, for example one combo per
(event, beam photon). Check that no (run, event, E_beam) appears in two t bins before you
rely on it. New edges must lie inside the source coverage, and `REBIN.json` records them.

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
OUT/sample=genmc/mass=<lo_hi>/tbin=<lo_hi>/part-0.parquet thrown MC, derived columns only (float32; t, t_hyperon, M_Kpi in float64)
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
| `*_GJ` (`theta_V`, `phi_V`, `theta_Y`, `phi_Y`, `n_x`, `n_y`, `n_z`) | the same in the Gottfried-Jackson frames: z = beam in the K\* rest frame, z = target proton in the hyperon rest frame; same y axes, so `n_y_GJ = n_y` and `Phi` is unchanged |
| `Phi_lab` | AmpTools `userVars[kBigPhi]` |
| `pol_angle_deg`, `Phi`, `P_gamma` | `Phi = Phi_lab + polAngle`; `P_gamma` from the TH1 at `E_beam` (under/overflow → 0); NaN / Phi_lab / 0 for AMO |
| `{beam,kp,pi0,proton,pim}_{E,px,py,pz}` | kinematic-fit lab four-vectors |

`accmc` has the same kinematic columns, plus `mc_*` angles (helicity and GJ) from the thrown branches of
the accepted events. It has no `Phi` or `P_gamma`, because those depend on the
orientation: `load.accmc_by_orientation` stacks one copy per orientation for the
normalization integrals Z_o.

The frame definitions are listed in [`fsroot2parquet/kinematics.py`](fsroot2parquet/kinematics.py)
and tagged `convention_version` in every file's metadata.

## Validation

1. `pytest`: boosts, invariants, rotation about the beam, GJ crossing angles against the invariant (s, t, u) formula, TH1 bin rules, an end-to-end conversion of a synthetic skim directory, and merge + rebin reproducing the per-bin conversion row for row.
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
