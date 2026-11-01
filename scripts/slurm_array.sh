#!/bin/bash
# One slurm array task per t-bin of every Mass_* directory under the config's source_root.
#     sbatch -A ACCOUNT -p PARTITION --array=0-$((NBINS-1)) scripts/slurm_array.sh CONFIG OUTDIR
# Submit from the repository root.  PY selects the interpreter (default: python3 on PATH).
#SBATCH --job-name=fsroot2parquet
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%A_%a.out
CFG=${1:?usage: sbatch scripts/slurm_array.sh CONFIG OUTDIR}
OUT=${2:?usage: sbatch scripts/slurm_array.sh CONFIG OUTDIR}
PY=${PY:-python3}
set -e
mkdir -p logs
export PYTHONPATH="$PWD"
SRC=$("$PY" -s -c "import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))['source_root'])" "$CFG")
BINS=($(cd "$SRC" && ls -d Mass_*/t_* | sort))
B=${BINS[$SLURM_ARRAY_TASK_ID]:?no bin for task $SLURM_ARRAY_TASK_ID (${#BINS[@]} bins)}
"$PY" -s -m fsroot2parquet.convert --config "$CFG" --out "$OUT" --mass "${B%%/*}" --tbin "${B##*/}"
