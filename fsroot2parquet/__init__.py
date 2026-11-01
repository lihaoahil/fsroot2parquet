"""fsroot2parquet: AmpTools-ready FSRoot trees -> partitioned Parquet for JAX likelihoods.

The angles reproduce ``KStarHyperonExtended::calcUserVars`` (halld_sim, twops_hao), so a
likelihood built on these columns sees exactly the events and frames the AmpTools fit saw.
"""
__version__ = "0.1.0"

from .kinematics import CONVENTION_VERSION, kstar_hyperon_angles
from .polarization import PolarizationTable
from .load import load_bin, accmc_by_orientation, pad_bins

__all__ = [
    "CONVENTION_VERSION", "kstar_hyperon_angles", "PolarizationTable",
    "load_bin", "accmc_by_orientation", "pad_bins",
]
