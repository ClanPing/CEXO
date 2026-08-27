"""Shared command-line helpers for CEXO runner scripts."""

from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .config import FACILITY_SPECS, generate_facility_mix


def parse_facility_mix(mix_spec: str) -> List[str]:
    """Parse a CLI facility mix such as core=2,crane=1,storage=2."""
    if not mix_spec or not mix_spec.strip():
        raise ValueError("Facility mix cannot be empty.")

    facilities: List[str] = []
    valid_types = set(FACILITY_SPECS)
    parts = [part.strip() for part in mix_spec.replace(";", ",").split(",") if part.strip()]

    for part in parts:
        if "=" in part:
            name, count_text = part.split("=", 1)
        elif ":" in part:
            name, count_text = part.split(":", 1)
        else:
            name, count_text = part, "1"

        facility_type = name.strip()
        if facility_type not in valid_types:
            valid = ", ".join(sorted(valid_types))
            raise ValueError(f"Unknown facility type '{facility_type}'. Valid types: {valid}.")

        try:
            count = int(count_text.strip())
        except ValueError as exc:
            raise ValueError(f"Invalid count for '{facility_type}': {count_text!r}.") from exc

        if count < 0:
            raise ValueError(f"Facility count for '{facility_type}' must be zero or greater.")

        facilities.extend([facility_type] * count)

    if not facilities:
        raise ValueError("Facility mix must contain at least one facility.")

    return facilities


def resolve_facility_types(
    facility_count: int,
    seed: Optional[int] = None,
    facility_mix: Optional[str] = None,
) -> List[str]:
    """Resolve either an explicit facility mix or a seeded generated mix."""
    if facility_mix:
        return parse_facility_mix(facility_mix)
    return generate_facility_mix(facility_count, seed=seed)


def make_run_output_dir(base_dir: str, run_label: str) -> str:
    """Create a timestamped run directory under the selected base directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base_dir) / f"{run_label}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return str(output_dir)
