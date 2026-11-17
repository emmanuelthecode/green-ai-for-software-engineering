"""Measuring energy and carbon for a block of code.

On Linux machines that expose Intel/AMD RAPL counters
(/sys/class/powercap/intel-rapl*), the tracker reads real package energy.
Everywhere else it falls back to an estimate:

    energy ~= CPU seconds used by this process x (CPU TDP / number of cores)

The estimate is rough. It ignores idle power, memory, turbo and so on, but it
scales with the work actually done, so it's good enough to *compare*
techniques against each other on the same machine, which is all this project
needs. Section 6.5 of the paper makes the same point: always report how
energy was measured.

Carbon = energy (kWh) x grid carbon intensity (gCO2e/kWh). Both the TDP and
the intensity are settings; the defaults are placeholders, not measurements.
"""

import glob
import os
import time
from dataclasses import dataclass, field

DEFAULT_TDP_WATTS = float(os.environ.get("GREENAI_TDP_WATTS", 65))
DEFAULT_INTENSITY = float(os.environ.get("GREENAI_GRID_INTENSITY", 200))   # gCO2e/kWh


def _rapl_files():
    files = []
    for path in glob.glob("/sys/class/powercap/intel-rapl:*/energy_uj"):
        # top-level packages only (intel-rapl:0, not intel-rapl:0:0)
        if path.count(":") == 1 and os.access(path, os.R_OK):
            files.append(path)
    return files


def _read_rapl(files):
    total = 0
    for f in files:
        with open(f) as fh:
            total += int(fh.read())
    return total / 1e6     # joules


@dataclass
class Measurement:
    label: str
    wall_s: float = 0.0
    cpu_s: float = 0.0
    energy_j: float = 0.0
    method: str = "estimate"
    extra: dict = field(default_factory=dict)

    @property
    def kwh(self):
        return self.energy_j / 3.6e6

    def co2_g(self, intensity=DEFAULT_INTENSITY):
        return self.kwh * intensity


class EnergyTracker:
    """Context manager:

        with EnergyTracker("baseline training") as m:
            train(...)
        print(m.energy_j, m.co2_g())
    """

    def __init__(self, label="", tdp_watts=DEFAULT_TDP_WATTS, cores=None):
        self.m = Measurement(label)
        self.tdp = tdp_watts
        self.cores = cores or os.cpu_count() or 1
        self.rapl = _rapl_files()

    def __enter__(self):
        self._wall = time.perf_counter()
        self._cpu = time.process_time()
        self._rapl0 = _read_rapl(self.rapl) if self.rapl else None
        return self.m

    def __exit__(self, *exc):
        self.m.wall_s = time.perf_counter() - self._wall
        self.m.cpu_s = time.process_time() - self._cpu
        if self._rapl0 is not None:
            self.m.energy_j = _read_rapl(self.rapl) - self._rapl0
            self.m.method = "rapl"
        else:
            self.m.energy_j = self.m.cpu_s * self.tdp / self.cores
            self.m.method = "estimate"
        return False
