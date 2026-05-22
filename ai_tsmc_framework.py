"""
AI Gigawatts → TSMC Front-End Wafer Starts Framework
=====================================================
Maps AI datacenter power capacity additions to TSMC leading-edge wafer demand.

Chain:
  GW/year new AI capacity
    → racks/month  (at rack TDP)
    → units/month per component (by per-rack attach rate)
    → wafers/month (GDW × yield, negative-binomial model)
    → wspm by node

Confidence labels on each chip spec:
  [H] confirmed die shot or TSMC node announcement
  [M] analyst estimate / teardown inference
  [L] derived / speculative

Usage:
  python ai_tsmc_framework.py                       # all scenarios, NVL72 HBM3e
  python ai_tsmc_framework.py NVL72_HBM4            # same, but HBM4 base on TSMC N3
  python ai_tsmc_framework.py NVL72_HBM4 10.0       # single 10 GW/yr scenario
  python ai_tsmc_framework.py DGX_H100 5.0          # H100 reference platform
"""

import math
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# ── Yield model ────────────────────────────────────────────────────────────────
# Negative binomial: Y = (1 + D0·A / α)^(−α)
# D0 = defect density (defects/cm²), A = die area (cm²), α = clustering factor

def gross_dies_per_wafer(
    die_area_mm2: float,
    wafer_diam_mm: float = 300.0,
    edge_excl_mm: float = 3.0,
) -> float:
    r = wafer_diam_mm / 2 - edge_excl_mm
    A = die_area_mm2
    return (math.pi * r**2 / A) - (2 * math.pi * r / math.sqrt(A))


def die_yield(die_area_mm2: float, d0: float, alpha: float = 10.0) -> float:
    A_cm2 = die_area_mm2 / 100.0
    return (1.0 + d0 * A_cm2 / alpha) ** (-alpha)


def net_dies_per_wafer(die_area_mm2: float, d0: float, alpha: float = 10.0) -> float:
    return gross_dies_per_wafer(die_area_mm2) * die_yield(die_area_mm2, d0, alpha)


# ── Process node parameters ────────────────────────────────────────────────────
# (defect_density defects/cm², display label, estimated wspm capacity 2025)
NODE: Dict[str, Tuple[float, str, int]] = {
    "N3B": (0.10, "TSMC N3B (3nm class)",  90_000),
    "N3":  (0.09, "TSMC N3  (3nm)",        90_000),
    "N4":  (0.07, "TSMC N4  (4nm class)",  70_000),
    "N4P": (0.07, "TSMC N4P (4nm perf)",   60_000),
    "N5":  (0.06, "TSMC N5  (5nm)",       100_000),
    "N6":  (0.05, "TSMC N6  (6nm)",        80_000),
    "N7":  (0.045,"TSMC N7  (7nm)",        90_000),
}


# ── Chip spec ──────────────────────────────────────────────────────────────────

@dataclass
class Chip:
    name: str
    node: str           # key into NODE dict, or "non-TSMC"
    die_area_mm2: float
    conf: str           # H / M / L
    notes: str = ""

    @property
    def is_tsmc(self) -> bool:
        return self.node != "non-TSMC"

    @property
    def d0(self) -> float:
        return NODE[self.node][0] if self.is_tsmc else 0.0

    @property
    def gdw(self) -> float:
        return gross_dies_per_wafer(self.die_area_mm2) if self.is_tsmc else float("inf")

    @property
    def yld(self) -> float:
        return die_yield(self.die_area_mm2, self.d0) if self.is_tsmc else 1.0

    @property
    def ndw(self) -> float:
        return self.gdw * self.yld if self.is_tsmc else float("inf")


CHIPS: Dict[str, Chip] = {

    # ── Main XPU ──────────────────────────────────────────────────────────────
    "B200": Chip(
        "NVIDIA B200 (Blackwell GPU)", "N3B", 1000, "M",
        "TSMC N3B; 208B transistors; die area from analyst estimates",
    ),
    "H100": Chip(
        "NVIDIA H100 SXM (Hopper GPU)", "N4", 814, "H",
        "TSMC 4N variant; 80B transistors; confirmed via die shots",
    ),
    "MI300X_XCD": Chip(
        "AMD MI300X XCD (compute die, ×3 per pkg)", "N5", 208, "H",
        "3 per MI300X; Zen 4 / CDNA3 CUs; TSMC N5",
    ),
    "MI300X_IOD": Chip(
        "AMD MI300X IOD (I/O die, ×4 per pkg)", "N6", 122, "M",
        "4 per MI300X; PCIe / xGMI fabric; TSMC N6",
    ),

    # ── Host CPU ──────────────────────────────────────────────────────────────
    "Grace": Chip(
        "NVIDIA Grace CPU", "N4", 382, "M",
        "72 Neoverse V2 cores; TSMC N4; 0.5 per B200 in NVL72",
    ),
    "EPYC_CCD": Chip(
        "AMD EPYC Genoa CCD (×12 max per socket)", "N5", 68, "H",
        "Zen 4 CCD; TSMC N5; up to 96C per socket",
    ),
    "EPYC_IOD": Chip(
        "AMD EPYC Genoa IOD (×1 per socket)", "N6", 400, "M",
        "PCIe Gen5 / memory controllers; TSMC N6",
    ),

    # ── HBM base die ─────────────────────────────────────────────────────────
    # HBM3e: SK Hynix / Samsung use their own 4nm-class logic process — NOT TSMC.
    # HBM4:  SK Hynix + TSMC partnership; base die expected on TSMC N3 (announced 2024).
    "HBM3e_base": Chip(
        "HBM3e base die (SK Hynix / Samsung internal)", "non-TSMC", 78, "M",
        "Vendor-proprietary 4nm logic; NOT TSMC → zero TSMC wspm",
    ),
    "HBM4_base": Chip(
        "HBM4 base die (TSMC N3, SK Hynix collab)", "N3", 100, "L",
        "Expected TSMC N3; SK Hynix+TSMC partnership; 2025+ ramp; "
        "8 stacks per B200-class XPU = key new N3 demand driver",
    ),

    # ── NVLink switch ─────────────────────────────────────────────────────────
    "NVSwitch_BW": Chip(
        "NVIDIA NVSwitch (Blackwell gen)", "N4", 440, "L",
        "18 per NVL72; est. ~900 GB/s per port; die area estimated",
    ),
    "NVSwitch_H": Chip(
        "NVIDIA NVSwitch 3 (Hopper gen)", "N4", 295, "M",
        "2 per DGX H100; 64-port; TSMC 4N; area from analyst estimates",
    ),

    # ── NIC ───────────────────────────────────────────────────────────────────
    "CX8": Chip(
        "NVIDIA ConnectX-8 NIC (400G IB/Eth)", "N5", 200, "L",
        "1 per compute node; estimated N5 ~200mm²",
    ),

    # ── Transceiver DSP ───────────────────────────────────────────────────────
    # 800G linear-drive (LPO) or coherent DSP for rack-to-spine uplinks.
    # Vendors: Broadcom (Humboldt), Marvell (Alaska), Coherent.
    "DSP_800G": Chip(
        "800G transceiver DSP (spine uplinks)", "N5", 120, "L",
        "1 per 800G OSFP port; Broadcom / Marvell / Coherent; est. N5 ~120mm²",
    ),

    # ── Spine switch silicon ──────────────────────────────────────────────────
    # Tomahawk 5: Broadcom BCM78800, 51.2 Tbps, 36.86B transistors, TSMC N5.
    # Tomahawk 6 (next gen, ~102.4T): expected TSMC N3B.
    "TH5": Chip(
        "Broadcom Tomahawk 5 (51.2T spine switch)", "N5", 650, "M",
        "BCM78800; 36.86B transistors; TSMC N5; 128×400G or 512×100G ports",
    ),
    "TH6": Chip(
        "Broadcom Tomahawk 6 (102.4T, est.)", "N3B", 700, "L",
        "Next-gen spine; expected TSMC N3B; die area speculative",
    ),
    "Jericho3": Chip(
        "Broadcom Jericho3-AI (WAN/DCI routing)", "N5", 1000, "M",
        "BCM88820; 31.5B transistors; TSMC N5; edge / long-haul routing",
    ),
}


# ── Rack / platform config ─────────────────────────────────────────────────────

@dataclass
class Platform:
    name: str
    rack_tdp_kw: float
    chips_per_rack: Dict[str, float]   # chip_key → count per rack
    hbm_stacks_per_xpu: int
    hbm_die_key: str
    xpu_key: str
    spine_uplink_ports_per_rack: int   # 800G OSFP ports to spine per rack
    spine_switch_key: str = "TH5"
    spine_switch_ports: int = 128      # usable 400G-equiv ports per switch
    spine_oversubscription: float = 8.0


PLATFORMS: Dict[str, Platform] = {

    # NVIDIA NVL72 (72× B200 + 36× Grace + 18× NVSwitch per liquid-cooled rack)
    # rack TDP ≈ 120 kW (liquid cooled)
    "NVL72_HBM3e": Platform(
        name="NVIDIA NVL72 — Blackwell, HBM3e (non-TSMC base die)",
        rack_tdp_kw=120,
        chips_per_rack={"B200": 72, "Grace": 36, "NVSwitch_BW": 18, "CX8": 36},
        hbm_stacks_per_xpu=8,    # 192 GB = 8 × 24 GB HBM3e
        hbm_die_key="HBM3e_base",
        xpu_key="B200",
        spine_uplink_ports_per_rack=36,
    ),
    "NVL72_HBM4": Platform(
        name="NVIDIA NVL72 — Blackwell, HBM4 (TSMC N3 base die)",
        rack_tdp_kw=130,
        chips_per_rack={"B200": 72, "Grace": 36, "NVSwitch_BW": 18, "CX8": 36},
        hbm_stacks_per_xpu=8,
        hbm_die_key="HBM4_base",
        xpu_key="B200",
        spine_uplink_ports_per_rack=36,
    ),
    "DGX_H100": Platform(
        name="NVIDIA DGX H100 (8-GPU, H100 SXM5)",
        rack_tdp_kw=10.2,
        chips_per_rack={"H100": 8, "NVSwitch_H": 2, "CX8": 8},
        hbm_stacks_per_xpu=5,   # 80 GB = 5 × 16 GB HBM3
        hbm_die_key="HBM3e_base",
        xpu_key="H100",
        spine_uplink_ports_per_rack=8,
    ),
}


# ── Core calculator ────────────────────────────────────────────────────────────

@dataclass
class ComponentRow:
    chip_key: str
    chip: Chip
    units_per_month: float
    wafers_per_month: float


def compute(gw_annual: float, platform_key: str = "NVL72_HBM3e") -> Dict:
    """
    Returns monthly TSMC wafer demand for a given annual AI power addition.

    gw_annual:    GW of new AI datacenter capacity deployed per year
    platform_key: key into PLATFORMS dict
    """
    p = PLATFORMS[platform_key]
    gw_monthly = gw_annual / 12.0
    racks_per_month = (gw_monthly * 1e6) / p.rack_tdp_kw   # GW → kW

    units: Dict[str, float] = {}

    # Core rack components
    for key, count in p.chips_per_rack.items():
        units[key] = racks_per_month * count

    # HBM base die (follows XPU count)
    xpu_units = units[p.xpu_key]
    units[p.hbm_die_key] = xpu_units * p.hbm_stacks_per_xpu

    # Transceiver DSP (1 per spine uplink port)
    units["DSP_800G"] = racks_per_month * p.spine_uplink_ports_per_rack

    # Spine switch: each switch handles (ports × oversubscription) rack uplinks
    effective_ports = p.spine_switch_ports * p.spine_oversubscription
    racks_per_switch = effective_ports / p.spine_uplink_ports_per_rack
    units[p.spine_switch_key] = racks_per_month / racks_per_switch

    # Build row list
    rows: List[ComponentRow] = []
    for key, count in units.items():
        chip = CHIPS[key]
        wafers = (count / chip.ndw) if chip.is_tsmc else 0.0
        rows.append(ComponentRow(key, chip, count, wafers))

    # Aggregate by node
    node_wafers: Dict[str, float] = {}
    for row in rows:
        if row.chip.is_tsmc:
            node_wafers[row.chip.node] = node_wafers.get(row.chip.node, 0.0) + row.wafers_per_month

    return {
        "platform": p.name,
        "gw_annual": gw_annual,
        "racks_per_month": racks_per_month,
        "xpu_per_month": xpu_units,
        "rows": rows,
        "node_wafers": node_wafers,
    }


# ── Report printer ─────────────────────────────────────────────────────────────

def print_report(result: Dict) -> None:
    W = 100
    print(f"\n{'═'*W}")
    print(f"  AI Power → TSMC Wafer Demand")
    print(f"  Platform      : {result['platform']}")
    print(f"  Input         : {result['gw_annual']:.1f} GW/year new AI capacity "
          f"({result['gw_annual']/12:.2f} GW/month)")
    print(f"  Racks/month   : {result['racks_per_month']:>9,.0f}")
    print(f"  XPUs/month    : {result['xpu_per_month']:>9,.0f}")
    print(f"{'═'*W}")

    # Component detail
    hdr = (f"  {'Component':<46} {'Node':<7} {'mm²':>5} {'GDW':>5} "
           f"{'Yld%':>5} {'NDW':>5} {'Units/mo':>10} {'Wfrs/mo':>9}  Conf")
    print(f"\n{hdr}")
    print(f"  {'─'*96}")

    for r in result["rows"]:
        c = r.chip
        if not c.is_tsmc:
            print(
                f"  {c.name:<46} {'non-TSMC':<7} {'—':>5} {'—':>5} "
                f"{'—':>5} {'—':>5} {r.units_per_month:>10,.0f} {'—':>9}  [{c.conf}]"
            )
        else:
            print(
                f"  {c.name:<46} {c.node:<7} {c.die_area_mm2:>5.0f} {c.gdw:>5.1f} "
                f"{c.yld*100:>4.1f}% {c.ndw:>5.1f} "
                f"{r.units_per_month:>10,.0f} {r.wafers_per_month:>9,.0f}  [{c.conf}]"
            )

    # Node summary
    print(f"\n  {'─'*60}")
    print(f"  TSMC wafer demand by node (wspm = wafer starts per month)")
    print(f"  {'─'*60}")

    total_tsmc = sum(result["node_wafers"].values())
    for node, wspm in sorted(result["node_wafers"].items()):
        d0, label, cap = NODE[node]
        pct = f"= {wspm/cap*100:4.1f}% of ~{cap//1000}k wspm est. cap" if cap else ""
        print(f"  {label:<35}  {wspm:>8,.0f} wspm  {pct}")

    print(f"  {'Leading-edge TSMC total':<35}  {total_tsmc:>8,.0f} wspm")
    print()


def print_chip_table() -> None:
    print(f"\n{'─'*88}")
    print("  Chip parameters (TSMC components only)")
    print(f"  {'─'*84}")
    hdr = f"  {'Chip':<46} {'Node':<7} {'mm²':>5} {'GDW':>5} {'Yld%':>5} {'NDW':>5}  Conf  Notes"
    print(hdr)
    print(f"  {'─'*84}")
    for key, c in CHIPS.items():
        if c.is_tsmc:
            short_notes = c.notes[:40] + "…" if len(c.notes) > 40 else c.notes
            print(
                f"  {c.name:<46} {c.node:<7} {c.die_area_mm2:>5.0f} "
                f"{c.gdw:>5.1f} {c.yld*100:>4.1f}% {c.ndw:>5.1f}  [{c.conf}]  {short_notes}"
            )
    print()


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    platform_key = sys.argv[1] if len(sys.argv) > 1 else "NVL72_HBM3e"

    if platform_key not in PLATFORMS:
        print(f"Unknown platform '{platform_key}'. Choices: {list(PLATFORMS)}")
        sys.exit(1)

    print(f"\n  AI Datacenter Power → TSMC Front-End Wafer Demand Framework")
    print(f"  ─────────────────────────────────────────────────────────────")
    print(f"  Yield model  : negative binomial  Y = (1 + D₀·A/α)^(−α),  α = 10")
    print(f"  Wafer size   : 300mm,  3mm edge exclusion")
    print(f"  Node D₀      : N3B={NODE['N3B'][0]}, N3={NODE['N3'][0]}, "
          f"N4={NODE['N4'][0]}, N5={NODE['N5'][0]} defects/cm²")
    print(f"  Spine oversubscription : {PLATFORMS[platform_key].spine_oversubscription:.0f}:1")
    print(f"  HBM3e base die         : non-TSMC (SK Hynix / Samsung proprietary 4nm)")
    print(f"  HBM4  base die         : TSMC N3  (SK Hynix + TSMC, ramping 2025+)")

    print_chip_table()

    if len(sys.argv) > 2:
        gw = float(sys.argv[2])
        print_report(compute(gw, platform_key))
    else:
        for gw in [1.0, 3.0, 5.0, 10.0, 20.0]:
            print_report(compute(gw, platform_key))


if __name__ == "__main__":
    main()
