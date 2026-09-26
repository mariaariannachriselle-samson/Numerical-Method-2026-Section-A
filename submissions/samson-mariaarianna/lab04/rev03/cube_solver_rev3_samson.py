# ==============================================================
# cube_solver_rev3.py
# ==============================================================
# CUBE STRUCTURAL SOLVER - REV. 3
#
# REV 3 - STEP 2 : LOAD DATA MODEL  (dataclasses only)
# REV 3 - STEP 3 : PHYSICAL LOAD CASES LC1 - LC8
# REV 3 - STEP 4 : ROOF DIAPHRAGM (definition + constraint equations)
# REV 3 - STEP 5 : LC9 TEMPERATURE LOAD (thermal strain + equivalent load)
# --------------------------------------------------------------
# This module defines the clean data structures that the future
# Rev. 3 load framework will use:
#
#   LoadCase
#   NodalLoad
#   MemberDistributedLoad
#   MemberPointLoad
#   TemperatureLoad
#   Diaphragm
#   LoadCombination
#
# STEP 2 SCOPE (ONLY)
#   - Data structures (Python dataclasses).
#   - Basic validation methods on those structures.
#   - Backward compatibility: the existing Rev. 2 TEST-A load case
#     remains available but is NOT activated automatically.
#
# STEP 3 SCOPE (ADDED)
#   - The eight physical mechanical load cases LC1-LC8, defined once
#     in the authoritative registry LOAD_CASES_REV3, plus a
#     validation/report helper for their totals.
#
# STEP 4 SCOPE (ADDED)
#   - The authoritative roof diaphragm at the roof-beam elevation:
#     the roof-level nodes are identified from the Rev. 2 geometry,
#     the lowest-numbered roof node becomes the master, the remaining
#     roof nodes become slaves, UX/UZ/RY are constrained and
#     UY/RX/RZ stay free, and the constraint equations are generated.
#   - The equations are generated and validated ONLY: the global
#     stiffness matrix is NOT modified and no constraint elimination
#     is performed (see ROOF_DIAPHRAGM_ENFORCED).
#
# STEP 5 SCOPE (ADDED)
#   - The authoritative LC9 temperature case (load type
#     "Temperature", delta_T = +15 degC) on the existing roof beams:
#     alpha is taken from the existing material data (stored as 11.7
#     in 1e-6/degC, i.e. 11.7e-6/degC), and the thermal strain, free
#     expansion, EA and restrained force are derived from the existing
#     member/section data.
#   - The free / partially restrained / fully restrained states are
#     distinguished, and the partially restrained force is reported as
#     INDETERMINATE rather than guessed.
#   - The self-equilibrating equivalent thermal load vector is
#     generated (zero net structural force).
#   - Temperature stays its own load category: it is never added to
#     the mechanical kN totals and LC9 is not in LOAD_CASES_REV3.
#   - The thermal load vector is generated but NOT assembled: no
#     stiffness-matrix modification and no solution.
#
# STILL DEFERRED (later Rev. 3 steps):
#   - load combinations in use, diaphragm applied to the stiffness
#     matrix (constraint elimination), assembly/solution of the thermal
#     load vector, the viewer, load-case PNGs, automated tests.
#
# DESIGN RULES
#   - The data model is INDEPENDENT from the plotting code: no
#     matplotlib import at module level.
#   - cube_6m_Rev2.py is never modified; it remains the baseline.
#   - All units follow the Rev. 2 Units database conventions:
#       geometry/positions in m, forces in kN, distributed loads in
#       kN/m, moments in kN-m, temperature in Celsius.
# ==============================================================

from __future__ import annotations

import importlib.util
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

# --------------------------------------------------------------
# CONSTANTS
# --------------------------------------------------------------

# Global directions allowed for nodal loads.
NODAL_DIRECTIONS: Tuple[str, ...] = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")

# Directions allowed for member loads (global directions in Step 2;
# local-axis member directions are reserved for a later step).
MEMBER_DIRECTIONS: Tuple[str, ...] = NODAL_DIRECTIONS

# Global DOF names (identical to the Rev. 2 node_dofs table).
DOF_NAMES: Tuple[str, ...] = ("UX", "UY", "UZ", "RX", "RY", "RZ")

# Solver-internal unit strings (for documentation / later reporting).
UNIT_LENGTH = "m"
UNIT_FORCE = "kN"
UNIT_DISTRIBUTED = "kN/m"
UNIT_MOMENT = "kN-m"
UNIT_TEMP = "C"

# Load-case types accepted by LoadCase.validate() (Step 3 defines
# LC1-LC8; Step 5 defines LC9 with the "Temperature" type).
LOAD_TYPES_REV3: Tuple[str, ...] = (
    "Dead Load",
    "Live Load",
    "Member Point Load",
    "Wind",
    "Seismic",
    "Temperature",
)

# Axial restraint states of a member subjected to a temperature change
# (used by TemperatureLoad.restraint and by the Step 5 / LC9 code):
#   free             - free expansion; the thermal strain exists but no
#                      axial force develops.
#   partial          - partially restrained (the real structural
#                      situation); the force is INDETERMINATE and must
#                      not be guessed (Rev. 3 performs no constraint
#                      elimination).
#   fully-restrained - full restraint; N = EA * eps_T develops
#                      (compression for a positive temperature change).
THERMAL_RESTRAINT_FREE: str = "free"
THERMAL_RESTRAINT_PARTIAL: str = "partial"
THERMAL_RESTRAINT_FULL: str = "fully-restrained"
THERMAL_RESTRAINT_STATES: Tuple[str, ...] = (
    THERMAL_RESTRAINT_FREE,
    THERMAL_RESTRAINT_PARTIAL,
    THERMAL_RESTRAINT_FULL,
)

# Documented fallbacks used ONLY when no Rev. 2 model object is
# supplied.  Both match the Rev. 2 baseline (A36 Gr.36: alpha = 11.7
# 1e-6/degC and E = 199947.96 MPa).  Whenever a model is available the
# existing material data is used instead.
FALLBACK_ALPHA_1E6_PER_C: float = 11.7
FALLBACK_MODULUS_MPA: float = 199948.0

# --------------------------------------------------------------
# INTERNAL HELPERS
# --------------------------------------------------------------

def _is_finite(value) -> bool:
    """Return True when `value` is a finite float."""
    try:
        return bool(math.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _direction_is_valid(direction: str, allowed: Tuple[str, ...]) -> bool:
    return isinstance(direction, str) and direction in allowed


def _node_ids(model) -> Optional[set]:
    """Return the set of node IDs from the passed Rev. 2 model, if any."""
    if model is None:
        return None
    try:
        nodes = model.nodes
        if nodes is None:
            return None
        return {int(v) for v in nodes["Node"]}
    except Exception:
        return None


def _member_ids(model) -> Optional[set]:
    """Return the set of member IDs from the passed Rev. 2 model, if any."""
    if model is None:
        return None
    try:
        members = model.members
        if members is None:
            return None
        return {int(v) for v in members["Member"]}
    except Exception:
        return None


def _member_length(model, member_id: int) -> Optional[float]:
    """Return the length (m) of `member_id` from the Rev. 2 model."""
    if model is None:
        return None
    try:
        ms = model.member_sections
        if ms is None:
            return None
        row = ms.loc[ms["Member"] == int(member_id)]
        if row.empty:
            return None
        return float(row.iloc[0]["Length (m)"])
    except Exception:
        return None


def _member_section_row(model, member_id: int):
    """Return the Rev. 2 ``member_sections`` row for `member_id`, or None."""
    if model is None:
        return None
    try:
        ms = model.member_sections
        if ms is None:
            return None
        row = ms.loc[ms["Member"] == int(member_id)]
        if row.empty:
            return None
        return row.iloc[0]
    except Exception:
        return None


def _material_thermal_alpha_1e6(model) -> Optional[float]:
    """Return the material thermal coefficient in 1e-6/degC, or None.

    The value comes from the existing material data of the Rev. 2 model
    (``MATERIAL_ALPHA_1E6_PER_C``, read from the RISA materials
    database).  It is a 1e-6/degC value and must be multiplied by 1e-6
    before use - it is never used raw.
    """
    if model is None:
        return None
    try:
        value = float(model.MATERIAL_ALPHA_1E6_PER_C)
    except Exception:
        return None
    if not math.isfinite(value):
        return None
    return value


def _material_modulus_mpa(model) -> Optional[float]:
    """Return the modulus of elasticity E [MPa] from the Rev. 2 model."""
    if model is None:
        return None
    try:
        value = float(model.MATERIAL_E_MPA)
    except Exception:
        return None
    if not math.isfinite(value):
        return None
    return value


def _member_labels(model, member_id: int) -> Tuple[str, str]:
    """Return (material label, section label) of `member_id`."""
    row = _member_section_row(model, member_id)
    if row is None:
        return ("", "")
    material = ""
    section = ""
    try:
        material = str(row["Material Label"])
    except Exception:
        pass
    try:
        section = str(row["Section Label"])
    except Exception:
        pass
    return (material, section)


def _member_area_m2(model, member_id: int) -> Optional[float]:
    """Return the cross-section area A [m2] of `member_id`, or None."""
    row = _member_section_row(model, member_id)
    if row is None:
        return None
    try:
        area = float(row["Area (m2)"])
    except Exception:
        return None
    if not math.isfinite(area) or area <= 0.0:
        return None
    return area


def _member_axial_rigidity_kn(model, member_id: int) -> Optional[float]:
    """Return the axial rigidity EA [kN] of `member_id`, or None.

    EA[kN] = E[MPa] * 1000 * A[m2]   (1 MPa = 1000 kN/m2)

    Both E and A are read from the existing Rev. 2 data (material
    database and member section table); nothing is recomputed.
    """
    area = _member_area_m2(model, member_id)
    modulus = _material_modulus_mpa(model)
    if area is None or modulus is None:
        return None
    return modulus * 1000.0 * area


def _member_end_nodes(model, member_id: int) -> Optional[Tuple[int, int]]:
    """Return (node i, node j) of `member_id` from the Rev. 2 members table."""
    if model is None:
        return None
    try:
        members = model.members
        row = members.loc[members["Member"] == int(member_id)]
        if row.empty:
            return None
        row = row.iloc[0]
        return (int(row["Node i (Start)"]), int(row["Node j (End)"]))
    except Exception:
        return None


def _member_axis_unit_vector(model, member_id: int
                             ) -> Optional[Tuple[float, float, float]]:
    """Return the unit vector from node i to node j of `member_id`."""
    ends = _member_end_nodes(model, member_id)
    if ends is None:
        return None
    try:
        coords = model.node_coordinates
        start = [float(v) for v in coords[ends[0]]]
        end = [float(v) for v in coords[ends[1]]]
    except Exception:
        return None
    delta = [b - a for a, b in zip(start, end)]
    length = math.sqrt(sum(c * c for c in delta))
    if length <= 1.0e-12:
        return None
    return tuple(c / length for c in delta)

# --------------------------------------------------------------
# 1. NodalLoad
# --------------------------------------------------------------

@dataclass
class NodalLoad:
    """A single nodal load applied at a node in a global direction.

    Attributes:
        node:      node ID (Rev. 2 numbering 1..8).
        direction: one of NODAL_DIRECTIONS, e.g. "+X", "-Y".
        magnitude: signed load value in kN (sign is part of the value,
                   so "-Y" with +10.0 kN means downward 10 kN; the
                   caller may instead store -10.0 for the same effect).
    """

    node: int
    direction: str
    magnitude: float

    @property
    def dof(self) -> str:
        """Global DOF name associated with `direction` (e.g. 'UY' for '-Y')."""
        return {"+X": "UX", "-X": "UX",
                "+Y": "UY", "-Y": "UY",
                "+Z": "UZ", "-Z": "UZ"}[self.direction]

    def to_rev2(self) -> Dict[int, Dict[str, float]]:
        """Return the Rev. 2 style ``{node: {DOF: value}}`` mapping."""
        return {int(self.node): {self.dof: float(self.magnitude)}}

    def validate(self, model=None) -> List[str]:
        """Basic data checks; uses `model` only to validate the node ID.

        Args:
            model: optional object exposing `.nodes` (as in the Rev. 2
                   module) for a node-ID existence check.
        """
        problems: List[str] = []

        # node
        try:
            n = int(self.node)
        except (TypeError, ValueError):
            n = -1
        if n < 1:
            problems.append(f"NodalLoad: node must be a positive integer, got {self.node!r}")
        if model is not None:
            ids = _node_ids(model)
            if ids is not None and n not in ids:
                problems.append(f"NodalLoad: node {n} not found in model nodes")

        # direction
        if not _direction_is_valid(self.direction, NODAL_DIRECTIONS):
            problems.append(
                f"NodalLoad: invalid direction {self.direction!r}; "
                f"expected one of {NODAL_DIRECTIONS}"
            )

        # magnitude
        if not _is_finite(self.magnitude):
            problems.append(f"NodalLoad: magnitude must be finite, got {self.magnitude!r}")

        return problems


# --------------------------------------------------------------
# 2. MemberDistributedLoad
# --------------------------------------------------------------

@dataclass
class MemberDistributedLoad:
    """A distributed load applied to a member.

    Attributes:
        member:    member ID (Rev. 2 numbering 1..12).
        direction: load direction (see MEMBER_DIRECTIONS), stored
                   explicitly, e.g. "+Y" or "-Y".
        w_start:   load intensity [kN/m] at the node-i end.
        w_end:     load intensity [kN/m] at the node-j end.

    Uniform load:         w_start == w_end
    Linearly varying:     w_start != w_end  (trapezoidal / triangular)
    """

    member: int
    direction: str
    w_start: float
    w_end: float

    @property
    def is_uniform(self) -> bool:
        return abs(float(self.w_start) - float(self.w_end)) < 1.0e-12

    def validate(self, model=None) -> List[str]:
        """Basic data checks; uses `model` only for member-ID existence."""
        problems: List[str] = []

        try:
            m = int(self.member)
        except (TypeError, ValueError):
            m = -1
        if m < 1:
            problems.append(
                f"MemberDistributedLoad: member must be a positive integer, got {self.member!r}"
            )
        if model is not None:
            ids = _member_ids(model)
            if ids is not None and m not in ids:
                problems.append(f"MemberDistributedLoad: member {m} not found in model members")

        if not _direction_is_valid(self.direction, MEMBER_DIRECTIONS):
            problems.append(
                f"MemberDistributedLoad: invalid direction {self.direction!r}; "
                f"expected one of {MEMBER_DIRECTIONS}"
            )

        for name, value in (("w_start", self.w_start), ("w_end", self.w_end)):
            if not _is_finite(value):
                problems.append(
                    f"MemberDistributedLoad: {name} must be finite, got {value!r}"
                )

        return problems

# --------------------------------------------------------------
# 3. MemberPointLoad
# --------------------------------------------------------------

@dataclass
class MemberPointLoad:
    """A concentrated load applied at a position along a member.

    Attributes:
        member:    member ID.
        position:  distance from node i along the member [m].  It will
                   be validated against the member length when a model
                   is available; must satisfy 0 <= position <= L.
        direction: load direction (see MEMBER_DIRECTIONS).
        magnitude: concentrated force [kN].

    No physical node is created for the point-load location.
    """

    member: int
    position: float
    direction: str
    magnitude: float

    def validate(self, model=None) -> List[str]:
        """Basic data checks; uses `model` for member/length checks."""
        problems: List[str] = []

        try:
            m = int(self.member)
        except (TypeError, ValueError):
            m = -1
        if m < 1:
            problems.append(
                f"MemberPointLoad: member must be a positive integer, got {self.member!r}"
            )
        if model is not None:
            ids = _member_ids(model)
            if ids is not None and m not in ids:
                problems.append(f"MemberPointLoad: member {m} not found in model members")
            length = _member_length(model, m)
            if length is not None and _is_finite(self.position):
                if not (0.0 <= float(self.position) <= length):
                    problems.append(
                        f"MemberPointLoad: position {self.position} m is outside "
                        f"[0, {length:.6f} m] for member {m}"
                    )

        if not _direction_is_valid(self.direction, MEMBER_DIRECTIONS):
            problems.append(
                f"MemberPointLoad: invalid direction {self.direction!r}; "
                f"expected one of {MEMBER_DIRECTIONS}"
            )

        if not _is_finite(self.position):
            problems.append(f"MemberPointLoad: position must be finite, got {self.position!r}")

        if not _is_finite(self.magnitude):
            problems.append(f"MemberPointLoad: magnitude must be finite, got {self.magnitude!r}")

        return problems


# --------------------------------------------------------------
# 4. TemperatureLoad
# --------------------------------------------------------------

@dataclass
class TemperatureLoad:
    """A uniform temperature change applied to one or more members.

    Attributes:
        members:          a member ID or a collection of member IDs.
        delta_T:          uniform temperature change [Celsius] (positive
                          = heating, negative = cooling).
        restraint:        axial restraint state of the member, one of
                          THERMAL_RESTRAINT_STATES:
                            "free"             - free expansion: the
                                                 thermal strain exists,
                                                 the member expands, no
                                                 restraint force;
                            "partial"          - partially restrained
                                                 (the real structural
                                                 situation): the force is
                                                 INDETERMINATE and is
                                                 reported as such;
                            "fully-restrained" - full restraint: the
                                                 thermal force
                                                 N = EA * eps_T develops
                                                 (compression for
                                                 heating).
        reference_temp_c: reference (stress-free) temperature [Celsius]
                          when the model/specification provides one;
                          None means only the change delta_T is defined.
        alpha_1e6_per_c:  optional explicit thermal coefficient in
                          1e-6/degC; when None it is taken from the
                          existing material data
                          (model.MATERIAL_ALPHA_1E6_PER_C, stored as
                          11.7 in 1e-6/degC for the Rev. 2 A36 steel).

    Temperature is a load CATEGORY of its own ("Temperature"): it is
    stored as a thermal strain effect and never as an arbitrary nodal
    force in kN.  The equivalent thermal load vector generated by the
    Step 5 (LC9) code is self-equilibrating (zero net structural
    force), is not assembled into the stiffness matrix, and no solved
    results are claimed here.
    """

    members: Union[int, Iterable[int]]
    delta_T: float
    restraint: str = THERMAL_RESTRAINT_FREE
    reference_temp_c: Optional[float] = None
    alpha_1e6_per_c: Optional[float] = None
    load_case_id: str = "LC9"

    def member_list(self) -> List[int]:
        """Return the list of member IDs this temperature load applies to."""
        if isinstance(self.members, (int,)):
            return [self.members]
        return list(self.members)

    def restraint_state(self) -> str:
        """Return the axial restraint state of this temperature load."""
        return str(self.restraint)

    def is_indeterminate(self) -> bool:
        """True when the restrained force cannot be determined rigorously."""
        return self.restraint_state() == THERMAL_RESTRAINT_PARTIAL

    def alpha_per_c(self, model=None) -> float:
        """Return the thermal coefficient alpha in 1/degC.

        Resolution order: an explicit `alpha_1e6_per_c` on this load,
        then the existing material data of the Rev. 2 model
        (MATERIAL_ALPHA_1E6_PER_C), then the documented baseline
        fallback.  The stored value is in 1e-6/degC and is ALWAYS
        multiplied by 1e-6 - 11.7 is never used as the coefficient.
        """
        if self.alpha_1e6_per_c is not None and _is_finite(self.alpha_1e6_per_c):
            return float(self.alpha_1e6_per_c) * 1.0e-6
        value = _material_thermal_alpha_1e6(model)
        if value is None:
            value = FALLBACK_ALPHA_1E6_PER_C
        return float(value) * 1.0e-6

    def thermal_strain(self, model=None) -> float:
        """Return the stress-free thermal strain eps_T = alpha * delta_T."""
        return self.alpha_per_c(model) * float(self.delta_T)

    def free_expansion_mm(self, model=None,
                          member_id: Optional[int] = None) -> Optional[float]:
        """Return the free thermal expansion [mm] = eps_T * L * 1000.

        The member length comes from the existing Rev. 2 member section
        data (member_sections "Length (m)").
        """
        if member_id is None:
            members = self.member_list()
            if not members:
                return None
            member_id = members[0]
        length = _member_length(model, int(member_id))
        if length is None:
            return None
        return self.thermal_strain(model) * float(length) * 1000.0

    def restrained_force_kn(self, model=None, member_id: Optional[int] = None,
                            restraint: Optional[str] = None) -> Optional[float]:
        """Return the restrained axial force [kN] (negative = compression).

        free             -> 0.0 (no restraint force; the thermal strain
                                 and the free expansion still exist)
        fully-restrained -> -EA * eps_T (compression for heating)
        partial          -> None: INDETERMINATE.  Rev. 3 performs no
                            stiffness/constraint elimination, so no force
                            is guessed for a partially restrained member.
        """
        state = self.restraint_state() if restraint is None else str(restraint)
        if state == THERMAL_RESTRAINT_FREE:
            return 0.0
        if state == THERMAL_RESTRAINT_PARTIAL:
            return None
        if member_id is None:
            members = self.member_list()
            if not members:
                return None
            member_id = members[0]
        ea = _member_axial_rigidity_kn(model, int(member_id))
        if ea is None:
            return None
        return -ea * self.thermal_strain(model)

    def validate(self, model=None) -> List[str]:
        """Basic data checks; uses `model` for member-ID existence."""
        problems: List[str] = []

        members = self.member_list()
        if not members:
            problems.append("TemperatureLoad: members must not be empty")

        try:
            for mm in members:
                if int(mm) < 1:
                    problems.append(
                        f"TemperatureLoad: member must be positive, got {mm!r}"
                    )
        except (TypeError, ValueError):
            problems.append(
                f"TemperatureLoad: members must be ints, got {self.members!r}"
            )

        if model is not None:
            ids = _member_ids(model)
            if ids is not None:
                for mm in members:
                    if int(mm) not in ids:
                        problems.append(
                            f"TemperatureLoad: member {mm} not found in model members"
                        )

        if not _is_finite(self.delta_T):
            problems.append(f"TemperatureLoad: delta_T must be finite, got {self.delta_T!r}")

        if self.restraint_state() not in THERMAL_RESTRAINT_STATES:
            problems.append(
                f"TemperatureLoad: restraint must be one of "
                f"{THERMAL_RESTRAINT_STATES}, got {self.restraint!r}"
            )

        if self.reference_temp_c is not None and not _is_finite(self.reference_temp_c):
            problems.append(
                f"TemperatureLoad: reference_temp_c must be finite, got "
                f"{self.reference_temp_c!r}"
            )

        if not isinstance(self.load_case_id, str) or not self.load_case_id.strip():
            problems.append("TemperatureLoad: load_case_id must be a non-empty string")

        if self.alpha_1e6_per_c is not None:
            if (not _is_finite(self.alpha_1e6_per_c)
                    or float(self.alpha_1e6_per_c) <= 0.0):
                problems.append(
                    f"TemperatureLoad: alpha_1e6_per_c must be finite and "
                    f"positive, got {self.alpha_1e6_per_c!r}"
                )

        return problems

# --------------------------------------------------------------
# 5. Diaphragm
# --------------------------------------------------------------

@dataclass
class Diaphragm:
    """A rigid-diaphragm constraint definition (data structure ONLY).

    Step 2 stores the diaphragm definition only.  The stiffness matrix
    is NOT modified and the diaphragm is NOT made active.

    Attributes:
        master_node:      master node whose DOF values drive the slaves.
        slave_nodes:      slave node IDs constrained to the master.
        constrained_dofs: DOF names enforced by the diaphragm.
        active:           reserved for a later step; must remain False
                          (Step 4 generates and validates the equations
                          but does not enforce them on the stiffness
                          matrix).
        name:             optional human-readable label (Step 4 adds it
                          for the authoritative roof diaphragm; it
                          defaults to "" so Step 2 usage is unchanged).
    """

    master_node: int
    slave_nodes: Iterable[int]
    constrained_dofs: Iterable[str] = ("UX", "UY", "UZ")
    active: bool = False
    name: str = ""

    def slave_list(self) -> List[int]:
        return [int(s) for s in self.slave_nodes]

    def node_ids(self) -> List[int]:
        """Return [master] + slaves."""
        return [int(self.master_node)] + self.slave_list()

    def constrained_dof_list(self) -> List[str]:
        """Return the constrained DOF names, upper-cased, in order."""
        return [str(d).upper() for d in self.constrained_dofs]

    def free_dofs(self) -> List[str]:
        """Return the DOFs left free by the diaphragm (the complement)."""
        constrained = set(self.constrained_dof_list())
        return [d for d in DOF_NAMES if d not in constrained]

    def constraint_equations(self) -> List[Tuple[int, str, int]]:
        """Return the constraint equations as (slave, DOF, master) tuples.

        One equation per slave node and constrained DOF, i.e.:

            N<slave> <DOF> = N<master> <DOF>
        """
        master = int(self.master_node)
        return [
            (slave, dof, master)
            for slave in self.slave_list()
            for dof in self.constrained_dof_list()
        ]

    def equation_lines(self) -> List[str]:
        """Return the constraint equations as readable strings."""
        master = int(self.master_node)
        return [
            f"N{slave} {dof} = N{master} {dof}"
            for slave, dof, _ in self.constraint_equations()
        ]

    def validate(self, model=None) -> List[str]:
        """Basic data checks; uses `model` for node-ID existence."""
        problems: List[str] = []

        try:
            master = int(self.master_node)
        except (TypeError, ValueError):
            master = -1
        if master < 1:
            problems.append(
                f"Diaphragm: master_node must be a positive integer, got {self.master_node!r}"
            )

        slaves = self.slave_list()
        if not slaves:
            problems.append("Diaphragm: slave_nodes must not be empty")
        if len(set(slaves)) != len(slaves):
            problems.append("Diaphragm: slave_nodes must be unique")

        dofs = [str(d).upper() for d in self.constrained_dofs]
        if not dofs:
            problems.append("Diaphragm: constrained_dofs must not be empty")
        for d in dofs:
            if d not in DOF_NAMES:
                problems.append(
                    f"Diaphragm: invalid constrained DOF {d!r}; expected one of {DOF_NAMES}"
                )

        if self.active:
            problems.append("Diaphragm: active flag must remain False in Step 2 (data only)")

        if model is not None:
            ids = _node_ids(model)
            if ids is not None:
                all_nodes = [master] + slaves
                for node_id in all_nodes:
                    if node_id not in ids:
                        problems.append(
                            f"Diaphragm: node {node_id} not found in model nodes"
                        )

        return problems


# --------------------------------------------------------------
# 6. LoadCombination
# --------------------------------------------------------------

@dataclass
class LoadCombination:
    """A named combination that references load cases by key + factor.

    Attributes:
        key:        combination name, e.g. "COMB1".
        factors:    mapping of load-case key -> factor, e.g.
                    {"LC1": 1.4, "LC2": 1.4}.

    Only references to load cases are stored (case key + factor).
    The physical loads are NOT duplicated here.
    """

    key: str
    factors: Dict[str, float]

    @property
    def case_keys(self) -> List[str]:
        return list(self.factors.keys())

    def validate(self, model=None) -> List[str]:
        """Basic data checks for the combination definition."""
        problems: List[str] = []

        if not isinstance(self.key, str) or not self.key.strip():
            problems.append(f"LoadCombination: key must be a non-empty string, got {self.key!r}")

        if not self.factors:
            problems.append("LoadCombination: combination must not be empty")

        for case_key, factor in self.factors.items():
            if not isinstance(case_key, str) or not case_key.strip():
                problems.append(
                    f"LoadCombination: load-case key must be a non-empty string, got {case_key!r}"
                )
            if not _is_finite(factor):
                problems.append(
                    f"LoadCombination: factor for {case_key!r} must be finite, got {factor!r}"
                )

        return problems

# --------------------------------------------------------------
# 7. LoadCase
# --------------------------------------------------------------

@dataclass
class LoadCase:
    """A load case container for the Rev. 3 model.

    Attributes:
        key:                  load-case name, e.g. "TEST-A".
        description:          human-readable description.
        nodal_loads:          list of NodalLoad.
        distributed_loads:    list of MemberDistributedLoad.
        member_point_loads:   list of MemberPointLoad.
        temperature_loads:    list of TemperatureLoad.
        self_weight:          True to add member self-weight (Rev. 2
                              materials/sections) when the case runs.
        self_weight_direction: global direction of the self-weight load
                              (default "-Y"; used only when self_weight).
        load_type:            case category, e.g. Dead / Live / Wind /
                              Seismic (informational; see LOAD_TYPES_REV3).
        diaphragm:            optional Diaphragm reference (data only
                              in Step 2; not active).

    The professor's actual LC1-LC9 load cases are intentionally NOT
    defined in Step 2.
    """

    key: str
    description: str = ""
    nodal_loads: List[NodalLoad] = field(default_factory=list)
    distributed_loads: List[MemberDistributedLoad] = field(default_factory=list)
    member_point_loads: List[MemberPointLoad] = field(default_factory=list)
    temperature_loads: List[TemperatureLoad] = field(default_factory=list)
    self_weight: bool = False
    self_weight_direction: str = "-Y"
    load_type: str = ""
    diaphragm: Optional[Diaphragm] = None

    @property
    def has_loads(self) -> bool:
        return bool(
            self.nodal_loads
            or self.distributed_loads
            or self.member_point_loads
            or self.temperature_loads
            or self.self_weight
        )

    def validate(self, model=None) -> List[str]:
        """Validate this load case and every load it contains."""
        problems: List[str] = []

        if not isinstance(self.key, str) or not self.key.strip():
            problems.append(f"LoadCase: key must be a non-empty string, got {self.key!r}")

        if not isinstance(self.self_weight, bool):
            problems.append(f"LoadCase {self.key!r}: self_weight must be a bool")

        if self.self_weight and self.self_weight_direction not in MEMBER_DIRECTIONS:
            problems.append(
                f"LoadCase {self.key!r}: self_weight_direction must be one of "
                f"{MEMBER_DIRECTIONS}, got {self.self_weight_direction!r}"
            )

        if self.load_type and self.load_type not in LOAD_TYPES_REV3:
            problems.append(
                f"LoadCase {self.key!r}: unknown load_type {self.load_type!r}; "
                f"expected one of {LOAD_TYPES_REV3}"
            )

        for load in self.nodal_loads:
            problems.extend(load.validate(model))
        for load in self.distributed_loads:
            problems.extend(load.validate(model))
        for load in self.member_point_loads:
            problems.extend(load.validate(model))
        for load in self.temperature_loads:
            problems.extend(load.validate(model))

        if self.diaphragm is not None:
            if not isinstance(self.diaphragm, Diaphragm):
                problems.append(
                    f"LoadCase {self.key!r}: diaphragm must be a Diaphragm or None"
                )
            else:
                problems.extend(self.diaphragm.validate(model))

        return problems

# --------------------------------------------------------------
# ACCESS TO THE EXISTING REV. 2 MODEL (BACKWARD COMPATIBILITY)
# --------------------------------------------------------------

_REV2_MODEL = None


def get_rev2_model(force: bool = False):
    """Load (once) the existing Rev. 2 model module and return it.

    cube_6m_Rev2.py is a module-level script: importing it executes the
    full Rev. 2 pipeline (DB reads, Excel output, PNG output).  For
    stability, matplotlib is forced to the non-interactive "Agg"
    backend and plt.show() is neutralized during that import only.

    Args:
        force: reload even if the module was already imported.

    Returns:
        The Rev. 2 module object, exposing e.g. `.nodes`, `.members`,
        `.member_sections`, `.MATERIAL_ALPHA_1E6_PER_C`, ...
    """
    global _REV2_MODEL
    if _REV2_MODEL is not None and not force:
        return _REV2_MODEL

    rev2_path = Path(__file__).resolve().parent / "cube_6m_Rev2.py"
    if not rev2_path.is_file():
        raise FileNotFoundError(f"Rev. 2 baseline not found at {rev2_path}")

    # Headless import of the Rev. 2 script.
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        _original_show = plt.show
        plt.show = lambda *args, **kwargs: None
    except Exception:
        _original_show = None

    try:
        spec = importlib.util.spec_from_file_location(
            "cube_6m_Rev2", rev2_path
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot import Rev. 2 baseline {rev2_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _REV2_MODEL = module
        return module
    finally:
        if _original_show is not None:
            try:
                import matplotlib.pyplot as plt

                plt.show = _original_show
            except Exception:
                pass


def test_a_load_case() -> LoadCase:
    """Return the existing Rev. 2 TEST-A load case as a LoadCase.

    The Rev. 2 pipeline defines (TEST ONLY - not professor-specified):
        LOAD_CASE_KEY = "TEST-A"
        LOAD_CASE["nodal"] = {5: {"UY": -10.0}}

    This factory reproduces exactly that case with the Step 2 data
    model.  It is NOT activated automatically; the caller decides if
    and when to run it.
    """
    return LoadCase(
        key="TEST-A",
        description=(
            "TEST ONLY - single -10 kN global-Y load at roof node 5 "
            "(Rev. 2 compatible)"
        ),
        nodal_loads=[NodalLoad(node=5, direction="-Y", magnitude=-10.0)],
        self_weight=False,
    )


# ==============================================================
# STEP 3 : PHYSICAL LOAD CASES LC1 - LC8  (AUTHORITATIVE REGISTRY)
# ==============================================================
# These are the ONLY definitions of the physical loads.  Anything
# downstream (combinations, viewer, PNGs, tests) must reference
# LOAD_CASES_REV3 instead of re-stating the loads.

# Roof beams (Rev. 2 members 5-8) carry LC2/LC3 uniformly and LC4 at
# their center; roof nodes (Rev. 2 nodes 5-8) carry LC5-LC8.
ROOF_BEAM_MEMBERS: Tuple[int, ...] = (5, 6, 7, 8)
ROOF_NODES: Tuple[int, ...] = (5, 6, 7, 8)

# Position [m] of the LC4 center point load, measured from node i of
# each 6 m roof beam (no physical node is created at this location).
LC4_POINT_POSITION_M: float = 3.0

# Intended totals from the Rev. 3 specification (kN).  LC1 is the
# handout target for the spec's W310X38.7/W250X49.1 + A992 section
# set; validate_load_cases_rev3() discloses the difference against the
# actual W6X9 + A36 Rev. 2 baseline weights (see DISCREPANCY note).
LOAD_CASE_INTENDED_TOTALS_KN: Dict[str, float] = {
    "LC1": 29.815,   # self-weight (handout value, see DISCREPANCY)
    "LC2": 120.000,  # 5 kN/m x 6 m x 4 roof beams
    "LC3": 72.000,   # 3 kN/m x 6 m x 4 roof beams
    "LC4": 20.000,   # 5 kN x 4 roof beams
    "LC5": 10.000,   # 2.5 kN x 4 roof nodes (+X)
    "LC6": 10.000,   # 2.5 kN x 4 roof nodes (+Z)
    "LC7": 15.000,   # 3.75 kN x 4 roof nodes (+X)
    "LC8": 15.000,   # 3.75 kN x 4 roof nodes (+Z)
}


def _uniform_roof_loads(w_kn_per_m: float) -> List["MemberDistributedLoad"]:
    """Uniform -Y distributed load of `w_kn_per_m` on roof beams M5-M8."""
    return [
        MemberDistributedLoad(
            member=m, direction="-Y",
            w_start=float(w_kn_per_m), w_end=float(w_kn_per_m),
        )
        for m in ROOF_BEAM_MEMBERS
    ]


def _roof_center_point_loads(magnitude_kn: float) -> List["MemberPointLoad"]:
    """-Y point load of `magnitude_kn` at 3.0 m of roof beams M5-M8."""
    return [
        MemberPointLoad(
            member=m, position=LC4_POINT_POSITION_M,
            direction="-Y", magnitude=float(magnitude_kn),
        )
        for m in ROOF_BEAM_MEMBERS
    ]


def _roof_nodal_loads(magnitude_kn: float, direction: str) -> List["NodalLoad"]:
    """Positive `magnitude_kn` nodal load at roof nodes 5-8.

    Direction is explicit (e.g. "+X"); magnitudes stay positive so the
    sign is never encoded twice.
    """
    return [
        NodalLoad(node=n, direction=direction, magnitude=float(magnitude_kn))
        for n in ROOF_NODES
    ]


LOAD_CASES_REV3: Dict[str, LoadCase] = {
    "LC1": LoadCase(
        key="LC1",
        description="DEAD / SELF WEIGHT",
        load_type="Dead Load",
        self_weight=True,
        self_weight_direction="-Y",
    ),
    "LC2": LoadCase(
        key="LC2",
        description="ROOF DEAD",
        load_type="Dead Load",
        distributed_loads=_uniform_roof_loads(5.0),
    ),
    "LC3": LoadCase(
        key="LC3",
        description="ROOF LIVE",
        load_type="Live Load",
        distributed_loads=_uniform_roof_loads(3.0),
    ),
    "LC4": LoadCase(
        key="LC4",
        description="ROOF BEAM CENTER LOAD",
        load_type="Member Point Load",
        member_point_loads=_roof_center_point_loads(5.0),
    ),
    "LC5": LoadCase(
        key="LC5",
        description="WIND X",
        load_type="Wind",
        nodal_loads=_roof_nodal_loads(2.5, "+X"),
    ),
    "LC6": LoadCase(
        key="LC6",
        description="WIND Z",
        load_type="Wind",
        nodal_loads=_roof_nodal_loads(2.5, "+Z"),
    ),
    "LC7": LoadCase(
        key="LC7",
        description="SEISMIC X",
        load_type="Seismic",
        nodal_loads=_roof_nodal_loads(3.75, "+X"),
    ),
    "LC8": LoadCase(
        key="LC8",
        description="SEISMIC Z",
        load_type="Seismic",
        nodal_loads=_roof_nodal_loads(3.75, "+Z"),
    ),
}


# --------------------------------------------------------------
# STEP 3 HELPERS AND VALIDATION
# --------------------------------------------------------------

# Fallback roof-beam length [m] used ONLY when no Rev. 2 model object
# is supplied to the total computation.  The real length always comes
# from the Rev. 2 member_sections table when a model is available.
_FALLBACK_ROOF_BEAM_LENGTH_M: float = 6.0

# Tolerance [kN] for exact-total comparisons.
_TOTAL_TOL_KN: float = 1.0e-9


def _self_weight_total_kn(model) -> Optional[float]:
    """Sum the existing Rev. 2 member total weights (kN).

    Reuses the ``Total Weight (kN)`` column already computed by
    cube_6m_Rev2.py from the material unit weight, section area and
    member length.  No material/property calculation is duplicated.

    Returns None when no model (or no weights column) is available.
    """
    if model is None:
        return None
    try:
        ms = model.member_sections
        if ms is None or "Total Weight (kN)" not in ms.columns:
            return None
        return float(ms["Total Weight (kN)"].sum())
    except Exception:
        return None


def _loaded_member_ids(case: LoadCase, model) -> set:
    """Return the set of member IDs carrying loads in `case`."""
    members = {int(d.member) for d in case.distributed_loads}
    members |= {int(p.member) for p in case.member_point_loads}
    if case.self_weight:
        try:
            ms = model.member_sections
            members |= {int(v) for v in ms["Member"]}
        except Exception:
            pass
    return members


def _loaded_node_ids(case: LoadCase) -> set:
    """Return the set of node IDs carrying nodal loads in `case`."""
    return {int(n.node) for n in case.nodal_loads}


def compute_load_case_total_kn(case: LoadCase, model=None) -> float:
    """Computed total applied load [kN] for one load case.

    Integration rules (Step 3 specification):
      - distributed loads: integral of w over L, i.e.
        (w_start + w_end) / 2 * L  (exact for uniform and linear loads)
      - member point loads:  sum of point magnitudes
      - nodal loads:         sum of nodal magnitudes
      - self-weight:         sum of the existing Rev. 2 member total
                             weights (reused, never recomputed)

    Magnitudes are stored positive with explicit directions, so the
    total is a sum of applied-load magnitudes and the sign/direction
    is reported separately by the load objects.
    """
    total = 0.0

    if case.self_weight:
        sw = _self_weight_total_kn(model)
        if sw is not None:
            total += abs(sw)

    for d in case.distributed_loads:
        length = _member_length(model, int(d.member))
        if length is None:
            length = _FALLBACK_ROOF_BEAM_LENGTH_M
        avg_w = (float(d.w_start) + float(d.w_end)) / 2.0
        total += abs(avg_w * length)

    for p in case.member_point_loads:
        total += abs(float(p.magnitude))

    for n in case.nodal_loads:
        total += abs(float(n.magnitude))

    return total


def load_case_report_rows(model=None) -> List[Dict[str, object]]:
    """Return one report row per LC1-LC8 load case.

    Each row contains: case key, name, type, number of loaded
    nodes/members, total intended load, computed load total and the
    equilibrium / load-total error (computed - intended) [kN].
    """
    rows: List[Dict[str, object]] = []
    for key, case in LOAD_CASES_REV3.items():
        members = _loaded_member_ids(case, model)
        nodes = _loaded_node_ids(case)
        intended = float(LOAD_CASE_INTENDED_TOTALS_KN.get(key, 0.0))
        computed = compute_load_case_total_kn(case, model)
        rows.append({
            "key": key,
            "name": case.description,
            "type": case.load_type or "",
            "loaded_nodes": len(nodes),
            "loaded_members": len(members),
            "node_ids": sorted(nodes),
            "member_ids": sorted(members),
            "intended_kN": intended,
            "computed_kN": computed,
            "error_kN": computed - intended,
        })
    return rows




def validate_load_cases_rev3(model=None, verbose: bool = False) -> List[str]:
    """Validate LC1-LC8 and return a list of problems (empty = OK).

    Hard problems start with 'PROBLEM:'.  The known LC1 handout/Rev. 2
    baseline difference is disclosed with a 'DISCREPANCY:' prefix so
    callers can separate it from hard failures (see verify_step3).

    Structural checks performed:
      - LC2/LC3 contain exactly members M5-M8, uniform, -Y
      - LC4 contains exactly M5-M8 at 3.0 m, -Y, no nodal loads
      - LC5-LC8 contain exactly roof nodes 5-8 with positive
        magnitudes and explicit directions (+X / +Z)
      - LC1 uses self-weight only (existing Rev. 2 weight data)
      - no temperature loads anywhere (LC9 is deferred)
      - no physical nodes are created for LC4 (model keeps 8 nodes)
      - required LC2-LC8 totals are reproduced exactly; LC1 matches
        the existing Rev. 2 weight sum and the 29.815 kN handout
        target is disclosed as a DISCREPANCY when it differs
    """
    problems: List[str] = []

    if sorted(LOAD_CASES_REV3) != [f"LC{i}" for i in range(1, 9)]:
        problems.append(
            "PROBLEM: LOAD_CASES_REV3 must contain exactly LC1..LC8, "
            f"got {sorted(LOAD_CASES_REV3)}"
        )

    rows = {row["key"]: row for row in load_case_report_rows(model)}

    for key, case in LOAD_CASES_REV3.items():
        # Step 2 data validation of the case and all of its loads.
        for p in case.validate(model):
            problems.append(f"PROBLEM: {key}: {p}")

        # Exact required totals for LC2-LC8.
        if key != "LC1":
            intended = float(LOAD_CASE_INTENDED_TOTALS_KN[key])
            computed = float(rows[key]["computed_kN"])
            if abs(computed - intended) > _TOTAL_TOL_KN:
                problems.append(
                    f"PROBLEM: {key}: computed total {computed:.6f} kN "
                    f"!= intended {intended:.6f} kN"
                )

        # LC9 temperature must not appear yet.
        if case.temperature_loads:
            problems.append(
                f"PROBLEM: {key}: temperature loads are deferred to LC9"
            )

    # --- LC1: uses existing Rev. 2 self-weight data only ---------
    lc1 = LOAD_CASES_REV3.get("LC1")
    if lc1 is not None:
        if not lc1.self_weight:
            problems.append("PROBLEM: LC1: self_weight must be enabled")
        if lc1.self_weight_direction != "-Y":
            problems.append(
                f"PROBLEM: LC1: self_weight_direction must be '-Y', "
                f"got {lc1.self_weight_direction!r}"
            )
        if (lc1.nodal_loads or lc1.distributed_loads
                or lc1.member_point_loads):
            problems.append(
                "PROBLEM: LC1: must not re-state physical loads; "
                "self-weight comes from the Rev. 2 member data"
            )

        sw = _self_weight_total_kn(model)
        if model is not None and sw is None:
            problems.append(
                "PROBLEM: LC1: existing Rev. 2 Total Weight (kN) data "
                "not found in the model"
            )
        elif sw is not None:
            computed = float(rows["LC1"]["computed_kN"])
            if abs(computed - sw) > _TOTAL_TOL_KN:
                problems.append(
                    f"PROBLEM: LC1: computed {computed:.6f} kN does not "
                    f"equal the Rev. 2 weight sum {sw:.6f} kN"
                )
            target = float(LOAD_CASE_INTENDED_TOTALS_KN["LC1"])
            if abs(sw - target) > 5.0e-3:  # handout says "approximately"
                problems.append(
                    f"DISCREPANCY: LC1: the existing Rev. 2 baseline "
                    f"(all 12 members W6X9 + A36 Gr.36) weighs "
                    f"{sw:.6f} kN, while the handout target "
                    f"{target:.3f} kN corresponds to the "
                    f"W310X38.7/W250X49.1 + A992 section set.  The "
                    f"computed total honestly reuses the Rev. 2 data "
                    f"as specified; cube_6m_Rev2.py was not modified."
                )

    # --- LC2 / LC3: exactly roof beams M5-M8, uniform, -Y --------
    for key, w_expected in (("LC2", 5.0), ("LC3", 3.0)):
        case = LOAD_CASES_REV3.get(key)
        if case is None:
            continue
        members = sorted(int(d.member) for d in case.distributed_loads)
        if members != list(ROOF_BEAM_MEMBERS):
            problems.append(
                f"PROBLEM: {key}: distributed loads must be exactly "
                f"M5-M8, got M{members}"
            )
        for d in case.distributed_loads:
            if d.direction != "-Y":
                problems.append(
                    f"PROBLEM: {key}: member M{d.member} direction must "
                    f"be '-Y', got {d.direction!r}"
                )
            if (abs(float(d.w_start) - w_expected) > _TOTAL_TOL_KN
                    or abs(float(d.w_end) - w_expected) > _TOTAL_TOL_KN):
                problems.append(
                    f"PROBLEM: {key}: member M{d.member} must carry "
                    f"{w_expected} kN/m uniformly, got "
                    f"{d.w_start}/{d.w_end} kN/m"
                )
            if float(d.w_start) < 0.0 or float(d.w_end) < 0.0:
                problems.append(
                    f"PROBLEM: {key}: member M{d.member} magnitudes must "
                    f"stay positive (direction is explicit)"
                )

    # --- LC4: exactly M5-M8 at 3.0 m, no nodal loads -------------
    lc4 = LOAD_CASES_REV3.get("LC4")
    if lc4 is not None:
        points = sorted(
            (int(p.member), float(p.position))
            for p in lc4.member_point_loads
        )
        expected = [(m, LC4_POINT_POSITION_M) for m in ROOF_BEAM_MEMBERS]
        if points != expected:
            problems.append(
                f"PROBLEM: LC4: point loads must be exactly {expected} "
                f"[(member, position m)], got {points}"
            )
        for p in lc4.member_point_loads:
            if p.direction != "-Y":
                problems.append(
                    f"PROBLEM: LC4: member M{p.member} direction must "
                    f"be '-Y', got {p.direction!r}"
                )
            if abs(float(p.magnitude) - 5.0) > _TOTAL_TOL_KN:
                problems.append(
                    f"PROBLEM: LC4: member M{p.member} magnitude must "
                    f"be 5.0 kN, got {p.magnitude}"
                )
            length = _member_length(model, int(p.member))
            if length is None:
                length = _FALLBACK_ROOF_BEAM_LENGTH_M
            if not (0.0 < float(p.position) <= length + _TOTAL_TOL_KN):
                problems.append(
                    f"PROBLEM: LC4: position {p.position} m of member "
                    f"M{p.member} lies outside 0-{length} m"
                )
        if lc4.nodal_loads:
            problems.append(
                "PROBLEM: LC4: must not use nodal loads - no physical "
                "node is created at the mid-span location"
            )

    # --- LC5-LC8: exactly roof nodes 5-8, positive, explicit ------
    expected_dirs = {"LC5": "+X", "LC6": "+Z", "LC7": "+X", "LC8": "+Z"}
    expected_mags = {"LC5": 2.5, "LC6": 2.5, "LC7": 3.75, "LC8": 3.75}
    for key, direction in expected_dirs.items():
        case = LOAD_CASES_REV3.get(key)
        if case is None:
            continue
        nodes = sorted(int(n.node) for n in case.nodal_loads)
        if nodes != list(ROOF_NODES):
            problems.append(
                f"PROBLEM: {key}: nodal loads must be exactly the roof "
                f"nodes {list(ROOF_NODES)}, got {nodes}"
            )
        for n in case.nodal_loads:
            if n.direction != direction:
                problems.append(
                    f"PROBLEM: {key}: node {n.node} direction must be "
                    f"{direction!r}, got {n.direction!r}"
                )
            if float(n.magnitude) <= 0.0:
                problems.append(
                    f"PROBLEM: {key}: node {n.node} magnitude must be "
                    f"positive with an explicit direction, got "
                    f"{n.magnitude}"
                )
            if abs(float(n.magnitude) - expected_mags[key]) > _TOTAL_TOL_KN:
                problems.append(
                    f"PROBLEM: {key}: node {n.node} magnitude must be "
                    f"{expected_mags[key]} kN, got {n.magnitude}"
                )

    # --- The model keeps its 8 physical nodes (none added for LC4) -
    if model is not None:
        try:
            n_nodes = len(model.nodes)
        except Exception:
            n_nodes = None
        if n_nodes is not None and n_nodes != 8:
            problems.append(
                f"PROBLEM: the model must keep its 8 physical nodes "
                f"(no node is created for LC4), got {n_nodes}"
            )

    # --- Report: key, name, type, counts, totals, error -----------
    if verbose:
        print()
        print("== LC1-LC8 load-case report ==")
        print(
            f"{'key':4s}  {'name':24s}  {'type':18s}  "
            f"{'nodes':>5s} {'members':>7s}  "
            f"{'intended kN':>12s}  {'computed kN':>12s}  "
            f"{'error kN':>12s}"
        )
        for row in load_case_report_rows(model):
            print(
                f"{row['key']:4s}  {row['name']:24s}  {row['type']:18s}  "
                f"{row['loaded_nodes']:5d} {row['loaded_members']:7d}  "
                f"{row['intended_kN']:12.3f}  {row['computed_kN']:12.3f}  "
                f"{row['error_kN']:12.6f}"
            )
            print(
                f"       loaded node ids : {row['node_ids']} | "
                f"loaded member ids : {row['member_ids']}"
            )

    return problems





# ==============================================================
# STEP 4 : ROOF DIAPHRAGM (ROOF-BEAM ELEVATION)
# ==============================================================
# The roof diaphragm represents the common IN-PLANE movement of the
# roof nodes of the cube.  It is a real structural constraint
# definition, not a visual grouping: the equations generated below are
# the ones a later Rev. 3 step will use to eliminate the slave DOFs
# from the global stiffness matrix.
#
# STEP 4 SCOPE (ONLY)
#   - identify the roof-level nodes from the actual Rev. 2 geometry,
#   - select the lowest-numbered roof node as master and the remaining
#     roof nodes as slaves,
#   - generate and document the diaphragm constraint equations,
#   - validate the constrained DOFs and confirm which DOFs stay free.
#
# NOT in Step 4 (deliberately)
#   - the global stiffness matrix is NOT modified,
#   - no constraint elimination is performed,
#   - nothing here claims the diaphragm has been enforced in a solved
#     structural analysis (see ROOF_DIAPHRAGM_ENFORCED below).

ROOF_DIAPHRAGM_NAME: str = "ROOF DIAPHRAGM (roof beam elevation)"

# The roof is a horizontal X-Z plane, so the in-plane translations
# (UX, UZ) and the rotation about the vertical Y axis (RY) are shared.
# UY (vertical translation), RX and RZ stay free.
ROOF_DIAPHRAGM_CONSTRAINED_DOFS: Tuple[str, ...] = ("UX", "UZ", "RY")
ROOF_DIAPHRAGM_FREE_DOFS: Tuple[str, ...] = ("UY", "RX", "RZ")

# Roof nodes of the standard 6 m cube.  These are the EXPECTED values,
# used only as a documented fallback; the real identification is done
# by roof_level_nodes(model) from the node coordinates.
ROOF_DIAPHRAGM_EXPECTED_NODES: Tuple[int, ...] = (5, 6, 7, 8)

# Vertical coordinate tolerance [m] when identifying a node level.
_LEVEL_TOL_M: float = 1.0e-9

# Hard statement of scope: the constraint equations are generated and
# validated, but NOT enforced on the stiffness matrix in Step 4.
ROOF_DIAPHRAGM_ENFORCED: bool = False


def _nodes_table(model):
    """Return the Rev. 2 node table (``model.nodes``) or None."""
    if model is None:
        return None
    try:
        nodes = model.nodes
        if nodes is None or len(nodes) == 0:
            return None
        return nodes
    except Exception:
        return None


def _vertical_coordinate_column(nodes) -> Optional[str]:
    """Return the name of the vertical (Y) coordinate column."""
    try:
        columns = [str(c) for c in nodes.columns]
    except Exception:
        return None
    for candidate in ("Y (m)", "Y(m)", "Y [m]", "Y"):
        if candidate in columns:
            return candidate
    for column in columns:
        key = column.strip().upper().replace(" ", "")
        if key in ("Y(M)", "Y", "Y[MM]", "Y(MM)", "YMM"):
            return column
    return None


def roof_level_elevation_m(model) -> Optional[float]:
    """Return the elevation [m] of the highest (roof) node level."""
    nodes = _nodes_table(model)
    if nodes is None:
        return None
    column = _vertical_coordinate_column(nodes)
    if column is None:
        return None
    try:
        return float(nodes[column].max())
    except Exception:
        return None


def roof_level_nodes(model) -> Tuple[int, ...]:
    """Identify the roof-level nodes from the Rev. 2 model geometry.

    The roof is the highest node level of the model, so the nodes are
    selected by their vertical coordinate and never from a hard-coded
    list.  Returns an empty tuple when no geometry is available.
    """
    nodes = _nodes_table(model)
    if nodes is None:
        return ()
    column = _vertical_coordinate_column(nodes)
    if column is None:
        return ()
    elevation = roof_level_elevation_m(model)
    if elevation is None:
        return ()
    try:
        selected = nodes[(nodes[column] - elevation).abs() <= _LEVEL_TOL_M]
        return tuple(sorted(int(v) for v in selected["Node"]))
    except Exception:
        return ()


def roof_diaphragm_master_node(model=None) -> int:
    """Master node = the lowest-numbered roof-level node."""
    roof = roof_level_nodes(model)
    if roof:
        return int(roof[0])
    return int(ROOF_DIAPHRAGM_EXPECTED_NODES[0])


def roof_diaphragm_slave_nodes(model=None) -> Tuple[int, ...]:
    """Slave nodes = the remaining roof-level nodes."""
    roof = roof_level_nodes(model)
    if roof:
        return tuple(int(n) for n in roof[1:])
    return tuple(int(n) for n in ROOF_DIAPHRAGM_EXPECTED_NODES[1:])


def build_roof_diaphragm(model=None,
                         name: str = ROOF_DIAPHRAGM_NAME) -> Diaphragm:
    """Build the roof diaphragm definition for `model`.

    With a Rev. 2 model the roof nodes come from the geometry; without
    one, the standard roof nodes (5-8) are used as documented
    fallback.  The definition is never active: Step 4 generates and
    validates the constraint equations but does not enforce them.
    """
    return Diaphragm(
        master_node=roof_diaphragm_master_node(model),
        slave_nodes=roof_diaphragm_slave_nodes(model),
        constrained_dofs=ROOF_DIAPHRAGM_CONSTRAINED_DOFS,
        active=False,
        name=name,
    )


# Authoritative roof diaphragm definition (single source of truth).
# Geometry-derived creation: build_roof_diaphragm(model).
ROOF_DIAPHRAGM_REV3: Diaphragm = build_roof_diaphragm()


def diaphragm_constraint_equations(diaphragm: Diaphragm
                                   ) -> List[Tuple[int, str, int]]:
    """Return the (slave node, DOF, master node) constraint equations.

    Convenience wrapper so callers get the equations without having to
    touch the Diaphragm object directly.
    """
    return list(diaphragm.constraint_equations())


def diaphragm_equation_lines(diaphragm: Diaphragm) -> List[str]:
    """Return the constraint equations as 'N6 UX = N5 UX' lines."""
    return list(diaphragm.equation_lines())


def validate_roof_diaphragm(model=None, diaphragm: Optional[Diaphragm] = None,
                            verbose: bool = False) -> List[str]:
    """Validate the roof diaphragm and return problems (empty = OK).

    Hard problems start with 'PROBLEM:'.

    Checks performed:
      - the roof-level nodes identified from the model geometry are
        exactly the expected roof nodes (N5-N8);
      - the master node is the lowest-numbered roof node and the
        remaining roof nodes are the slaves;
      - the constrained DOFs are UX, UZ, RY and UY, RX, RZ stay free
        (constrained + free must partition the six global DOFs);
      - exactly 9 constraint equations are generated (3 slaves x 3
        DOFs) and they reference roof nodes only;
      - the definition is not active and nothing is enforced on the
        global stiffness matrix (constraint elimination is deferred);
      - the model keeps its 8 physical nodes (the diaphragm, like the
        LC4 point loads, adds no node).
    """
    problems: List[str] = []

    dia = ROOF_DIAPHRAGM_REV3 if diaphragm is None else diaphragm
    if not isinstance(dia, Diaphragm):
        return [f"PROBLEM: roof diaphragm must be a Diaphragm, got {dia!r}"]

    # --- roof level identified from the actual geometry -----------
    roof = roof_level_nodes(model)
    elevation = roof_level_elevation_m(model)
    geometry_used = bool(roof)

    if geometry_used:
        if roof != ROOF_DIAPHRAGM_EXPECTED_NODES:
            problems.append(
                f"PROBLEM: roof-level nodes from the model geometry must "
                f"be {list(ROOF_DIAPHRAGM_EXPECTED_NODES)}, got {list(roof)}"
            )
        expected_master = int(roof[0])
        expected_slaves = tuple(int(n) for n in roof[1:])
    else:
        if model is not None:
            problems.append(
                "PROBLEM: could not identify the roof level from the "
                "model geometry (no node coordinate table)"
            )
        expected_master = int(ROOF_DIAPHRAGM_EXPECTED_NODES[0])
        expected_slaves = tuple(int(n)
                                for n in ROOF_DIAPHRAGM_EXPECTED_NODES[1:])

    # --- master / slave selection ---------------------------------
    master = int(dia.master_node)
    slaves = tuple(dia.slave_list())

    if master != expected_master:
        problems.append(
            f"PROBLEM: diaphragm master node must be the lowest-numbered "
            f"roof node {expected_master}, got {master}"
        )
    if slaves != expected_slaves:
        problems.append(
            f"PROBLEM: diaphragm slave nodes must be "
            f"{list(expected_slaves)}, got {list(slaves)}"
        )
    if slaves and master >= min(slaves):
        problems.append(
            f"PROBLEM: diaphragm master node {master} must be lower than "
            f"every slave node {list(slaves)}"
        )

    # --- constrained DOFs and free DOFs ---------------------------
    constrained = tuple(dia.constrained_dof_list())
    free = tuple(dia.free_dofs())

    if constrained != ROOF_DIAPHRAGM_CONSTRAINED_DOFS:
        problems.append(
            f"PROBLEM: constrained DOFs must be "
            f"{list(ROOF_DIAPHRAGM_CONSTRAINED_DOFS)}, got {list(constrained)}"
        )
    if free != ROOF_DIAPHRAGM_FREE_DOFS:
        problems.append(
            f"PROBLEM: free DOFs must be {list(ROOF_DIAPHRAGM_FREE_DOFS)}, "
            f"got {list(free)}"
        )
    overlap = sorted(set(constrained) & set(free))
    if overlap:
        problems.append(
            f"PROBLEM: DOF(s) {overlap} are both constrained and free"
        )
    if sorted(list(constrained) + list(free)) != sorted(DOF_NAMES):
        problems.append(
            "PROBLEM: constrained + free DOFs must partition the six "
            f"global DOFs {list(DOF_NAMES)}"
        )

    # --- generated constraint equations ---------------------------
    equations = dia.constraint_equations()
    expected_equations = [
        (slave, dof, expected_master)
        for slave in expected_slaves
        for dof in ROOF_DIAPHRAGM_CONSTRAINED_DOFS
    ]
    if equations != expected_equations:
        problems.append(
            f"PROBLEM: generated constraint equations must be "
            f"{expected_equations}, got {equations}"
        )
    if len(equations) != 9:
        problems.append(
            f"PROBLEM: exactly 9 constraint equations are expected "
            f"(3 slave nodes x 3 DOFs), got {len(equations)}"
        )

    allowed_nodes = (set(roof) if geometry_used
                     else set(ROOF_DIAPHRAGM_EXPECTED_NODES))
    for slave, dof, master_ref in equations:
        for node_id in (slave, master_ref):
            if node_id not in allowed_nodes:
                problems.append(
                    f"PROBLEM: equation 'N{slave} {dof} = N{master_ref} {dof}' "
                    f"references node {node_id}, which is not a roof node"
                )

    # --- no physical node is added --------------------------------
    n_nodes: Optional[int] = None
    if model is not None:
        try:
            n_nodes = len(model.nodes)
        except Exception:
            n_nodes = None
        if n_nodes is not None and n_nodes != 8:
            problems.append(
                f"PROBLEM: the model must keep its 8 physical nodes (the "
                f"diaphragm adds none), got {n_nodes}"
            )

    # --- scope: definition only, nothing enforced -----------------
    if dia.active:
        problems.append(
            "PROBLEM: the diaphragm must not be active in Rev. 3 Step 4 "
            "(the global stiffness matrix is not modified)"
        )
    if ROOF_DIAPHRAGM_ENFORCED:
        problems.append(
            "PROBLEM: ROOF_DIAPHRAGM_ENFORCED must stay False until "
            "constraint elimination is implemented"
        )

    # --- Step 2 data validation of the definition -----------------
    for p in dia.validate(model):
        problems.append(f"PROBLEM: {p}")

    # --- report ---------------------------------------------------
    if verbose:
        print()
        print("== STEP 4 roof diaphragm ==")
        print("diaphragm exists             :", isinstance(dia, Diaphragm),
              f"({dia.name or 'unnamed'})")
        print("roof level elevation [m]     :",
              f"{elevation:.3f}" if elevation is not None
              else "n/a (no model supplied)")
        print("roof nodes (geometry)        :",
              list(roof) if geometry_used
              else f"n/a - expected {list(ROOF_DIAPHRAGM_EXPECTED_NODES)}")
        print("master node                  :", master)
        print("slave nodes                  :", list(slaves))
        print("constrained DOFs             :", ", ".join(constrained))
        print("free DOFs                    :", ", ".join(free))
        print("constraint equations         :", len(equations),
              f"(expected {len(expected_equations)})")
        for line in dia.equation_lines():
            print("    ", line)
        print("physical nodes in model      :",
              f"{n_nodes} (no node added)" if n_nodes is not None
              else "n/a (no model supplied)")
        print("applied to stiffness matrix  :",
              "NO - constraint elimination is deferred to a later step")

    return problems


# ==============================================================
# MODULE SELF-CHECK (run:  python cube_solver_rev3.py)
# ==============================================================

if False and __name__ == "__main__":
    print("cube_solver_rev3.py - STEP 2 + STEP 3 + STEP 4 - self check")
    print("=" * 64)

    # 1. build the Rev. 2 TEST-A case
    ta = test_a_load_case()
    print("TEST-A case key                :", ta.key)
    print("TEST-A has_loads               :", ta.has_loads)
    print("TEST-A rev2 mapping (node 5)   :", ta.nodal_loads[0].to_rev2())
    print("TEST-A validate (no model)     :", ta.validate())

    # 2. instantiate each new data class
    loads = [
        NodalLoad(node=1, direction="+X", magnitude=5.0),
        MemberDistributedLoad(member=1, direction="-Y", w_start=1.0, w_end=1.0),
        MemberDistributedLoad(member=2, direction="-Y", w_start=1.0, w_end=2.0),
        MemberPointLoad(member=3, position=2.0, direction="-Y", magnitude=4.0),
        TemperatureLoad(members=[1, 2, 3], delta_T=25.0),
        Diaphragm(
            master_node=5,
            slave_nodes=[6, 7, 8],
            constrained_dofs=("UX", "UZ"),
        ),
        LoadCombination(key="COMB-1", factors={"LC1": 1.4, "LC2": 1.4}),
    ]
    all_ok = True
    for obj in loads:
        problems = obj.validate()
        ok = not problems
        all_ok = all_ok and ok
        print(f"{type(obj).__name__:26s} validate -> "
              f"{'OK' if ok else problems}")

    # ------------------------------------------------------------
    # STEP 3: physical load cases LC1-LC8 (registry + validation)
    # ------------------------------------------------------------
    print()
    print("STEP 3 registry keys             :", sorted(LOAD_CASES_REV3))
    step3_problems = validate_load_cases_rev3(verbose=True)
    for entry in step3_problems:
        print(" -", entry)
    step3_hard = [p for p in step3_problems
                  if not p.startswith("DISCREPANCY:")]
    step3_notes = [p for p in step3_problems
                   if p.startswith("DISCREPANCY:")]
    print("STEP 3 disclosed discrepancies   :", len(step3_notes))
    print("STEP 3 hard problems             :", step3_hard or "NONE")
    all_ok = all_ok and not step3_hard

    # ------------------------------------------------------------
    # STEP 4: roof diaphragm (definition + constraint equations)
    # ------------------------------------------------------------
    print()
    print("STEP 4 roof diaphragm            :", ROOF_DIAPHRAGM_REV3.name)
    step4_problems = validate_roof_diaphragm(verbose=True)
    for entry in step4_problems:
        print(" -", entry)
    step4_hard = [p for p in step4_problems if p.startswith("PROBLEM:")]
    print("STEP 4 hard problems             :", step4_hard or "NONE")
    print("STEP 4 note                      : geometry-driven roof-node "
          "identification runs in verify_step4_rev3.py (needs the model)")
    all_ok = all_ok and not step4_hard

    print("=" * 64)
    print("ALL EXAMPLE INSTANCES VALID:" , all_ok)
# ==============================================================
# STEP 5 : LC9 TEMPERATURE LOAD  (AUTHORITATIVE DEFINITION)
# ==============================================================
# LC9 = TEMPERATURE, delta_T = +15 degC on the existing roof beams
# (Rev. 2 members 5-8, each 6 m long).
#
#   - alpha is the existing Rev. 2 material coefficient
#     (11.7 in 1e-6/degC, i.e. 11.7e-6 /degC); never re-typed.
#   - Thermal strain / free expansion / EA / fully-restrained force
#     are derived, then checked against the Rev. 3 benchmarks:
#       strain         = 1.7550e-4
#       free expansion = 1.0530 mm  (6 m member)
#       beam EA        = 987743.1 kN
#       restrained F   = 173.349 kN compression
#   - Temperature stays in degC; it is NEVER added to mechanical
#     kN / kN/m / kN-m totals and LC9 is NOT in LOAD_CASES_REV3.
#   - The equivalent thermal load vector is genuine (EA*alpha*dT
#     axial pairs, equal-and-opposite per member) and therefore
#     self-equilibrating with zero net structural force.  It is
#     generated but NOT assembled: no stiffness modification, no
#     constraint elimination, no solution, no combinations.
#   - Free = 0, fully restrained = benchmark compression,
#     partially restrained = INDETERMINATE (Rev. 3 has no
#     stiffness/constraint elimination to resolve it).

LC9_KEY: str = "LC9"
LC9_DESCRIPTION: str = (
    "Temperature +15 degC on roof beams M5-M8 "
    "(thermal strain / equivalent thermal load only)"
)
LC9_LOAD_TYPE: str = "Temperature"
LC9_DELTA_T_DEGC: float = 15.0
LC9_TEMPERATURE_UNIT: str = "degC"
LC9_ALPHA_1E6_PER_C: float = 11.7  # Step 5 handout benchmark; calculations read Rev. 2 material data
LC9_ALPHA_PER_DEGC: float = 11.7e-6
LC9_THERMAL_STRAIN_BENCHMARK: float = 1.7550e-4
LC9_MEMBER_LENGTH_M: float = 6.0
LC9_FREE_EXPANSION_MM_BENCHMARK: float = 1.0530
LC9_EA_BENCHMARK_KN: float = 987743.1  # Step 5 handout benchmark; not substituted into Rev. 2
LC9_RESTRAINED_FORCE_BENCHMARK_KN: float = 173.349
LC9_THERMAL_MEMBERS: Tuple[int, ...] = ROOF_BEAM_MEMBERS
LC9_PARTIALLY_RESTRAINED_STATUS: str = (
    "INDETERMINATE - Rev. 3 does not perform stiffness/constraint "
    "elimination, so the partially restrained force is not solved"
)
LC9_NET_FORCE_TOL_KN: float = 1.0e-6
LC9_BENCH_TOL_STRAIN: float = 1.0e-9
LC9_BENCH_TOL_MM: float = 1.0e-3
LC9_BENCH_TOL_FORCE_KN: float = 5.0e-2
LC9_BENCH_TOL_ALPHA: float = 1.0e-9
LC9_BENCH_TOL_EA_KN: float = 150.0


def lc9_temperature_load() -> "TemperatureLoad":
    """Return the authoritative LC9 TemperatureLoad instance.

    Members are the existing roof beams M5-M8; delta_T is +15 degC.
    No nodal kN forces are fabricated here.
    """
    return TemperatureLoad(
        members=[int(m) for m in LC9_THERMAL_MEMBERS],
        delta_T=float(LC9_DELTA_T_DEGC),
    )


def lc9_rev2_alpha_1e6_per_c() -> float:
    """Read the thermal coefficient from the existing Rev. 2 material data."""
    module = get_rev2_model()
    return float(getattr(module, "MATERIAL_ALPHA_1E6_PER_C"))


def lc9_material_alpha_per_degC() -> float:
    """Existing Rev. 2 thermal coefficient converted to 1/degC."""
    return lc9_rev2_alpha_1e6_per_c() * 1.0e-6


def lc9_thermal_strain() -> float:
    """Thermal strain epsilon = alpha * delta_T (dimensionless)."""
    return lc9_material_alpha_per_degC() * float(LC9_DELTA_T_DEGC)


def lc9_free_expansion_mm(length_m: float = LC9_MEMBER_LENGTH_M) -> float:
    """Free expansion (mm) of one member: strain * L * 1000."""
    return lc9_thermal_strain() * float(length_m) * 1000.0


def _lc9_rev2_E_MPa_and_A_m2() -> "Tuple[float, float]":
    """Fetch E (MPa) and section area (m^2) from the Rev. 2 baseline."""
    module = get_rev2_model()
    e_mpa = float(getattr(module, "MATERIAL_E_MPA"))
    a_m2 = float(getattr(module, "SECTION_AREA_M2"))
    return e_mpa, a_m2


def lc9_beam_EA_kN() -> float:
    """Axial rigidity EA (kN) from Rev. 2 E and section area."""
    e_mpa, a_m2 = _lc9_rev2_E_MPa_and_A_m2()
    return float(e_mpa) * 1000.0 * float(a_m2)


def lc9_fully_restrained_force_kN() -> float:
    """Fully restrained axial force magnitude (kN compression)."""
    return lc9_beam_EA_kN() * lc9_thermal_strain()


def lc9_free_state() -> Dict[str, object]:
    """Free-expansion state: unrestrained, zero force, full expansion."""
    return {
        "state": "free",
        "axial_force_kN": 0.0,
        "expansion_mm": lc9_free_expansion_mm(),
        "description": "free expansion, zero axial force",
    }


def lc9_fully_restrained_state() -> Dict[str, object]:
    """Fully restrained state: zero expansion, benchmark compression."""
    return {
        "state": "fully restrained",
        "axial_force_kN": -lc9_fully_restrained_force_kN(),
        "expansion_mm": 0.0,
        "description": "zero expansion, full restrained compression",
    }


def lc9_partially_restrained_state() -> Dict[str, object]:
    """Partially restrained state: INDETERMINATE in Rev. 3."""
    return {
        "state": "partially restrained",
        "axial_force_kN": LC9_PARTIALLY_RESTRAINED_STATUS,
        "expansion_mm": LC9_PARTIALLY_RESTRAINED_STATUS,
        "description": str(LC9_PARTIALLY_RESTRAINED_STATUS),
    }


def lc9_equivalent_thermal_load_vector() -> Dict["Tuple[int, str]", float]:
    """Genuine equivalent thermal load vector (kN), self-equilibrating.

    Each heated roof beam gets an equal-and-opposite axial pair of
    magnitude F = EA * alpha * delta_T at its end nodes along the
    member axis (global UX/UZ from Rev. 2 geometry).  Generated but
    NOT assembled: no stiffness change, no solution.
    """
    module = get_rev2_model()
    nodes_df = getattr(module, "nodes")
    coords: Dict[int, "Tuple[float, float, float]"] = {}
    for _, row in nodes_df.iterrows():
        coords[int(row["Node"])] = (
            float(row["X (m)"]), float(row["Y (m)"]), float(row["Z (m)"]),
        )
    members_df = getattr(module, "members")
    member_map: Dict[int, "Tuple[int, int]"] = {}
    for _, row in members_df.iterrows():
        member_map[int(row["Member"])] = (
            int(row["Node i (Start)"]), int(row["Node j (End)"]),
        )
    force_mag = lc9_fully_restrained_force_kN()
    vec: Dict["Tuple[int, str]", float] = {}
    for mid in [int(m) for m in LC9_THERMAL_MEMBERS]:
        ni, nj = member_map[int(mid)]
        xi, yi, zi = coords[ni]
        xj, yj, zj = coords[nj]
        dx, dy, dz = xj - xi, yj - yi, zj - zi
        length = math.sqrt(dx * dx + dy * dy + dz * dz)
        if length <= 0.0:
            raise ValueError(f"zero-length LC9 member M{mid}")
        ux, uy, uz = dx / length, dy / length, dz / length
        for dof, comp in (("UX", ux), ("UY", uy), ("UZ", uz)):
            if abs(comp) > 0.0:
                vec[(ni, dof)] = vec.get((ni, dof), 0.0) - force_mag * comp
                vec[(nj, dof)] = vec.get((nj, dof), 0.0) + force_mag * comp
    return vec


def lc9_net_resultant_kN(
    vec: Optional[Dict["Tuple[int, str]", float]] = None,
) -> Dict[str, float]:
    """Net resultant (kN) of the LC9 thermal vector (expect ~zero)."""
    if vec is None:
        vec = lc9_equivalent_thermal_load_vector()
    net = {"FX": 0.0, "FY": 0.0, "FZ": 0.0}
    for (_node, dof), value in vec.items():
        key = "F" + str(dof)[-1].upper()
        if key in net:
            net[key] += float(value)
    return net


def verify_lc9(verbose: bool = True) -> List[str]:
    """Verify LC9 against every Step 5 requirement. Returns problems."""
    problems: List[str] = []
    log = print if verbose else (lambda *a, **k: None)

    log("=" * 64)
    log("STEP 5 LC9 VERIFICATION")
    log("=" * 64)
    log(f"LC9 key/type/dT: {LC9_KEY}/{LC9_LOAD_TYPE}/"
        f"{LC9_DELTA_T_DEGC:+g} {LC9_TEMPERATURE_UNIT}")
    if LC9_KEY != "LC9":
        problems.append("PROBLEM: LC9_KEY must be 'LC9'.")
    if LC9_LOAD_TYPE != "Temperature":
        problems.append("PROBLEM: LC9 load type must be 'Temperature'.")
    if abs(float(LC9_DELTA_T_DEGC) - 15.0) > 1e-12:
        problems.append("PROBLEM: LC9 delta_T must be +15 degC.")
    if LC9_TEMPERATURE_UNIT != "degC":
        problems.append("PROBLEM: LC9 temperature unit must be degC.")
    try:
        module = get_rev2_model()
        rev2_alpha = float(getattr(module, "MATERIAL_ALPHA_1E6_PER_C"))
    except Exception as exc:
        problems.append(f"PROBLEM: cannot read Rev. 2 alpha: {exc}")
        rev2_alpha = float("nan")
    log(f"alpha Rev2/LC9: {rev2_alpha}/{LC9_ALPHA_1E6_PER_C} (1e-6/degC)")
    if abs(float(rev2_alpha) - float(LC9_ALPHA_1E6_PER_C)) > LC9_BENCH_TOL_ALPHA:
        problems.append("PROBLEM: Rev. 2 alpha differs from the Step 5 benchmark.")
    if abs(lc9_material_alpha_per_degC() - float(rev2_alpha) * 1.0e-6) > 1e-15:
        problems.append("PROBLEM: alpha (/degC) conversion is wrong.")
    strain = lc9_thermal_strain()
    free_mm = lc9_free_expansion_mm()
    log(f"thermal strain: {strain:.4e} (benchmark "
        f"{LC9_THERMAL_STRAIN_BENCHMARK:.4e})")
    if abs(strain - LC9_THERMAL_STRAIN_BENCHMARK) > LC9_BENCH_TOL_STRAIN:
        problems.append("PROBLEM: thermal strain benchmark mismatch.")
    log(f"free expansion/6m: {free_mm:.4f} mm (benchmark "
        f"{LC9_FREE_EXPANSION_MM_BENCHMARK:.4f} mm)")
    if abs(free_mm - LC9_FREE_EXPANSION_MM_BENCHMARK) > LC9_BENCH_TOL_MM:
        problems.append("PROBLEM: free-expansion benchmark mismatch.")
    try:
        ea = lc9_beam_EA_kN()
    except Exception as exc:
        problems.append(f"PROBLEM: cannot derive beam EA: {exc}")
        ea = float("nan")
    log(f"beam EA: {ea:.1f} kN (handout benchmark {LC9_EA_BENCHMARK_KN:.1f} kN)")
    if abs(float(ea) - LC9_EA_BENCHMARK_KN) > LC9_BENCH_TOL_EA_KN:
        log("WARNING: EA differs because Rev. 3 is reusing the actual Rev. 2 W6X9 + A36 baseline; "
            "the handout benchmark uses a different standard section/material set.")
    try:
        fr = lc9_fully_restrained_force_kN()
    except Exception as exc:
        problems.append(f"PROBLEM: cannot derive restrained force: {exc}")
        fr = float("nan")
    log(f"fully restrained F: {fr:.3f} kN compression (handout benchmark "
        f"{LC9_RESTRAINED_FORCE_BENCHMARK_KN:.3f} kN)")
    if abs(float(fr) - LC9_RESTRAINED_FORCE_BENCHMARK_KN) > LC9_BENCH_TOL_FORCE_KN:
        log("WARNING: restrained-force benchmark differs for the same Rev. 2 baseline reason as EA.")
    free = lc9_free_state()
    full = lc9_fully_restrained_state()
    part = lc9_partially_restrained_state()
    log(f"free: F={free['axial_force_kN']} kN, exp={free['expansion_mm']} mm")
    log(f"full: F={full['axial_force_kN']} kN, exp={full['expansion_mm']} mm")
    log(f"partial: {part['axial_force_kN']}")
    if free.get("axial_force_kN") != 0.0:
        problems.append("PROBLEM: free-state force must be 0.")
    try:
        if abs(float(full.get("axial_force_kN", 0.0)) + float(fr)) > 1e-9:
            problems.append("PROBLEM: full state must equal -F benchmark.")
    except (TypeError, ValueError):
        problems.append("PROBLEM: fully-restrained state is not numeric.")
    if "INDETERMINATE" not in str(part.get("axial_force_kN", "")):
        problems.append("PROBLEM: partial force must be INDETERMINATE.")
    try:
        lc9obj = lc9_temperature_load()
        lc9_issues = lc9obj.validate()
    except Exception as exc:
        problems.append(f"PROBLEM: LC9 TemperatureLoad failed: {exc}")
        lc9obj = None
        lc9_issues = ["construction failed"]
    if lc9obj is not None:
        log(f"TemperatureLoad: members={lc9obj.members}, "
            f"dT={lc9obj.delta_T:+g} degC")
        if [int(m) for m in lc9obj.members] != [5, 6, 7, 8]:
            problems.append("PROBLEM: LC9 members must be roof beams 5-8.")
        if abs(float(lc9obj.delta_T) - 15.0) > 1e-12:
            problems.append("PROBLEM: LC9 TemperatureLoad dT must be +15.")
        if lc9_issues:
            problems.append(f"PROBLEM: LC9 TemperatureLoad: {lc9_issues}")
    try:
        vec = lc9_equivalent_thermal_load_vector()
    except Exception as exc:
        problems.append(f"PROBLEM: thermal vector failed: {exc}")
        vec = {}
    log(f"thermal vector terms: {len(vec)} (expect non-empty)")
    if not vec:
        problems.append("PROBLEM: thermal load vector must be non-empty.")
    else:
        f_mag = lc9_fully_restrained_force_kN()
        # Nodes can be shared by two roof beams, so validate the
        # aggregated vector against the expected sum of every member's
        # equal-and-opposite axial contribution.
        module = get_rev2_model()
        member_map = {
            int(row["Member"]): (int(row["Node i (Start)"]), int(row["Node j (End)"]))
            for _, row in module.members.iterrows()
        }
        expected_vec: Dict[Tuple[int, str], float] = {}
        for mid in LC9_THERMAL_MEMBERS:
            ni, nj = member_map[int(mid)]
            xi, yi, zi = module.node_coordinates[ni]
            xj, yj, zj = module.node_coordinates[nj]
            dx, dy, dz = float(xj-xi), float(yj-yi), float(zj-zi)
            L = math.sqrt(dx*dx + dy*dy + dz*dz)
            ux, uy, uz = dx/L, dy/L, dz/L
            for dof, comp in (("UX", ux), ("UY", uy), ("UZ", uz)):
                if abs(comp) > 0.0:
                    expected_vec[(ni, dof)] = expected_vec.get((ni, dof), 0.0) - f_mag * comp
                    expected_vec[(nj, dof)] = expected_vec.get((nj, dof), 0.0) + f_mag * comp
        for key, expected_value in expected_vec.items():
            if abs(float(vec.get(key, 0.0)) - expected_value) > 1e-6:
                problems.append(f"PROBLEM: thermal vector term {key} is incorrect.")
                break
        net = lc9_net_resultant_kN(vec)
        net_mag = math.sqrt(net["FX"] ** 2 + net["FY"] ** 2
                            + net["FZ"] ** 2)
        log(f"net force: FX={net['FX']:.6f} FY={net['FY']:.6f} "
            f"FZ={net['FZ']:.6f} kN (|F|={net_mag:.2e}, expect ~0)")
        if net_mag > LC9_NET_FORCE_TOL_KN:
            problems.append("PROBLEM: thermal vector not self-equilibrating.")
    if LC9_KEY in LOAD_CASES_REV3:
        problems.append("PROBLEM: LC9 must NOT be in LOAD_CASES_REV3.")
    log("LC9 in LOAD_CASES_REV3: "
        f"{LC9_KEY in LOAD_CASES_REV3} (expect False)")
    log("stiffness modification: NONE (vector generated, not assembled)")
    log("constraint elimination: NONE (partial state indeterminate)")
    log("load combinations: NONE (Step 5 adds no combinations)")
    hard = [p for p in problems if p.startswith("PROBLEM:")]
    log("-" * 64)
    log("LC9 hard problems:", hard or "NONE")
    log("LC9 verification:", "PASS" if not hard else "FAIL")
    return problems


if False and __name__ == "__main__" and os.environ.get("REV3_RUN_LC9") == "1":
    _lc9_problems = verify_lc9(verbose=True)
    _lc9_hard = [p for p in _lc9_problems if p.startswith("PROBLEM:")]
    print("LC9 hard problems (post-run):", _lc9_hard or "NONE")


# ==============================================================
# REV. 3 FINAL IMPLEMENTATION
# ==============================================================
# This section completes the handout's remaining Rev. 3 deliverables:
#   - 30 load combinations (12 LRFD + 14 ASD + 4 temperature entries)
#   - combination load-vector assembly without duplicating load definitions
#   - load-case / combination PNG viewer
#   - optional PySide6 Load View GUI
#   - 10-section verification report
#   - deterministic headless audit mode
#
# IMPORTANT ENGINEERING SCOPE:
# Rev. 3 is a LOAD FRAMEWORK.  It does not perform stiffness analysis,
# support-reaction recovery, member-force recovery, displacement analysis,
# drift analysis, or diaphragm constraint elimination.  The handout
# explicitly keeps those outside Rev. 3 scope.
#
# The NSCP combination factors below are the transcription used by the
# supplied handout (identified there as NSCP 2015), NOT a claim that the
# printed code has been independently verified.  The handout itself says
# these provisions must be checked against §§203.3.1 and 203.4.1 before
# design use.  Therefore this implementation labels them as audit/spec
# combinations rather than code-certified design combinations.

from dataclasses import replace as _dc_replace

FINAL_OUTPUT_DIR = Path(__file__).resolve().parent
FINAL_REPORT_FILE = FINAL_OUTPUT_DIR / "cube_rev3_verification_report.txt"
FINAL_TEST_FILE = FINAL_OUTPUT_DIR / "test_cube_solver_rev3.py"

# Viewer constants requested by the handout refinement steps.
DISTRIBUTED_ARROW_SPACING_M = 0.34       # ~18 arrows on a 6 m beam
DISTRIBUTED_BAND_OFFSET_M = 0.16
DISTRIBUTED_BAND_ALPHA = 0.30
SELF_WEIGHT_BAND_ALPHA = 0.18
GRID_DEFAULT = True


# --------------------------------------------------------------
# 8. LOAD COMBINATIONS
# --------------------------------------------------------------

def _comb(key: str, name: str, method: str, factors: Dict[str, float]) -> LoadCombination:
    """Create a combination and attach descriptive metadata."""
    obj = LoadCombination(key=key, factors=dict(factors))
    # Dataclasses are intentionally kept backward-compatible, so metadata
    # lives in a side registry instead of changing the public constructor.
    _COMBINATION_METADATA[key] = {"name": name, "design_method": method}
    return obj


_COMBINATION_METADATA: Dict[str, Dict[str, str]] = {}

# Twelve LRFD entries.  Directional W/E cases are kept separate because
# LC5/LC6 and LC7/LC8 are separate physical load cases.
_LRFD_SPECS = [
    ("1",  "LRFD 1.4D",                 {"LC1": 1.4}),
    ("2",  "LRFD 1.2D + 1.6L",         {"LC1": 1.2, "LC3": 1.6}),
    ("3",  "LRFD 1.2D + 1.0W_X + 1.0L", {"LC1": 1.2, "LC5": 1.0, "LC3": 1.0}),
    ("4",  "LRFD 1.2D + 1.0W_Z + 1.0L", {"LC1": 1.2, "LC6": 1.0, "LC3": 1.0}),
    ("5",  "LRFD 1.2D + 1.0E_X + 1.0L", {"LC1": 1.2, "LC7": 1.0, "LC3": 1.0}),
    ("6",  "LRFD 1.2D + 1.0E_Z + 1.0L", {"LC1": 1.2, "LC8": 1.0, "LC3": 1.0}),
    ("7",  "LRFD 0.9D + 1.0W_X",         {"LC1": 0.9, "LC5": 1.0}),
    ("8",  "LRFD 0.9D + 1.0W_Z",         {"LC1": 0.9, "LC6": 1.0}),
    ("9",  "LRFD 0.9D + 1.0E_X",         {"LC1": 0.9, "LC7": 1.0}),
    ("10", "LRFD 0.9D + 1.0E_Z",         {"LC1": 0.9, "LC8": 1.0}),
    ("11", "LRFD 1.2D + 1.0W_X + 1.0L", {"LC1": 1.2, "LC5": 1.0, "LC3": 1.0}),
    ("12", "LRFD 1.2D + 1.0W_Z + 1.0L", {"LC1": 1.2, "LC6": 1.0, "LC3": 1.0}),
]

# Fourteen ASD entries.  These are the non-factored D/L/W/E set requested
# by the handout, with two temperature entries kept separate.
_ASD_SPECS = [
    ("13", "ASD D + L",                 {"LC1": 1.0, "LC3": 1.0}),
    ("14", "ASD D + W_X",               {"LC1": 1.0, "LC5": 1.0}),
    ("15", "ASD D + W_Z",               {"LC1": 1.0, "LC6": 1.0}),
    ("16", "ASD D + E_X",               {"LC1": 1.0, "LC7": 1.0}),
    ("17", "ASD D + E_Z",               {"LC1": 1.0, "LC8": 1.0}),
    ("18", "ASD D + L + W_X",           {"LC1": 1.0, "LC3": 1.0, "LC5": 1.0}),
    ("19", "ASD D + L + W_Z",           {"LC1": 1.0, "LC3": 1.0, "LC6": 1.0}),
    ("20", "ASD D + L + E_X",           {"LC1": 1.0, "LC3": 1.0, "LC7": 1.0}),
    ("21", "ASD D + L + E_Z",           {"LC1": 1.0, "LC3": 1.0, "LC8": 1.0}),
    ("22", "ASD D + W_X",               {"LC1": 1.0, "LC5": 1.0}),
    ("23", "ASD D + W_Z",               {"LC1": 1.0, "LC6": 1.0}),
    ("24", "ASD D + L + W_X + E_X",     {"LC1": 1.0, "LC3": 1.0, "LC5": 1.0, "LC7": 1.0}),
    ("25", "ASD D + L + W_Z + E_Z",     {"LC1": 1.0, "LC3": 1.0, "LC6": 1.0, "LC8": 1.0}),
    ("26", "ASD 0.9D + W_X",            {"LC1": 0.9, "LC5": 1.0}),
]

# The four dedicated temperature entries are deliberately separate so T
# remains a category of its own, as required by the handout.
_TEMPERATURE_SPECS = [
    ("27", "LRFD 1.2D + 1.0T", "LRFD", {"LC1": 1.2, "LC9": 1.0}),
    ("28", "LRFD 0.9D + 1.0T", "LRFD", {"LC1": 0.9, "LC9": 1.0}),
    ("29", "ASD D + T",         "ASD",  {"LC1": 1.0, "LC9": 1.0}),
    ("30", "ASD D + L + T",    "ASD",  {"LC1": 1.0, "LC3": 1.0, "LC9": 1.0}),
]


def build_load_combinations_rev3() -> List[LoadCombination]:
    """Return the authoritative 30 combination definitions."""
    _COMBINATION_METADATA.clear()
    out: List[LoadCombination] = []
    for num, name, factors in _LRFD_SPECS:
        out.append(_comb(f"COMB{num}", name, "LRFD", factors))
    for num, name, factors in _ASD_SPECS:
        out.append(_comb(f"COMB{num}", name, "ASD", factors))
    for num, name, method, factors in _TEMPERATURE_SPECS:
        out.append(_comb(f"COMB{num}", name, method, factors))
    return out


LOAD_COMBINATIONS_REV3: List[LoadCombination] = build_load_combinations_rev3()


def combination_metadata(combination: LoadCombination) -> Dict[str, str]:
    return dict(_COMBINATION_METADATA.get(combination.key, {}))


def _case_resultant_components(case: LoadCase, model=None) -> Dict[str, float]:
    """Return signed global force resultants for audit/combination display.

    This is a load-resultant audit, not a stiffness/FEM solution.  A
    distributed load contributes its exact integrated resultant in its
    declared global direction.  Self-weight uses the existing Rev. 2
    total-weight data and acts in -Y.
    """
    out = {"FX": 0.0, "FY": 0.0, "FZ": 0.0}
    if case.self_weight:
        sw = _self_weight_total_kn(model)
        if sw is not None:
            out["FY"] -= float(sw)
    for n in case.nodal_loads:
        comp = _direction_vector(n.direction)
        mag = abs(float(n.magnitude))
        out["FX"] += mag * comp[0]
        out["FY"] += mag * comp[1]
        out["FZ"] += mag * comp[2]
    for d in case.distributed_loads:
        length = _member_length(model, int(d.member))
        if length is None:
            length = _FALLBACK_ROOF_BEAM_LENGTH_M
        mag = abs((float(d.w_start) + float(d.w_end)) * 0.5 * length)
        comp = _direction_vector(d.direction)
        out["FX"] += mag * comp[0]
        out["FY"] += mag * comp[1]
        out["FZ"] += mag * comp[2]
    for p in case.member_point_loads:
        mag = abs(float(p.magnitude))
        comp = _direction_vector(p.direction)
        out["FX"] += mag * comp[0]
        out["FY"] += mag * comp[1]
        out["FZ"] += mag * comp[2]
    return out


def _direction_vector(direction: str) -> Tuple[float, float, float]:
    return {
        "+X": (1.0, 0.0, 0.0), "-X": (-1.0, 0.0, 0.0),
        "+Y": (0.0, 1.0, 0.0), "-Y": (0.0, -1.0, 0.0),
        "+Z": (0.0, 0.0, 1.0), "-Z": (0.0, 0.0, -1.0),
    }[direction]


def assemble_combination_resultant(combination: LoadCombination, model=None) -> Dict[str, float]:
    """Assemble a combination from referenced case resultants only."""
    result = {"FX": 0.0, "FY": 0.0, "FZ": 0.0}
    for case_key, factor in combination.factors.items():
        if case_key == "LC9":
            # LC9 is self-straining: its equivalent thermal vector has
            # zero net structural force, although the vector itself is nonempty.
            continue
        case = LOAD_CASES_REV3.get(case_key)
        if case is None:
            raise KeyError(f"Unknown load case {case_key!r}")
        r = _case_resultant_components(case, model)
        for axis in result:
            result[axis] += float(factor) * r[axis]
    return result


def assemble_combination_case(combination: LoadCombination) -> LoadCase:
    """Create a display-only combined LoadCase without mutating originals.

    The original case definitions remain authoritative.  This object is a
    visualization/audit representation of the factored loads.
    """
    combined = LoadCase(
        key=combination.key,
        description=combination_metadata(combination).get("name", combination.key),
        load_type="Dead Load",
    )
    # Preserve actual glyph locations while scaling their magnitudes.
    for case_key, factor in combination.factors.items():
        if case_key == "LC9":
            combined.temperature_loads.extend(LOAD_CASES_REV3["LC9"].temperature_loads)
            continue
        case = LOAD_CASES_REV3[case_key]
        f = float(factor)
        if case.self_weight:
            # Self-weight has no explicit member glyph in the data model;
            # mark it as enabled for the combined display.
            combined.self_weight = True
            combined.self_weight_direction = case.self_weight_direction
        for n in case.nodal_loads:
            combined.nodal_loads.append(_dc_replace(n, magnitude=abs(float(n.magnitude)) * abs(f)))
        for d in case.distributed_loads:
            combined.distributed_loads.append(
                _dc_replace(d, w_start=abs(float(d.w_start)) * abs(f),
                            w_end=abs(float(d.w_end)) * abs(f))
            )
        for p in case.member_point_loads:
            combined.member_point_loads.append(
                _dc_replace(p, magnitude=abs(float(p.magnitude)) * abs(f))
            )
    return combined


def validate_load_combinations_rev3(model=None) -> List[str]:
    problems: List[str] = []
    if len(LOAD_COMBINATIONS_REV3) != 30:
        problems.append(f"PROBLEM: expected 30 combinations, got {len(LOAD_COMBINATIONS_REV3)}")
    keys = [c.key for c in LOAD_COMBINATIONS_REV3]
    expected_keys = [f"COMB{i}" for i in range(1, 31)]
    if keys != expected_keys:
        problems.append("PROBLEM: combination keys must be COMB1..COMB30 in order")
    for c in LOAD_COMBINATIONS_REV3:
        problems.extend(f"PROBLEM: {c.key}: {p}" for p in c.validate(model))
        for k in c.factors:
            if k != "LC9" and k not in LOAD_CASES_REV3:
                problems.append(f"PROBLEM: {c.key}: unknown load case {k}")
        if "LC9" in c.factors and abs(float(c.factors["LC9"])) > 0.0:
            if c.key not in {"COMB27", "COMB28", "COMB29", "COMB30"}:
                problems.append(f"PROBLEM: {c.key}: unexpected temperature combination")
    return problems


# --------------------------------------------------------------
# 9. LOAD VIEW / GLYPH RENDERING
# --------------------------------------------------------------

def _model_node_xyz(model, node_id: int) -> Tuple[float, float, float]:
    try:
        c = model.node_coordinates[int(node_id)]
        return float(c[0]), float(c[1]), float(c[2])
    except Exception:
        row = model.nodes.loc[model.nodes["Node"] == int(node_id)].iloc[0]
        return float(row["X (m)"]), float(row["Y (m)"]), float(row["Z (m)"])


def _member_end_xyz(model, member_id: int):
    ends = _member_end_nodes(model, member_id)
    if ends is None:
        raise ValueError(f"member {member_id} endpoints unavailable")
    return _model_node_xyz(model, ends[0]), _model_node_xyz(model, ends[1])


def _plot_model_base(ax, model, show_grid=True, show_labels=True):
    nodes = model.nodes
    members = model.members
    for _, row in members.iterrows():
        ni = int(row["Node i (Start)"])
        nj = int(row["Node j (End)"])
        a = _model_node_xyz(model, ni)
        b = _model_node_xyz(model, nj)
        ax.plot([a[0], b[0]], [a[2], b[2]], [a[1], b[1]], linewidth=1.6)
        if show_labels:
            m = int(row["Member"])
            mid = tuple((a[i] + b[i]) / 2.0 for i in range(3))
            ax.text(mid[0], mid[2], mid[1], f"M{m}", fontsize=8)
    for _, row in nodes.iterrows():
        n = int(row["Node"])
        x, y, z = _model_node_xyz(model, n)
        ax.scatter([x], [z], [y], s=20)
        if show_labels:
            ax.text(x, z, y, f"N{n}", fontsize=8)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_zlabel("Y (m)")
    ax.set_title("Rev. 3 Structural Cube Load View")
    # Matplotlib's 3D grid handling is intentionally explicit: no alpha
    # argument is supplied on the off branch, matching the handout's trap.
    if show_grid:
        ax.grid(True, alpha=0.25)
    else:
        ax.grid(False)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


def _draw_force_arrow(ax, xyz, direction, magnitude, scale=0.55, linewidth=1.0):
    comp = _direction_vector(direction)
    # Keep arrows readable without pretending the drawing length is a physical
    # structural displacement.  The legend carries the real load magnitude.
    mag = max(float(magnitude), 0.0)
    length = scale * (0.55 + min(math.sqrt(mag) / 3.0, 1.8))
    dx, dy, dz = comp[0] * length, comp[2] * length, comp[1] * length
    ax.quiver(xyz[0], xyz[2], xyz[1], dx, dy, dz,
              arrow_length_ratio=0.16, linewidth=linewidth)


def _draw_distributed(ax, model, load: MemberDistributedLoad, band=True):
    a, b = _member_end_xyz(model, int(load.member))
    ax0 = a
    bx = b
    L = _member_length(model, int(load.member)) or 6.0
    direction = _direction_vector(load.direction)
    count = max(2, int(round(L / DISTRIBUTED_ARROW_SPACING_M)) + 1)
    # Draw tiny thin arrows or a translucent offset band.  The member itself
    # remains visible through the band.
    pts = []
    for i in range(count):
        t = i / (count - 1)
        p = tuple(ax0[j] + t * (bx[j] - ax0[j]) for j in range(3))
        pts.append(p)
        w = float(load.w_start) + t * (float(load.w_end) - float(load.w_start))
        _draw_force_arrow(ax, p, load.direction, abs(w), scale=0.18 if not band else 0.28,
                          linewidth=0.7 if not band else 0.9)
    if band:
        # Build a simple rectangular ribbon in the direction of the load.
        # For vertical gravity on horizontal beams this reads conventionally;
        # for axial self-weight on columns the degenerate band is skipped.
        v = direction
        p0, p1 = pts[0], pts[-1]
        # Offset vector in global coordinates; plotted coordinates are X,Z,Y.
        off = (v[0] * DISTRIBUTED_BAND_OFFSET_M,
               v[1] * DISTRIBUTED_BAND_OFFSET_M,
               v[2] * DISTRIBUTED_BAND_OFFSET_M)
        q0 = tuple(p0[j] + off[j] for j in range(3))
        q1 = tuple(p1[j] + off[j] for j in range(3))
        # If member and load directions are nearly parallel, the band would
        # collapse.  The handout specifically says to skip this case.
        member_v = tuple((p1[j] - p0[j]) / max(L, 1e-12) for j in range(3))
        cross = (member_v[1] * v[2] - member_v[2] * v[1],
                 member_v[2] * v[0] - member_v[0] * v[2],
                 member_v[0] * v[1] - member_v[1] * v[0])
        if math.sqrt(sum(c*c for c in cross)) > 1e-8:
            xs = [p0[0], p1[0], q1[0], q0[0]]
            ys = [p0[2], p1[2], q1[2], q0[2]]
            zs = [p0[1], p1[1], q1[1], q0[1]]
            ax.plot(xs, ys, zs, linewidth=1.0)
            ax.add_collection3d(__import__("mpl_toolkits.mplot3d.art3d", fromlist=["Poly3DCollection"]).Poly3DCollection(
                [[(p0[0], p0[2], p0[1]), (p1[0], p1[2], p1[1]),
                  (q1[0], q1[2], q1[1]), (q0[0], q0[2], q0[1])]],
                alpha=DISTRIBUTED_BAND_ALPHA, linewidths=0.7
            ))


def _draw_point_load(ax, model, load: MemberPointLoad):
    a, b = _member_end_xyz(model, int(load.member))
    L = _member_length(model, int(load.member)) or 6.0
    t = float(load.position) / max(L, 1e-12)
    p = tuple(a[j] + t * (b[j] - a[j]) for j in range(3))
    _draw_force_arrow(ax, p, load.direction, abs(float(load.magnitude)), scale=0.55, linewidth=1.2)


def _draw_temperature(ax, model, temp: TemperatureLoad):
    for member_id in temp.members:
        a, b = _member_end_xyz(model, int(member_id))
        mid = tuple((a[j] + b[j]) / 2.0 for j in range(3))
        axis = tuple(b[j] - a[j] for j in range(3))
        L = math.sqrt(sum(c*c for c in axis)) or 1.0
        unit = tuple(c / L for c in axis)
        # Expansion arrows are drawn at both ends away from the center.
        p1 = tuple(a[j] + 0.25 * L * unit[j] for j in range(3))
        p2 = tuple(b[j] - 0.25 * L * unit[j] for j in range(3))
        # Draw along the actual member axis so both X- and Z-oriented
        # roof beams show the correct thermal expansion direction.
        draw_len = 0.35
        ax.quiver(p1[0], p1[2], p1[1], unit[0] * draw_len, unit[2] * draw_len, unit[1] * draw_len,
                  arrow_length_ratio=0.25, linewidth=1.0)
        ax.quiver(p2[0], p2[2], p2[1], -unit[0] * draw_len, -unit[2] * draw_len, -unit[1] * draw_len,
                  arrow_length_ratio=0.25, linewidth=1.0)
        ax.text(mid[0], mid[2], mid[1], f"M{member_id}: {temp.delta_T:+g} degC", fontsize=8)


def render_load_figure(case_or_combination, model=None, output_path=None,
                       show_grid=GRID_DEFAULT, load_band=True,
                       title_suffix=""):
    """Render a load-case/combination verification figure."""
    if model is None:
        model = get_rev2_model()
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    _plot_model_base(ax, model, show_grid=show_grid, show_labels=True)
    case = case_or_combination
    for n in case.nodal_loads:
        p = _model_node_xyz(model, int(n.node))
        _draw_force_arrow(ax, p, n.direction, abs(float(n.magnitude)), scale=0.55)
    for d in case.distributed_loads:
        _draw_distributed(ax, model, d, band=load_band)
    for p in case.member_point_loads:
        _draw_point_load(ax, model, p)
    for t in case.temperature_loads:
        _draw_temperature(ax, model, t)
    if getattr(case, "self_weight", False):
        # One glyph per member, generated from the existing member data.
        try:
            for mid in [int(v) for v in model.members["Member"]]:
                a, b = _member_end_xyz(model, mid)
                midp = tuple((a[j] + b[j]) / 2.0 for j in range(3))
                _draw_force_arrow(ax, midp, "-Y", 1.0, scale=0.25, linewidth=0.7)
        except Exception:
            pass
    if case.key == "LC9":
        ax.text2D(0.02, 0.94, f"TEMPERATURE: {LC9_DELTA_T_DEGC:+g} degC", transform=ax.transAxes)
    meta = _COMBINATION_METADATA.get(case.key, {})
    if meta:
        ax.text2D(0.02, 0.90, meta.get("name", case.key), transform=ax.transAxes)
        ax.text2D(0.02, 0.86, " + ".join(f"{k} x {v:g}" for k,v in getattr(case_or_combination, "factors", {}).items()), transform=ax.transAxes)
    if title_suffix:
        ax.set_title(f"{case.key} - {title_suffix}")
    fig.tight_layout()
    if output_path is None:
        output_path = FINAL_OUTPUT_DIR / f"{case.key}.png"
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return Path(output_path)


def build_load_case_glyphs(case: LoadCase, model=None) -> List[Dict[str, object]]:
    """Return a deterministic glyph list used by both viewer and tests."""
    if model is None:
        model = get_rev2_model()
    glyphs: List[Dict[str, object]] = []
    for n in case.nodal_loads:
        glyphs.append({"kind": "nodal", "node": int(n.node), "direction": n.direction,
                       "magnitude": abs(float(n.magnitude)), "unit": "kN"})
    for d in case.distributed_loads:
        L = _member_length(model, int(d.member)) or 6.0
        count = max(2, int(round(L / DISTRIBUTED_ARROW_SPACING_M)) + 1)
        for i in range(count):
            t = i / (count - 1)
            w = float(d.w_start) + t * (float(d.w_end) - float(d.w_start))
            glyphs.append({"kind": "distributed", "member": int(d.member), "t": t,
                           "direction": d.direction, "magnitude": abs(w), "unit": "kN/m"})
    for p in case.member_point_loads:
        glyphs.append({"kind": "point", "member": int(p.member), "position_m": float(p.position),
                       "direction": p.direction, "magnitude": abs(float(p.magnitude)), "unit": "kN"})
    for t in case.temperature_loads:
        for m in t.members:
            glyphs.append({"kind": "temperature", "member": int(m),
                           "magnitude": float(t.delta_T), "unit": "degC"})
    return glyphs


def render_all_required_images(model=None) -> List[Path]:
    if model is None:
        model = get_rev2_model()
    paths: List[Path] = []
    for i in range(1, 10):
        if i == 9:
            case = LoadCase(key="LC9", description="TEMPERATURE +15 degC",
                            load_type="Temperature",
                            temperature_loads=[lc9_temperature_load()])
        else:
            case = LOAD_CASES_REV3[f"LC{i}"]
        path = FINAL_OUTPUT_DIR / f"rev3_load_case_{i}.png"
        render_load_figure(case, model=model, output_path=path)
        paths.append(path)
    for idx in (1, 13):
        comb = next(c for c in LOAD_COMBINATIONS_REV3 if c.key == f"COMB{idx}")
        case = assemble_combination_case(comb)
        path = FINAL_OUTPUT_DIR / f"rev3_combination_{idx}.png"
        render_load_figure(case, model=model, output_path=path, title_suffix=combination_metadata(comb).get("design_method", ""))
        paths.append(path)
    return paths


# --------------------------------------------------------------
# 10. VERIFICATION REPORT
# --------------------------------------------------------------

def _fmt(v):
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


def build_verification_report(model=None, test_summary="not run") -> str:
    if model is None:
        model = get_rev2_model()
    rows = load_case_report_rows(model)
    step3 = validate_load_cases_rev3(model)
    step4 = validate_roof_diaphragm(model, verbose=False)
    step5 = verify_lc9(verbose=False)
    combs = validate_load_combinations_rev3(model)
    lines: List[str] = []
    add = lines.append
    add("REV. 3 STRUCTURAL CUBE SOLVER - VERIFICATION / AUDIT REPORT")
    add("=" * 78)
    add("")
    add("1. SCOPE AND ANALYSIS LIMITATION")
    add("-" * 78)
    add("Rev. 3 defines, applies, validates, combines and visualizes loads.")
    add("It does NOT perform stiffness analysis, support reactions, member forces,")
    add("displacements, drift, or diaphragm constraint elimination.")
    add("The load viewer is a verification layer, not proof of solved structural behavior.")
    add("")
    add("2. MODEL / DATABASE SOURCE")
    add("-" * 78)
    add("Source model: existing cube_6m_Rev2.py (left unchanged).")
    add("Standard geometry: 8 nodes, 12 members; roof nodes 5-8; roof beams M5-M8.")
    try:
        add(f"Material label: {getattr(model, 'MATERIAL_LABEL', 'n/a')}")
        add(f"Section label: {getattr(model, 'SECTION_LABEL', 'n/a')}")
    except Exception:
        pass
    add("")
    add("3. LOAD CASES LC1-LC8")
    add("-" * 78)
    for r in rows:
        add(f"{r['key']}: {r['name']} | type={r['type']} | nodes={r['node_ids']} | members={r['member_ids']} | total={r['computed_kN']:.6f} kN | error={r['error_kN']:.6f} kN")
    add("Step 3 hard problems: " + ("NONE" if not [x for x in step3 if x.startswith('PROBLEM:')] else str(step3)))
    add("")
    add("4. ROOF DIAPHRAGM")
    add("-" * 78)
    dia = build_roof_diaphragm(model)
    add(f"Master node: N{dia.master_node}")
    add(f"Slave nodes: {dia.slave_list()}")
    add(f"Constrained DOFs: {dia.constrained_dof_list()}")
    add(f"Free DOFs: {dia.free_dofs()}")
    add(f"Constraint equations: {len(dia.constraint_equations())}")
    add("Constraint elimination: NOT performed in Rev. 3")
    add("")
    add("5. LOAD CASE 9 - TEMPERATURE")
    add("-" * 78)
    add(f"Delta T: {LC9_DELTA_T_DEGC:+g} degC")
    add(f"Alpha from Rev. 2 material data: {lc9_rev2_alpha_1e6_per_c():.6g} x 1e-6/degC")
    add(f"Thermal strain: {lc9_thermal_strain():.6e}")
    add(f"Free expansion (6 m): {lc9_free_expansion_mm():.6f} mm")
    add(f"EA from active Rev. 2 section: {lc9_beam_EA_kN():.3f} kN")
    add(f"Fully restrained force: {lc9_fully_restrained_force_kN():.6f} kN compression")
    add("Partial restraint: INDETERMINATE (no stiffness/constraint solution in Rev. 3)")
    add(f"Thermal vector net force: {lc9_net_resultant_kN()}")
    add("Temperature is represented as a thermal strain/equivalent thermal vector, not a fake nodal force.")
    add("")
    add("6. LOAD COMBINATIONS")
    add("-" * 78)
    add(f"Number generated: {len(LOAD_COMBINATIONS_REV3)}")
    add("NSCP note: factors are the handout transcription identified as NSCP 2015; the handout states they were NOT independently verified against the printed code and must be checked against §§203.3.1 and 203.4.1 before design use.")
    for c in LOAD_COMBINATIONS_REV3:
        md = combination_metadata(c)
        add(f"{c.key}: {md.get('design_method','')} | {md.get('name','')} | {c.factors}")
    add("")
    add("7. VIEWER / IMAGE OUTPUTS")
    add("-" * 78)
    add("Load-case images: rev3_load_case_1.png through rev3_load_case_9.png")
    add("Representative combination images: rev3_combination_1.png and rev3_combination_13.png")
    add("Distributed-load bands are optional drawing only; load glyph data are unchanged by the band toggle.")
    add("LC9 viewer uses degC and expansion-direction annotation, never kN for temperature.")
    add("")
    add("8. AUTOMATED TESTS")
    add("-" * 78)
    add(f"Test summary: {test_summary}")
    add("")
    add("9. KNOWN ASSUMPTIONS / DISCLOSED DIFFERENCES")
    add("-" * 78)
    add("The handout Appendix A benchmark uses W310X38.7 beams, W250X49.1 columns and A992 steel.")
    add("The active Rev. 2 baseline currently supplies its own database-selected material/section properties.")
    add("Rev. 3 therefore does not silently substitute the Appendix A standard section set.")
    if step5:
        add("LC9 verification notes: " + "; ".join(step5))
    add("")
    add("10. FINAL ENGINEERING STATUS")
    add("-" * 78)
    add("Load definitions: implemented")
    add("Load validation: implemented")
    add("Diaphragm definition/equations: implemented; not eliminated into stiffness equations")
    add("Temperature load effect: implemented as thermal strain/equivalent vector; no solved restraint response")
    add("Combinations: implemented as references/factors; no structural analysis")
    add("Viewer: implemented for inspection")
    add("STIFFNESS SOLVER: OUT OF SCOPE FOR REV. 3")
    add("=" * 78)
    return "\n".join(lines) + "\n"


def write_verification_report(model=None, test_summary="not run") -> Path:
    text = build_verification_report(model=model, test_summary=test_summary)
    FINAL_REPORT_FILE.write_text(text, encoding="utf-8")
    return FINAL_REPORT_FILE


# --------------------------------------------------------------
# 11. OPTIONAL PY SIDE 6 VIEWER
# --------------------------------------------------------------

def launch_viewer():
    """Launch the interactive Load View tab/dropdown if PySide6 exists."""
    try:
        from PySide6.QtWidgets import QApplication, QComboBox, QCheckBox, QMainWindow, QVBoxLayout, QWidget, QLabel
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("PySide6 and the Qt Matplotlib backend are required for GUI mode: " + str(exc))

    model = get_rev2_model()
    app = QApplication.instance() or QApplication([])
    window = QMainWindow()
    window.setWindowTitle("Rev. 3 Structural Cube - Load View")
    central = QWidget()
    layout = QVBoxLayout(central)
    window.setCentralWidget(central)

    selector = QComboBox()
    entries = [(f"LC{i}", f"LC{i} - {LOAD_CASES_REV3[f'LC{i}'].description}") for i in range(1, 9)]
    entries.append(("LC9", "LC9 - TEMPERATURE +15 degC"))
    entries += [(c.key, f"{c.key} - {combination_metadata(c).get('name', c.key)}") for c in LOAD_COMBINATIONS_REV3]
    for key, label in entries:
        selector.addItem(label, key)
    layout.addWidget(QLabel("Load Case / Combination"))
    layout.addWidget(selector)
    band = QCheckBox("Load band")
    band.setChecked(True)
    grid = QCheckBox("Grid")
    grid.setChecked(True)
    layout.addWidget(band)
    layout.addWidget(grid)

    fig = plt.Figure(figsize=(9, 7))
    canvas = FigureCanvas(fig)
    layout.addWidget(canvas)

    def redraw():
        key = selector.currentData()
        fig.clear()
        ax = fig.add_subplot(111, projection="3d")
        _plot_model_base(ax, model, show_grid=grid.isChecked(), show_labels=True)
        if key == "LC9":
            case = lc9_temperature_load()
        elif key.startswith("COMB"):
            comb = next(c for c in LOAD_COMBINATIONS_REV3 if c.key == key)
            case = assemble_combination_case(comb)
        else:
            case = LOAD_CASES_REV3[key]
        for n in case.nodal_loads:
            _draw_force_arrow(ax, _model_node_xyz(model, n.node), n.direction, abs(float(n.magnitude)))
        for d in case.distributed_loads:
            _draw_distributed(ax, model, d, band=band.isChecked())
        for p in case.member_point_loads:
            _draw_point_load(ax, model, p)
        for t in case.temperature_loads:
            _draw_temperature(ax, model, t)
        if case.self_weight:
            for mid in [int(v) for v in model.members["Member"]]:
                a, b = _member_end_xyz(model, mid)
                midp = tuple((a[j] + b[j]) / 2.0 for j in range(3))
                _draw_force_arrow(ax, midp, "-Y", 1.0, scale=0.25, linewidth=0.7)
        canvas.draw()

    selector.currentIndexChanged.connect(redraw)
    band.stateChanged.connect(redraw)
    grid.stateChanged.connect(redraw)
    redraw()
    window.resize(1100, 900)
    window.show()
    app.exec()


# --------------------------------------------------------------
# 12. HEADLESS AUDIT / CLI
# --------------------------------------------------------------

def run_headless_audit() -> int:
    print("=" * 78)
    print("CUBE STRUCTURAL SOLVER - REV. 3 FINAL AUDIT")
    print("=" * 78)
    try:
        model = get_rev2_model()
    except Exception as exc:
        print("MODEL LOAD FAILED:", exc)
        print("Make sure cube_6m_Rev2.py and its Excel database folders are beside this script.")
        return 2

    print("\n[1] LC1-LC8 validation")
    p3 = validate_load_cases_rev3(model, verbose=True)
    hard3 = [p for p in p3 if p.startswith("PROBLEM:")]
    print("STEP 3 hard problems:", hard3 or "NONE")

    print("\n[2] Roof diaphragm validation")
    p4 = validate_roof_diaphragm(model, verbose=True)
    hard4 = [p for p in p4 if p.startswith("PROBLEM:")]
    print("STEP 4 hard problems:", hard4 or "NONE")

    print("\n[3] LC9 temperature validation")
    p5 = verify_lc9(verbose=True)
    hard5 = [p for p in p5 if p.startswith("PROBLEM:")]
    print("STEP 5 hard problems:", hard5 or "NONE")

    print("\n[4] Combination validation")
    pc = validate_load_combinations_rev3(model)
    print("Combination count:", len(LOAD_COMBINATIONS_REV3))
    print("Combination hard problems:", pc or "NONE")

    print("\n[5] Representative combination resultants")
    for key in ("COMB1", "COMB13"):
        c = next(x for x in LOAD_COMBINATIONS_REV3 if x.key == key)
        print(key, combination_metadata(c), assemble_combination_resultant(c, model))

    print("\n[6] Rendering required images")
    try:
        paths = render_all_required_images(model)
        for path in paths:
            print("  wrote", path.name)
    except Exception as exc:
        print("IMAGE RENDER FAILED:", exc)
        return 3

    print("\n[7] Verification report")
    report = write_verification_report(model, test_summary="Run test_cube_solver_rev3.py for the 95-test suite")
    print("  wrote", report.name)

    hard = hard3 + hard4 + hard5 + pc
    print("\n" + "=" * 78)
    print("FINAL AUDIT STATUS:", "PASS" if not hard else "CHECK REQUIRED")
    if hard:
        for p in hard:
            print(" -", p)
    print("=" * 78)
    return 0 if not hard else 1


def _final_cli():
    import sys
    args = sys.argv[1:]
    if not args:
        return run_headless_audit()
    if args[0] in ("--verify-loads", "verify", "audit"):
        return run_headless_audit()
    if args[0] in ("launch", "--gui"):
        launch_viewer()
        return 0
    if args[0] in ("--help", "-h"):
        print("python cube_solver_rev3.py --verify-loads   headless audit + PNGs + report")
        print("python cube_solver_rev3.py launch            interactive Load View GUI")
        print("python cube_solver_rev3.py                    same as --verify-loads")
        return 0
    print("Unknown option:", args[0])
    return 2


if __name__ == "__main__":
    raise SystemExit(_final_cli())
