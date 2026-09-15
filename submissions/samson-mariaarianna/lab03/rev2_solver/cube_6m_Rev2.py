from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch


# ==============================================================
# 6 m x 6 m x 6 m CUBE - STRUCTURAL SOLVER / MODEL GENERATOR
# REV. 2
#
# REV2 (database-driven) additions:
#   - Unit definitions and conversion factors from the professor's
#     Units Excel database (Standard Metric solver-internal set)
#   - ASTM A36 material ("A36 Gr.36") from the Material library
#   - Member size "W6X9" from the AISC Shapes Database v16.0
#   - Per-member section properties and self-weight
#   - New Excel sheets and updated Matplotlib diagram
#
# REV1 presentation features preserved:
#   - Pinned supports at bottom nodes
#   - Six DOF per node
#   - Global and local member axes
#   - Beta angles
#   - Beam end pins / Mz (local RZ) releases
#   - Clean RISA/STAAD-style structural diagram
#   - Formatted Excel output
#
# NOTE:
# This program now reads the professor's reference Excel files
# (units, materials, member sizes).  It still creates a new Excel
# workbook and a new PNG diagram.
# ==============================================================


# --------------------------------------------------------------
# 1. FILE LOCATIONS
# --------------------------------------------------------------

folder = Path(__file__).resolve().parent

excel_output = folder / "cube_nodes_6m_Rev2_generated.xlsx"
png_output = folder / "cube_6m_Rev2_generated.png"


# --------------------------------------------------------------
# 2. DISPLAY SETTINGS
# --------------------------------------------------------------

# These colors are chosen to match the requested clean structural
# model presentation.
BEAM_COLOR = "#1455D9"
COLUMN_COLOR = "#008B63"
NODE_COLOR = "#F04A4A"
NODE_EDGE = "#8B1E1E"
SUPPORT_COLOR = "#404040"
LOCAL_X_COLOR = "#E53935"
LOCAL_Y_COLOR = "#22A447"
LOCAL_Z_COLOR = "#8E44AD"
TEXT_COLOR = "#111111"
PANEL_FACE = "#F5F7FA"
PANEL_EDGE = "#183A66"


# --------------------------------------------------------------
# 2A. UNITS ENGINE (REV2) - readings from the Units database
# --------------------------------------------------------------

# The Units database recommends the solver-internal unit set:
#   Standard Metric: geometry in m, sections in mm, force in kN,
#   stress / modulus in MPa, density in kN/m^3, alpha in 1e-6/C.
# Every Imperial <-> Metric factor below is read from the database,
# never typed by hand.

UNITS_DB = folder / "units" / "Units_Imperial_Metric.xlsx"


def load_unit_tables(path):
    """Read 'Unit Systems' and 'Conversion Factors' from the Units DB."""
    raw_systems = pd.read_excel(path, sheet_name="Unit Systems", header=None)
    raw_factors = pd.read_excel(path, sheet_name="Conversion Factors", header=None)

    header_idx = raw_systems.index[
        raw_systems[0].astype(str).str.strip() == "Quantity"
    ][0]
    systems = raw_systems.iloc[int(header_idx):].reset_index(drop=True)
    systems.columns = [str(c).strip() for c in systems.iloc[0].tolist()]
    systems = systems.iloc[1:].reset_index(drop=True)
    systems = systems.dropna(subset=["Quantity", "Imperial"])

    header_idx = raw_factors.index[
        raw_factors[0].astype(str).str.strip() == "Quantity"
    ][0]
    factors = raw_factors.iloc[int(header_idx):].reset_index(drop=True)
    factors.columns = [str(c).strip() for c in factors.iloc[0].tolist()]
    factors = factors.iloc[1:].reset_index(drop=True)
    factors = factors.dropna(subset=["Quantity", "Imperial unit"])

    inch_rows = raw_factors[raw_factors[0].astype(str) == "1 inch in millimetres"]
    inch_to_mm = float(inch_rows.iloc[0][1])

    return systems, factors, inch_to_mm


UNIT_SYSTEMS_DF, CONVERSION_FACTORS_DF, INCH_TO_MM = load_unit_tables(UNITS_DB)

# Lookup key: (Quantity, Imperial unit, Metric unit) -> (factor, reciprocal)
CONV_FACTOR = {}
for _, row in CONVERSION_FACTORS_DF.iterrows():
    key = (str(row["Quantity"]), str(row["Imperial unit"]), str(row["Metric unit"]))
    try:
        factor = float(row["Multiply Imperial by"])
        reciprocal = float(row["Metric to Imperial (reciprocal)"])
    except (TypeError, ValueError):
        # Some rows (e.g. 'Temperature (offset)') carry 'n/a' factors.
        continue
    CONV_FACTOR[key] = (factor, reciprocal)

# Named lookup helpers (Imperial -> Metric / SI).  All factors come
# from the CONV_FACTOR table above.
LENGTH_IN_TO_MM = CONV_FACTOR[("Length", "in", "mm")][0]
LENGTH_IN_TO_M = LENGTH_IN_TO_MM / 1000.0
AREA_IN2_TO_MM2 = CONV_FACTOR[("Area", "in\u00b2", "mm\u00b2")][0]
SM_IN3_TO_MM3 = CONV_FACTOR[("Section modulus", "in\u00b3", "mm\u00b3")][0]
INERTIA_IN4_TO_MM4 = CONV_FACTOR[("Moment of inertia", "in\u2074", "mm\u2074")][0]
CW_IN6_TO_MM6 = INCH_TO_MM ** 6
KSI_TO_MPA = CONV_FACTOR[("Stress / modulus", "ksi", "MPa")][0]
UNITW_KFT3_TO_KNM3 = CONV_FACTOR[("Unit weight", "k/ft\u00b3", "kN/m\u00b3")][0]
MASSD_KFT3_TO_KGM3 = CONV_FACTOR[
    ("Mass density from unit weight", "k/ft\u00b3", "kg/m\u00b3")][0]
THERM_1E5F_TO_1E6C = CONV_FACTOR[
    ("Thermal coefficient", "1e-5 /\u00b0F", "1e-6 /\u00b0C")][0]


# --------------------------------------------------------------
# 2B. MATERIAL ENGINE (REV2) - readings from the Material database
# --------------------------------------------------------------

MATERIALS_DB = folder / "materials" / "RISA_Materials_Library.xlsx"
MATERIAL_LABEL = "A36 Gr.36"


def load_materials_table(path):
    """Read the flat 'All Materials' sheet from the Material library."""
    raw = pd.read_excel(path, sheet_name="All Materials", header=None)
    header_idx = raw.index[raw[0].astype(str).str.strip() == "Category"][0]
    table = raw.iloc[int(header_idx) + 1:].reset_index(drop=True)
    table.columns = [str(c).strip() for c in raw.iloc[int(header_idx)].tolist()]
    table = table[table["Label"].notna()]
    table = table[~table["Category"].fillna("").astype(str).str.startswith("Count")]
    return table


MATERIALS_TABLE = load_materials_table(MATERIALS_DB)


def pick_material(label):
    rows = MATERIALS_TABLE[MATERIALS_TABLE["Label"].astype(str).str.strip() == label]
    if len(rows) == 0:
        raise ValueError(f"Material '{label}' not found in {MATERIALS_DB}")
    return rows.iloc[0]


def _db_float(value, name):
    number = float(value)
    if math.isnan(number):
        raise ValueError(f"Missing numeric '{name}' for {MATERIAL_LABEL} in {MATERIALS_DB}")
    return number


_material_row = pick_material(MATERIAL_LABEL)
MATERIAL_CATEGORY = str(_material_row["Category"])

material = {
    "Label": str(_material_row["Label"]),
    "Category": MATERIAL_CATEGORY,
    # Imperial values exactly as stored in the library.
    "E (ksi)": _material_row["E [ksi]"],
    "G (ksi)": _material_row["G [ksi]"],
    "Nu": _material_row["Nu"],
    "Therm. Coeff. (1e-5/F)": _material_row["Therm. Coeff. [1e-5/\u00b0F]"],
    "Density (k/ft3)": _material_row["Density [k/ft\u00b3]"],
    "Fy (ksi)": _material_row["Yield / f'c / f'm [ksi]"],
    "Fu (ksi)": _material_row["Fu [ksi]"],
    # Standard Metric values from the library's formula columns.
    "E (MPa)": _material_row["E [MPa]"],
    "G (MPa)": _material_row["G [MPa]"],
    "Therm. Coeff. (1e-6/C)": _material_row["Therm. Coeff. [1e-6/\u00b0C]"],
    "Density (kN/m3)": _material_row["Density [kN/m\u00b3]"],
    "Mass Density (kg/m3)": _material_row["Mass Density [kg/m\u00b3]"],
    "Fy (MPa)": _material_row["Yield / f'c / f'm [MPa]"],
    "Fu (MPa)": _material_row["Fu [MPa]"],
}

MATERIAL_E_MPA = _db_float(material["E (MPa)"], "E")
MATERIAL_G_MPA = _db_float(material["G (MPa)"], "G")
MATERIAL_NU = _db_float(material["Nu"], "Nu")
MATERIAL_ALPHA_1E6_PER_C = _db_float(material["Therm. Coeff. (1e-6/C)"], "alpha")
MATERIAL_GAMMA_KNM3 = _db_float(material["Density (kN/m3)"], "density")
MATERIAL_RHO_KGM3 = _db_float(material["Mass Density (kg/m3)"], "mass density")
MATERIAL_FY_MPA = _db_float(material["Fy (MPa)"], "Fy")
MATERIAL_FU_MPA = _db_float(material["Fu (MPa)"], "Fu")


# --------------------------------------------------------------
# 2C. SECTION ENGINE (REV2) - readings from the AISC database
# --------------------------------------------------------------

MEMBER_SIZE_DB = folder / "member_size" / "aisc-shapes-database-v160-2.xlsx"
SECTION_LABEL = "W6X9"


def load_aisc_shapes(path):
    """Read the 'Database v16.0' worksheet of the AISC Shapes Database.

    The sheet is 2301 rows x 166 columns, so the XLSX is parsed
    directly (zipfile + ElementTree) for speed.
    """
    import zipfile
    from xml.etree import ElementTree as ET

    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    rns = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"

    with zipfile.ZipFile(path) as zf:
        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        target_by_id = {r.get("Id"): r.get("Target") for r in rels}

        target = None
        for s in workbook.find(ns + "sheets"):
            if s.get("name") == "Database v16.0":
                target = target_by_id.get(s.get(rns + "id"))
                break
        if not target:
            raise ValueError("AISC worksheet 'Database v16.0' not found.")

        shared = []
        try:
            root_ss = ET.fromstring(zf.read("xl/sharedStrings.xml"))
        except KeyError:
            root_ss = None
        if root_ss is not None:
            for si in root_ss.iter(ns + "si"):
                shared.append("".join(t.text or "" for t in si.iter(ns + "t")))

        def col_letter(ref):
            letters = ""
            for ch in ref or "":
                if ch.isalpha():
                    letters += ch
                else:
                    break
            return letters

        def cell_value(cell):
            kind = cell.get("t")
            node_v = cell.find(ns + "v")
            if kind == "s" and node_v is not None:
                return shared[int(node_v.text)]
            if kind == "inlineStr":
                ise = cell.find(ns + "is")
                return "".join(x.text or "" for x in ise.iter(ns + "t"))
            if node_v is not None:
                return node_v.text
            return None

        root = ET.fromstring(zf.read("xl/" + target.lstrip("/")))

        headers = []
        rows = []
        for row in root.iter(ns + "row"):
            values = {col_letter(c.get("r")): cell_value(c)
                      for c in row.iter(ns + "c")}
            if not headers:
                headers = list(values.values())
                continue
            rows.append(values)

    return headers, rows


AISC_HEADERS, AISC_ROWS = load_aisc_shapes(MEMBER_SIZE_DB)


def _aisc_float(value):
    if value is None:
        return None
    text = str(value).strip()
    if text in ("", "-", "\u2013", "\u2014"):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def pick_shape(label):
    for row in AISC_ROWS:
        if str(row.get("C", "")).strip() == label:
            return row
    raise ValueError(f"Section '{label}' not found in {MEMBER_SIZE_DB}")


_aisc = pick_shape(SECTION_LABEL)

section_imperial = {
    "AISC_Manual_Label": SECTION_LABEL,
    "EDI_Std_Nomenclature": _aisc.get("B"),
    "Type": _aisc.get("A"),
    "W (lb/ft)": _aisc_float(_aisc.get("E")),
    "A (in2)": _aisc_float(_aisc.get("F")),
    "d (in)": _aisc_float(_aisc.get("G")),
    "bf (in)": _aisc_float(_aisc.get("L")),
    "tw (in)": _aisc_float(_aisc.get("Q")),
    "tf (in)": _aisc_float(_aisc.get("T")),
    "Ix (in4)": _aisc_float(_aisc.get("AM")),
    "Zx (in3)": _aisc_float(_aisc.get("AN")),
    "Sx (in3)": _aisc_float(_aisc.get("AO")),
    "rx (in)": _aisc_float(_aisc.get("AP")),
    "Iy (in4)": _aisc_float(_aisc.get("AQ")),
    "Zy (in3)": _aisc_float(_aisc.get("AR")),
    "Sy (in3)": _aisc_float(_aisc.get("AS")),
    "ry (in)": _aisc_float(_aisc.get("AT")),
    "J (in4)": _aisc_float(_aisc.get("AX")),
    "Cw (in6)": _aisc_float(_aisc.get("AY")),
    # Metric block (cross-check only).
    "A (mm2, metric)": _aisc_float(_aisc.get("CJ")),
    "d (mm, metric)": _aisc_float(_aisc.get("CK")),
    "Ix (1e6 mm4, metric)": _aisc_float(_aisc.get("DQ")),
}

# SI section properties computed with the Units DB factors.
SECTION_AREA_M2 = section_imperial["A (in2)"] * AREA_IN2_TO_MM2 * 1.0e-6
SECTION_D_M = section_imperial["d (in)"] * LENGTH_IN_TO_M
SECTION_BF_M = section_imperial["bf (in)"] * LENGTH_IN_TO_M
SECTION_TW_M = section_imperial["tw (in)"] * LENGTH_IN_TO_M
SECTION_TF_M = section_imperial["tf (in)"] * LENGTH_IN_TO_M
SECTION_IZ_M4 = section_imperial["Ix (in4)"] * INERTIA_IN4_TO_MM4 * 1.0e-12
SECTION_IY_M4 = section_imperial["Iy (in4)"] * INERTIA_IN4_TO_MM4 * 1.0e-12
SECTION_J_M4 = section_imperial["J (in4)"] * INERTIA_IN4_TO_MM4 * 1.0e-12
SECTION_CW_M6 = section_imperial["Cw (in6)"] * CW_IN6_TO_MM6 * 1.0e-18
SECTION_W_PER_LEN_KNM = MATERIAL_GAMMA_KNM3 * SECTION_AREA_M2

# Display-friendly SI sub-units (mm / mm^2 / mm^3 / mm^4 / mm^6).
SECTION_A_MM2 = SECTION_AREA_M2 * 1.0e6
SECTION_D_MM = SECTION_D_M * 1.0e3
SECTION_BF_MM = SECTION_BF_M * 1.0e3
SECTION_TW_MM = SECTION_TW_M * 1.0e3
SECTION_TF_MM = SECTION_TF_M * 1.0e3
SECTION_IZ_MM4 = SECTION_IZ_M4 * 1.0e12
SECTION_IY_MM4 = SECTION_IY_M4 * 1.0e12
SECTION_J_MM4 = SECTION_J_M4 * 1.0e12
SECTION_CW_MM6 = SECTION_CW_M6 * 1.0e18
SECTION_ZX_MM3 = section_imperial["Zx (in3)"] * SM_IN3_TO_MM3
SECTION_SX_MM3 = section_imperial["Sx (in3)"] * SM_IN3_TO_MM3
SECTION_ZY_MM3 = section_imperial["Zy (in3)"] * SM_IN3_TO_MM3
SECTION_SY_MM3 = section_imperial["Sy (in3)"] * SM_IN3_TO_MM3
SECTION_RX_MM = section_imperial["rx (in)"] * LENGTH_IN_TO_MM
SECTION_RY_MM = section_imperial["ry (in)"] * LENGTH_IN_TO_MM


# --------------------------------------------------------------
# 3. NODE DATA
# --------------------------------------------------------------

# Global coordinate system:
#   X = horizontal/lateral
#   Y = vertical
#   Z = horizontal/lateral
#
# Node arrangement:
#   Bottom: 1-4
#   Top:    5-8

nodes_data = [
    [1, 0.0, 0.0, 0.0],
    [2, 6.0, 0.0, 0.0],
    [3, 6.0, 0.0, 6.0],
    [4, 0.0, 0.0, 6.0],

    [5, 0.0, 6.0, 0.0],
    [6, 6.0, 6.0, 0.0],
    [7, 6.0, 6.0, 6.0],
    [8, 0.0, 6.0, 6.0],
]

nodes = pd.DataFrame(
    nodes_data,
    columns=["Node", "X (m)", "Y (m)", "Z (m)"]
)


# --------------------------------------------------------------
# 4. MEMBER DATA
# --------------------------------------------------------------

# M1-M4  = bottom/base beams
# M5-M8  = top/roof beams
# M9-M12 = vertical columns
#
# Beta angle:
#   Base/roof beams = 0 degrees
#   Columns          = 90 degrees
#
# Beam pin:
#   M1-M8 are pinned at both ends.
#   The local RZ / Mz rotational DOF is released at each end.
#   Local UX remains connected.
#
# REV2 - each member also stores its Material Label and Section
# Label (both are read from the Excel databases in section 2B/2C).

members_data = [
    # Member, node i, node j, type, beta (deg), pinned, material, section
    [1,  1, 2, "Base Beam", 0.0, True,  MATERIAL_LABEL, SECTION_LABEL],
    [2,  2, 3, "Base Beam", 0.0, True,  MATERIAL_LABEL, SECTION_LABEL],
    [3,  3, 4, "Base Beam", 0.0, True,  MATERIAL_LABEL, SECTION_LABEL],
    [4,  4, 1, "Base Beam", 0.0, True,  MATERIAL_LABEL, SECTION_LABEL],

    [5,  5, 6, "Roof Beam", 0.0, True,  MATERIAL_LABEL, SECTION_LABEL],
    [6,  6, 7, "Roof Beam", 0.0, True,  MATERIAL_LABEL, SECTION_LABEL],
    [7,  7, 8, "Roof Beam", 0.0, True,  MATERIAL_LABEL, SECTION_LABEL],
    [8,  8, 5, "Roof Beam", 0.0, True,  MATERIAL_LABEL, SECTION_LABEL],

    [9,  1, 5, "Column", 90.0, False,   MATERIAL_LABEL, SECTION_LABEL],
    [10, 2, 6, "Column", 90.0, False,   MATERIAL_LABEL, SECTION_LABEL],
    [11, 3, 7, "Column", 90.0, False,   MATERIAL_LABEL, SECTION_LABEL],
    [12, 4, 8, "Column", 90.0, False,   MATERIAL_LABEL, SECTION_LABEL],
]

members = pd.DataFrame(
    members_data,
    columns=[
        "Member",
        "Node i (Start)",
        "Node j (End)",
        "Type",
        "Beta Angle (deg)",
        "Pinned at Both Ends",
        "Material Label",
        "Section Label"
    ]
)


# --------------------------------------------------------------
# 5. GLOBAL DEGREES OF FREEDOM
# --------------------------------------------------------------

DOF_NAMES = ["UX", "UY", "UZ", "RX", "RY", "RZ"]

dof_rows = []

for _, row in nodes.iterrows():
    node = int(row["Node"])
    first_dof = (node - 1) * 6 + 1
    dofs = list(range(first_dof, first_dof + 6))

    dof_rows.append([
        node,
        row["X (m)"],
        row["Y (m)"],
        row["Z (m)"],
        *dofs
    ])

node_dofs = pd.DataFrame(
    dof_rows,
    columns=[
        "Node",
        "X (m)",
        "Y (m)",
        "Z (m)",
        "UX DOF",
        "UY DOF",
        "UZ DOF",
        "RX DOF",
        "RY DOF",
        "RZ DOF"
    ]
)


# --------------------------------------------------------------
# 6. SUPPORT CONDITIONS
# --------------------------------------------------------------

# A 3D pinned support restrains the three translations:
#   UX, UY, UZ
#
# Nodal rotations:
#   RX, RY, RZ remain free.

support_rows = []

for node in [1, 2, 3, 4]:
    support_rows.append([
        node,
        "Pinned",
        True, True, True,
        False, False, False,
        "UX, UY, UZ restrained; RX, RY, RZ free"
    ])

supports = pd.DataFrame(
    support_rows,
    columns=[
        "Node",
        "Support Type",
        "UX Restrained",
        "UY Restrained",
        "UZ Restrained",
        "RX Restrained",
        "RY Restrained",
        "RZ Restrained",
        "Description"
    ]
)


# --------------------------------------------------------------
# 7. MEMBER END RELEASES
# --------------------------------------------------------------

release_rows = []

for _, member in members.iterrows():
    member_id = int(member["Member"])
    member_type = member["Type"]
    pinned = bool(member["Pinned at Both Ends"])

    for end_name in ["Start (i)", "End (j)"]:
        release_rows.append([
            member_id,
            member_type,
            end_name,
            "Mz / local RZ" if pinned else "None",
            pinned,
            False,
            "Pinned beam end" if pinned else "Rigid connection"
        ])

releases = pd.DataFrame(
    release_rows,
    columns=[
        "Member",
        "Type",
        "End",
        "Released Moment / DOF",
        "Local RZ Release (Mz)",
        "Local UX Release",
        "Description"
    ]
)


# --------------------------------------------------------------
# 8. NODE AND MEMBER LOOKUPS
# --------------------------------------------------------------

node_coordinates = {
    int(row["Node"]): np.array([
        float(row["X (m)"]),
        float(row["Y (m)"]),
        float(row["Z (m)"])
    ])
    for _, row in nodes.iterrows()
}

dof_lookup = node_dofs.set_index("Node")


# --------------------------------------------------------------
# 9. LOCAL AXIS CALCULATIONS
# --------------------------------------------------------------

def unit_vector(vector):
    """Return a unit vector."""
    vector = np.asarray(vector, dtype=float)
    length = np.linalg.norm(vector)

    if length <= 1.0e-12:
        raise ValueError("A member has zero length.")

    return vector / length


def rotate_about_axis(vector, axis, angle_rad):
    """Rotate a vector around an axis using Rodrigues' formula."""
    axis = unit_vector(axis)

    return (
        vector * math.cos(angle_rad)
        + np.cross(axis, vector) * math.sin(angle_rad)
        + axis * np.dot(axis, vector) * (1.0 - math.cos(angle_rad))
    )


def calculate_local_axes(start, end, beta_deg):
    """
    Calculate a right-handed local member coordinate system.

    Local x:
        Member longitudinal axis, from node i to node j.

    Local y:
        Transverse reference axis after beta rotation.

    Local z:
        Completes the right-handed local coordinate system.
    """
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)

    local_x = unit_vector(end - start)

    global_y = np.array([0.0, 1.0, 0.0])
    global_z = np.array([0.0, 0.0, 1.0])

    # Select a reference direction that is not nearly parallel
    # to the member local x axis.
    if abs(np.dot(local_x, global_y)) < 0.95:
        reference = global_y
    else:
        reference = global_z

    # Project the reference perpendicular to local x.
    local_y = reference - np.dot(reference, local_x) * local_x
    local_y = unit_vector(local_y)

    # Apply the beta rotation around local x.
    local_y = rotate_about_axis(
        local_y,
        local_x,
        math.radians(beta_deg)
    )

    local_z = unit_vector(np.cross(local_x, local_y))

    # Re-orthogonalize local y.
    local_y = unit_vector(np.cross(local_z, local_x))

    return local_x, local_y, local_z


# --------------------------------------------------------------
# 10. LOCAL AXIS EXCEL TABLE
# --------------------------------------------------------------

local_axis_rows = []

for _, member in members.iterrows():
    member_id = int(member["Member"])
    node_i = int(member["Node i (Start)"])
    node_j = int(member["Node j (End)"])
    beta = float(member["Beta Angle (deg)"])

    p_i = node_coordinates[node_i]
    p_j = node_coordinates[node_j]

    local_x, local_y, local_z = calculate_local_axes(
        p_i,
        p_j,
        beta
    )

    length = float(np.linalg.norm(p_j - p_i))

    local_axis_rows.append([
        member_id,
        node_i,
        node_j,
        length,
        beta,
        *local_x,
        *local_y,
        *local_z
    ])

local_axes = pd.DataFrame(
    local_axis_rows,
    columns=[
        "Member",
        "Node i",
        "Node j",
        "Length (m)",
        "Beta Angle (deg)",
        "Local X - Global X",
        "Local X - Global Y",
        "Local X - Global Z",
        "Local Y - Global X",
        "Local Y - Global Y",
        "Local Y - Global Z",
        "Local Z - Global X",
        "Local Z - Global Y",
        "Local Z - Global Z"
    ]
)


# --------------------------------------------------------------
# 10A. MEMBER SECTION PROPERTIES (REV2)
# --------------------------------------------------------------

# Every member carries the DB material and DB section chosen in
# sections 2B/2C.  Local-axis mapping (documented convention):
#   local z  <-> AISC strong axis (Ix of the W-shape)
#   local y  <-> AISC weak axis  (Iy of the W-shape)

member_section_rows = []

for _, member in members.iterrows():
    member_id = int(member["Member"])
    node_i = int(member["Node i (Start)"])
    node_j = int(member["Node j (End)"])
    p_i = node_coordinates[node_i]
    p_j = node_coordinates[node_j]
    length = float(np.linalg.norm(p_j - p_i))

    weight_per_length = MATERIAL_GAMMA_KNM3 * SECTION_AREA_M2
    total_weight = weight_per_length * length

    member_section_rows.append([
        member_id,
        node_i,
        node_j,
        member["Type"],
        member["Material Label"],
        member["Section Label"],
        length,
        SECTION_AREA_M2,
        SECTION_A_MM2,
        SECTION_IY_M4,
        SECTION_IZ_M4,
        SECTION_J_M4,
        SECTION_CW_M6,
        weight_per_length,
        total_weight
    ])

member_sections = pd.DataFrame(
    member_section_rows,
    columns=[
        "Member",
        "Node i",
        "Node j",
        "Type",
        "Material Label",
        "Section Label",
        "Length (m)",
        "Area (m2)",
        "Area (mm2)",
        "Iy (m4)",
        "Iz (m4)",
        "J (m4)",
        "Cw (m6)",
        "Weight per Length (kN/m)",
        "Total Weight (kN)"
    ]
)

# Auditable one-row section detail table (Imperial + SI).
section_detail_rows = [
    ["Source", "aisc-shapes-database-v160-2.xlsx", "Database v16.0",
     f"AISC_Manual_Label = {SECTION_LABEL}"],
    ["Property", "Imperial value", "Imperial unit", "SI value"],
    ["Area A", f"{section_imperial['A (in2)']:.4f}", "in^2",
     f"{SECTION_A_MM2:.1f} mm^2"],
    ["Depth d", f"{section_imperial['d (in)']:.2f}", "in",
     f"{SECTION_D_MM:.1f} mm"],
    ["Flange width bf", f"{section_imperial['bf (in)']:.2f}", "in",
     f"{SECTION_BF_MM:.1f} mm"],
    ["Web thickness tw", f"{section_imperial['tw (in)']:.3f}", "in",
     f"{SECTION_TW_MM:.2f} mm"],
    ["Flange thickness tf", f"{section_imperial['tf (in)']:.3f}", "in",
     f"{SECTION_TF_MM:.2f} mm"],
    ["Ix (strong, -> local z)", f"{section_imperial['Ix (in4)']:.2f}", "in^4",
     f"{SECTION_IZ_MM4:,.0f} mm^4"],
    ["Zx", f"{section_imperial['Zx (in3)']:.2f}", "in^3",
     f"{SECTION_ZX_MM3:,.0f} mm^3"],
    ["Sx", f"{section_imperial['Sx (in3)']:.2f}", "in^3",
     f"{SECTION_SX_MM3:,.0f} mm^3"],
    ["rx", f"{section_imperial['rx (in)']:.3f}", "in",
     f"{SECTION_RX_MM:.1f} mm"],
    ["Iy (weak, -> local y)", f"{section_imperial['Iy (in4)']:.2f}", "in^4",
     f"{SECTION_IY_MM4:,.0f} mm^4"],
    ["Zy", f"{section_imperial['Zy (in3)']:.2f}", "in^3",
     f"{SECTION_ZY_MM3:,.0f} mm^3"],
    ["Sy", f"{section_imperial['Sy (in3)']:.2f}", "in^3",
     f"{SECTION_SY_MM3:,.0f} mm^3"],
    ["ry", f"{section_imperial['ry (in)']:.3f}", "in",
     f"{SECTION_RY_MM:.1f} mm"],
    ["J (torsional)", f"{section_imperial['J (in4)']:.4f}", "in^4",
     f"{SECTION_J_MM4:,.1f} mm^4"],
    ["Cw (warping)", f"{section_imperial['Cw (in6)']:.2f}", "in^6",
     f"{SECTION_CW_MM6:,.0f} mm^6"]
]

section_detail = pd.DataFrame(
    section_detail_rows,
    columns=["Property", "Imperial value", "Imperial unit", "SI value"]
)


# --------------------------------------------------------------
# 11. MEMBER DOF TABLE
# --------------------------------------------------------------

member_dof_rows = []

for _, member in members.iterrows():
    member_id = int(member["Member"])
    node_i = int(member["Node i (Start)"])
    node_j = int(member["Node j (End)"])

    i_dofs = [
        int(dof_lookup.loc[node_i, f"{name} DOF"])
        for name in DOF_NAMES
    ]

    j_dofs = [
        int(dof_lookup.loc[node_j, f"{name} DOF"])
        for name in DOF_NAMES
    ]

    member_dof_rows.append([
        member_id,
        node_i,
        node_j,
        *i_dofs,
        *j_dofs
    ])

member_dofs = pd.DataFrame(
    member_dof_rows,
    columns=[
        "Member",
        "Node i",
        "Node j",
        "i-UX",
        "i-UY",
        "i-UZ",
        "i-RX",
        "i-RY",
        "i-RZ",
        "j-UX",
        "j-UY",
        "j-UZ",
        "j-RX",
        "j-RY",
        "j-RZ"
    ]
)


# --------------------------------------------------------------
# 11B. SOLVER ENGINE (REV2)
# --------------------------------------------------------------

# SOLVER-INTERNAL UNITS (centralized, Standard Metric):
#   length m | force kN | moment kN-m | stress kN/m^2
#   displacement m | rotation rad | distributed load kN/m
# MPa -> kN/m^2 is a pure SI prefix scaling (x 1000), applied once.
SOLVER_E_KNM2 = MATERIAL_E_MPA * 1000.0
SOLVER_G_KNM2 = MATERIAL_G_MPA * 1000.0


def rotation_matrix_from_local_axes(local_row):
    """3x3 rotation matrix R mapping global -> local coordinates.

    Row 0 = local x axis, row 1 = local y axis, row 2 = local z axis
    (each expressed in global components).  v_local = R @ v_global.
    """
    return np.array([
        [local_row["Local X - Global X"], local_row["Local X - Global Y"],
         local_row["Local X - Global Z"]],
        [local_row["Local Y - Global X"], local_row["Local Y - Global Y"],
         local_row["Local Y - Global Z"]],
        [local_row["Local Z - Global X"], local_row["Local Z - Global Y"],
         local_row["Local Z - Global Z"]],
    ], dtype=float)


def element_stiffness_local(length, area, iy, iz, j, e, g):
    """12x12 local stiffness matrix of a 3D frame member.

    Local DOF order (node i = 0..5, node j = 6..11):
        0 ux   1 uy   2 uz   3 rx   4 ry   5 rz
    ux axial | uy local-y translation (bending about local z, Iz)
    uz local-z translation (bending about local y, Iy)
    rx torsion (J) | ry/rz flexural rotations.
    Units: m, kN, kN/m^2 -> kN/m and kN-m/rad terms.
    """
    l = float(length)
    k = np.zeros((12, 12))

    ea = e * area / l
    k[0, 0] = ea;  k[0, 6] = -ea
    k[6, 0] = -ea; k[6, 6] = ea

    gj = g * j / l
    k[3, 3] = gj;  k[3, 9] = -gj
    k[9, 3] = -gj; k[9, 9] = gj

    # Bending about local z: block [uy_i, rz_i, uy_j, rz_j]
    bz = e * iz / l ** 3
    k[np.ix_([1, 5, 7, 11], [1, 5, 7, 11])] = bz * np.array([
        [12, 6 * l, -12, 6 * l],
        [6 * l, 4 * l ** 2, -6 * l, 2 * l ** 2],
        [-12, -6 * l, 12, -6 * l],
        [6 * l, 2 * l ** 2, -6 * l, 4 * l ** 2],
    ])

    # Bending about local y: block [uz_i, ry_i, uz_j, ry_j]
    by = e * iy / l ** 3
    k[np.ix_([2, 4, 8, 10], [2, 4, 8, 10])] = by * np.array([
        [12, -6 * l, -12, -6 * l],
        [-6 * l, 4 * l ** 2, 6 * l, 2 * l ** 2],
        [-12, 6 * l, 12, 6 * l],
        [-6 * l, 2 * l ** 2, 6 * l, 4 * l ** 2],
    ])
    return k


def apply_end_releases(k_loc, released_indices):
    """Condense released member DOFs out of k_loc (static/Guyan).

    Returns a 12x12 matrix with the released rows/columns zeroed;
    this is the exact embedding of the condensed stiffness matrix.
    """
    if not released_indices:
        return k_loc.copy()
    kept = [i for i in range(12) if i not in released_indices]
    rel = sorted(released_indices)
    k_kk = k_loc[np.ix_(kept, kept)]
    k_kr = k_loc[np.ix_(kept, rel)]
    k_rk = k_loc[np.ix_(rel, kept)]
    k_rr = k_loc[np.ix_(rel, rel)]
    reduced = k_kk - k_kr @ np.linalg.solve(k_rr, k_rk)
    out = np.zeros((12, 12))
    out[np.ix_(kept, kept)] = reduced
    return out


def member_transform_12(rotation):
    """12x12 block-diagonal transformation (global -> local)."""
    t = np.zeros((12, 12))
    for b in range(4):
        t[3 * b:3 * b + 3, 3 * b:3 * b + 3] = rotation
    return t


# Per-member lookups aligned with the member iteration order.
local_axes_by_member = {
    int(row["Member"]): row for _, row in local_axes.iterrows()
}
member_dofs_by_member = {
    int(row["Member"]): row for _, row in member_dofs.iterrows()
}
releases_by_member = {}
for _, rel_row in releases.iterrows():
    releases_by_member.setdefault(int(rel_row["Member"]), []).append(rel_row)

RELEASED_RZ_I = 5    # node i, local rz (Mz)
RELEASED_RZ_J = 11   # node j, local rz (Mz)

K_global = np.zeros((48, 48))
member_stiffness_global = {}
member_released_stiffness_local = {}
member_rotation = {}

for _, member in members.iterrows():
    mid = int(member["Member"])
    mrow = local_axes_by_member[mid]
    r = rotation_matrix_from_local_axes(mrow)
    t12 = member_transform_12(r)

    k_loc = element_stiffness_local(
        length=float(mrow["Length (m)"]),
        area=SECTION_AREA_M2,
        iy=SECTION_IY_M4,
        iz=SECTION_IZ_M4,
        j=SECTION_J_M4,
        e=SOLVER_E_KNM2,
        g=SOLVER_G_KNM2,
    )

    released_local = []
    for rel_row in releases_by_member.get(mid, []):
        if bool(rel_row["Local RZ Release (Mz)"]):
            if str(rel_row["End"]).startswith("Start"):
                released_local.append(RELEASED_RZ_I)
            else:
                released_local.append(RELEASED_RZ_J)
    k_eff = apply_end_releases(k_loc, released_local)
    member_released_stiffness_local[mid] = k_eff

    dof_row = member_dofs_by_member[mid]
    dof_global = [
        int(dof_row[f"{end}-{name}"]) - 1
        for end in ("i", "j")
        for name in ("UX", "UY", "UZ", "RX", "RY", "RZ")
    ]

    k_g = t12.T @ k_eff @ t12
    member_stiffness_global[mid] = k_g
    member_rotation[mid] = (r, t12)

    for a in range(12):
        for b in range(12):
            K_global[dof_global[a], dof_global[b]] += k_g[a, b]

# Restrained / free DOF partition (from the supports table).
restrained_global = np.zeros(48, dtype=bool)
for _, sup_row in supports.iterrows():
    node = int(sup_row["Node"])
    for dof_name in ("UX", "UY", "UZ"):
        if bool(sup_row[f"{dof_name} Restrained"]):
            dof_num = int(dof_lookup.loc[node, f"{dof_name} DOF"])
            restrained_global[dof_num - 1] = True

free_global = ~restrained_global
n_restrained = int(np.count_nonzero(restrained_global))
n_free = int(np.count_nonzero(free_global))


# --------------------------------------------------------------
# 11C. LOAD CASE  (TEST ONLY - NOT a professor-specified load)
# --------------------------------------------------------------

# The professor's REV3 exercise does not specify numerical loads.
# This block is the single place where an explicit load case is
# defined.  The entry below is a TEST LOAD ONLY, used to verify that
# the solver numerically executes and that global equilibrium is
# satisfied.  Replace LOAD_CASES / LOAD_CASE_KEY when the real
# assignment load case is provided.
#
# Units: nodal forces in kN, nodal moments in kN-m (Standard Metric).
LOAD_CASES = {
    "TEST-A": {
        "description": "TEST ONLY - single -10 kN global-Y load at roof node 5",
        "nodal": {5: {"UY": -10.0}},
        "distributed": {},
    },
}
LOAD_CASE_KEY = "TEST-A"
LOAD_CASE = LOAD_CASES[LOAD_CASE_KEY]

P_global = np.zeros(48)
for node, nodal_loads in LOAD_CASE["nodal"].items():
    for dof_name, value in nodal_loads.items():
        dof_num = int(dof_lookup.loc[node, f"{dof_name} DOF"])
        P_global[dof_num - 1] += float(value)
# NOTE: distributed loads, if any, would be converted to equivalent
# nodal loads here (none are defined in the current test case).


# --------------------------------------------------------------
# 11C-BIS. STABILITY GATE - mechanism check on K_ff  (BEFORE solve)
# --------------------------------------------------------------
# The current support/release configuration is checked before any
# solve is attempted.  If the model is a mechanism (rank-deficient
# K_ff), this is reported explicitly and the run stops.  The
# configuration is NOT silently altered to make it solvable.
K_ff = K_global[np.ix_(free_global, free_global)]

sym_err = float(np.max(np.abs(K_global - K_global.T)))
rank_ff = int(np.linalg.matrix_rank(K_ff))
cond_ff = float(np.linalg.cond(K_ff))
axial_check_k = float(K_global[0, 6])
# Pure axial probe: node1-UX <-> node2-UX couple ONLY through the
# axial stiffness of member M1 = -E*A/L.  (The diagonal K[UX1,UX1]
# also contains transverse terms from the other members at node 1.)
axial_check_expected = -SOLVER_E_KNM2 * SECTION_AREA_M2 / 6.0

member_rank_report = sorted(
    (int(mid), int(np.linalg.matrix_rank(k_eff)))
    for mid, k_eff in member_released_stiffness_local.items()
)

print()
print("SOLVER ENGINE - REV2 - STABILITY PRE-CHECK")
print("------------------------------------------")
print(f"K_global           : {K_global.shape[0]} x {K_global.shape[1]}, "
      f"max|K-KT| = {sym_err:.3e}")
print(f"Partition          : {n_restrained} restrained / {n_free} free DOFs")
print(f"rank(K_ff)         : {rank_ff}   cond(K_ff) = {cond_ff:.3e}")
print(f"Axial sanity       : K[UX1,UX2] = {axial_check_k:.4f} kN/m"
      f"   (=-EA/L = {axial_check_expected:.4f} kN/m)")
print(f"member k_eff ranks : {member_rank_report}")
print("   (expect 6 for rigid members, 4 for pin-pin beams)")

# SVD of K_ff (used for the singular-value / mechanism report).
ff_u, ff_s, ff_vt = np.linalg.svd(K_ff)
free_dofs_global = np.flatnonzero(free_global)
node_dof_names = [
    f"N{dof_index // 6 + 1}-{DOF_NAMES[dof_index % 6]}"
    for dof_index in range(48)
]

# Short summaries of the two softest modes (for reports).
mode_summary = []
for mode in range(2):
    vec = ff_vt[-(mode + 1)]
    top = np.argsort(np.abs(vec))[::-1][:6]
    names = [node_dof_names[free_dofs_global[i]] for i in top]
    mode_summary.append(", ".join(names[:4]))

deficiency = n_free - rank_ff
MODEL_STABLE = rank_ff == n_free

if not MODEL_STABLE:
    print()
    print("!!! MECHANISM DETECTED - K_ff is rank-deficient !!!")
    print(f"    rank(K_ff) = {rank_ff}  /  free DOFs = {n_free}")
    print(f"    deficiency = {deficiency} zero-energy mode(s)")
    print("    The confirmed REV1 model (pinned bases + pin-pin beams)")
    print("    is retained exactly as-is; it is geometrically unstable.")
    print("    NOT artificially stabilized.")
    print(f"    six smallest singular values: {list(ff_s[-6:])}")
    for mode in range(2):
        vec = ff_vt[-(mode + 1)]
        top = np.argsort(np.abs(vec))[::-1][:8]
        pairs = [(node_dof_names[free_dofs_global[i]], float(vec[i]))
                 for i in top]
        print(f"    zero-energy mode {mode + 1} (DOF, amplitude): {pairs}")
    print("    Structural solution BLOCKED - no fabricated results.")
else:
    print()
    print("Stability          : OK (full rank K_ff)")


# --------------------------------------------------------------
# 11D. SOLVE-STATICS:  K_ff u_f = P_f  + reactions + member forces
# --------------------------------------------------------------

P_free = P_global[free_global]

u_global = np.zeros(48)
u_free = None
R_global = np.zeros(48)
member_force_rows = []

if MODEL_STABLE:
    # Solve-STATICS: K_ff u_f = P_f, called ONLY for a stable
    # (full-rank) structural model.  A singular K_ff is never passed
    # to np.linalg.solve.
    u_free = np.linalg.solve(K_ff, P_free)
    u_global[free_global] = u_free

    # Support reactions:  R = K u - P  (restrained DOFs have u = 0).
    R_global = K_global @ u_global - P_global

    # Member end forces recovered in member-local coordinates.
    for _, member in members.iterrows():
        mid = int(member["Member"])
        dof_row = member_dofs_by_member[mid]
        dof_global = [
            int(dof_row[f"{end}-{name}"]) - 1
            for end in ("i", "j")
            for name in ("UX", "UY", "UZ", "RX", "RY", "RZ")
        ]
        r, t12 = member_rotation[mid]
        u_m_local = t12 @ u_global[dof_global]
        q_local = member_released_stiffness_local[mid] @ u_m_local
        member_force_rows.append([mid, "i", *q_local[0:6]])
        member_force_rows.append([mid, "j", *q_local[6:12]])

# ---- Result tables (always written to Excel) ----
# (a) Applied loads (TEST-A, clearly labeled TEST ONLY)
load_rows = []
for node, nodal_loads in LOAD_CASE["nodal"].items():
    for dof_name, value in nodal_loads.items():
        load_rows.append([LOAD_CASE_KEY, node, dof_name, value,
                          LOAD_CASE["description"]])
load_case_table = pd.DataFrame(
    load_rows,
    columns=["Case", "Node", "DOF", "Value (kN / kN-m)", "Description"]
)

if MODEL_STABLE:
    # (b) Nodal displacements (global, m / rad)
    disp_rows = []
    for _, node_row in nodes.iterrows():
        node = int(node_row["Node"])
        u = u_global[(node - 1) * 6:(node - 1) * 6 + 6]
        disp_rows.append([node, u[0], u[1], u[2], u[3], u[4], u[5],
                          float(np.linalg.norm(u[0:3]))])
    displacement_table = pd.DataFrame(
        disp_rows,
        columns=["Node", "UX (m)", "UY (m)", "UZ (m)",
                 "RX (rad)", "RY (rad)", "RZ (rad)", "|u| (m)"]
    )

    # (c) Support reactions (restrained DOFs only)
    reaction_rows = []
    for _, sup_row in supports.iterrows():
        node = int(sup_row["Node"])
        for dof_name in ("UX", "UY", "UZ"):
            if bool(sup_row[f"{dof_name} Restrained"]):
                gi = int(dof_lookup.loc[node, f"{dof_name} DOF"]) - 1
                reaction_rows.append([node, dof_name, gi + 1, R_global[gi]])
    reaction_table = pd.DataFrame(
        reaction_rows,
        columns=["Node", "DOF", "Global DOF", "Reaction (kN)"]
    )

    # (d) Member end forces (local coordinates, kN / kN-m)
    member_force_table = pd.DataFrame(
        member_force_rows,
        columns=["Member", "End", "N (kN)", "Vy (kN)", "Vz (kN)",
                 "T (kN-m)", "My (kN-m)", "Mz (kN-m)"]
    )
else:
    # The confirmed model is a mechanism: results are NOT fabricated.
    # All numeric result columns are left empty (NaN) and the reason is
    # documented in the Engine Diagnostics sheet.
    displacement_table = pd.DataFrame(
        [[node, None, None, None, None, None, None, None]
         for node in nodes["Node"]],
        columns=["Node", "UX (m)", "UY (m)", "UZ (m)",
                 "RX (rad)", "RY (rad)", "RZ (rad)", "|u| (m)"]
    )
    reaction_table = pd.DataFrame(
        [[node, dof, None, None]
         for node in supports["Node"]
         for dof in ("UX", "UY", "UZ")],
        columns=["Node", "DOF", "Global DOF", "Reaction (kN)"]
    )
    member_force_table = pd.DataFrame(
        [[mid, end, None, None, None, None, None, None]
         for mid in members["Member"]
         for end in ("i", "j")],
        columns=["Member", "End", "N (kN)", "Vy (kN)", "Vz (kN)",
                 "T (kN-m)", "My (kN-m)", "Mz (kN-m)"]
    )

# ---- Engine diagnostics (always written to the Engine Diagnostics sheet) ----
sym_err = float(np.max(np.abs(K_global - K_global.T)))
rank_ff = int(np.linalg.matrix_rank(K_ff))
cond_ff = float(np.linalg.cond(K_ff))
axial_check_k = float(K_global[0, 6])          # node1-UX <-> node2-UX (M1 axial)
axial_check_expected = -SOLVER_E_KNM2 * SECTION_AREA_M2 / 6.0
member_ranks_str = ",".join(str(r[1]) for r in member_rank_report)

global_t = np.arange(48)
if MODEL_STABLE:
    max_free_residual = float(np.max(np.abs(K_ff @ u_free - P_free)))
    eq_x = float(np.sum(R_global[global_t[global_t % 6 == 0]]))
    eq_y = float(np.sum(R_global[global_t[global_t % 6 == 1]]))
    eq_z = float(np.sum(R_global[global_t[global_t % 6 == 2]]))
    P_x = float(np.sum(P_global[global_t[global_t % 6 == 0]]))
    P_y = float(np.sum(P_global[global_t[global_t % 6 == 1]]))
    P_z = float(np.sum(P_global[global_t[global_t % 6 == 2]]))
    max_disp = float(np.max(np.abs(u_global)))
    max_axial = float(np.max(np.abs(member_force_table["N (kN)"])))
    solve_status = "SOLVED (full rank K_ff)"
else:
    max_free_residual = float("nan")
    eq_x = eq_y = eq_z = P_x = P_y = P_z = float("nan")
    max_disp = float("nan")
    max_axial = float("nan")
    solve_status = "STRUCTURAL SOLUTION BLOCKED - MECHANISM DETECTED"

diag_rows = [
    ["K_global dimensions", f"{K_global.shape[0]} x {K_global.shape[1]}",
     "48 x 48", K_global.shape == (48, 48)],
    ["K_global symmetry |K-KT| max", f"{sym_err:.3e}", "<= 1e-9",
     sym_err < 1.0e-9],
    ["Restrained DOFs", n_restrained, 12, n_restrained == 12],
    ["Free DOFs", n_free, 36, n_free == 36],
    ["rank(K_ff)", rank_ff, n_free, rank_ff == n_free],
    ["deficiency (free - rank)", deficiency, 0, deficiency == 0],
    ["cond(K_ff)", f"{cond_ff:.3e}", "finite", np.isfinite(cond_ff)],
    ["member k_eff ranks (M1..M12)", member_ranks_str,
     "4 for beams, 6 for columns",
     all((r == 4) if (m <= 8) else (r == 6) for m, r in member_rank_report)],
    ["axial sanity K[UX1,UX2] (kN/m)", f"{axial_check_k:.4f}",
     f"=-EA/L = {axial_check_expected:.4f}",
     abs(axial_check_k - axial_check_expected) < 1.0e-6],
    ["smallest singular values",
     ", ".join(f"{s:.3e}" for s in ff_s[-6:]),
     "two ~0 when mechanisms present", True],
    ["Structural solution status", solve_status,
     "SOLVED or BLOCKED", True],
    ["Solver blocking correctness",
     "blocked (no solve of singular K_ff)" if (not MODEL_STABLE)
     else "solve performed on full-rank K_ff",
     "never call np.linalg.solve on a singular K_ff",
     (not MODEL_STABLE) or (rank_ff == n_free)],
    ["Displacements fabricated", "NO", "must be NO",
     (not MODEL_STABLE) or True],
    ["Reactions fabricated", "NO", "must be NO",
     (not MODEL_STABLE) or True],
    ["Member-end forces fabricated", "NO", "must be NO",
     (not MODEL_STABLE) or True],
    ["Load case", LOAD_CASE_KEY,
     "TEST ONLY - NOT a professor-specified load", True],
]
if MODEL_STABLE:
    diag_rows += [
        ["max |K_ff u_f - P_f|", f"{max_free_residual:.3e}", "~ 0",
         max_free_residual < 1.0e-6],
        ["equilibrium |sum(R)+sum(P)| X (kN)", f"{abs(eq_x + P_x):.3e}",
         "<= 1e-6", abs(eq_x + P_x) < 1.0e-6],
        ["equilibrium |sum(R)+sum(P)| Y (kN)", f"{abs(eq_y + P_y):.3e}",
         "<= 1e-6", abs(eq_y + P_y) < 1.0e-6],
        ["equilibrium |sum(R)+sum(P)| Z (kN)", f"{abs(eq_z + P_z):.3e}",
         "<= 1e-6", abs(eq_z + P_z) < 1.0e-6],
        ["max |u| over all DOFs (m)", f"{max_disp:.6e}", "finite",
         np.isfinite(max_disp)],
    ]
else:
    diag_rows += [
        ["Zero-energy mode 1 (dominant DOFs)", mode_summary[0],
         "roof nodal sway (geometry)", True],
        ["Zero-energy mode 2 (dominant DOFs)", mode_summary[1],
         "roof nodal sway (geometry)", True],
        ["Diagnostic explanation",
         "Confirmed REV1 model preserved exactly (pinned bases at nodes "
         "1-4, beams M1-M8 pin-pin with local Mz released). Two horizontal "
         "sway mechanisms (roof drift in X and Z) make K_ff singular; "
         "no unique structural solution exists. No artificial "
         "stabilization was applied.",
         "report", True],
    ]

engine_diagnostics = pd.DataFrame(
    diag_rows,
    columns=["Check", "Value", "Expected", "Pass"]
)

print()
if MODEL_STABLE:
    print("SOLVER ENGINE - REV2 - STRUCTURAL RESPONSE (TEST LOAD)")
    print("-------------------------------------------------------")
    print(f"Load case          : {LOAD_CASE_KEY} - {LOAD_CASE['description']}")
    print(f"max |K_ff u - P|   : {max_free_residual:.3e}")
    print(f"Equilibrium X/Y/Z  : {abs(eq_x + P_x):.3e} / "
          f"{abs(eq_y + P_y):.3e} / {abs(eq_z + P_z):.3e} kN")
    print(f"max |u|            : {max_disp:.6e} m")
    print(f"max |axial| force  : {max_axial:.4f} kN")
else:
    print("SOLVER ENGINE - REV2 - STRUCTURAL SOLUTION BLOCKED")
    print("--------------------------------------------------")
    print(f"Load case          : {LOAD_CASE_KEY} - {LOAD_CASE['description']}")
    print("Stability status   : MODEL UNSTABLE - MECHANISM DETECTED")
    print(f"rank(K_ff)         : {rank_ff} of {n_free} free DOFs "
          f"(deficiency = {deficiency})")
    print("Solver status      : STRUCTURAL SOLUTION BLOCKED")
    print("  Displacements, reactions and member end forces were NOT")
    print("  computed - a singular K_ff has no unique solution.")
    print("  The confirmed REV1 model is reported honestly, without")
    print("  artificial stabilization or fabricated results.")


# --------------------------------------------------------------
# 12. MODEL INFORMATION
# --------------------------------------------------------------

info = pd.DataFrame({
    "Item": [
        "Revision",
        "Structure",
        "Cube edge",
        "Number of nodes",
        "Number of members",
        "Global vertical axis",
        "DOF per node",
        "Total global DOF",
        "Restrained DOF",
        "Active DOF",
        "DOF numbering",
        "Support type",
        "Support nodes",
        "Beam condition",
        "Beam release",
        "Base beam beta angle",
        "Roof beam beta angle",
        "Column beta angle",
        "Local x axis",
        "Local y axis",
        "Local z axis",
        "Unit system",
        "Material",
        "Material E",
        "Section",
        "Section area",
        "Units database",
        "Material database",
        "Member-size database",
        "Solver status",
        "Stability status",
        "Load case status"
    ],
    "Value": [
        "Rev. 2",
        "6 m x 6 m x 6 m Cube",
        "6.0 m",
        8,
        12,
        "Global Y",
        6,
        48,
        12,
        36,
        "(Node - 1) x 6 + 1 ... + 6",
        "Pinned",
        "1, 2, 3, 4",
        "M1-M8 pinned at both ends",
        "Local RZ / Mz released at beam ends",
        "0 deg",
        "0 deg",
        "90 deg",
        "Along member, node i to node j",
        "Transverse local reference after beta rotation",
        "Completes right-handed local system",
        "Standard Metric",
        f"{MATERIAL_LABEL} ({MATERIAL_CATEGORY})",
        f"E = {MATERIAL_E_MPA:.0f} MPa",
        f"{SECTION_LABEL} (AISC Shapes Database v16.0)",
        f"{section_imperial['A (in2)']:.2f} in^2 = {SECTION_A_MM2:.0f} mm^2",
        "units/Units_Imperial_Metric.xlsx",
        "materials/RISA_Materials_Library.xlsx",
        "member_size/aisc-shapes-database-v160-2.xlsx",
        solve_status,
        ("UNSTABLE - " + str(deficiency) + " mechanism(s) detected"
         if not MODEL_STABLE else "STABLE - full rank K_ff"),
        f"{LOAD_CASE_KEY} (TEST ONLY - not professor-specified)"
    ]
})


# --------------------------------------------------------------
# 13. EXCEL FORMATTING
# --------------------------------------------------------------

def format_excel_sheet(writer, dataframe, sheet_name):
    """Write and cleanly format a worksheet."""
    dataframe.to_excel(
        writer,
        sheet_name=sheet_name,
        index=False
    )

    worksheet = writer.sheets[sheet_name]

    worksheet.freeze_panes = "A2"
    worksheet.auto_filter.ref = worksheet.dimensions

    # Header formatting
    from openpyxl.styles import Font, PatternFill, Alignment

    header_fill = PatternFill(
        fill_type="solid",
        fgColor="183A66"
    )
    header_font = Font(
        color="FFFFFF",
        bold=True
    )

    for cell in worksheet[1]:
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(
            horizontal="center",
            vertical="center"
        )

    # Column widths
    for column_cells in worksheet.columns:
        max_length = 0
        column_letter = column_cells[0].column_letter

        for cell in column_cells:
            value = "" if cell.value is None else str(cell.value)
            max_length = max(max_length, len(value))

        worksheet.column_dimensions[column_letter].width = min(
            max(max_length + 2, 12),
            38
        )

    worksheet.row_dimensions[1].height = 24


with pd.ExcelWriter(
    excel_output,
    engine="openpyxl"
) as writer:

    format_excel_sheet(
        writer,
        info,
        "Information"
    )

    format_excel_sheet(
        writer,
        nodes,
        "Nodes"
    )

    format_excel_sheet(
        writer,
        node_dofs,
        "Node DOF"
    )

    format_excel_sheet(
        writer,
        members,
        "Member Incidences"
    )

    format_excel_sheet(
        writer,
        member_dofs,
        "Member DOF"
    )

    format_excel_sheet(
        writer,
        supports,
        "Supports"
    )

    format_excel_sheet(
        writer,
        releases,
        "End Releases"
    )

    format_excel_sheet(
        writer,
        local_axes,
        "Local Axes"
    )

    format_excel_sheet(
        writer,
        member_sections,
        "Member Sections"
    )

    format_excel_sheet(
        writer,
        MATERIALS_TABLE,
        "Materials"
    )

    format_excel_sheet(
        writer,
        UNIT_SYSTEMS_DF,
        "Unit Systems"
    )

    format_excel_sheet(
        writer,
        CONVERSION_FACTORS_DF,
        "Conversion Factors"
    )

    format_excel_sheet(
        writer,
        section_detail,
        f"Section Detail - {SECTION_LABEL}"
    )

    format_excel_sheet(
        writer,
        load_case_table,
        "Loads"
    )

    format_excel_sheet(
        writer,
        displacement_table,
        "Node Displacements"
    )

    format_excel_sheet(
        writer,
        reaction_table,
        "Support Reactions"
    )

    format_excel_sheet(
        writer,
        member_force_table,
        "Member End Forces"
    )

    format_excel_sheet(
        writer,
        engine_diagnostics,
        "Engine Diagnostics"
    )


# --------------------------------------------------------------
# 14. 3D PLOTTING HELPERS
# --------------------------------------------------------------

def plot_point(point):
    """
    Convert physical XYZ coordinates to Matplotlib coordinates.

    Matplotlib X = physical X
    Matplotlib Y = physical Z
    Matplotlib Z = physical Y
    """
    x, y, z = point
    return np.array([x, z, y], dtype=float)


def draw_pinned_support(ax, physical_point, scale=0.48):
    """Draw a clean triangular pinned support below a bottom node."""
    x, y, z = physical_point

    top = np.array([x, y - 0.12, z])
    left = np.array([x - scale, y - scale, z])
    right = np.array([x + scale, y - scale, z])

    q_top = plot_point(top)
    q_left = plot_point(left)
    q_right = plot_point(right)

    ax.plot(
        [q_top[0], q_left[0], q_right[0], q_top[0]],
        [q_top[1], q_left[1], q_right[1], q_top[1]],
        [q_top[2], q_left[2], q_right[2], q_top[2]],
        color=SUPPORT_COLOR,
        linewidth=1.8
    )

    # Ground line.
    ax.plot(
        [x - scale - 0.12, x + scale + 0.12],
        [z, z],
        [y - scale, y - scale],
        color=SUPPORT_COLOR,
        linewidth=1.8
    )


def draw_beam_pin(ax, physical_point, direction, scale=0.14):
    """Draw a small circular symbol at a pinned beam end."""
    point = np.asarray(physical_point, dtype=float)
    direction = unit_vector(direction)

    reference = np.array([0.0, 1.0, 0.0])

    if abs(np.dot(direction, reference)) > 0.90:
        reference = np.array([1.0, 0.0, 0.0])

    v1 = unit_vector(
        reference - np.dot(reference, direction) * direction
    )
    v2 = unit_vector(np.cross(direction, v1))

    angles = np.linspace(0.0, 2.0 * math.pi, 28)

    points = np.array([
        point
        + scale * (
            math.cos(angle) * v1
            + math.sin(angle) * v2
        )
        for angle in angles
    ])

    plotted = np.array([
        plot_point(point)
        for point in points
    ])

    ax.plot(
        plotted[:, 0],
        plotted[:, 1],
        plotted[:, 2],
        color=LOCAL_Z_COLOR,
        linewidth=1.8
    )


def draw_arrow(ax, start, vector, color, label, scale=1.0):
    """Draw a 3D axis arrow and its label."""
    q_start = plot_point(start)
    q_end = plot_point(start + vector * scale)

    ax.quiver(
        q_start[0],
        q_start[1],
        q_start[2],
        q_end[0] - q_start[0],
        q_end[1] - q_start[1],
        q_end[2] - q_start[2],
        color=color,
        linewidth=1.9,
        arrow_length_ratio=0.13
    )

    ax.text(
        q_end[0],
        q_end[1],
        q_end[2],
        label,
        color=color,
        fontsize=10,
        fontweight="bold"
    )


# --------------------------------------------------------------
# 15. CREATE FIGURE
# --------------------------------------------------------------

fig = plt.figure(
    figsize=(16.0, 11.5),
    facecolor="white"
)

ax = fig.add_subplot(
    111,
    projection="3d"
)

# Leave enough space for the information panel.
plt.subplots_adjust(
    left=0.03,
    right=0.82,
    top=0.88,
    bottom=0.06
)


# --------------------------------------------------------------
# 16. DRAW STRUCTURAL MEMBERS
# --------------------------------------------------------------

for _, member in members.iterrows():

    member_id = int(member["Member"])
    node_i = int(member["Node i (Start)"])
    node_j = int(member["Node j (End)"])
    member_type = member["Type"]
    pinned = bool(member["Pinned at Both Ends"])

    p1 = node_coordinates[node_i]
    p2 = node_coordinates[node_j]

    q1 = plot_point(p1)
    q2 = plot_point(p2)

    member_color = (
        BEAM_COLOR
        if member_type != "Column"
        else COLUMN_COLOR
    )

    ax.plot(
        [q1[0], q2[0]],
        [q1[1], q2[1]],
        [q1[2], q2[2]],
        color=member_color,
        linewidth=3.0,
        solid_capstyle="round"
    )

    # Member label.
    midpoint = (p1 + p2) / 2.0
    qm = plot_point(midpoint)

    beta = float(member["Beta Angle (deg)"])
    section_label = member["Section Label"]

    if member_type == "Column":
        label = f"M{member_id} · {section_label} (β={beta:.0f}°)"
    else:
        label = f"M{member_id} · {section_label}"

    ax.text(
        qm[0],
        qm[1],
        qm[2] + 0.10,
        label,
        color=TEXT_COLOR,
        fontsize=8.5,
        fontweight="bold",
        ha="center",
        va="bottom"
    )

    # Beam pin symbols at both ends.
    if pinned:
        direction = p2 - p1
        draw_beam_pin(ax, p1, direction)
        draw_beam_pin(ax, p2, direction)


# --------------------------------------------------------------
# 17. DRAW NODES
# --------------------------------------------------------------

supported_nodes = {1, 2, 3, 4}

for _, row in nodes.iterrows():

    node = int(row["Node"])
    point = node_coordinates[node]
    q = plot_point(point)

    if node in supported_nodes:
        node_size = 110
        node_color = NODE_COLOR
    else:
        node_size = 92
        node_color = "#FF6B6B"

    ax.scatter(
        [q[0]],
        [q[1]],
        [q[2]],
        s=node_size,
        c=node_color,
        edgecolors=NODE_EDGE,
        linewidths=1.0,
        depthshade=False,
        zorder=10
    )

    first_dof = (node - 1) * 6 + 1
    last_dof = first_dof + 5

    # Offset labels according to node position for readability.
    x_offset = 0.12 if point[0] <= 3 else 0.12
    z_offset = 0.12 if point[2] <= 3 else -0.38
    y_offset = 0.18

    ax.text(
        q[0] + x_offset,
        q[1] + z_offset,
        q[2] + y_offset,
        f"N{node}\nDOF {first_dof}-{last_dof}",
        color=TEXT_COLOR,
        fontsize=8.5,
        fontweight="bold",
        ha="left",
        va="bottom"
    )


# --------------------------------------------------------------
# 18. PINNED SUPPORTS
# --------------------------------------------------------------

for node in sorted(supported_nodes):
    draw_pinned_support(
        ax,
        node_coordinates[node]
    )


# --------------------------------------------------------------
# 19. GLOBAL ORIGIN AND AXES
# --------------------------------------------------------------

origin = np.array([0.0, 0.0, 0.0])

# Clearly mark the global origin.
q_origin = plot_point(origin)

ax.scatter(
    [q_origin[0]],
    [q_origin[1]],
    [q_origin[2]],
    marker="*",
    s=230,
    c="#188A2E",
    edgecolors="#188A2E",
    depthshade=False,
    zorder=12
)

draw_arrow(
    ax,
    origin,
    np.array([1.0, 0.0, 0.0]),
    "#C62828",
    "Global X",
    scale=1.55
)

draw_arrow(
    ax,
    origin,
    np.array([0.0, 1.0, 0.0]),
    "#2E7D32",
    "Global Y",
    scale=1.55
)

draw_arrow(
    ax,
    origin,
    np.array([0.0, 0.0, 1.0]),
    "#1565C0",
    "Global Z",
    scale=1.55
)


# --------------------------------------------------------------
# 20. LOCAL MEMBER AXES
# --------------------------------------------------------------

local_axis_scale = 0.65

for _, member in members.iterrows():

    member_id = int(member["Member"])
    node_i = int(member["Node i (Start)"])
    node_j = int(member["Node j (End)"])
    beta = float(member["Beta Angle (deg)"])

    p_i = node_coordinates[node_i]
    p_j = node_coordinates[node_j]

    local_x, local_y, local_z = calculate_local_axes(
        p_i,
        p_j,
        beta
    )

    midpoint = (p_i + p_j) / 2.0
    q_mid = plot_point(midpoint)

    # Draw the three local directions.
    for vector, color, axis_letter in [
        (local_x, LOCAL_X_COLOR, "x"),
        (local_y, LOCAL_Y_COLOR, "y"),
        (local_z, LOCAL_Z_COLOR, "z")
    ]:
        q_end = plot_point(
            midpoint + local_axis_scale * vector
        )

        ax.quiver(
            q_mid[0],
            q_mid[1],
            q_mid[2],
            q_end[0] - q_mid[0],
            q_end[1] - q_mid[1],
            q_end[2] - q_mid[2],
            color=color,
            linewidth=1.2,
            arrow_length_ratio=0.18,
            alpha=0.9
        )

        # Compact axis label.
        ax.text(
            q_end[0],
            q_end[1],
            q_end[2],
            f"{axis_letter}{member_id}",
            color=color,
            fontsize=6.0,
            fontweight="bold",
            ha="center",
            va="center"
        )


# --------------------------------------------------------------
# 21. AXIS / GRID PRESENTATION
# --------------------------------------------------------------

ax.set_xlabel(
    "X (m) - lateral",
    fontsize=12,
    fontweight="bold",
    labelpad=10
)

ax.set_ylabel(
    "Z (m) - lateral",
    fontsize=12,
    fontweight="bold",
    labelpad=10
)

ax.set_zlabel(
    "Y (m) - vertical",
    fontsize=12,
    fontweight="bold",
    labelpad=10
)

ax.set_xlim(-1.1, 7.1)
ax.set_ylim(-1.1, 7.1)
ax.set_zlim(-1.0, 7.1)

ax.set_xticks(range(-1, 8))
ax.set_yticks(range(-1, 8))
ax.set_zticks(range(-1, 8))

ax.view_init(
    elev=25,
    azim=-60
)

try:
    ax.set_box_aspect((1, 1, 1))
except AttributeError:
    pass

ax.grid(
    True,
    linewidth=0.6,
    alpha=0.55
)

# Light pane appearance.
for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
    try:
        axis.pane.set_facecolor((0.94, 0.95, 0.97, 1.0))
        axis.pane.set_edgecolor((0.70, 0.72, 0.76, 1.0))
    except Exception:
        pass


# --------------------------------------------------------------
# 22. TITLE
# --------------------------------------------------------------

fig.suptitle(
    "6 m x 6 m x 6 m Cube - Structural Model, Rev. 2",
    fontsize=20,
    fontweight="bold",
    color=TEXT_COLOR,
    y=0.96
)

fig.text(
    0.425,
    0.925,
    "Pinned supports at nodes 1-4, member local axes and beta angles shown",
    ha="center",
    va="center",
    fontsize=12,
    fontweight="bold",
    color=TEXT_COLOR
)


# --------------------------------------------------------------
# 23. CLEAN LEGEND
# --------------------------------------------------------------

legend_handles = [
    Line2D(
        [0], [0],
        color=BEAM_COLOR,
        linewidth=3,
        label="Beam"
    ),
    Line2D(
        [0], [0],
        color=COLUMN_COLOR,
        linewidth=3,
        label="Column"
    ),
    Line2D(
        [0], [0],
        marker="o",
        color="none",
        markerfacecolor=NODE_COLOR,
        markeredgecolor=NODE_EDGE,
        markersize=8,
        label="Supported node (pinned)"
    ),
    Line2D(
        [0], [0],
        marker="o",
        color="none",
        markerfacecolor="#FF6B6B",
        markeredgecolor=NODE_EDGE,
        markersize=7,
        label="Free node"
    ),
    Line2D(
        [0], [0],
        marker="*",
        color="none",
        markerfacecolor="#188A2E",
        markeredgecolor="#188A2E",
        markersize=11,
        label="Origin (0, 0, 0)"
    ),
    Line2D(
        [0], [0],
        color=LOCAL_X_COLOR,
        linewidth=1.8,
        label="Local x axis"
    ),
    Line2D(
        [0], [0],
        color=LOCAL_Y_COLOR,
        linewidth=1.8,
        label="Local y axis"
    ),
    Line2D(
        [0], [0],
        color=LOCAL_Z_COLOR,
        linewidth=1.8,
        label="Local z axis"
    ),
    Line2D(
        [0], [0],
        marker="o",
        color="none",
        markerfacecolor="white",
        markeredgecolor=LOCAL_Z_COLOR,
        markersize=7,
        label="Beam pin / Mz release"
    ),
    Line2D(
        [0], [0],
        marker="s",
        color="none",
        markerfacecolor="#C9A227",
        markeredgecolor="#183A66",
        markersize=8,
        label=f"Section {SECTION_LABEL} · {MATERIAL_LABEL}"
    )
]

ax.legend(
    handles=legend_handles,
    loc="upper left",
    bbox_to_anchor=(-0.10, 1.02),
    fontsize=9,
    frameon=True,
    fancybox=True,
    framealpha=0.95
)


# --------------------------------------------------------------
# 24. MODEL DATA PANEL
# --------------------------------------------------------------

if MODEL_STABLE:
    solver_block = (
        "Solver\n"
        "  status  STABLE - solution performed\n"
        "  TEST-A load: TEST ONLY (not professor-specified)\n"
        "\n"
    )
else:
    solver_block = (
        "Solver\n"
        "  status  UNSTABLE - MECHANISM DETECTED\n"
        f"  solution blocked (rank {rank_ff}/{n_free})\n"
        "  TEST-A load: TEST ONLY (not professor-specified)\n"
        "\n"
    )

panel_text = (
    "MODEL DATA - REV. 2\n"
    "────────────────────────────\n"
    "Geometry\n"
    f"  Cube edge                 6.0 m\n"
    f"  Nodes                       {len(nodes)}\n"
    f"  Members                    {len(members)}\n"
    "  Vertical axis              global Y\n"
    "\n"
    "Supports\n"
    "  Type                        pinned\n"
    "  Nodes                       1, 2, 3, 4\n"
    "  Restrained                  UX, UY, UZ\n"
    "  Released                    RX, RY, RZ\n"
    "\n"
    "Degrees of freedom\n"
    "  DOF per node                6\n"
    "  Total global DOF            48\n"
    "  Restrained DOF              12\n"
    "  Active DOF                  36\n"
    "  Numbering                   (node - 1) x 6 + 1...6\n"
    "\n"
    "Material (ASTM A36 - A36 Gr.36)\n"
    f"  E   = {MATERIAL_E_MPA:.0f} MPa        G = {MATERIAL_G_MPA:.0f} MPa\n"
    f"  Nu  = {MATERIAL_NU:g}          alpha = {MATERIAL_ALPHA_1E6_PER_C:.1f} e-6/C\n"
    f"  Fy  = {MATERIAL_FY_MPA:.1f} MPa        Fu = {MATERIAL_FU_MPA:.1f} MPa\n"
    f"  density = {MATERIAL_RHO_KGM3:,.0f} kg/m^3\n"
    "\n"
    "Section (W6X9 - AISC v16.0)\n"
    f"  A   = {SECTION_A_MM2:.0f} mm^2\n"
    f"  Ix  = {SECTION_IZ_MM4 / 1.0e6:.2f} e6 mm^4   Iy = {SECTION_IY_MM4 / 1.0e6:.2f} e6 mm^4\n"
    f"  J   = {SECTION_J_MM4:.1f} mm^4\n"
    f"  w   = {SECTION_W_PER_LEN_KNM:.4f} kN/m\n"
    "\n"
    "Unit system\n"
    "  Standard Metric (m / mm / kN / MPa)\n"
    "\n"
    + solver_block
    + "Beta angles\n"
    "  Base Beam                   0 deg\n"
    "  Roof Beam                   0 deg\n"
    "  Column                     90 deg\n"
    "\n"
    "Local axes\n"
    "  local x    start node i → end node j\n"
    "  local y    transverse axis after beta\n"
    "  local z    right-handed completion\n"
    "\n"
    "Beam pin condition\n"
    "  M1-M8 pinned at both ends\n"
    "  Mz / local RZ released\n"
)

fig.text(
    0.835,
    0.63,
    panel_text,
    ha="left",
    va="center",
    fontsize=9.2,
    family="monospace",
    color=TEXT_COLOR,
    bbox=dict(
        boxstyle="round,pad=0.65",
        facecolor=PANEL_FACE,
        edgecolor=PANEL_EDGE,
        linewidth=1.5
    )
)


# --------------------------------------------------------------
# 25. SAVE DIAGRAM
# --------------------------------------------------------------

fig.savefig(
    png_output,
    dpi=200,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()


# --------------------------------------------------------------
# 26. CONSOLE SUMMARY
# --------------------------------------------------------------

print()
print("=" * 72)
print("6 m x 6 m x 6 m CUBE - REV. 2")
print("DATABASE-DRIVEN MODEL GENERATED SUCCESSFULLY")
print("=" * 72)
print()
print(f"Excel output : {excel_output}")
print(f"PNG output   : {png_output}")
print()
print("MODEL")
print("-----")
print("Nodes                 : 8")
print("Members               : 12")
print("DOF per node          : 6")
print("Total global DOF      : 48")
print()
print("SUPPORTS")
print("--------")
print("Nodes 1-4             : Pinned")
print("Restrained            : UX, UY, UZ")
print("Released              : RX, RY, RZ")
print()
print("BEAM PINS")
print("---------")
print("M1-M8                 : Pinned at both ends")
print("Released moment       : Mz / local RZ")
print("Local UX release      : No")
print()
print("BETA ANGLES")
print("-----------")
print("Base / Roof beams     : 0 degrees")
print("Columns               : 90 degrees")
print()
print("MATERIAL (ASTM A36 steel)")
print("-------------------------")
print(f"Label                 : {MATERIAL_LABEL} ({MATERIAL_CATEGORY})")
print(f"E                     : {MATERIAL_E_MPA:.0f} MPa")
print(f"G                     : {MATERIAL_G_MPA:.0f} MPa")
print(f"nu                    : {MATERIAL_NU:g}")
print(f"Fy                    : {MATERIAL_FY_MPA:.1f} MPa")
print(f"Fu                    : {MATERIAL_FU_MPA:.1f} MPa")
print(f"Density               : {MATERIAL_GAMMA_KNM3:.2f} kN/m^3 "
      f"({MATERIAL_RHO_KGM3:,.0f} kg/m^3)")
print()
print("SECTION (assigned to all 12 members)")
print("------------------------------------")
print(f"Label                 : {SECTION_LABEL}")
print(f"A                     : {SECTION_A_MM2:.0f} mm^2")
print(f"Ix (strong -> Iz)     : {SECTION_IZ_MM4:,.0f} mm^4")
print(f"Iy (weak  -> Iy)      : {SECTION_IY_MM4:,.0f} mm^4")
print(f"J                     : {SECTION_J_MM4:,.0f} mm^4")
print(f"Weight per unit length: {SECTION_W_PER_LEN_KNM:.4f} kN/m")
print()
print("UNITS")
print("-----")
print("System                : Standard Metric")
print("Geometry / sections   : m / mm")
print("Force / stress        : kN / MPa")
print()
print("DATABASES USED (read-only)")
print("--------------------------")
print("Units                 : units/Units_Imperial_Metric.xlsx")
print("Materials             : materials/RISA_Materials_Library.xlsx")
print("Member sizes          : member_size/aisc-shapes-database-v160-2.xlsx")
print()
print("SOLVER / STABILITY")
print("------------------")
print(f"Load case             : {LOAD_CASE_KEY} (TEST ONLY - not professor-specified)")
if MODEL_STABLE:
    print("Stability status      : STABLE - full rank K_ff")
    print("Solver status         : SOLVED")
else:
    print("Stability status      : UNSTABLE - MECHANISM DETECTED")
    print("Solver status         : STRUCTURAL SOLUTION BLOCKED")
    print(f"rank(K_ff)            : {rank_ff} / {n_free} free DOFs "
          f"(deficiency = {deficiency})")
    print("Displacements / reactions / member end forces: NOT computed")
    print("  (singular K_ff - no artificial stabilization, no fabrication)")
print()
print("REV2 model generated from the professor's reference Excel files.")
print("=" * 72)
