from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch


# ==============================================================
# 6 m x 6 m x 6 m CUBE - STRUCTURAL SOLVER / MODEL GENERATOR
# REV. 1
#
# Presentation features:
#   - Pinned supports at bottom nodes
#   - Six DOF per node
#   - Global and local member axes
#   - Beta angles
#   - Beam end pins / Mz (local RZ) releases
#   - Clean RISA/STAAD-style structural diagram
#   - Formatted Excel output
#
# NOTE:
# This program does not read the professor's reference Excel file.
# It creates a new Excel workbook and a new PNG diagram.
# ==============================================================


# --------------------------------------------------------------
# 1. FILE LOCATIONS
# --------------------------------------------------------------

folder = Path(__file__).resolve().parent

excel_output = folder / "cube_nodes_6m_Rev1_generated.xlsx"
png_output = folder / "cube_6m_Rev1_generated.png"


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

members_data = [
    [1,  1, 2, "Base Beam", 0.0, True],
    [2,  2, 3, "Base Beam", 0.0, True],
    [3,  3, 4, "Base Beam", 0.0, True],
    [4,  4, 1, "Base Beam", 0.0, True],

    [5,  5, 6, "Roof Beam", 0.0, True],
    [6,  6, 7, "Roof Beam", 0.0, True],
    [7,  7, 8, "Roof Beam", 0.0, True],
    [8,  8, 5, "Roof Beam", 0.0, True],

    [9,  1, 5, "Column", 90.0, False],
    [10, 2, 6, "Column", 90.0, False],
    [11, 3, 7, "Column", 90.0, False],
    [12, 4, 8, "Column", 90.0, False],
]

members = pd.DataFrame(
    members_data,
    columns=[
        "Member",
        "Node i (Start)",
        "Node j (End)",
        "Type",
        "Beta Angle (deg)",
        "Pinned at Both Ends"
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
        "Local z axis"
    ],
    "Value": [
        "Rev. 1",
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
        "Completes right-handed local system"
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

    if member_type == "Column":
        label = f"M{member_id} (β={beta:.0f}°)"
    else:
        label = f"M{member_id}"

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
    "6 m x 6 m x 6 m Cube - Structural Model, Rev. 1",
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

panel_text = (
    "MODEL DATA - REV. 1\n"
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
    "Beta angles\n"
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
print("6 m x 6 m x 6 m CUBE - REV. 1")
print("STRUCTURAL MODEL GENERATED SUCCESSFULLY")
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
print("The professor's reference Excel file was not used.")
print("=" * 72)
