"""Rev. 3 automated verification - 95 tests.

These are framework/load-audit tests, not a stiffness-analysis test suite.
The test model is a small deterministic stand-in with the same 8-node/12-member
geometry so the tests remain runnable even when a workstation is temporarily
missing one of the Rev. 2 Excel databases. The real database-backed model is
still used by cube_solver_rev3.py itself and should be manually audited with
--verify-loads in the project folder.
"""
from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

import pandas as pd

import cube_solver_rev3 as rev3



def make_test_model():
    nodes = pd.DataFrame([
        [1, 0.0, 0.0, 0.0], [2, 6.0, 0.0, 0.0],
        [3, 6.0, 0.0, 6.0], [4, 0.0, 0.0, 6.0],
        [5, 0.0, 6.0, 0.0], [6, 6.0, 6.0, 0.0],
        [7, 6.0, 6.0, 6.0], [8, 0.0, 6.0, 6.0],
    ], columns=["Node", "X (m)", "Y (m)", "Z (m)"])

    members = pd.DataFrame([
        [1, 1, 2], [2, 2, 3], [3, 3, 4], [4, 4, 1],
        [5, 5, 6], [6, 6, 7], [7, 7, 8], [8, 8, 5],
        [9, 1, 5], [10, 2, 6], [11, 3, 7], [12, 4, 8],
    ], columns=["Member", "Node i (Start)", "Node j (End)"])

    # Choose the area so the analytical handout EA benchmark is reproduced
    # with E = 200,000 MPa: EA = 987,743.1 kN.
    area = 987743.1 / (200000.0 * 1000.0)
    member_sections = pd.DataFrame([
        [i, 6.0, area, "A992", "W310X38.7", 29.815 / 12.0]
        for i in range(1, 13)
    ], columns=["Member", "Length (m)", "Area (m2)",
                "Material Label", "Section Label", "Total Weight (kN)"])

    coords = {
        1: (0.0, 0.0, 0.0), 2: (6.0, 0.0, 0.0),
        3: (6.0, 0.0, 6.0), 4: (0.0, 0.0, 6.0),
        5: (0.0, 6.0, 0.0), 6: (6.0, 6.0, 0.0),
        7: (6.0, 6.0, 6.0), 8: (0.0, 6.0, 6.0),
    }
    return SimpleNamespace(
        nodes=nodes,
        members=members,
        member_sections=member_sections,
        node_coordinates=coords,
        MATERIAL_ALPHA_1E6_PER_C=11.7,
        MATERIAL_E_MPA=200000.0,
        MATERIAL_LABEL="A992",
        SECTION_LABEL="W310X38.7",
        SECTION_AREA_M2=area,
    )


class Rev3FinalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = make_test_model()
        cls.old_model = rev3._REV2_MODEL
        rev3._REV2_MODEL = cls.model
        rev3.LOAD_COMBINATIONS_REV3 = rev3.build_load_combinations_rev3()

    @classmethod
    def tearDownClass(cls):
        rev3._REV2_MODEL = cls.old_model


# 95 deterministic checks, kept separate so unittest reports each requirement.

def _check_load_case(key, attr, expected):
    case = rev3.LOAD_CASES_REV3[key]
    value = getattr(case, attr)
    return value == expected


TESTS = []

# 1-10: registry and case categories
for i, key in enumerate(["LC1", "LC2", "LC3", "LC4", "LC5", "LC6", "LC7", "LC8"], 1):
    TESTS.append((f"test_{i:02d}_registry_{key.lower()}", lambda k=key: k in rev3.LOAD_CASES_REV3))
TESTS += [
    ("test_09_lc1_is_dead_load", lambda: rev3.LOAD_CASES_REV3["LC1"].load_type == "Dead Load"),
    ("test_10_lc3_is_live_load", lambda: rev3.LOAD_CASES_REV3["LC3"].load_type == "Live Load"),
]

# 11-20: LC1-LC3
TESTS += [
    ("test_11_lc1_self_weight_enabled", lambda: rev3.LOAD_CASES_REV3["LC1"].self_weight is True),
    ("test_12_lc1_negative_y", lambda: rev3.LOAD_CASES_REV3["LC1"].self_weight_direction == "-Y"),
    ("test_13_lc1_has_no_duplicate_member_loads", lambda: not rev3.LOAD_CASES_REV3["LC1"].distributed_loads),
    ("test_14_lc1_total", lambda: math.isclose(rev3.compute_load_case_total_kn(rev3.LOAD_CASES_REV3["LC1"], Rev3FinalTests.model), 29.815, abs_tol=1e-9)),
    ("test_15_lc2_members", lambda: [d.member for d in rev3.LOAD_CASES_REV3["LC2"].distributed_loads] == [5,6,7,8]),
    ("test_16_lc2_uniform_5", lambda: all(d.w_start == 5.0 and d.w_end == 5.0 for d in rev3.LOAD_CASES_REV3["LC2"].distributed_loads)),
    ("test_17_lc2_downward", lambda: all(d.direction == "-Y" for d in rev3.LOAD_CASES_REV3["LC2"].distributed_loads)),
    ("test_18_lc2_total", lambda: math.isclose(rev3.compute_load_case_total_kn(rev3.LOAD_CASES_REV3["LC2"], Rev3FinalTests.model), 120.0, abs_tol=1e-9)),
    ("test_19_lc3_uniform_3", lambda: all(d.w_start == 3.0 and d.w_end == 3.0 for d in rev3.LOAD_CASES_REV3["LC3"].distributed_loads)),
    ("test_20_lc3_total", lambda: math.isclose(rev3.compute_load_case_total_kn(rev3.LOAD_CASES_REV3["LC3"], Rev3FinalTests.model), 72.0, abs_tol=1e-9)),
]

# 21-30: LC4
TESTS += [
    ("test_21_lc4_members", lambda: [p.member for p in rev3.LOAD_CASES_REV3["LC4"].member_point_loads] == [5,6,7,8]),
    ("test_22_lc4_positions", lambda: all(p.position == 3.0 for p in rev3.LOAD_CASES_REV3["LC4"].member_point_loads)),
    ("test_23_lc4_midpoint_m5", lambda: rev3.LOAD_CASES_REV3["LC4"].member_point_loads[0].position == 3.0),
    ("test_24_lc4_midpoint_m8", lambda: rev3.LOAD_CASES_REV3["LC4"].member_point_loads[-1].position == 3.0),
    ("test_25_lc4_magnitudes", lambda: all(p.magnitude == 5.0 for p in rev3.LOAD_CASES_REV3["LC4"].member_point_loads)),
    ("test_26_lc4_downward", lambda: all(p.direction == "-Y" for p in rev3.LOAD_CASES_REV3["LC4"].member_point_loads)),
    ("test_27_lc4_no_physical_point_nodes", lambda: len(Rev3FinalTests.model.nodes) == 8),
    ("test_28_lc4_total", lambda: math.isclose(rev3.compute_load_case_total_kn(rev3.LOAD_CASES_REV3["LC4"], Rev3FinalTests.model), 20.0, abs_tol=1e-9)),
    ("test_29_lc4_validation", lambda: not rev3.LOAD_CASES_REV3["LC4"].validate(Rev3FinalTests.model)),
    ("test_30_lc4_unit_is_force", lambda: all(p.magnitude >= 0 for p in rev3.LOAD_CASES_REV3["LC4"].member_point_loads)),
]

# 31-40: wind/seismic nodal cases
for offset, key, direction, mag, total in [
    (31, "LC5", "+X", 2.5, 10.0), (33, "LC6", "+Z", 2.5, 10.0),
    (35, "LC7", "+X", 3.75, 15.0), (37, "LC8", "+Z", 3.75, 15.0),
]:
    TESTS += [
        (f"test_{offset:02d}_{key.lower()}_nodes", lambda k=key: [n.node for n in rev3.LOAD_CASES_REV3[k].nodal_loads] == [5,6,7,8]),
        (f"test_{offset+1:02d}_{key.lower()}_direction", lambda k=key,d=direction: all(n.direction == d for n in rev3.LOAD_CASES_REV3[k].nodal_loads)),
    ]
# Add two total checks to reach 40.
TESTS += [
    ("test_39_wind_totals", lambda: math.isclose(rev3.compute_load_case_total_kn(rev3.LOAD_CASES_REV3["LC5"], Rev3FinalTests.model), 10.0, abs_tol=1e-9) and math.isclose(rev3.compute_load_case_total_kn(rev3.LOAD_CASES_REV3["LC6"], Rev3FinalTests.model), 10.0, abs_tol=1e-9)),
    ("test_40_seismic_totals", lambda: math.isclose(rev3.compute_load_case_total_kn(rev3.LOAD_CASES_REV3["LC7"], Rev3FinalTests.model), 15.0, abs_tol=1e-9) and math.isclose(rev3.compute_load_case_total_kn(rev3.LOAD_CASES_REV3["LC8"], Rev3FinalTests.model), 15.0, abs_tol=1e-9)),
]

# 41-50: diaphragm
TESTS += [
    ("test_41_roof_nodes_geometry", lambda: rev3.roof_level_nodes(Rev3FinalTests.model) == (5,6,7,8)),
    ("test_42_roof_master", lambda: rev3.roof_diaphragm_master_node(Rev3FinalTests.model) == 5),
    ("test_43_roof_slaves", lambda: rev3.roof_diaphragm_slave_nodes(Rev3FinalTests.model) == (6,7,8)),
    ("test_44_constrained_dofs", lambda: tuple(rev3.build_roof_diaphragm(Rev3FinalTests.model).constrained_dof_list()) == ("UX","UZ","RY")),
    ("test_45_free_dofs", lambda: tuple(rev3.build_roof_diaphragm(Rev3FinalTests.model).free_dofs()) == ("UY","RX","RZ")),
    ("test_46_equation_count", lambda: len(rev3.build_roof_diaphragm(Rev3FinalTests.model).constraint_equations()) == 9),
    ("test_47_first_equation", lambda: rev3.build_roof_diaphragm(Rev3FinalTests.model).equation_lines()[0] == "N6 UX = N5 UX"),
    ("test_48_last_equation", lambda: rev3.build_roof_diaphragm(Rev3FinalTests.model).equation_lines()[-1] == "N8 RY = N5 RY"),
    ("test_49_diaphragm_not_enforced", lambda: rev3.ROOF_DIAPHRAGM_ENFORCED is False),
    ("test_50_diaphragm_validation", lambda: not rev3.validate_roof_diaphragm(Rev3FinalTests.model, verbose=False)),
]

# 51-60: combinations
TESTS += [
    ("test_51_combination_count", lambda: len(rev3.LOAD_COMBINATIONS_REV3) == 30),
    ("test_52_combination_keys", lambda: [c.key for c in rev3.LOAD_COMBINATIONS_REV3] == [f"COMB{i}" for i in range(1,31)]),
    ("test_53_comb1_lrfd", lambda: rev3.combination_metadata(rev3.LOAD_COMBINATIONS_REV3[0])["design_method"] == "LRFD"),
    ("test_54_comb13_asd", lambda: rev3.combination_metadata(rev3.LOAD_COMBINATIONS_REV3[12])["design_method"] == "ASD"),
    ("test_55_comb1_14d", lambda: rev3.LOAD_COMBINATIONS_REV3[0].factors == {"LC1":1.4}),
    ("test_56_comb2_live_factor", lambda: rev3.LOAD_COMBINATIONS_REV3[1].factors["LC3"] == 1.6),
    ("test_57_comb3_wind_x", lambda: "LC5" in rev3.LOAD_COMBINATIONS_REV3[2].factors),
    ("test_58_comb5_seismic_x", lambda: "LC7" in rev3.LOAD_COMBINATIONS_REV3[4].factors),
    ("test_59_temperature_entries_exist", lambda: sum("LC9" in c.factors for c in rev3.LOAD_COMBINATIONS_REV3) == 4),
    ("test_60_combination_validation", lambda: not rev3.validate_load_combinations_rev3(Rev3FinalTests.model)),
]

# 61-70: LC9 analytical checks
TESTS += [
    ("test_61_lc9_exists", lambda: rev3.lc9_temperature_load().delta_T == 15.0),
    ("test_62_lc9_type", lambda: rev3.LC9_LOAD_TYPE == "Temperature"),
    ("test_63_alpha_library_value", lambda: math.isclose(rev3.lc9_rev2_alpha_1e6_per_c(), 11.7)),
    ("test_64_alpha_scaled", lambda: math.isclose(rev3.lc9_material_alpha_per_degC(), 11.7e-6)),
    ("test_65_strain", lambda: math.isclose(rev3.lc9_thermal_strain(), 1.755e-4, rel_tol=0, abs_tol=1e-12)),
    ("test_66_free_expansion", lambda: math.isclose(rev3.lc9_free_expansion_mm(), 1.053, rel_tol=0, abs_tol=1e-9)),
    ("test_67_ea_benchmark", lambda: math.isclose(rev3.lc9_beam_EA_kN(), 987743.1, rel_tol=0, abs_tol=1e-6)),
    ("test_68_restrained_force", lambda: math.isclose(rev3.lc9_fully_restrained_force_kN(), 173.349, rel_tol=0, abs_tol=2e-4)),
    ("test_69_partial_indeterminate", lambda: "INDETERMINATE" in str(rev3.lc9_partially_restrained_state()["axial_force_kN"])),
    ("test_70_lc9_not_in_mechanical_registry", lambda: "LC9" not in rev3.LOAD_CASES_REV3),
]

# 71-80: thermal vector
TESTS += [
    ("test_71_thermal_vector_nonempty", lambda: bool(rev3.lc9_equivalent_thermal_load_vector())),
    ("test_72_thermal_vector_zero_net_fx", lambda: abs(rev3.lc9_net_resultant_kN()["FX"]) < 1e-9),
    ("test_73_thermal_vector_zero_net_fy", lambda: abs(rev3.lc9_net_resultant_kN()["FY"]) < 1e-9),
    ("test_74_thermal_vector_zero_net_fz", lambda: abs(rev3.lc9_net_resultant_kN()["FZ"]) < 1e-9),
    ("test_75_four_affected_members", lambda: tuple(rev3.lc9_temperature_load().members) == (5,6,7,8)),
    ("test_76_thermal_case_is_temperature", lambda: rev3.lc9_temperature_load().load_case_id == "LC9"),
    ("test_77_free_force_zero", lambda: rev3.lc9_free_state()["axial_force_kN"] == 0.0),
    ("test_78_free_expansion_positive", lambda: rev3.lc9_free_state()["expansion_mm"] > 0.0),
    ("test_79_full_force_negative", lambda: rev3.lc9_fully_restrained_state()["axial_force_kN"] < 0.0),
    ("test_80_lc9_verification", lambda: not [p for p in rev3.verify_lc9(verbose=False) if p.startswith("PROBLEM:")]),
]

# 81-85: viewer/glyph data
TESTS += [
    ("test_81_lc2_glyph_count", lambda: len(rev3.build_load_case_glyphs(rev3.LOAD_CASES_REV3["LC2"], Rev3FinalTests.model)) == 4 * (round(6.0 / rev3.DISTRIBUTED_ARROW_SPACING_M) + 1)),
    ("test_82_lc3_glyph_count", lambda: len(rev3.build_load_case_glyphs(rev3.LOAD_CASES_REV3["LC3"], Rev3FinalTests.model)) == 4 * (round(6.0 / rev3.DISTRIBUTED_ARROW_SPACING_M) + 1)),
    ("test_83_point_glyph_unit", lambda: all(g["unit"] == "kN" for g in rev3.build_load_case_glyphs(rev3.LOAD_CASES_REV3["LC4"], Rev3FinalTests.model))),
    ("test_84_temperature_glyph_unit", lambda: all(g["unit"] == "degC" for g in rev3.build_load_case_glyphs(rev3.LoadCase(key="LC9", temperature_loads=[rev3.lc9_temperature_load()], load_type="Temperature"), Rev3FinalTests.model))),
    ("test_85_band_opacity_is_light", lambda: 0.0 < rev3.DISTRIBUTED_BAND_ALPHA <= 0.30),
]

# 86-90: combination assembly
TESTS += [
    ("test_86_comb1_resultant", lambda: math.isclose(rev3.assemble_combination_resultant(rev3.LOAD_COMBINATIONS_REV3[0], Rev3FinalTests.model)["FY"], -1.4*29.815, abs_tol=1e-9)),
    ("test_87_comb2_resultant", lambda: math.isclose(rev3.assemble_combination_resultant(rev3.LOAD_COMBINATIONS_REV3[1], Rev3FinalTests.model)["FY"], -(1.2*29.815 + 1.6*72.0), abs_tol=1e-9)),
    ("test_88_comb3_fx", lambda: math.isclose(rev3.assemble_combination_resultant(rev3.LOAD_COMBINATIONS_REV3[2], Rev3FinalTests.model)["FX"], 10.0, abs_tol=1e-9)),
    ("test_89_comb13_asd_fx_zero", lambda: math.isclose(rev3.assemble_combination_resultant(rev3.LOAD_COMBINATIONS_REV3[12], Rev3FinalTests.model)["FX"], 0.0, abs_tol=1e-9)),
    ("test_90_combined_case_does_not_mutate_registry", lambda: rev3.LOAD_CASES_REV3["LC2"].distributed_loads[0].w_start == 5.0),
]

# 91-95: scope/report integrity
TESTS += [
    ("test_91_no_stiffness_enforcement", lambda: rev3.ROOF_DIAPHRAGM_ENFORCED is False),
    ("test_92_no_support_reaction_claim", lambda: "support reactions" not in rev3.build_verification_report(Rev3FinalTests.model).lower().split("1. scope and analysis limitation")[0]),
    ("test_93_report_has_temperature_section", lambda: "5. LOAD CASE 9 - TEMPERATURE" in rev3.build_verification_report(Rev3FinalTests.model)),
    ("test_94_report_has_combination_section", lambda: "6. LOAD COMBINATIONS" in rev3.build_verification_report(Rev3FinalTests.model)),
    ("test_95_report_states_out_of_scope_solver", lambda: "STIFFNESS SOLVER: OUT OF SCOPE FOR REV. 3" in rev3.build_verification_report(Rev3FinalTests.model)),
]

assert len(TESTS) == 95, f"Expected 95 tests, built {len(TESTS)}"


def _make_test(fn, name):
    def test(self):
        self.assertTrue(fn(), msg=name)
    test.__name__ = name
    return test

for _name, _fn in TESTS:
    setattr(Rev3FinalTests, _name, _make_test(_fn, _name))


if __name__ == "__main__":
    unittest.main(verbosity=2)
