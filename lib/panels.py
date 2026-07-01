"""The four comparison panels and the class colours used throughout the paper.

The four panels arise as follows (see Methods and Table 3):

  Complement-5 -- one of three zero-error panels under all-68-LOPPO
                  cross-validation; shares the {CFB, IGHV3-64, MINPP1, PI16}
                  four-protein backbone with one of the other two and adds
                  L-selectin (SELL, leukocyte adhesion). Also zero-error
                  under all-68-LOOCV.
  Metabolic-4  -- the third zero-error all-68-LOPPO panel; the only one of
                  the three that does NOT contain CFB, MINPP1 or PI16
                  (size 4, ADIPOQ + FGB + IGHV3-64 + IGLV2-11). Represents
                  an alternative adipokine/coagulation biological axis.
  Core-3       -- the three proteins CFB, IGHV3-64 and PI16, the most
                  frequent recurrent set among the top-10 all-68-LOPPO panels
                  (CFB in 9/10; IGHV3-64 and PI16 in 8/10 each; all three
                  jointly in 7/10).
  Clinical-3   -- clinician-curated set of three proteins with routine
                  clinical-laboratory assays available.
"""

PANELS = {
    "Complement-5": ["CFB", "IGHV3-64", "MINPP1", "PI16", "SELL"],
    "Metabolic-4":  ["ADIPOQ", "FGB", "IGHV3-64", "IGLV2-11"],
    "Core-3":       ["CFB", "IGHV3-64", "PI16"],
    "Clinical-3":   ["F2", "CFB", "TTR"],
}

PANEL_DESCRIPTIONS = {
    "Complement-5": "one of three zero-error all-68-LOPPO panels; shares the "
                    "{CFB, IGHV3-64, MINPP1, PI16} backbone with the leading "
                    "panel and adds L-selectin (SELL, leukocyte adhesion); "
                    "also zero-error under all-68-LOOCV",
    "Metabolic-4":  "the third zero-error all-68-LOPPO panel; the only one of "
                    "the three that does not contain CFB, MINPP1 or PI16; "
                    "represents an alternative adipokine/coagulation axis",
    "Core-3":       "CFB, IGHV3-64 and PI16 recur across the top-10 all-68-LOPPO "
                    "panels (CFB in 9/10; IGHV3-64 and PI16 in 8/10 each; "
                    "all three jointly in 7/10)",
    "Clinical-3":   "clinician-curated, routine-assay proteins",
}

# class colours used throughout the figures
POS = "#b2182b"   # healer (class 1)
NEG = "#2166ac"   # non-healer (class 0)
