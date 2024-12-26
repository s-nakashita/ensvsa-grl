#!/bin/sh
set -e

# Figure 1
python comp_en.py -sm 1 -em 3 -fh 0

# Figure 2
python comp_tevol_nl.py -sm 1 -em 3 -qadj

# Figure 3
python comp_en_nl.py -e msm2msm3_da -sm 3 -qadj -fs 0 -fe 15 -ifh 3

# Figure 4, S3
python panel_prtbspectra_nl.py -e rsm2rsm27_da -sm 1 -qadj -fs 0 -fe 15 -ifh 3
python panel_prtbspectra_nl.py -e rsm2msm9_da -sm 1 -qadj -fs 0 -fe 15 -ifh 3
python panel_prtbspectra_nl.py -e msm2msm3_da -sm 1 -qadj -fs 0 -fe 15 -ifh 3

# Figure S1
python comp_prtb.py -sm 1 -em 3 -fh 0

# Figure S2
python comp_linearity.py -sm 1 -em 3 -qadj

# Table S1
python comp_wgt.py 
Rscript create_table.r

