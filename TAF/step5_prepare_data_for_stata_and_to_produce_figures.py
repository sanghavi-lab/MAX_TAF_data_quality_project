#----------------------------------------------------------------------------------------------------------------------#
# Project: Medicaid Data Quality Project
# Authors: Jessy Nguyen
# Last Updated: August 12, 2021
# Description: The goal of this script is to prepare the TAF file exported by ICDPICR for Stata using Pandas.
#----------------------------------------------------------------------------------------------------------------------#

################################################ IMPORT MODULES ########################################################

# Read in relevant libraries
import numpy as np
import pandas as pd

######################## PREPARE FILE FOR STATA TO RUN MODEL AND TO PRODUCE FIGURES ####################################

#___ Define Functions to Calculate NISS___#

# Define function to calculate NISS where AIS 6 -> 5 then calculate normally. Need lambda function for this function to work
def NISS_6_to_5(df,num_sev_col):
    ais_list = []  # Create empty list to store AIS values for each claims
    for i in range(1, num_sev_col):  # Define loop function to store all AIS values to the ais_list. Starting at one but ending at -1 since we have one extra column "unique_id"
        ais_list.append(int(df[f'sev_{i}']))
    ais_list = [x for x in ais_list if x != 9]  # Remove AIS of 9 (i.e. invalid trauma code)
    ais_list = [5 if x == 6 else x for x in ais_list]  # Convert any 6 to 5's
    sorted_ais_list = sorted(ais_list, reverse=True)[:3]  # Sort in descending order and keep only the top three values
    results = sum([i ** 2 for i in sorted_ais_list])  # Calculate the sum of squares for the three high values
    return results

#___ Read in and process data for stata and to produce figures ___#

# Define columns
col_mcaid = ['BENE_ID', 'MSIS_ID', 'STATE_CD', 'SEX_CD', 'RACE_ETHNCTY_CD', 'death_ind_discharge', 'AGE', 'niss'] + [f'sev_{s}' for s in range(1,13)] + [f'dx{a}' for a in range(1,13)]
col_mcare = ['BENE_ID', 'BENE_RSDNC_SSA_STATE_CD', 'SEX_IDENT_CD', 'RTI_RACE_CD', 'death_ind_discharge', 'AGE', 'niss','BENE_DSCHRG_STUS_CD'] + [f'sev_{s}' for s in range(1,27)]

# Read in data as string
medicaid_ffs = pd.read_csv('/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicaid_ffs_mc/data_for_prediction_model/icdpicr/taf_ffs_from_icdpicr_w_niss.csv',usecols=col_mcaid,dtype=str)
medicaid_mc = pd.read_csv('/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicaid_ffs_mc/data_for_prediction_model/icdpicr/taf_mc_from_icdpicr_w_niss.csv',usecols=col_mcaid,dtype=str)
medicare_ab = pd.read_csv('/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicare_ab/data_for_prediction_model/icdpicr/medicare_16_from_icdpicr_w_niss.csv',dtype=str)

# Convert numeric columns to float
num_col_mcaid = ['death_ind_discharge', 'AGE', 'niss'] + [f'sev_{s}' for s in range(1,13)]
num_col_mcare = ['death_ind_discharge', 'AGE', 'niss'] + [f'sev_{s}' for s in range(1,27)]
for n in num_col_mcaid:
    medicaid_ffs[f'{n}'] = medicaid_ffs[f'{n}'].astype(float)
    medicaid_mc[f'{n}'] = medicaid_mc[f'{n}'].astype(float)
for m in num_col_mcare:
    medicare_ab[f'{m}'] = medicare_ab[f'{m}'].astype(float)

# Count number of sev columns (plus 1 accounts for all sev columns)
num_sev_col_mcaid = len([column for column in medicaid_ffs if column.startswith('sev')]) + 1 # We can use either ffs or mc since both have same number of sev cols
num_sev_col_mcare = len([column for column in medicare_ab if column.startswith('sev')]) + 1

# Create a list that starts with sev in the dataframe
sev_col_mcaid = medicaid_ffs.columns[medicaid_ffs.columns.str.startswith('sev')].tolist() # We can use either ffs or mc since both have same sev cols
sev_col_mcare = medicare_ab.columns[medicare_ab.columns.str.startswith('sev')].tolist()

# Fill all nan's in sev columns with 0
medicaid_ffs[sev_col_mcaid] = medicaid_ffs[sev_col_mcaid].fillna(0)
medicaid_mc[sev_col_mcaid] = medicaid_mc[sev_col_mcaid].fillna(0)
medicare_ab[sev_col_mcare] = medicare_ab[sev_col_mcare].fillna(0)

#--- Version 1: Calculating NISS where AIS 6 -> 5 then calculate normally ---#

# Create column 'niss_6_to_5' which calculates the sum of squares for the three highest ais values regardless of body region
medicaid_ffs["niss_6_to_5"] = medicaid_ffs.apply(lambda medicaid_ffs: NISS_6_to_5(medicaid_ffs, num_sev_col_mcaid),axis=1)
medicaid_mc["niss_6_to_5"] = medicaid_mc.apply(lambda medicaid_mc: NISS_6_to_5(medicaid_mc, num_sev_col_mcaid),axis=1)
medicare_ab["niss_6_to_5"] = medicare_ab.apply(lambda medicare_ab: NISS_6_to_5(medicare_ab, num_sev_col_mcare),axis=1)

#--- Clean DF to export to stata (dta) and csv (to produce figures) ---#

# Clean DF
medicaid_ffs = medicaid_ffs.drop(sev_col_mcaid,axis=1)
medicaid_mc = medicaid_mc.drop(sev_col_mcaid,axis=1)
medicare_ab = medicare_ab.drop(sev_col_mcare,axis=1)

# Export as csv to produce figures 1 and 2
medicaid_ffs.to_csv('/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicaid_ffs_mc/data_for_prediction_model/icdpicr/taf_ffs_from_icdpicr_w_niss_six_to_five.csv',index=False,index_label=False)
medicaid_mc.to_csv('/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicaid_ffs_mc/data_for_prediction_model/icdpicr/taf_mc_from_icdpicr_w_niss_six_to_five.csv',index=False,index_label=False)
medicare_ab.to_csv('/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicare_ab/data_for_prediction_model/icdpicr/medicare_16_from_icdpicr_w_niss_six_to_five.csv',index=False,index_label=False)

# Clean DF for Stata
medicaid_ffs_stata = medicaid_ffs[['death_ind_discharge', 'AGE', 'niss', 'niss_6_to_5', 'SEX_CD', 'RACE_ETHNCTY_CD', 'STATE_CD']]
medicaid_mc_stata = medicaid_mc[['death_ind_discharge', 'AGE', 'niss', 'niss_6_to_5', 'SEX_CD', 'RACE_ETHNCTY_CD', 'STATE_CD']]
medicare_ab_stata = medicare_ab[['death_ind_discharge', 'AGE', 'niss', 'niss_6_to_5', 'RTI_RACE_CD', 'SEX_IDENT_CD', 'BENE_RSDNC_SSA_STATE_CD']]

# Export as dta for stata
medicaid_ffs_stata.to_stata('/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicaid_ffs_mc/data_for_prediction_model/icdpicr/taf_ffs_from_icdpicr_w_niss_six_to_five.dta',write_index=False)
medicaid_mc_stata.to_stata('/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicaid_ffs_mc/data_for_prediction_model/icdpicr/taf_mc_from_icdpicr_w_niss_six_to_five.dta',write_index=False)
medicare_ab_stata.to_stata('/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicare_ab/data_for_prediction_model/icdpicr/medicare_16_from_icdpicr_w_niss_six_to_five.dta',write_index=False)