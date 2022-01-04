#----------------------------------------------------------------------------------------------------------------------#
# Project: Medicaid Data Quality Project
# Author: Jessy Nguyen
# Last Updated: August 12, 2021
# Description: The goal of this script is to create CSV files for the ICDPIC-R software. We first identified patients
#              with trauma then combine all years and states for TAF FFS, TAF Encounter, and MCARE A and B. Then,
#              we used Pandas to drop any icd10 duplicates and converted the file to CSV.
#----------------------------------------------------------------------------------------------------------------------#

################################################ IMPORT PACKAGES #######################################################

# Read in relevant libraries
import dask.dataframe as dd
from datetime import datetime, timedelta
import pandas as pd
from pandas.tseries.offsets import MonthEnd
import numpy as np
from dateutil.relativedelta import relativedelta # to get years from subtracting dates in datetime format
from itertools import chain
import math

################################################ MODULE FOR CLUSTER ####################################################

# Read in libraries to use cluster
from dask.distributed import Client
client = Client('[insert_ip_address_for_cluster]')

#################################################### MEDICAID ##########################################################

#-----------------------------------------Define functions for Medicaid------------------------------------------------#

# Define function to identify trauma cases
def identify_trauma(year,mcaid_payment_type,state):

    # Define columns
    columns_ip = ['BENE_ID', 'MSIS_ID', 'STATE_CD', 'PTNT_DSCHRG_STUS_CD', 'ip_ind',
                  'SRVC_END_DT', 'BIRTH_DT', 'SEX_CD', 'RACE_ETHNCTY_CD'] + [f'DGNS_CD_{i}' for i in range(1, 13)]

    # Read in data
    amb_hos_df = dd.read_parquet(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicaid_ffs_mc/{year}/{mcaid_payment_type}_merged_amb_hos_claims_ninetyonedays/{state}/',
        engine='fastparquet', columns=columns_ip)

    # Keep only those merged with ip
    ip = amb_hos_df[amb_hos_df['ip_ind']==1]

    # Recover memory
    del amb_hos_df

    # Convert claim thru date to datetime
    ip['SRVC_END_DT'] = dd.to_datetime(ip['SRVC_END_DT'])
    ip['BIRTH_DT'] = dd.to_datetime(ip['BIRTH_DT'])

    #--- Filter out trauma ---#

    # ICD10 Append to lst_include_codes: S00-S99, T07-T34, T36-T50 (ignoring sixth character of 5, 6, or some X's),T51-T76, T79, M97, T8404, O9A2-O9A5
    lst_include_codes = ['S0{}'.format(i) for i in range(0, 10)] + ['S{}'.format(i) for i in range(10, 100)      # S00-S99
                         ] + ['T0{}'.format(i) for i in range(7, 10)] + ['T{}'.format(i) for i in range(10, 35)  # T07-T34
                            ] + ['T{}'.format(i) for i in range(36, 51)                                          # T36-T50 (will exclude sixth character of 5, 6, or some X's later)
                          ] + ['T{}'.format(i) for i in range(51, 77)                                            # T51-T76
                            ] + ['T79', 'M97', 'T8404', 'O9A2', 'O9A3', 'O9A4','O9A5']                           # T79, M97, T8404, O9A2-O9A5

    # Define list of all diagnosis and ecodes columns
    diag_ecode_col = [f'DGNS_CD_{i}' for i in range(1, 13)]

    # Define list of first four diagnosis columns (four in case first is same as second column)
    diag_first_four_cols = [f'DGNS_CD_{i}' for i in range(1, 5)]

    # Convert all diagnosis and ecodes columns to string
    ip[diag_ecode_col] = ip[diag_ecode_col].astype(str)

    # First, we filter based on lst_include_codes
    amb_ip_trauma = ip.loc[(ip[diag_first_four_cols].applymap(lambda x: x.startswith(tuple(lst_include_codes))).any(axis='columns'))]

    # Recover Memory
    del ip

    # Second, for icd10, we exclude the sixth (thus we use str[5]) character of 5 or 6 from the T36-T50 series from first four columns
    amb_ip_trauma = amb_ip_trauma[~(
            ((amb_ip_trauma['DGNS_CD_1'].str.startswith(tuple([f'T{i}' for i in range(36, 51)]))) & ((amb_ip_trauma['DGNS_CD_1'].str[5] == '5') | (amb_ip_trauma['DGNS_CD_1'].str[5] == '6'))) |
            ((amb_ip_trauma['DGNS_CD_2'].str.startswith(tuple([f'T{i}' for i in range(36, 51)]))) & ((amb_ip_trauma['DGNS_CD_2'].str[5] == '5') | (amb_ip_trauma['DGNS_CD_2'].str[5] == '6'))) |
            ((amb_ip_trauma['DGNS_CD_3'].str.startswith(tuple([f'T{i}' for i in range(36, 51)]))) & ((amb_ip_trauma['DGNS_CD_3'].str[5] == '5') | (amb_ip_trauma['DGNS_CD_3'].str[5] == '6'))) |
            ((amb_ip_trauma['DGNS_CD_4'].str.startswith(tuple([f'T{i}' for i in range(36, 51)]))) & ((amb_ip_trauma['DGNS_CD_4'].str[5] == '5') | (amb_ip_trauma['DGNS_CD_4'].str[5] == '6')))
    )]

    # Third, we exclude some T36-T50 series where the sixth (str[5]) is a character of X (see HCUP definition for specifics: https://www.hcup-us.ahrq.gov/db/vars/siddistnote.jsp?var=i10_multinjury)
    lst_sixth_X_include = ['T369', 'T379', 'T399', 'T414', 'T427', 'T439', 'T459', 'T479','T499']  # create a list to include
    amb_ip_trauma['incld_some_sixth_X_ind'] = 0  # Create indicator columns starting with zero's first
    amb_ip_trauma['incld_some_sixth_X_ind'] = amb_ip_trauma['incld_some_sixth_X_ind'].mask((amb_ip_trauma['DGNS_CD_1'].str.startswith(tuple(lst_sixth_X_include))) & (amb_ip_trauma['DGNS_CD_1'].str[5] == 'X') & (amb_ip_trauma['DGNS_CD_1'].str[4].isin(['1', '2', '3', '4'])), 1)  # Replace incld_some_sixth_X_ind based on condition
    amb_ip_trauma['incld_some_sixth_X_ind'] = amb_ip_trauma['incld_some_sixth_X_ind'].mask((amb_ip_trauma['DGNS_CD_2'].str.startswith(tuple(lst_sixth_X_include))) & (amb_ip_trauma['DGNS_CD_2'].str[5] == 'X') & (amb_ip_trauma['DGNS_CD_2'].str[4].isin(['1', '2', '3', '4'])), 1)  # Replace incld_some_sixth_X_ind based on condition
    amb_ip_trauma['incld_some_sixth_X_ind'] = amb_ip_trauma['incld_some_sixth_X_ind'].mask((amb_ip_trauma['DGNS_CD_3'].str.startswith(tuple(lst_sixth_X_include))) & (amb_ip_trauma['DGNS_CD_3'].str[5] == 'X') & (amb_ip_trauma['DGNS_CD_3'].str[4].isin(['1', '2', '3', '4'])), 1)  # Replace incld_some_sixth_X_ind based on condition
    amb_ip_trauma['incld_some_sixth_X_ind'] = amb_ip_trauma['incld_some_sixth_X_ind'].mask((amb_ip_trauma['DGNS_CD_4'].str.startswith(tuple(lst_sixth_X_include))) & (amb_ip_trauma['DGNS_CD_4'].str[5] == 'X') & (amb_ip_trauma['DGNS_CD_4'].str[4].isin(['1', '2', '3', '4'])), 1)  # Replace incld_some_sixth_X_ind based on condition
    amb_ip_trauma = amb_ip_trauma[~(
            ((amb_ip_trauma['DGNS_CD_1'].str.startswith(tuple([f'T{i}' for i in range(36, 51)]))) & (amb_ip_trauma['DGNS_CD_1'].str[5] == 'X') & (amb_ip_trauma['incld_some_sixth_X_ind'] != 1)) |
            ((amb_ip_trauma['DGNS_CD_2'].str.startswith(tuple([f'T{i}' for i in range(36, 51)]))) & (amb_ip_trauma['DGNS_CD_2'].str[5] == 'X') & (amb_ip_trauma['incld_some_sixth_X_ind'] != 1)) |
            ((amb_ip_trauma['DGNS_CD_3'].str.startswith(tuple([f'T{i}' for i in range(36, 51)]))) & (amb_ip_trauma['DGNS_CD_3'].str[5] == 'X') & (amb_ip_trauma['incld_some_sixth_X_ind'] != 1)) |
            ((amb_ip_trauma['DGNS_CD_4'].str.startswith(tuple([f'T{i}' for i in range(36, 51)]))) & (amb_ip_trauma['DGNS_CD_4'].str[5] == 'X') & (amb_ip_trauma['incld_some_sixth_X_ind'] != 1))
    )]  # Drop any T36-T50 if 6th is X and indicator is not 1 (i.e. not the codes where 6th character is an X that we want to keep)

    # Drop column
    amb_ip_trauma = amb_ip_trauma.drop(['incld_some_sixth_X_ind'], axis=1)

    # Create indicator if eligible died based on variables in IP and PS
    amb_ip_trauma['death_ind_discharge'] = 0
    amb_ip_trauma['death_ind_discharge'] = amb_ip_trauma['death_ind_discharge'].mask((amb_ip_trauma['PTNT_DSCHRG_STUS_CD']=='20') | (amb_ip_trauma['PTNT_DSCHRG_STUS_CD']=='40') |
                                                         (amb_ip_trauma['PTNT_DSCHRG_STUS_CD']=='41') | (amb_ip_trauma['PTNT_DSCHRG_STUS_CD']=='42'), 1)

    # Clean Dataset to run icdpic. All diag columns were renamed with dx and pushed down one number to for the admitting diag code.
    amb_ip_trauma = amb_ip_trauma.rename(columns={'DGNS_CD_1':'dx1','DGNS_CD_2':'dx2', 'DGNS_CD_3':'dx3', 'DGNS_CD_4':'dx4', 'DGNS_CD_5':'dx5', 'DGNS_CD_6':'dx6',
                                'DGNS_CD_7':'dx7', 'DGNS_CD_8':'dx8', 'DGNS_CD_9':'dx9','DGNS_CD_10':'dx10','DGNS_CD_11':'dx11','DGNS_CD_12':'dx12'})

    # Return
    return amb_ip_trauma

#------------------------------------Run Defined function for Medicaid-------------------------------------------------#

# Specify all States that we want to combine to one df
all_states = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS',
              'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC',
              'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY']

# Define empty dictionary to store each state's DF
df_dict_ffs = {}
df_dict_mc = {}

# Define empty list to store and concat each state (with each year) to one df
df_list_ffs = []
df_list_mc = []

for s in all_states:

    # Run function for FFS and MC
    df_dict_ffs[s] = identify_trauma(2016,'ffs',s)
    df_dict_mc[s] = identify_trauma(2016,'mc',s)

    # Append each df (i.e. each state's df) to the empty list (should be 50 states plus DC) for FFS and MC
    df_list_ffs.append(df_dict_ffs[s])
    df_list_mc.append(df_dict_mc[s])

# Concat for FFS and MC
ffs_concat_all_states_years_trauma = dd.concat(df_list_ffs,axis=0)
mc_concat_all_states_years_trauma = dd.concat(df_list_mc,axis=0)

# Read out file because pandas cannot work with dask dataframe for FFS and MC
ffs_concat_all_states_years_trauma.to_parquet(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicaid_ffs_mc/taf_ffs_medicaid_16_allstates_ip_merged_amb/', compression='gzip', engine='fastparquet')
mc_concat_all_states_years_trauma.to_parquet(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicaid_ffs_mc/taf_mc_medicaid_16_allstates_ip_merged_amb/', compression='gzip', engine='fastparquet')

#______________ The following will drop duplicated icd codes in pandas and convert to CSV for ICDPICR _________________#
# Run this only after you have identified trauma and concatenated all years and states into one df                     #
########################################################################################################################

#-----------------------------------------Define function for Medicaid-------------------------------------------------#

# Define function to read in df (with all the years and states concatenated) and prepare file for icdpicr
def prepare_for_icdpicr(mcaid_payment_type):

    # Specify Columns
    columns = ['BENE_ID', 'MSIS_ID', 'STATE_CD', 'SRVC_END_DT', 'PTNT_DSCHRG_STUS_CD', 'dx1', 'dx2', 'dx3', 'dx4', 'dx5',
               'dx6', 'dx7', 'dx8', 'dx9','dx10','dx11','dx12', 'ip_ind', 'BIRTH_DT', 'SEX_CD', 'RACE_ETHNCTY_CD', 'death_ind_discharge']

    # Read in only columns to drop dup icd codes using pandas
    mcaid_ip_trauma = pd.read_parquet(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicaid_ffs_mc/taf_{mcaid_payment_type}_medicaid_16_allstates_ip_merged_amb/',
        engine='fastparquet', columns=columns)

    # Specify columns to drop dup
    diag_col = ['dx{}'.format(i) for i in range(1, 13)]

    # First convert all dx columns to strings (needs to be string for ICDPICR)
    for d in diag_col:
        mcaid_ip_trauma[f'{d}'] = mcaid_ip_trauma[f'{d}'].astype(str)

    # Drop dup icd codes
    mcaid_ip_trauma[diag_col] = mcaid_ip_trauma[diag_col].apply(lambda x: x.drop_duplicates(), axis=1)

    # Calculate Age
    mcaid_ip_trauma['SRVC_END_DT'] = pd.to_datetime(mcaid_ip_trauma['SRVC_END_DT'])
    mcaid_ip_trauma['BIRTH_DT'] = pd.to_datetime(mcaid_ip_trauma['BIRTH_DT'])
    mcaid_ip_trauma['AGE'] = (mcaid_ip_trauma['SRVC_END_DT'] - mcaid_ip_trauma['BIRTH_DT']) / np.timedelta64(1, 'Y')

    # Read out
    mcaid_ip_trauma.to_csv(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicaid_ffs_mc/data_for_prediction_model/icdpicr/taf_{mcaid_payment_type}_for_icdpicr.csv',
        index=False, index_label=False)

#------------------------------------Run Defined function for Medicaid-------------------------------------------------#

# Run defined function for FFS
prepare_for_icdpicr('ffs')

# Run defined function for MC
prepare_for_icdpicr('mc')

############################################### Medicare A and B ########################################################

# Define columns
columns_ip = ['BENE_ID', 'ADMSN_DT', 'BENE_BIRTH_DT', 'BENE_RSDNC_SSA_STATE_CD', 'RTI_RACE_CD', 'SEX_IDENT_CD',
              'BENE_DSCHRG_STUS_CD', 'ADMTG_DGNS_CD'] + ['DGNS_{}_CD'.format(i) for i in range(1, 26)] + \
             ['DGNS_E_{}_CD'.format(k) for k in range(1, 13)]

# Read in data for 2016
mcare_amb_ip=dd.read_parquet('/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicare_ab/2016/merged_amb_hos_claims/ip_merged_amb/',engine='fastparquet',columns=columns_ip)

# Convert claim thru date to datetime
mcare_amb_ip['ADMSN_DT'] = dd.to_datetime(mcare_amb_ip['ADMSN_DT'])

# --- Filter out trauma ---#

# ICD10 Append to lst_include_codes: S00-S99, T07-T34, T36-T50 (ignoring sixth character of 5, 6, or some X's),T51-T76, T79, M97, T8404, O9A2-O9A5
lst_include_codes = ['S0{}'.format(i) for i in range(0, 10)] + ['S{}'.format(i) for i in range(10, 100)     # S00-S99
                    ] + ['T0{}'.format(i) for i in range(7, 10)] + ['T{}'.format(i) for i in range(10, 35)  # T07-T34
                    ] + ['T{}'.format(i) for i in range(36, 51)                                             # T36-T50 (will exclude sixth character of 5, 6, or some X's later)
                    ] + ['T{}'.format(i) for i in range(51, 77)                                             # T51-T76
                    ] + ['T79', 'M97', 'T8404', 'O9A2', 'O9A3', 'O9A4','O9A5']                              # T79, M97, T8404, O9A2-O9A5

# Define list of all diagnosis and ecodes columns (38 columns total)
diag_ecode_col = ['ADMTG_DGNS_CD'] + ['DGNS_{}_CD'.format(i) for i in range(1, 26)] + ['DGNS_E_{}_CD'.format(k) for k in range(1, 13)]

# Define list of first three diagnosis columns plus the principal/admission column (4 columns total)
diag_first_four_cols = ['ADMTG_DGNS_CD'] + ['DGNS_{}_CD'.format(i) for i in range(1, 4)]

# Convert all diagnosis and ecodes columns to string
mcare_amb_ip[diag_ecode_col] = mcare_amb_ip[diag_ecode_col].astype(str)

# First, we filter based on lst_include_codes
mcare_amb_ip_trauma = mcare_amb_ip.loc[
    (mcare_amb_ip[diag_first_four_cols].applymap(lambda x: x.startswith(tuple(lst_include_codes))).any(axis='columns'))]

# Recover Memory
del mcare_amb_ip

# Second, for icd10, we exclude the sixth (thus we use str[5]) character of 5 or 6 from the T36-T50 series from first four columns
mcare_amb_ip_trauma = mcare_amb_ip_trauma[~(
        ((mcare_amb_ip_trauma['DGNS_1_CD'].str.startswith(tuple([f'T{i}' for i in range(36, 51)]))) & (
                    (mcare_amb_ip_trauma['DGNS_1_CD'].str[5] == '5') | (mcare_amb_ip_trauma['DGNS_1_CD'].str[5] == '6'))) |
        ((mcare_amb_ip_trauma['DGNS_2_CD'].str.startswith(tuple([f'T{i}' for i in range(36, 51)]))) & (
                    (mcare_amb_ip_trauma['DGNS_2_CD'].str[5] == '5') | (mcare_amb_ip_trauma['DGNS_2_CD'].str[5] == '6'))) |
        ((mcare_amb_ip_trauma['DGNS_3_CD'].str.startswith(tuple([f'T{i}' for i in range(36, 51)]))) & (
                    (mcare_amb_ip_trauma['DGNS_3_CD'].str[5] == '5') | (mcare_amb_ip_trauma['DGNS_3_CD'].str[5] == '6'))) |
        ((mcare_amb_ip_trauma['DGNS_4_CD'].str.startswith(tuple([f'T{i}' for i in range(36, 51)]))) & (
                    (mcare_amb_ip_trauma['DGNS_4_CD'].str[5] == '5') | (mcare_amb_ip_trauma['DGNS_4_CD'].str[5] == '6')))
)]

# Third, we exclude some T36-T50 series where the sixth (str[5]) is a character of X (see HCUP definition for specifics: https://www.hcup-us.ahrq.gov/db/vars/siddistnote.jsp?var=i10_multinjury)
lst_sixth_X_include = ['T369', 'T379', 'T399', 'T414', 'T427', 'T439', 'T459', 'T479','T499']  # create a list to include
mcare_amb_ip_trauma['incld_some_sixth_X_ind'] = 0  # Create indicator columns starting with zero's first
mcare_amb_ip_trauma['incld_some_sixth_X_ind'] = mcare_amb_ip_trauma['incld_some_sixth_X_ind'].mask((mcare_amb_ip_trauma['DGNS_1_CD'].str.startswith(tuple(lst_sixth_X_include))) & (mcare_amb_ip_trauma['DGNS_1_CD'].str[5] == 'X') & (mcare_amb_ip_trauma['DGNS_1_CD'].str[4].isin(['1', '2', '3', '4'])),1)  # Replace incld_some_sixth_X_ind based on condition
mcare_amb_ip_trauma['incld_some_sixth_X_ind'] = mcare_amb_ip_trauma['incld_some_sixth_X_ind'].mask((mcare_amb_ip_trauma['DGNS_2_CD'].str.startswith(tuple(lst_sixth_X_include))) & (mcare_amb_ip_trauma['DGNS_2_CD'].str[5] == 'X') & (mcare_amb_ip_trauma['DGNS_2_CD'].str[4].isin(['1', '2', '3', '4'])),1)  # Replace incld_some_sixth_X_ind based on condition
mcare_amb_ip_trauma['incld_some_sixth_X_ind'] = mcare_amb_ip_trauma['incld_some_sixth_X_ind'].mask((mcare_amb_ip_trauma['DGNS_3_CD'].str.startswith(tuple(lst_sixth_X_include))) & (mcare_amb_ip_trauma['DGNS_3_CD'].str[5] == 'X') & (mcare_amb_ip_trauma['DGNS_3_CD'].str[4].isin(['1', '2', '3', '4'])),1)  # Replace incld_some_sixth_X_ind based on condition
mcare_amb_ip_trauma['incld_some_sixth_X_ind'] = mcare_amb_ip_trauma['incld_some_sixth_X_ind'].mask((mcare_amb_ip_trauma['DGNS_4_CD'].str.startswith(tuple(lst_sixth_X_include))) & (mcare_amb_ip_trauma['DGNS_4_CD'].str[5] == 'X') & (mcare_amb_ip_trauma['DGNS_4_CD'].str[4].isin(['1', '2', '3', '4'])),1)  # Replace incld_some_sixth_X_ind based on condition
mcare_amb_ip_trauma = mcare_amb_ip_trauma[~(
                      ((mcare_amb_ip_trauma['DGNS_1_CD'].str.startswith(tuple([f'T{i}' for i in range(36, 51)]))) & (
                        mcare_amb_ip_trauma['DGNS_1_CD'].str[5] == 'X') & (mcare_amb_ip_trauma['incld_some_sixth_X_ind'] != 1)) |
                      ((mcare_amb_ip_trauma['DGNS_2_CD'].str.startswith(tuple([f'T{i}' for i in range(36, 51)]))) & (
                        mcare_amb_ip_trauma['DGNS_2_CD'].str[5] == 'X') & (mcare_amb_ip_trauma['incld_some_sixth_X_ind'] != 1)) |
                      ((mcare_amb_ip_trauma['DGNS_3_CD'].str.startswith(tuple([f'T{i}' for i in range(36, 51)]))) & (
                        mcare_amb_ip_trauma['DGNS_3_CD'].str[5] == 'X') & (mcare_amb_ip_trauma['incld_some_sixth_X_ind'] != 1)) |
                      ((mcare_amb_ip_trauma['DGNS_4_CD'].str.startswith(tuple([f'T{i}' for i in range(36, 51)]))) & (
                        mcare_amb_ip_trauma['DGNS_4_CD'].str[5] == 'X') & (mcare_amb_ip_trauma['incld_some_sixth_X_ind'] != 1))
                        )]  # Drop any T36-T50 if 6th is X and indicator is not 1 (i.e. not the codes where 6th character is an X that we want to keep)

# Drop column
mcare_amb_ip_trauma = mcare_amb_ip_trauma.drop(['incld_some_sixth_X_ind'], axis=1)

# Create indicator if eligible died based on variables in IP and PS
mcare_amb_ip_trauma['death_ind_discharge'] = 0
mcare_amb_ip_trauma['death_ind_discharge'] = mcare_amb_ip_trauma['death_ind_discharge'].mask((mcare_amb_ip_trauma['BENE_DSCHRG_STUS_CD']=='B'), 1)

# Clean Dataset to run icdpic. All diag columns were renamed with dx and pushed down one number to for the admitting diag code.
mcare_amb_ip_trauma = mcare_amb_ip_trauma.rename(columns={'ADMTG_DGNS_CD':'dx1','DGNS_1_CD':'dx2', 'DGNS_2_CD':'dx3', 'DGNS_3_CD':'dx4', 'DGNS_4_CD':'dx5', 'DGNS_5_CD':'dx6',
                            'DGNS_6_CD':'dx7', 'DGNS_7_CD':'dx8', 'DGNS_8_CD':'dx9', 'DGNS_9_CD':'dx10','DGNS_10_CD':'dx11','DGNS_11_CD':'dx12', 'DGNS_12_CD':'dx13',
                            'DGNS_13_CD':'dx14', 'DGNS_14_CD':'dx15', 'DGNS_15_CD':'dx16','DGNS_16_CD':'dx17', 'DGNS_17_CD':'dx18', 'DGNS_18_CD':'dx19', 'DGNS_19_CD':'dx20',
                            'DGNS_20_CD':'dx21','DGNS_21_CD':'dx22','DGNS_22_CD':'dx23','DGNS_23_CD':'dx24','DGNS_24_CD':'dx25','DGNS_25_CD':'dx26'})

# Read out file
mcare_amb_ip_trauma.to_parquet('/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicare_ab/ab_medicare_16_allstates_ip_merged_amb/',compression='gzip',engine='fastparquet')

#______________ The following will drop duplicated icd codes in pandas and convert to CSV for ICDPICR _________________#
# Run this only after you have identified trauma and concatenated all years and states into one df                     #
########################################################################################################################

# Specify Columns
columns=['BENE_ID', 'ADMSN_DT', 'BENE_BIRTH_DT', 'BENE_RSDNC_SSA_STATE_CD', 'RTI_RACE_CD', 'SEX_IDENT_CD', 'BENE_DSCHRG_STUS_CD',
         'dx1', 'dx2', 'dx3', 'dx4', 'dx5', 'dx6', 'dx7', 'dx8', 'dx9', 'dx10', 'dx11', 'dx12', 'dx13', 'dx14', 'dx15', 'dx16',
         'dx17', 'dx18', 'dx19', 'dx20', 'dx21', 'dx22', 'dx23', 'dx24', 'dx25', 'dx26', 'death_ind_discharge'] + ['DGNS_E_{}_CD'.format(k) for k in range(1, 13)]

# Read in only columns to drop dup icd codes
mcare_ip_trauma = pd.read_parquet('/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicare_ab/ab_medicare_16_allstates_ip_merged_amb/',engine='fastparquet',columns=columns)

# Specify columns to drop dup
diag_col=['dx{}'.format(i) for i in range(1,27)]

# First convert all dx columns to strings (needs to be string for ICDPICR)
for d in diag_col:
    mcare_ip_trauma[f'{d}'] = mcare_ip_trauma[f'{d}'].astype(str)

# Drop dup icd codes
mcare_ip_trauma[diag_col] = mcare_ip_trauma[diag_col].apply(lambda x: x.drop_duplicates(), axis=1)

# Calculate Age
mcare_ip_trauma['ADMSN_DT'] = pd.to_datetime(mcare_ip_trauma['ADMSN_DT'])
mcare_ip_trauma['BENE_BIRTH_DT'] = pd.to_datetime(mcare_ip_trauma['BENE_BIRTH_DT'])
mcare_ip_trauma['AGE']=(mcare_ip_trauma['ADMSN_DT']-mcare_ip_trauma['BENE_BIRTH_DT'])/np.timedelta64(1, 'Y')  # should equal to age count variable from medpar

# Read out
mcare_ip_trauma.to_csv('/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicare_ab/data_for_prediction_model/icdpicr/medicare_16_for_icdpicr.csv',index=False,index_label=False)




