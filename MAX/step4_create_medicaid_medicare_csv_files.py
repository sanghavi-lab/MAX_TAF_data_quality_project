#----------------------------------------------------------------------------------------------------------------------#
# Project: Medicaid Data Quality Project
# Author: Jessy Nguyen
# Last Updated: August 12, 2021
# Description: The goal of this script is to create CSV files for the ICDPIC-R software. We first identified patients
#              with trauma then combine all years and states for MAX FFS, MAX Encounter, and MCARE A and B. Then,
#              we used Pandas to drop any icd9 duplicates and converted the file to CSV.
#----------------------------------------------------------------------------------------------------------------------#

################################################ IMPORT PACKAGES #######################################################

# Read in relevant libraries
import dask.dataframe as dd
from datetime import datetime, timedelta
import pandas as pd
from pandas.tseries.offsets import MonthEnd
import numpy as np
from dateutil.relativedelta import relativedelta    # to get years from subtracting dates in datetime format
from itertools import chain
import math

################################################ MODULE FOR CLUSTER ####################################################

# Read in libraries to use cluster
from dask.distributed import Client
client = Client('[insert_ip_address_for_cluster]')

#################################################### MEDICAID ##########################################################
# The goal is to concat all years 11-14 and states into one df, drop any duplicated icd codes, and convert to csv for  #
# icdpic-r.                                                                                                            #
########################################################################################################################

#-----------------------------------------Define functions for Medicaid------------------------------------------------#

# Define function to identify trauma cases
def identify_trauma(amb_hos_df):

    # Keep only those merged with ip
    ip = amb_hos_df[amb_hos_df['ip_ind']==1]

    # Recover memory
    del amb_hos_df

    # Convert claim thru date to datetime
    ip['SRVC_END_DT'] = dd.to_datetime(ip['SRVC_END_DT'])
    ip['EL_DOB'] = dd.to_datetime(ip['EL_DOB'])

    # Create list of HCUP Trauma: 800-909.2, 909.4, 909.9; 910-994.9; 995.5-995.59; 995.80-995.85
    lst_include_codes = [str(cd) for cd in chain(range(800, 910),        # 800-909.9 (will exclude 909.3, 909.5-909.8 in lst_ignore_codes below)
                                                range(910, 995))         # 910-994.9
                                            ] + ['9955',                 # 995.5-995.59
                                                 '9958']                 # 995.80-995.89 (will exclude 995.86-995.89 in lst_ignore_codes below)
    lst_ignore_codes = ['9093', '9095', '9096', '9097', '9098', '99586','99587', '99588', '99589'] # List to ignore codes 909.3, 909.5-909.8, & 995.86-995.89

    # Create list of Ecodes to remove claims: E849.0-E849.9; E967.0-E967.9; E869.4; E870-E879; E930-E949
    ecode_to_remove = ['E849',                                        # E849.0-E849.9
                       'E967',                                        # E967.0-E967.9
                       'E8694'                                        # E869.4
                       ] + ['E{}'.format(i) for i in range(870,880)   # E870-E879
                       ] + ['E{}'.format(i) for i in range(930,950)]  # E930-E949

    # Define list of all diagnosis and ecodes columns
    diag_ecode_col = ['DIAG_CD_{}'.format(i) for i in range(1, 10)]

    # Define list of first three diagnosis columns
    diag_first_three_cols = ['DIAG_CD_{}'.format(i) for i in range(1, 4)]

    # Convert all diagnosis and ecodes columns to string
    ip[diag_ecode_col] = ip[diag_ecode_col].astype(str)

    # First, we filter based on lst_include_codes, while ignoring lst_ignore_codes.
    mcaid_ip_trauma = ip.loc[(ip[diag_first_three_cols].applymap(lambda x: x.startswith(tuple(lst_include_codes)) & (~x.startswith(tuple(lst_ignore_codes)))).any(axis='columns'))]

    # Recover Memory
    del ip

    # Second, we obtain our final subset by excluding (with the "~" sign) the claims using the Ecodes defined above (ecode_to_remove)
    mcaid_ip_trauma = mcaid_ip_trauma.loc[~(mcaid_ip_trauma[diag_ecode_col].applymap(lambda x: x.startswith(tuple(ecode_to_remove))).any(axis='columns'))]

    # Create indicator if eligible died based on variables in IP and PS
    mcaid_ip_trauma['death_ind_discharge'] = 0
    mcaid_ip_trauma['death_ind_discharge'] = mcaid_ip_trauma['death_ind_discharge'].mask((mcaid_ip_trauma['PATIENT_STATUS_CD']=='20') | (mcaid_ip_trauma['PATIENT_STATUS_CD']=='40') |
                                                         (mcaid_ip_trauma['PATIENT_STATUS_CD']=='41') | (mcaid_ip_trauma['PATIENT_STATUS_CD']=='42'), 1)

    # Clean Dataset to run icdpic. All diag columns were renamed with dx and pushed down one number to for the admitting diag code.
    mcaid_ip_trauma = mcaid_ip_trauma.rename(columns={'DIAG_CD_1':'dx1','DIAG_CD_2':'dx2', 'DIAG_CD_3':'dx3', 'DIAG_CD_4':'dx4', 'DIAG_CD_5':'dx5', 'DIAG_CD_6':'dx6',
                                'DIAG_CD_7':'dx7', 'DIAG_CD_8':'dx8', 'DIAG_CD_9':'dx9'})

    # Return dataframe after identifying trauma
    return mcaid_ip_trauma

# Define function to combine all states and all years for medicaid
def combine_all_years_each_state(state,mcaid_payment_type):

    # Specify States available for 2011-2014
    states_11_14 = ['CA', 'GA', 'IA', 'LA', 'MI', 'MN', 'MO', 'MS', 'NJ', 'PA', 'SD', 'TN', 'UT', 'VT', 'WV', 'WY']

    # Specify States available for 2011-2013
    states_11_13 = ['AR', 'AZ', 'CT', 'HI', 'IN', 'MA', 'NY', 'OH', 'OK', 'OR', 'WA']

    # Specify States available for 2011-2012
    states_11_12 = ['AL', 'AK', 'CO', 'DC', 'DE', 'FL', 'IL', 'KS', 'KY', 'ME', 'MD', 'MT', 'NE', 'NV', 'NH', 'NM',
                    'NC', 'ND', 'RI', 'SC', 'TX', 'VA', 'WI']

    # Specify States available for 2012-2014
    states_12_14 = ['ID']

    # Define columns
    columns_ip = ['BENE_ID', 'MSIS_ID', 'STATE_CD', 'PATIENT_STATUS_CD', 'DIAG_CD_1', 'DIAG_CD_2', 'DIAG_CD_3',
                  'DIAG_CD_4', 'DIAG_CD_5', 'DIAG_CD_6', 'DIAG_CD_7', 'DIAG_CD_8', 'DIAG_CD_9', 'ip_ind',
                  'SRVC_END_DT', 'EL_DOB', 'EL_SEX_CD', 'EL_RACE_ETHNCY_CD']

    # Due to missing states in some years, we use if/then statement
    if state in states_11_14:

        # Append 2011-2014 file paths to list
        years=[2011,2012,2013,2014]

        # Define empty dictionary to store each state's DF
        df_dict_each_year = {}

        # Define empty list to store and concat each state (with each year) to one df
        df_list_each_year = []

        # Create Loop for each year
        for y in years:

            # Read in each year for one state and append to empty dictionary
            df_dict_each_year[y] = dd.read_parquet(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicaid_ffs_mc/{y}/{mcaid_payment_type}_merged_amb_hos_claims_ninetyonedays/{state}/',
                                                   engine='fastparquet', columns=columns_ip)

            # Append each df of each year (for one state) to the empty list
            df_list_each_year.append(df_dict_each_year[y])

        # Concate all years of one state to one df
        amb_merge_hos_allyears_perstate = dd.concat(df_list_each_year,axis=0)

        # Embed identify trauma function here
        df_w_trauma_per_state = identify_trauma(amb_merge_hos_allyears_perstate)

    elif state in states_11_13:

        # Append 2011-2013 file paths to list
        years=[2011,2012,2013]

        # Define empty dictionary to store each state's DF
        df_dict_each_year = {}

        # Define empty list to store and concat each state (with each year) to one df
        df_list_each_year = []

        # Create Loop for each year
        for y in years:

            # Read in each year for one state and append to empty dictionary
            df_dict_each_year[y] = dd.read_parquet(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicaid_ffs_mc/{y}/{mcaid_payment_type}_merged_amb_hos_claims_ninetyonedays/{state}/',
                                                   engine='fastparquet', columns=columns_ip)

            # Append each df of each year (for one state) to the empty list
            df_list_each_year.append(df_dict_each_year[y])

        # Concate all years of one state to one df
        amb_merge_hos_allyears_perstate = dd.concat(df_list_each_year,axis=0)

        # Embed identify trauma function here
        df_w_trauma_per_state = identify_trauma(amb_merge_hos_allyears_perstate)

    elif state in states_11_12:

        # Append 2011-2012 file paths to list
        years=[2011,2012]

        # Define empty dictionary to store each state's DF
        df_dict_each_year = {}

        # Define empty list to store and concat each state (with each year) to one df
        df_list_each_year = []

        # Create Loop for each year
        for y in years:

            # Read in each year for one state and append to empty dictionary
            df_dict_each_year[y] = dd.read_parquet(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicaid_ffs_mc/{y}/{mcaid_payment_type}_merged_amb_hos_claims_ninetyonedays/{state}/',
                                                   engine='fastparquet', columns=columns_ip)

            # Append each df of each year (for one state) to the empty list
            df_list_each_year.append(df_dict_each_year[y])

        # Concate all years of one state to one df
        amb_merge_hos_allyears_perstate = dd.concat(df_list_each_year,axis=0)

        # Embed identify trauma function here
        df_w_trauma_per_state = identify_trauma(amb_merge_hos_allyears_perstate)

    elif state in states_12_14:

        # Append 2012-2014 file paths to list
        years=[2012,2013,2014]

        # Define empty dictionary to store each state's DF
        df_dict_each_year = {}

        # Define empty list to store and concat each state (with each year) to one df
        df_list_each_year = []

        # Create Loop for each year
        for y in years:

            # Read in each year for one state and append to empty dictionary
            df_dict_each_year[y] = dd.read_parquet(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicaid_ffs_mc/{y}/{mcaid_payment_type}_merged_amb_hos_claims_ninetyonedays/{state}/',
                                                   engine='fastparquet', columns=columns_ip)

            # Append each df of each year (for one state) to the empty list
            df_list_each_year.append(df_dict_each_year[y])

        # Concate all years of one state to one df
        amb_merge_hos_allyears_perstate = dd.concat(df_list_each_year,axis=0)

        # Embed identify trauma function here
        df_w_trauma_per_state = identify_trauma(amb_merge_hos_allyears_perstate)

    # Return df that concatenated all years of one state
    return df_w_trauma_per_state

#------------------------------------Run Defined function for Medicaid-------------------------------------------------#
# Goal here is to concatenate all states (with all of the years concatenated from the function) to one df then read it #
# out.                                                                                                                 #
########################################################################################################################

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
    df_dict_ffs[s] = combine_all_years_each_state(s,'ffs')
    df_dict_mc[s] = combine_all_years_each_state(s,'mc')

    # Append each df (i.e. each state's df) to the empty list (should be 50 states plus DC) for FFS and MC
    df_list_ffs.append(df_dict_ffs[s])
    df_list_mc.append(df_dict_mc[s])

# Concat for FFS and MC
ffs_concat_all_states_years_trauma = dd.concat(df_list_ffs,axis=0)
mc_concat_all_states_years_trauma = dd.concat(df_list_mc,axis=0)

# Read out file because pandas cannot work with dask dataframe for FFS and MC
ffs_concat_all_states_years_trauma.to_parquet(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicaid_ffs_mc/ffs_medicaid_11_14_allstates_ip_merged_amb/', compression='gzip', engine='fastparquet')
mc_concat_all_states_years_trauma.to_parquet(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicaid_ffs_mc/mc_medicaid_11_14_allstates_ip_merged_amb/', compression='gzip', engine='fastparquet')

#______________ The following will drop duplicated icd codes in pandas and convert to CSV for ICDPICR _________________#
# Run this only after you have identified trauma and concatenated all years and states into one df                     #
########################################################################################################################

#-----------------------------------------Define function for Medicaid-------------------------------------------------#

# Define function to read in df (with all the years and states concatenated) and prepare file for icdpicr
def prepare_for_icdpicr(mcaid_payment_type):

    # Specify Columns
    columns = ['BENE_ID', 'MSIS_ID', 'STATE_CD', 'SRVC_END_DT', 'PATIENT_STATUS_CD', 'dx1', 'dx2', 'dx3', 'dx4', 'dx5',
               'dx6', 'dx7', 'dx8', 'dx9', 'ip_ind', 'EL_DOB', 'EL_SEX_CD', 'EL_RACE_ETHNCY_CD', 'death_ind_discharge']

    # Read in only columns to drop dup icd codes using pandas
    mcaid_ip_trauma = pd.read_parquet(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicaid_ffs_mc/{mcaid_payment_type}_medicaid_11_14_allstates_ip_merged_amb/',
        engine='fastparquet', columns=columns)

    # Specify columns to drop dup
    diag_col = ['dx{}'.format(i) for i in range(1, 10)]

    # First convert all dx columns to strings (needs to be string for ICDPICR)
    for d in diag_col:
        mcaid_ip_trauma[f'{d}'] = mcaid_ip_trauma[f'{d}'].astype(str)

    # Drop dup icd codes
    mcaid_ip_trauma[diag_col] = mcaid_ip_trauma[diag_col].apply(lambda x: x.drop_duplicates(), axis=1)

    # Calculate Age
    mcaid_ip_trauma['SRVC_END_DT'] = pd.to_datetime(mcaid_ip_trauma['SRVC_END_DT'])
    mcaid_ip_trauma['EL_DOB'] = pd.to_datetime(mcaid_ip_trauma['EL_DOB'])
    mcaid_ip_trauma['AGE'] = (mcaid_ip_trauma['SRVC_END_DT'] - mcaid_ip_trauma['EL_DOB']) / np.timedelta64(1, 'Y')

    # Read out
    mcaid_ip_trauma.to_csv(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicaid_ffs_mc/data_for_prediction_model/icdpicr/{mcaid_payment_type}_for_icdpicr.csv',
        index=False, index_label=False)

#------------------------------------Run Defined function for Medicaid-------------------------------------------------#

# Run defined function for FFS
prepare_for_icdpicr('ffs')

# Run defined function for MC
prepare_for_icdpicr('mc')

############################################### Medicare A and B #######################################################
# The goal is to combine all years 11-14 into one df, drop any duplicated icd codes, and convert to csv for icdpicr    #
########################################################################################################################

# Define years
years=[2011,2012,2013,2014]

# Create empty list to keep file paths
file_paths = []

# Append 2011-2014 file paths to list
for y in years:
    file_paths.append(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicare_ab/{y}/merged_amb_hos_claims/ip_merged_amb/',)

# Define columns
columns_ip = ['BENE_ID', 'ADMSN_DT', 'BENE_BIRTH_DT', 'BENE_RSDNC_SSA_STATE_CD', 'RTI_RACE_CD', 'SEX_IDENT_CD',
              'BENE_DSCHRG_STUS_CD', 'ADMTG_DGNS_CD'] + ['DGNS_{}_CD'.format(i) for i in range(1, 26)] + \
             ['DGNS_E_{}_CD'.format(k) for k in range(1, 13)]

# Read in data
ip=dd.read_parquet(file_paths,engine='fastparquet',columns=columns_ip)

# Convert claim thru date to datetime
ip['ADMSN_DT'] = dd.to_datetime(ip['ADMSN_DT'])

# Create list of HCUP Trauma: 800-909.2, 909.4, 909.9; 910-994.9; 995.5-995.59; 995.80-995.85
lst_include_codes = [str(cd) for cd in chain(range(800, 910),        # 800-909.9 (will exclude 909.3, 909.5-909.8 in lst_ignore_codes below)
                                            range(910, 995))         # 910-994.9
                                        ] + ['9955',                 # 995.5-995.59
                                             '9958']                 # 995.80-995.89 (will exclude 995.86-995.89 in lst_ignore_codes below)
lst_ignore_codes = ['9093', '9095', '9096', '9097', '9098', '99586','99587', '99588', '99589'] # List to ignore codes 909.3, 909.5-909.8, & 995.86-995.89

# Create list of Ecodes to remove claims: E849.0-E849.9; E967.0-E967.9; E869.4; E870-E879; E930-E949
ecode_to_remove = ['E849',                                        # E849.0-E849.9
                   'E967',                                        # E967.0-E967.9
                   'E8694'                                        # E869.4
                   ] + ['E{}'.format(i) for i in range(870,880)   # E870-E879
                   ] + ['E{}'.format(i) for i in range(930,950)]  # E930-E949

# Define list of all diagnosis and ecodes columns
diag_ecode_col = ['ADMTG_DGNS_CD'] + ['DGNS_{}_CD'.format(i) for i in range(1, 26)] + ['DGNS_E_{}_CD'.format(k) for k in range(1, 13)]

# Define list of first three diagnosis columns plus admission column
diag_first_three_cols = ['ADMTG_DGNS_CD'] + ['DGNS_{}_CD'.format(i) for i in range(1, 4)]

# Convert all diagnosis and ecodes columns to string
ip[diag_ecode_col] = ip[diag_ecode_col].astype(str)

# First, we filter based on lst_include_codes, while ignoring lst_ignore_codes.
mcare_ip_trauma = ip.loc[(ip[diag_first_three_cols].applymap(lambda x: x.startswith(tuple(lst_include_codes)) & (~x.startswith(tuple(lst_ignore_codes)))).any(axis='columns'))]

# Recover Memory
del ip

# Second, we obtain our final subset by excluding (with the "~" sign) the claims using the Ecodes defined above (ecode_to_remove)
mcare_ip_trauma = mcare_ip_trauma.loc[~(mcare_ip_trauma[diag_ecode_col].applymap(lambda x: x.startswith(tuple(ecode_to_remove))).any(axis='columns'))]

# Create indicator if eligible died based on variables in IP and PS
mcare_ip_trauma['death_ind_discharge'] = 0
mcare_ip_trauma['death_ind_discharge'] = mcare_ip_trauma['death_ind_discharge'].mask((mcare_ip_trauma['BENE_DSCHRG_STUS_CD']=='B'), 1)

# Clean Dataset to run icdpic. All diag columns were renamed with dx and pushed down one number to for the admitting diag code.
mcare_ip_trauma = mcare_ip_trauma.rename(columns={'ADMTG_DGNS_CD':'dx1','DGNS_1_CD':'dx2', 'DGNS_2_CD':'dx3', 'DGNS_3_CD':'dx4', 'DGNS_4_CD':'dx5', 'DGNS_5_CD':'dx6',
                            'DGNS_6_CD':'dx7', 'DGNS_7_CD':'dx8', 'DGNS_8_CD':'dx9', 'DGNS_9_CD':'dx10','DGNS_10_CD':'dx11','DGNS_11_CD':'dx12', 'DGNS_12_CD':'dx13',
                            'DGNS_13_CD':'dx14', 'DGNS_14_CD':'dx15', 'DGNS_15_CD':'dx16','DGNS_16_CD':'dx17', 'DGNS_17_CD':'dx18', 'DGNS_18_CD':'dx19', 'DGNS_19_CD':'dx20',
                            'DGNS_20_CD':'dx21','DGNS_21_CD':'dx22','DGNS_22_CD':'dx23','DGNS_23_CD':'dx24','DGNS_24_CD':'dx25','DGNS_25_CD':'dx26'})

# Read out file
mcare_ip_trauma.to_parquet('/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicare_ab/ab_medicare_11_14_allstates_ip_merged_amb/',compression='gzip',engine='fastparquet')

#______________ The following will drop duplicated icd codes in pandas and convert to CSV for ICDPICR _________________#
# Run this only after you have identified trauma and concatenated all years and states into one df                     #
########################################################################################################################

# Specify Columns
columns=['BENE_ID', 'ADMSN_DT', 'BENE_BIRTH_DT', 'BENE_RSDNC_SSA_STATE_CD', 'RTI_RACE_CD', 'SEX_IDENT_CD', 'BENE_DSCHRG_STUS_CD',
         'dx1', 'dx2', 'dx3', 'dx4', 'dx5', 'dx6', 'dx7', 'dx8', 'dx9', 'dx10', 'dx11', 'dx12', 'dx13', 'dx14', 'dx15', 'dx16',
         'dx17', 'dx18', 'dx19', 'dx20', 'dx21', 'dx22', 'dx23', 'dx24', 'dx25', 'dx26', 'death_ind_discharge'] + ['DGNS_E_{}_CD'.format(k) for k in range(1, 13)]

# Read in only columns to drop dup icd codes
mcare_ip_trauma = pd.read_parquet('/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicare_ab/ab_medicare_11_14_allstates_ip_merged_amb/',engine='fastparquet',columns=columns)

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
mcare_ip_trauma.to_csv('/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicare_ab/data_for_prediction_model/icdpicr/medicare_for_icdpicr.csv',index=False,index_label=False)





