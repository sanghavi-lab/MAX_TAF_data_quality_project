#----------------------------------------------------------------------------------------------------------------------#
# Project: Medicaid Data Quality Project
# Authors: Jessy Nguyen
# Last Updated: August 12, 2021
# Description: The goal of this script is to link (1) ambulance claims with mileage information and (2) ambulance
#              claims with hospital claims by states for each year for Medicaid TAF. Since we are linking within state,
#              we dropped linkages that delivered patients to a hospital outside of the beneficiary's state. Lastly,
#              we removed all individuals who were not in medicaid for at least 91 days and did not died at discharge.
#----------------------------------------------------------------------------------------------------------------------#

################################################ IMPORT MODULES ########################################################

# Read in relevant libraries
from datetime import datetime, timedelta
import numpy as np
from pandas.tseries.offsets import MonthEnd
import dask.dataframe as dd
import pandas as pd

################################################ MODULE FOR CLUSTER ####################################################

# Read in libraries to use cluster
from dask.distributed import Client
client = Client('[insert_ip_address_for_cluster]')

####################################### AMBULANCE CLAIMS MERGE WITH MILEAGE ############################################
# The following script matches the exported ambulance claims with the corresponding mileage information. We matched    #
# the claims for FFS and managed-care, separately. When exporting the merged file, it was convenient to separate the   #
# claims with missing BENE_ID's and claims with BENE_ID's. We did not match using pickup/dropoff codes since these are #
# all single rides. Finally, we dropped those who were not in Medicaid for at least 91 days.                           #
########################################################################################################################

#___________________________________________Define function____________________________________________________________#

# Define a function to match ambulance claims with mileage information
def amb_match_mileage(year,state,mcaid_payment_type):

    #---Import Ambulance---#

    # Specify columns for ambulance claims
    columns_amb=['CLM_ID','BENE_ID','MSIS_ID','STATE_CD','SRVC_BGN_DT','SRVC_END_DT','LINE_PRCDR_CD'] + [f'LINE_PRCDR_MDFR_CD_{i}' for i in range(1,5)] +[
                 'BIRTH_DT','DEATH_DT','EL_DOD_PS_NEXT3M','SEX_CD','RACE_ETHNCTY_CD','EL_DAYS_EL_CNT_13',
                 'EL_DAYS_EL_CNT_14','EL_DAYS_EL_CNT_15']+[f'MDCD_ENRLMT_DAYS_{m:02}' for m in range(1, 13)]

    # Read in Ambulance
    amb = dd.read_parquet(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicaid_ffs_mc/{year}/amb_{mcaid_payment_type}/{state}/',engine='fastparquet',columns=columns_amb)

    # Convert to Datetime
    amb['SRVC_BGN_DT'] = dd.to_datetime(amb['SRVC_BGN_DT'])
    amb['SRVC_END_DT'] = dd.to_datetime(amb['SRVC_END_DT'])
    amb['BIRTH_DT'] = dd.to_datetime(amb['BIRTH_DT'])
    amb['DEATH_DT'] = dd.to_datetime(amb['DEATH_DT'])
    amb['EL_DOD_PS_NEXT3M'] = dd.to_datetime(amb['EL_DOD_PS_NEXT3M'])

    # Fill na with blanks to keep DF consistent
    amb['CLM_ID'] = amb['CLM_ID'].fillna('')
    amb['CLM_ID'] = amb['CLM_ID'].astype(str)

    # Change column name
    amb = amb.rename(columns={'EL_DAYS_EL_CNT_13':'MDCD_ENRLMT_DAYS_13','EL_DAYS_EL_CNT_14':'MDCD_ENRLMT_DAYS_14',
                              'EL_DAYS_EL_CNT_15':'MDCD_ENRLMT_DAYS_15'})

    #-----------------Keep those at least 91 days in Medicaid----------------------#

    #---------Codes to count number of days in first month---------#

    # Convert columns to floats
    for i in range(1,16):
        amb[f'MDCD_ENRLMT_DAYS_{i:02}'] = amb[f'MDCD_ENRLMT_DAYS_{i:02}'].astype('float')

    # Find the end of the month from service begin date
    amb['EndOfMonth'] =  dd.to_datetime(amb['SRVC_BGN_DT']) + MonthEnd(1)

    # Find number of days from service begin date to end of month
    amb['Days_Until_End_Month'] = amb['EndOfMonth'] - amb['SRVC_BGN_DT']

    # Convert from days/timedelta to integer
    amb['Days_Until_End_Month'] = amb['Days_Until_End_Month'].dt.days.astype('int64')

    # Create column for days enrolled for that month based on service begin date
    amb['days_enrolled'] = ''
    for i in range(1,16):
        amb['days_enrolled'] = amb['days_enrolled'].mask((amb['SRVC_BGN_DT'].dt.month==i), amb[f'MDCD_ENRLMT_DAYS_{i:02}'])

    # Convert to float
    amb['days_enrolled'] = amb['days_enrolled'].astype('float')

    # Filter only those with days enrolled more than days until end of month (i.e. for the first month, individual needs to be enrolled more than the time since they took the amb ride for the first month)
    amb = amb[amb['days_enrolled']>=amb['Days_Until_End_Month']]

    #---Codes to count number of days enrolled in Medicaid in the next months---#

    # Create new column to account for the subsequent months after initial month
    amb['days_enrolled_after_three_months'] = ''

    # For next months: Add subsequent 3 months for number of days enrolled and put into new column
    for i in range(1,13):
        amb['days_enrolled_after_three_months'] = amb['days_enrolled_after_three_months'].mask((amb['SRVC_BGN_DT'].dt.month==i), amb[f'MDCD_ENRLMT_DAYS_{i+1:02}'] + \
                                                                    amb[f'MDCD_ENRLMT_DAYS_{i+2:02}'] + amb[f'MDCD_ENRLMT_DAYS_{i+3:02}'])

    # Convert to float
    amb['days_enrolled_after_three_months'] = amb['days_enrolled_after_three_months'].astype('float')

    #---Codes to filter individuals with at least 91 days in Medicaid---#

    # Add to see if individuals enrolled at least 91 days
    amb['total_enrolled_after_4_months'] = amb['days_enrolled_after_three_months'] + amb['Days_Until_End_Month']

    # Filter based on if individuals with service date from Jan-Dec have at least 91 days in medicaid
    amb = amb[(amb['total_enrolled_after_4_months'] > 90)]

    # Clean DF before exporting
    amb = amb.drop([f'MDCD_ENRLMT_DAYS_{m:02}' for m in range(1, 16)] +
                   ['EndOfMonth','Days_Until_End_Month','days_enrolled','days_enrolled_after_three_months','total_enrolled_after_4_months'],axis=1)

    #---Import and Match with Mileage---#

    # Specify Columns for mileage
    columns_mi = ['CLM_ID','BENE_ID','MSIS_ID','STATE_CD','LINE_SRVC_BGN_DT','ACTL_SRVC_QTY','LINE_PRCDR_CD']

    # Read in Mileage
    mileage = dd.read_parquet(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicaid_ffs_mc/{year}/mileage/{state}/',engine='fastparquet',columns=columns_mi)

    # Rename col
    mileage = mileage.rename(columns={'LINE_SRVC_BGN_DT':'SRVC_BGN_DT'})

    # Convert to datetie
    mileage['SRVC_BGN_DT'] = dd.to_datetime(mileage['SRVC_BGN_DT'])

    # Fill na with blanks to keep DF consistent
    mileage['CLM_ID'] = mileage['CLM_ID'].fillna('')
    mileage['CLM_ID'] = mileage['CLM_ID'].astype(str)

    # Create column to count number matched
    mileage['ind_for_mi_match'] = 1

    # Split into missing vs not missing bene_id
    amb_missingid = amb[(amb['BENE_ID']=='')]
    amb_notmissingid = amb[(amb['BENE_ID']!='')]

    # Add column of consecutive numbers. Need this to drop additional duplicates due to input errors
    amb_missingid = amb_missingid.reset_index(drop=True)
    amb_notmissingid = amb_notmissingid.reset_index(drop=True)
    amb_missingid['for_drop_dup'] = amb_missingid.reset_index().index
    amb_notmissingid['for_drop_dup'] = amb_notmissingid.reset_index().index

    # Merge the mileage info with the ambulance claims
    amb_missingid_mi = dd.merge(mileage,amb_missingid, on=['MSIS_ID','STATE_CD','SRVC_BGN_DT'],suffixes=['_MI','_AMB'],how='right')
    amb_notmissingid_mi = dd.merge(mileage,amb_notmissingid,on=['BENE_ID','SRVC_BGN_DT'],suffixes=['_MI','_AMB'],how='right')

    # Drop all duplicates due to input errors
    amb_missingid_mi = amb_missingid_mi.drop_duplicates(subset=['MSIS_ID','STATE_CD','SRVC_BGN_DT','for_drop_dup'], keep = 'last')
    amb_notmissingid_mi = amb_notmissingid_mi.drop_duplicates(subset=['BENE_ID','SRVC_BGN_DT','for_drop_dup'], keep = 'last')

    # Concat
    amb_mi = dd.concat([amb_missingid_mi,amb_notmissingid_mi],axis=0)

    # Read out Data. Due to differences in column names from merging, we separated the claims with missing BENE_ID's from those with BENE_ID's
    amb_mi.to_parquet(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicaid_ffs_mc/{year}/{mcaid_payment_type}_merged_amb_mileage/{state}/',
                                compression='gzip', engine='fastparquet')

#________________________________________Run Defined functions_________________________________________________________#

# Specify years (only using 2016 for now)
years=[2016]

# Specify States
states_16=['AL','AK','AZ','AR','CA','CO','CT','DC','DE','FL','GA','HI','ID','IL','IN','IA','KS','KY','LA','ME','MD','MA',
           'MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX',
           'UT','VT','VA','WA','WV','WI','WY'] # better to run CA without cluster

# Create loop (only using 2016 for now)
for y in years:

    # States available in 2016
    for s in states_16:

        # Run for FFS
        amb_match_mileage(y, s, 'ffs')

        # Run for MC/Encounter
        amb_match_mileage(y, s, 'mc')

######################################## MATCH IP AND OP WITH AMB CLAIMS ###############################################
# The following script links the exported ambulance claims with hospital claims. We linked with IP same day, next day, #
# and the following day first then repeat the process with OP. We linked within states; this means that the denominator#
# does not include individuals who lived in one state but transported to a hospital in another state.                  #
#######################################################################################################################

#___________________________________________Define function____________________________________________________________#

# Define function to link ambulance to hospital claims
def amb_match_hos(year,state,mcaid_payment_type):

    #---Import Amb---#

    # Specify columns needed
    columns_amb = ['BENE_ID','MSIS_ID','STATE_CD','SRVC_END_DT','BIRTH_DT','DEATH_DT','SEX_CD','RACE_ETHNCTY_CD']

    # Import Ambulance
    amb = dd.read_parquet(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicaid_ffs_mc/{year}/amb_{mcaid_payment_type}/{state}/', engine='fastparquet', columns=columns_amb)

    # Create DF for missing_beneid and notmissing_beneid
    amb_missingid = amb[(amb['BENE_ID']=='')]
    amb_notmissingid = amb[(amb['BENE_ID']!='')]

    # Recover memory
    del amb

    # Fill na with blanks to keep DF consistent
    amb_missingid['BENE_ID'] = amb_missingid['BENE_ID'].fillna('')
    amb_missingid['MSIS_ID'] = amb_missingid['MSIS_ID'].fillna('')
    amb_notmissingid['BENE_ID'] = amb_notmissingid['BENE_ID'].fillna('')
    amb_notmissingid['MSIS_ID'] = amb_notmissingid['MSIS_ID'].fillna('')

    #---Import IP---#

    # Define columns for IP
    columns_ip=['BENE_ID','MSIS_ID','STATE_CD','ADMSN_DT','PTNT_DSCHRG_STUS_CD'
                ] + [f'DGNS_CD_{i}' for i in range(1, 13)]

    # Read in IP same year
    ip = dd.read_csv(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicaid_ffs_mc/{year}/ip_stay/{state}.csv', usecols=columns_ip,dtype=str)

    # Rename columns
    ip = ip.rename(columns={'ADMSN_DT':'SRVC_BGN_DT'})

    #---Match Amb with IP (same day)---#

    # Convert all to datetime before merging
    ip['SRVC_BGN_DT'] = dd.to_datetime(ip['SRVC_BGN_DT'], errors = 'coerce')
    amb_missingid['SRVC_END_DT'] = dd.to_datetime(amb_missingid['SRVC_END_DT'], errors = 'coerce')
    amb_notmissingid['SRVC_END_DT'] = dd.to_datetime(amb_notmissingid['SRVC_END_DT'], errors = 'coerce')

    # Add columns of one's in IP and OP DF to eventually drop unmatched
    ip['ind_for_hos_match'] = 1

    # Create indicator for IP vs OP (used to check data later regarding trauma cases)
    ip['ip_ind'] = 1

    # Add column of consecutive numbers. Needed to drop additional duplicates in IP due to input errors
    amb_missingid = amb_missingid.reset_index(drop=True)
    amb_notmissingid = amb_notmissingid.reset_index(drop=True)
    amb_missingid['for_drop_dup'] = amb_missingid.reset_index().index
    amb_notmissingid['for_drop_dup'] = amb_notmissingid.reset_index().index

    # First, merge with IP
    merge_with_ip_missing = dd.merge(ip,amb_missingid, left_on=['MSIS_ID','STATE_CD','SRVC_BGN_DT'],right_on=['MSIS_ID','STATE_CD','SRVC_END_DT'],suffixes=['_HOS','_AMB'], how = 'right')
    merge_w_ip_notmissing = dd.merge(ip,amb_notmissingid, left_on=['BENE_ID','SRVC_BGN_DT'],right_on=['BENE_ID','SRVC_END_DT'],suffixes=['_HOS','_AMB'], how = 'right')

    # Recover Memory
    del amb_missingid
    del amb_notmissingid

    # Drop all duplicates due to input errors
    merge_with_ip_missing = merge_with_ip_missing.drop_duplicates(subset=['MSIS_ID','STATE_CD','SRVC_END_DT','for_drop_dup'], keep = 'last')
    merge_w_ip_notmissing = merge_w_ip_notmissing.drop_duplicates(subset=['BENE_ID','SRVC_END_DT','for_drop_dup'], keep = 'last')

    # Create DF of those not matched with IP
    hos_missing_unmatched = merge_with_ip_missing[merge_with_ip_missing['ind_for_hos_match'].isna()]
    hos_notmissing_unmatched = merge_w_ip_notmissing[merge_w_ip_notmissing['ind_for_hos_match'].isna()]

    # Clean unmatched dataset to remerge after adding +1
    hos_missing_unmatched = hos_missing_unmatched.drop(['BENE_ID_HOS', 'PTNT_DSCHRG_STUS_CD', 'ind_for_hos_match', 'ip_ind',
                                                        'SRVC_BGN_DT', 'for_drop_dup']+[f'DGNS_CD_{i}' for i in range(1, 13)],axis=1)
    hos_notmissing_unmatched = hos_notmissing_unmatched.drop(['MSIS_ID_HOS', 'STATE_CD_HOS', 'PTNT_DSCHRG_STUS_CD', 'ind_for_hos_match', 'ip_ind',
                                                        'SRVC_BGN_DT', 'for_drop_dup']+[f'DGNS_CD_{i}' for i in range(1, 13)],axis=1)
    hos_missing_unmatched = hos_missing_unmatched.rename(columns={'BENE_ID_AMB':'BENE_ID'})
    hos_notmissing_unmatched = hos_notmissing_unmatched.rename(columns={'MSIS_ID_AMB':'MSIS_ID','STATE_CD_AMB':'STATE_CD'})

    # Create DF of those matched with IP
    merge_with_ip_missing = merge_with_ip_missing[merge_with_ip_missing['ind_for_hos_match']==1]
    merge_w_ip_notmissing = merge_w_ip_notmissing[merge_w_ip_notmissing['ind_for_hos_match']==1]

    # Clean DF for IP matched before concatenating and exporting
    merge_with_ip_missing = merge_with_ip_missing.drop(['BENE_ID_HOS','for_drop_dup'],axis=1)
    merge_w_ip_notmissing = merge_w_ip_notmissing.drop(['MSIS_ID_HOS','STATE_CD_HOS','for_drop_dup'],axis=1)
    merge_with_ip_missing = merge_with_ip_missing.rename(columns={'BENE_ID_AMB':'BENE_ID'})
    merge_w_ip_notmissing = merge_w_ip_notmissing.rename(columns={'MSIS_ID_AMB':'MSIS_ID','STATE_CD_AMB':'STATE_CD'})

    #---Match Amb with IP (next day)---#

    # Add +1 to unmatched amb claims
    hos_missing_unmatched['SRVC_END_DT_PLUSONE'] = hos_missing_unmatched['SRVC_END_DT'] + timedelta(days=1)
    hos_notmissing_unmatched['SRVC_END_DT_PLUSONE'] = hos_notmissing_unmatched['SRVC_END_DT'] + timedelta(days=1)

    # Add column of consecutive numbers. Needed to drop additional duplicates in IP due to input errors
    hos_missing_unmatched = hos_missing_unmatched.reset_index(drop=True)
    hos_notmissing_unmatched = hos_notmissing_unmatched.reset_index(drop=True)
    hos_missing_unmatched['for_drop_dup'] = hos_missing_unmatched.reset_index().index
    hos_notmissing_unmatched['for_drop_dup'] = hos_notmissing_unmatched.reset_index().index

    # First, merge +1's with IP
    merge_with_ip_missing_plus1 = dd.merge(ip,hos_missing_unmatched, left_on=['MSIS_ID','STATE_CD','SRVC_BGN_DT'],
                                           right_on=['MSIS_ID','STATE_CD','SRVC_END_DT_PLUSONE'], suffixes=['_HOS','_AMB'], how = 'right')
    merge_w_ip_notmissing_plus1 = dd.merge(ip,hos_notmissing_unmatched, left_on=['BENE_ID','SRVC_BGN_DT'],
                                           right_on=['BENE_ID','SRVC_END_DT_PLUSONE'], suffixes=['_HOS','_AMB'], how = 'right')

    # Recover Memory
    del hos_missing_unmatched
    del hos_notmissing_unmatched

    # Drop all duplicates due to input errors
    merge_with_ip_missing_plus1 = merge_with_ip_missing_plus1.drop_duplicates(subset=['MSIS_ID','STATE_CD','SRVC_END_DT_PLUSONE','for_drop_dup'], keep = 'last')
    merge_w_ip_notmissing_plus1 = merge_w_ip_notmissing_plus1.drop_duplicates(subset=['BENE_ID','SRVC_END_DT_PLUSONE','for_drop_dup'], keep = 'last')

    # Create DF of those not matched with IP
    hos_missing_unmatched_plus1 = merge_with_ip_missing_plus1[merge_with_ip_missing_plus1['ind_for_hos_match'].isna()]
    hos_notmissing_unmatched_plus1 = merge_w_ip_notmissing_plus1[merge_w_ip_notmissing_plus1['ind_for_hos_match'].isna()]

    # Clean unmatched dataset
    hos_missing_unmatched_plus1 = hos_missing_unmatched_plus1.drop(['SRVC_END_DT_PLUSONE','BENE_ID_HOS', 'PTNT_DSCHRG_STUS_CD', 'ind_for_hos_match', 'ip_ind',
                                                        'SRVC_BGN_DT', 'for_drop_dup']+[f'DGNS_CD_{i}' for i in range(1, 13)],axis=1)
    hos_notmissing_unmatched_plus1 = hos_notmissing_unmatched_plus1.drop(['SRVC_END_DT_PLUSONE','MSIS_ID_HOS', 'STATE_CD_HOS', 'PTNT_DSCHRG_STUS_CD', 'ind_for_hos_match', 'ip_ind',
                                                        'SRVC_BGN_DT', 'for_drop_dup']+[f'DGNS_CD_{i}' for i in range(1, 13)],axis=1)
    hos_missing_unmatched_plus1 = hos_missing_unmatched_plus1.rename(columns={'BENE_ID_AMB':'BENE_ID','SRVC_BGN_DT_AMB':'SRVC_BGN_DT'})
    hos_notmissing_unmatched_plus1 = hos_notmissing_unmatched_plus1.rename(columns={'MSIS_ID_AMB':'MSIS_ID','STATE_CD_AMB':'STATE_CD',
                                                                                    'SRVC_BGN_DT_AMB':'SRVC_BGN_DT'})

    # Create DF of those matched with IP
    merge_with_ip_missing_plus1 = merge_with_ip_missing_plus1[merge_with_ip_missing_plus1['ind_for_hos_match']==1]
    merge_w_ip_notmissing_plus1 = merge_w_ip_notmissing_plus1[merge_w_ip_notmissing_plus1['ind_for_hos_match']==1]

    # Clean DF for IP matched before concatenating and exporting
    merge_with_ip_missing_plus1 = merge_with_ip_missing_plus1.drop(['BENE_ID_HOS','for_drop_dup','SRVC_END_DT_PLUSONE'],axis=1)
    merge_w_ip_notmissing_plus1 = merge_w_ip_notmissing_plus1.drop(['MSIS_ID_HOS','STATE_CD_HOS','for_drop_dup','SRVC_END_DT_PLUSONE'],axis=1)
    merge_with_ip_missing_plus1 = merge_with_ip_missing_plus1.rename(columns={'BENE_ID_AMB':'BENE_ID','SRVC_BGN_DT_HOS':'SRVC_BGN_DT'})
    merge_w_ip_notmissing_plus1 = merge_w_ip_notmissing_plus1.rename(columns={'MSIS_ID_AMB':'MSIS_ID','STATE_CD_AMB':'STATE_CD','SRVC_BGN_DT_HOS':'SRVC_BGN_DT'})

    #---Match Amb with IP (following day (+2))---#

    # Add +2 to unmatched amb claims
    hos_missing_unmatched_plus1['SRVC_END_DT_PLUSTWO'] = hos_missing_unmatched_plus1['SRVC_END_DT'] + timedelta(days=2)
    hos_notmissing_unmatched_plus1['SRVC_END_DT_PLUSTWO'] = hos_notmissing_unmatched_plus1['SRVC_END_DT'] + timedelta(days=2)

    # Add column of consecutive numbers. Needed to drop additional duplicates in IP due to input errors
    hos_missing_unmatched_plus1 = hos_missing_unmatched_plus1.reset_index(drop=True)
    hos_notmissing_unmatched_plus1 = hos_notmissing_unmatched_plus1.reset_index(drop=True)
    hos_missing_unmatched_plus1['for_drop_dup'] = hos_missing_unmatched_plus1.reset_index().index
    hos_notmissing_unmatched_plus1['for_drop_dup'] = hos_notmissing_unmatched_plus1.reset_index().index

    # First, merge +1's with IP
    merge_with_ip_missing_plus2 = dd.merge(ip,hos_missing_unmatched_plus1, left_on=['MSIS_ID','STATE_CD','SRVC_BGN_DT'],
                                           right_on=['MSIS_ID','STATE_CD','SRVC_END_DT_PLUSTWO'], suffixes=['_HOS','_AMB'], how = 'right')
    merge_w_ip_notmissing_plus2 = dd.merge(ip,hos_notmissing_unmatched_plus1, left_on=['BENE_ID','SRVC_BGN_DT'],
                                           right_on=['BENE_ID','SRVC_END_DT_PLUSTWO'], suffixes=['_HOS','_AMB'], how = 'right')

    # Recover Memory
    del hos_missing_unmatched_plus1
    del hos_notmissing_unmatched_plus1

    # Drop all duplicates due to input errors
    merge_with_ip_missing_plus2 = merge_with_ip_missing_plus2.drop_duplicates(subset=['MSIS_ID','STATE_CD','SRVC_END_DT_PLUSTWO','for_drop_dup'], keep = 'last')
    merge_w_ip_notmissing_plus2 = merge_w_ip_notmissing_plus2.drop_duplicates(subset=['BENE_ID','SRVC_END_DT_PLUSTWO','for_drop_dup'], keep = 'last')

    # Create DF of those not matched with IP
    hos_missing_unmatched_plus2 = merge_with_ip_missing_plus2[merge_with_ip_missing_plus2['ind_for_hos_match'].isna()]
    hos_notmissing_unmatched_plus2 = merge_w_ip_notmissing_plus2[merge_w_ip_notmissing_plus2['ind_for_hos_match'].isna()]

    # Clean unmatched dataset
    hos_missing_unmatched_plus2 = hos_missing_unmatched_plus2.drop(['SRVC_END_DT_PLUSTWO','BENE_ID_HOS', 'PTNT_DSCHRG_STUS_CD', 'ind_for_hos_match', 'ip_ind',
                                                        'SRVC_BGN_DT', 'for_drop_dup']+[f'DGNS_CD_{i}' for i in range(1, 13)],axis=1)
    hos_notmissing_unmatched_plus2 = hos_notmissing_unmatched_plus2.drop(['SRVC_END_DT_PLUSTWO','MSIS_ID_HOS', 'STATE_CD_HOS', 'PTNT_DSCHRG_STUS_CD', 'ind_for_hos_match', 'ip_ind',
                                                        'SRVC_BGN_DT', 'for_drop_dup']+[f'DGNS_CD_{i}' for i in range(1, 13)],axis=1)
    hos_missing_unmatched_plus2 = hos_missing_unmatched_plus2.rename(columns={'BENE_ID_AMB':'BENE_ID','SRVC_BGN_DT_AMB':'SRVC_BGN_DT'})
    hos_notmissing_unmatched_plus2 = hos_notmissing_unmatched_plus2.rename(columns={'MSIS_ID_AMB':'MSIS_ID','STATE_CD_AMB':'STATE_CD',
                                                                                    'SRVC_BGN_DT_AMB':'SRVC_BGN_DT'})

    # Create DF of those matched with IP
    merge_with_ip_missing_plus2 = merge_with_ip_missing_plus2[merge_with_ip_missing_plus2['ind_for_hos_match']==1]
    merge_w_ip_notmissing_plus2 = merge_w_ip_notmissing_plus2[merge_w_ip_notmissing_plus2['ind_for_hos_match']==1]

    # Clean DF for IP matched before concatenating and exporting
    merge_with_ip_missing_plus2 = merge_with_ip_missing_plus2.drop(['BENE_ID_HOS','for_drop_dup','SRVC_END_DT_PLUSTWO'],axis=1)
    merge_w_ip_notmissing_plus2 = merge_w_ip_notmissing_plus2.drop(['MSIS_ID_HOS','STATE_CD_HOS','for_drop_dup','SRVC_END_DT_PLUSTWO'],axis=1)
    merge_with_ip_missing_plus2 = merge_with_ip_missing_plus2.rename(columns={'BENE_ID_AMB':'BENE_ID','SRVC_BGN_DT_HOS':'SRVC_BGN_DT'})
    merge_w_ip_notmissing_plus2 = merge_w_ip_notmissing_plus2.rename(columns={'MSIS_ID_AMB':'MSIS_ID','STATE_CD_AMB':'STATE_CD','SRVC_BGN_DT_HOS':'SRVC_BGN_DT'})

    # Recover memory
    del ip

    #---Import OP---#

    # Define columns for OP
    columns_op=['BENE_ID', 'MSIS_ID', 'STATE_CD', 'SRVC_BGN_DT'] + [
               f'DGNS_CD_{i}' for i in range(1, 3)]

    if state in ['CA']: # CA was so large so I saved the op file as a different name for CA
        op = dd.read_parquet(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicaid_ffs_mc/{year}/op/{state}_all_op/',
            engine='fastparquet', columns=columns_op)
    else:
        # Read in OP same year
        op = dd.read_parquet(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicaid_ffs_mc/{year}/op/{state}/',
            engine='fastparquet', columns=columns_op)

    #---Match Amb with OP (same day)---#

    # Convert all to datetime before merging
    op['SRVC_BGN_DT'] = dd.to_datetime(op['SRVC_BGN_DT'])

    # Add columns of one's in OP and OP DF to filter out unmatched
    op['ind_for_hos_match'] = 1

    # Create indicator for IP vs OP (used to check data later regarding trauma cases)
    op['ip_ind'] = 0

    # Add column of consecutive numbers. Needed to drop additional duplicates in OP due to input errors
    hos_missing_unmatched_plus2 = hos_missing_unmatched_plus2.reset_index(drop=True)
    hos_notmissing_unmatched_plus2 = hos_notmissing_unmatched_plus2.reset_index(drop=True)
    hos_missing_unmatched_plus2['for_drop_dup'] = hos_missing_unmatched_plus2.reset_index().index
    hos_notmissing_unmatched_plus2['for_drop_dup'] = hos_notmissing_unmatched_plus2.reset_index().index

    # First, merge with OP
    merge_with_op_missing = dd.merge(op,hos_missing_unmatched_plus2, left_on=['MSIS_ID','STATE_CD','SRVC_BGN_DT'], right_on=['MSIS_ID','STATE_CD','SRVC_END_DT'],suffixes=['_HOS','_AMB'], how = 'right')
    merge_w_op_notmissing = dd.merge(op,hos_notmissing_unmatched_plus2, left_on=['BENE_ID','SRVC_BGN_DT'], right_on=['BENE_ID','SRVC_END_DT'],suffixes=['_HOS','_AMB'], how = 'right')

    # Recover Memory
    del hos_missing_unmatched_plus2
    del hos_notmissing_unmatched_plus2

    # Drop all duplicates due to input errors
    merge_with_op_missing = merge_with_op_missing.drop_duplicates(subset=['MSIS_ID','STATE_CD','SRVC_END_DT','for_drop_dup'], keep = 'first')
    merge_w_op_notmissing = merge_w_op_notmissing.drop_duplicates(subset=['BENE_ID','SRVC_END_DT','for_drop_dup'], keep = 'first')

    # Create DF of those not matched with OP
    hos_missing_unmatched = merge_with_op_missing[merge_with_op_missing['ind_for_hos_match'].isna()]
    hos_notmissing_unmatched = merge_w_op_notmissing[merge_w_op_notmissing['ind_for_hos_match'].isna()]

    # Clean unmatched dataset
    hos_missing_unmatched = hos_missing_unmatched.drop(['BENE_ID_HOS', 'DGNS_CD_1','DGNS_CD_2', 'ind_for_hos_match',
                                                        'ip_ind','SRVC_BGN_DT', 'for_drop_dup'],axis=1)
    hos_notmissing_unmatched = hos_notmissing_unmatched.drop(['MSIS_ID_HOS', 'STATE_CD_HOS', 'DGNS_CD_1','DGNS_CD_2', 'ind_for_hos_match',
                                                              'ip_ind','SRVC_BGN_DT', 'for_drop_dup'],axis=1)
    hos_missing_unmatched = hos_missing_unmatched.rename(columns={'BENE_ID_AMB':'BENE_ID','SRVC_BGN_DT_AMB':'SRVC_BGN_DT'})
    hos_notmissing_unmatched = hos_notmissing_unmatched.rename(columns={'MSIS_ID_AMB':'MSIS_ID','STATE_CD_AMB':'STATE_CD','SRVC_BGN_DT_AMB':'SRVC_BGN_DT'})

    # Create DF of those matched with OP
    merge_with_op_missing = merge_with_op_missing[merge_with_op_missing['ind_for_hos_match']==1]
    merge_w_op_notmissing = merge_w_op_notmissing[merge_w_op_notmissing['ind_for_hos_match']==1]

    # Clean DF for OP matched before concatenating and exporting
    merge_with_op_missing = merge_with_op_missing.drop(['BENE_ID_HOS','for_drop_dup'],axis=1)
    merge_w_op_notmissing = merge_w_op_notmissing.drop(['MSIS_ID_HOS','STATE_CD_HOS','for_drop_dup'],axis=1)
    merge_with_op_missing = merge_with_op_missing.rename(columns={'BENE_ID_AMB':'BENE_ID','SRVC_BGN_DT_AMB':'SRVC_BGN_DT'})
    merge_w_op_notmissing = merge_w_op_notmissing.rename(columns={'MSIS_ID_AMB':'MSIS_ID','STATE_CD_AMB':'STATE_CD','SRVC_BGN_DT_AMB':'SRVC_BGN_DT'})

    #---Match Amb with OP (next day)---#

    # Add +1 to unmatched amb claims
    hos_missing_unmatched['SRVC_END_DT_PLUSONE'] = hos_missing_unmatched['SRVC_END_DT'] + timedelta(days=1)
    hos_notmissing_unmatched['SRVC_END_DT_PLUSONE'] = hos_notmissing_unmatched['SRVC_END_DT'] + timedelta(days=1)

    # Add column of consecutive numbers. Needed to drop additional duplicates in OP due to input errors
    hos_missing_unmatched = hos_missing_unmatched.reset_index(drop=True)
    hos_notmissing_unmatched = hos_notmissing_unmatched.reset_index(drop=True)
    hos_missing_unmatched['for_drop_dup'] = hos_missing_unmatched.reset_index().index
    hos_notmissing_unmatched['for_drop_dup'] = hos_notmissing_unmatched.reset_index().index

    # First, merge +1's with OP
    merge_with_op_missing_plus1 = dd.merge(op,hos_missing_unmatched, left_on=['MSIS_ID','STATE_CD','SRVC_BGN_DT'],
                                           right_on=['MSIS_ID','STATE_CD','SRVC_END_DT_PLUSONE'], suffixes=['_HOS','_AMB'], how = 'right')
    merge_w_op_notmissing_plus1 = dd.merge(op,hos_notmissing_unmatched, left_on=['BENE_ID','SRVC_BGN_DT'],
                                           right_on=['BENE_ID','SRVC_END_DT_PLUSONE'], suffixes=['_HOS','_AMB'], how = 'right')

    # Recover Memory
    del hos_missing_unmatched
    del hos_notmissing_unmatched

    # Drop all duplicates due to input errors
    merge_with_op_missing_plus1 = merge_with_op_missing_plus1.drop_duplicates(subset=['MSIS_ID','STATE_CD','SRVC_END_DT_PLUSONE','for_drop_dup'], keep = 'first')
    merge_w_op_notmissing_plus1 = merge_w_op_notmissing_plus1.drop_duplicates(subset=['BENE_ID','SRVC_END_DT_PLUSONE','for_drop_dup'], keep = 'first')

    # Create DF of those not matched with OP
    hos_missing_unmatched_plus1 = merge_with_op_missing_plus1[merge_with_op_missing_plus1['ind_for_hos_match'].isna()]
    hos_notmissing_unmatched_plus1 = merge_w_op_notmissing_plus1[merge_w_op_notmissing_plus1['ind_for_hos_match'].isna()]

    # Clean unmatched dataset
    hos_missing_unmatched_plus1 = hos_missing_unmatched_plus1.drop(['SRVC_END_DT_PLUSONE','BENE_ID_HOS', 'DGNS_CD_1','DGNS_CD_2', 'ind_for_hos_match',
                                                        'ip_ind','SRVC_BGN_DT', 'for_drop_dup'],axis=1)
    hos_notmissing_unmatched_plus1 = hos_notmissing_unmatched_plus1.drop(['SRVC_END_DT_PLUSONE','MSIS_ID_HOS', 'STATE_CD_HOS', 'DGNS_CD_1','DGNS_CD_2', 'ind_for_hos_match',
                                                              'ip_ind','SRVC_BGN_DT', 'for_drop_dup'],axis=1)
    hos_missing_unmatched_plus1 = hos_missing_unmatched_plus1.rename(columns={'BENE_ID_AMB':'BENE_ID','SRVC_BGN_DT_AMB':'SRVC_BGN_DT'})
    hos_notmissing_unmatched_plus1 = hos_notmissing_unmatched_plus1.rename(columns={'MSIS_ID_AMB':'MSIS_ID','STATE_CD_AMB':'STATE_CD',
                                                                                    'SRVC_BGN_DT_AMB':'SRVC_BGN_DT'})

    # Create DF of those matched with OP
    merge_with_op_missing_plus1 = merge_with_op_missing_plus1[merge_with_op_missing_plus1['ind_for_hos_match']==1]
    merge_w_op_notmissing_plus1 = merge_w_op_notmissing_plus1[merge_w_op_notmissing_plus1['ind_for_hos_match']==1]

    # Clean DF for OP matched before concatenating and exporting
    merge_with_op_missing_plus1 = merge_with_op_missing_plus1.drop(['BENE_ID_HOS','SRVC_END_DT_PLUSONE','for_drop_dup'],axis=1)
    merge_w_op_notmissing_plus1 = merge_w_op_notmissing_plus1.drop(['MSIS_ID_HOS','STATE_CD_HOS','SRVC_END_DT_PLUSONE','for_drop_dup'],axis=1)
    merge_with_op_missing_plus1 = merge_with_op_missing_plus1.rename(columns={'BENE_ID_AMB':'BENE_ID','SRVC_BGN_DT_HOS':'SRVC_BGN_DT'})
    merge_w_op_notmissing_plus1 = merge_w_op_notmissing_plus1.rename(columns={'MSIS_ID_AMB':'MSIS_ID','STATE_CD_AMB':'STATE_CD','SRVC_BGN_DT_HOS':'SRVC_BGN_DT'})

    #---Match Amb with OP (following day (+2))---#

    # Add +2 unmatched amb claims
    hos_missing_unmatched_plus1['SRVC_END_DT_PLUSTWO'] = hos_missing_unmatched_plus1['SRVC_END_DT'] + timedelta(days=2)
    hos_notmissing_unmatched_plus1['SRVC_END_DT_PLUSTWO'] = hos_notmissing_unmatched_plus1['SRVC_END_DT'] + timedelta(days=2)

    # Add column of consecutive numbers. Needed to drop additional duplicates in OP due to input errors
    hos_missing_unmatched_plus1 = hos_missing_unmatched_plus1.reset_index(drop=True)
    hos_notmissing_unmatched_plus1 = hos_notmissing_unmatched_plus1.reset_index(drop=True)
    hos_missing_unmatched_plus1['for_drop_dup'] = hos_missing_unmatched_plus1.reset_index().index
    hos_notmissing_unmatched_plus1['for_drop_dup'] = hos_notmissing_unmatched_plus1.reset_index().index

    # First, merge +1's with OP
    merge_with_op_missing_plus2 = dd.merge(op,hos_missing_unmatched_plus1, left_on=['MSIS_ID','STATE_CD','SRVC_BGN_DT'],
                                           right_on=['MSIS_ID','STATE_CD','SRVC_END_DT_PLUSTWO'], suffixes=['_HOS','_AMB'], how = 'right')
    merge_w_op_notmissing_plus2 = dd.merge(op,hos_notmissing_unmatched_plus1, left_on=['BENE_ID','SRVC_BGN_DT'],
                                           right_on=['BENE_ID','SRVC_END_DT_PLUSTWO'], suffixes=['_HOS','_AMB'], how = 'right')

    # Recover Memory
    del hos_missing_unmatched_plus1
    del hos_notmissing_unmatched_plus1

    # Drop all duplicates due to input errors
    merge_with_op_missing_plus2 = merge_with_op_missing_plus2.drop_duplicates(subset=['MSIS_ID','STATE_CD','SRVC_END_DT_PLUSTWO','for_drop_dup'], keep = 'first')
    merge_w_op_notmissing_plus2 = merge_w_op_notmissing_plus2.drop_duplicates(subset=['BENE_ID','SRVC_END_DT_PLUSTWO','for_drop_dup'], keep = 'first')

    # Create DF of those not matched with OP
    hos_missing_unmatched_plus2 = merge_with_op_missing_plus2[merge_with_op_missing_plus2['ind_for_hos_match'].isna()]
    hos_notmissing_unmatched_plus2 = merge_w_op_notmissing_plus2[merge_w_op_notmissing_plus2['ind_for_hos_match'].isna()]

    # Clean unmatched dataset
    hos_missing_unmatched_plus2 = hos_missing_unmatched_plus2.drop(['SRVC_END_DT_PLUSTWO','BENE_ID_HOS', 'DGNS_CD_1','DGNS_CD_2', 'ind_for_hos_match',
                                                        'ip_ind','SRVC_BGN_DT', 'for_drop_dup'],axis=1)
    hos_notmissing_unmatched_plus2 = hos_notmissing_unmatched_plus2.drop(['SRVC_END_DT_PLUSTWO','MSIS_ID_HOS', 'STATE_CD_HOS', 'DGNS_CD_1','DGNS_CD_2', 'ind_for_hos_match',
                                                              'ip_ind','SRVC_BGN_DT', 'for_drop_dup'],axis=1)
    hos_missing_unmatched_plus2 = hos_missing_unmatched_plus2.rename(columns={'BENE_ID_AMB':'BENE_ID','SRVC_BGN_DT_AMB':'SRVC_BGN_DT'})
    hos_notmissing_unmatched_plus2 = hos_notmissing_unmatched_plus2.rename(columns={'MSIS_ID_AMB':'MSIS_ID','STATE_CD_AMB':'STATE_CD',
                                                                                    'SRVC_BGN_DT_AMB':'SRVC_BGN_DT'})

    # Create DF of those matched with OP
    merge_with_op_missing_plus2 = merge_with_op_missing_plus2[merge_with_op_missing_plus2['ind_for_hos_match']==1]
    merge_w_op_notmissing_plus2 = merge_w_op_notmissing_plus2[merge_w_op_notmissing_plus2['ind_for_hos_match']==1]

    # Clean DF for OP matched before concatenating and exporting
    merge_with_op_missing_plus2 = merge_with_op_missing_plus2.drop(['BENE_ID_HOS','SRVC_END_DT_PLUSTWO','for_drop_dup'],axis=1)
    merge_w_op_notmissing_plus2 = merge_w_op_notmissing_plus2.drop(['MSIS_ID_HOS','STATE_CD_HOS','SRVC_END_DT_PLUSTWO','for_drop_dup'],axis=1)
    merge_with_op_missing_plus2 = merge_with_op_missing_plus2.rename(columns={'BENE_ID_AMB':'BENE_ID','SRVC_BGN_DT_HOS':'SRVC_BGN_DT'})
    merge_w_op_notmissing_plus2 = merge_w_op_notmissing_plus2.rename(columns={'MSIS_ID_AMB':'MSIS_ID','STATE_CD_AMB':'STATE_CD','SRVC_BGN_DT_HOS':'SRVC_BGN_DT'})

    # Recover memory
    del op

    #---Concat all matched and unmatched DF---#

    # Concat matched (includes amb claims that were not matched
    amb_hos_concat = dd.concat([merge_with_ip_missing,merge_with_ip_missing_plus1,merge_with_ip_missing_plus2,
                                merge_with_op_missing,merge_with_op_missing_plus1,merge_with_op_missing_plus2,
                                merge_w_ip_notmissing,merge_w_ip_notmissing_plus1,merge_w_ip_notmissing_plus2,
                                merge_w_op_notmissing,merge_w_op_notmissing_plus1,merge_w_op_notmissing_plus2,
                                hos_missing_unmatched_plus2,hos_notmissing_unmatched_plus2],axis=0)

    # Recover Memory
    del merge_with_ip_missing
    del merge_with_ip_missing_plus1
    del merge_with_ip_missing_plus2
    del merge_with_op_missing
    del merge_with_op_missing_plus1
    del merge_with_op_missing_plus2
    del merge_w_ip_notmissing
    del merge_w_ip_notmissing_plus1
    del merge_w_ip_notmissing_plus2
    del merge_w_op_notmissing
    del merge_w_op_notmissing_plus1
    del merge_w_op_notmissing_plus2
    del hos_missing_unmatched_plus2
    del hos_notmissing_unmatched_plus2

    #---Read Out Data---#

    # Read Out Data. Careful that some are na's and some are blanks
    amb_hos_concat.to_parquet(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicaid_ffs_mc/{year}/{mcaid_payment_type}_merged_amb_hos_claims/{state}/', compression='gzip', engine='fastparquet')

    # Recover Memory
    del amb_hos_concat

#________________________________________Run Defined functions_________________________________________________________#

# Specify years (only using 2016 for now)
years=[2016]

# Specify States
states_16=['AL','AK','AZ','AR','CA','CO','CT','DC','DE','FL','GA','HI','ID','IL','IN','IA','KS','KY','LA','ME','MD','MA',
           'MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX',
           'UT','VT','VA','WA','WV','WI','WY'] # better to run CA without cluster

# Create loop (only using 2016 for now)
for y in years:

    for s in states_16:

        # Run for FFS
        amb_match_hos(y,s,'ffs')

        # Run for MC/Encounter
        amb_match_hos(y,s,'mc')

####################### KEEP THOSE AT LEAST 91 CONSECUTIVE DAYS IN MCAID FROM THE AMB-HOS MERGE ########################
# Here, I imported all ambulance claims that were matched and not matched between ambulance and hospital claims and    #
# remove those with less than 91 consecutive days in MCAID and did not have a death at discharge status.               #                                                                                                 #
########################################################################################################################

#___________________________________________Define function____________________________________________________________#

# Define function to link ambulance to hospital claims
def keep_at_least_ninetyone_days(year,state,mcaid_payment_type):

    # Specify columns to use
    columns_amb = ['BENE_ID', 'MSIS_ID', 'STATE_CD', 'PTNT_DSCHRG_STUS_CD', 'ind_for_hos_match', 'ip_ind',
                   'SRVC_END_DT','BIRTH_DT','SEX_CD', 'RACE_ETHNCTY_CD'] + [f'DGNS_CD_{i}' for i in range(1, 13)]

    # Read in Data
    amb_hos_concat = dd.read_parquet(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicaid_ffs_mc/{year}/{mcaid_payment_type}_merged_amb_hos_claims/{state}/',
                                     compression='gzip', engine='fastparquet', columns=columns_amb)

    # Fill all na's in ID's with blanks
    amb_hos_concat['BENE_ID'] = amb_hos_concat['BENE_ID'].fillna('')
    amb_hos_concat['MSIS_ID'] = amb_hos_concat['MSIS_ID'].fillna('')

    # Convert all to datetime
    amb_hos_concat['SRVC_END_DT'] = dd.to_datetime(amb_hos_concat['SRVC_END_DT'])

    # ------------ Merge PS to remove those with less than 91 consecutive days in medicaid --------------#

    # Define Columns
    columns_ps = ['BENE_ID', 'MSIS_ID', 'STATE_CD'] + [f'MDCD_ENRLMT_DAYS_{m:02}' for m in range(1, 13)]

    # Read in data using dask
    ps = dd.read_parquet(f'/mnt/labshares/sanghavi-lab/Rhys/medicaid_data_extraction/{year}/TAFDEBSE/parquet/{state}/', engine='fastparquet', columns=columns_ps)

    # Separate into missing bene_id vs not missing bene_id to merge with PS
    amb_missingbeneid = amb_hos_concat[amb_hos_concat['BENE_ID'] == '']
    amb_notmissingbeneid = amb_hos_concat[amb_hos_concat['BENE_ID'] != '']

    # Merge current year PS with amb
    amb_missingbeneid_ps = dd.merge(amb_missingbeneid, ps, on=['MSIS_ID', 'STATE_CD'], suffixes=['_AMB', '_PS'],
                                    how='left')
    amb_notmissingbeneid_ps = dd.merge(amb_notmissingbeneid, ps, on=['BENE_ID'], suffixes=['_AMB', '_PS'],
                                       how='left')

    # Clean DF
    amb_missingbeneid_ps = amb_missingbeneid_ps.drop(['BENE_ID_PS'], axis=1)
    amb_missingbeneid_ps = amb_missingbeneid_ps.rename(columns={'BENE_ID_AMB': 'BENE_ID'})
    amb_notmissingbeneid_ps = amb_notmissingbeneid_ps.drop(['MSIS_ID_PS', 'STATE_CD_PS'], axis=1)
    amb_notmissingbeneid_ps = amb_notmissingbeneid_ps.rename(
        columns={'STATE_CD_AMB': 'STATE_CD', 'MSIS_ID_AMB': 'MSIS_ID'})

    # Recover memory
    del ps
    del amb_missingbeneid
    del amb_notmissingbeneid

    # Add columns for States that do not have data for the following year (i.e. we do not have 2017 data so we cannot create the next 3 months columns for the following year)
    amb_missingbeneid_ps['MDCD_ENRLMT_DAYS_13'] = '0'
    amb_missingbeneid_ps['MDCD_ENRLMT_DAYS_14'] = '0'
    amb_missingbeneid_ps['MDCD_ENRLMT_DAYS_15'] = '0'
    amb_notmissingbeneid_ps['MDCD_ENRLMT_DAYS_13'] = '0'
    amb_notmissingbeneid_ps['MDCD_ENRLMT_DAYS_14'] = '0'
    amb_notmissingbeneid_ps['MDCD_ENRLMT_DAYS_15'] = '0'

    # Concat
    amb_hos_concat = dd.concat([amb_missingbeneid_ps, amb_notmissingbeneid_ps], axis=0)

    # Fill in all na's with zero
    for i in range(1, 16):
        amb_hos_concat[f'MDCD_ENRLMT_DAYS_{i:02}'] = amb_hos_concat[f'MDCD_ENRLMT_DAYS_{i:02}'].fillna('0')

    # -----------------Keep those at least 91 days in Medicaid----------------------#

    # ---------Codes to count number of days in first month---------#

    # Convert all to datetime
    amb_hos_concat['SRVC_END_DT'] = dd.to_datetime(amb_hos_concat['SRVC_END_DT'])

    # Remove any claims that have nan in service begin date
    amb_hos_concat = amb_hos_concat[~amb_hos_concat['SRVC_END_DT'].isna()]

    # Convert columns to floats (the range up to 16 to account for those who had service date oct-dec)
    for i in range(1, 16):
        amb_hos_concat[f'MDCD_ENRLMT_DAYS_{i:02}'] = amb_hos_concat[f'MDCD_ENRLMT_DAYS_{i:02}'].astype('float')

    # Find the end of the month from service begin date
    amb_hos_concat['EndOfMonth'] = dd.to_datetime(amb_hos_concat['SRVC_END_DT']) + MonthEnd(1)

    # Find number of days from service begin date to end of month
    amb_hos_concat['Days_Until_End_Month'] = amb_hos_concat['EndOfMonth'] - amb_hos_concat['SRVC_END_DT']

    # Convert from days/timedelta to float
    amb_hos_concat['Days_Until_End_Month'] = amb_hos_concat['Days_Until_End_Month'].dt.days.astype('float')

    # Create column for days enrolled for that month based on service begin date
    amb_hos_concat['days_enrolled'] = ''
    for i in range(1, 13):
        amb_hos_concat['days_enrolled'] = amb_hos_concat['days_enrolled'].mask(
            (amb_hos_concat['SRVC_END_DT'].dt.month == i), amb_hos_concat[f'MDCD_ENRLMT_DAYS_{i:02}'])

    # Convert to float
    amb_hos_concat['days_enrolled'] = amb_hos_concat['days_enrolled'].astype('float')

    # Filter only those with days enrolled more than days until end of month (i.e. for the first month, individual needs to be enrolled more than the time since they took the amb ride for the first month)
    amb_hos_concat = amb_hos_concat[amb_hos_concat['days_enrolled'] >= amb_hos_concat['Days_Until_End_Month']]

    # ---Codes to count number of days enrolled in Medicaid in the next months---#

    # Create new column to account for the subsequent months after initial month
    amb_hos_concat['days_enrolled_after_three_months'] = ''

    # For next months: Add subsequent 3 months for number of days enrolled and put into new column
    for i in range(1, 13):
        amb_hos_concat['days_enrolled_after_three_months'] = amb_hos_concat['days_enrolled_after_three_months'].mask(
            (amb_hos_concat['SRVC_END_DT'].dt.month == i), amb_hos_concat[f'MDCD_ENRLMT_DAYS_{i+1:02}'] + \
            amb_hos_concat[f'MDCD_ENRLMT_DAYS_{i+2:02}'] + amb_hos_concat[f'MDCD_ENRLMT_DAYS_{i+3:02}'])

    # Convert to float
    amb_hos_concat['days_enrolled_after_three_months'] = amb_hos_concat['days_enrolled_after_three_months'].astype(
        'float')

    # ---Codes to filter individuals with at least 91 days in Medicaid---#

    # Add to see if individuals enrolled at least 91 days
    amb_hos_concat['total_enrolled_after_4_months'] = amb_hos_concat['days_enrolled_after_three_months'] + \
                                                      amb_hos_concat['Days_Until_End_Month']

    # Filter based on if individuals with service date from Jan-Dec have at least 91 days in medicaid or are dead within the 90 days
    amb_hos_concat = amb_hos_concat[
        (amb_hos_concat['total_enrolled_after_4_months'] > 90) | (amb_hos_concat['PTNT_DSCHRG_STUS_CD'] == '20') | (
                    amb_hos_concat['PTNT_DSCHRG_STUS_CD'] == '40') |
        (amb_hos_concat['PTNT_DSCHRG_STUS_CD'] == '41') | (amb_hos_concat['PTNT_DSCHRG_STUS_CD'] == '42')]

    # Clean DF before exporting
    amb_hos_concat = amb_hos_concat.drop([f'MDCD_ENRLMT_DAYS_{i:02}' for i in range(1, 16)] +
                                         ['EndOfMonth', 'Days_Until_End_Month', 'days_enrolled',
                                          'days_enrolled_after_three_months', 'total_enrolled_after_4_months'], axis=1)

    # Read Out
    amb_hos_concat.to_parquet(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicaid_ffs_mc/{year}/{mcaid_payment_type}_merged_amb_hos_claims_ninetyonedays/{state}/',
                              compression='gzip', engine='fastparquet')

#________________________________________Run Defined functions_________________________________________________________#

# Specify years (use only 2016 for now)
years=[2016]

# Specify States
states_16=['AL','AK','AZ','AR','CA','CO','CT','DC','DE','FL','GA','HI','ID','IL','IN','IA','KS','KY','LA','ME','MD','MA',
           'MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX',
           'UT','VT','VA','WA','WV','WI','WY'] # better to run CA without cluster

# Create loop (use only 2016 for now)
for y in years:

    # States available in 2016
    for s in states_16:

        # Run for FFS
        keep_at_least_ninetyone_days(y,s,'ffs')

        # Run for MC/Encounter
        keep_at_least_ninetyone_days(y,s,'mc')





