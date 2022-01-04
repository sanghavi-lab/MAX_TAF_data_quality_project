#----------------------------------------------------------------------------------------------------------------------#
# Project: Medicaid Data Quality Project
# Authors: Jessy Nguyen
# Last Updated: August 12, 2021
# Description: The goal of this script is to link (1) ambulance claims with mileage information and (2) ambulance
#              claims with hospital claims by states for each year for Medicaid. Since we are linking within state, we
#              are dropping linkages that delivered patients to a hospital outside of the beneficiary's state. Lastly,
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
    columns_amb=['BENE_ID','MSIS_ID','STATE_CD','SRVC_BGN_DT','SRVC_END_DT','PRCDR_SRVC_MDFR_CD']+['EL_DOB','EL_DOD','EL_DOD_PS_NEXT3M','EL_SEX_CD',
                 'EL_RACE_ETHNCY_CD']+['EL_DAYS_EL_CNT_{}'.format(i) for i in range(1,16)]

    # Read in Ambulance
    amb = dd.read_parquet(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicaid_ffs_mc/{year}/amb_{mcaid_payment_type}/{state}/',engine='fastparquet',columns=columns_amb)

    # Convert to Datetime
    amb['SRVC_BGN_DT'] = dd.to_datetime(amb['SRVC_BGN_DT'])
    amb['SRVC_END_DT'] = dd.to_datetime(amb['SRVC_END_DT'])
    amb['EL_DOB'] = dd.to_datetime(amb['EL_DOB'])
    amb['EL_DOD'] = dd.to_datetime(amb['EL_DOD'])
    amb['EL_DOD_PS_NEXT3M'] = dd.to_datetime(amb['EL_DOD_PS_NEXT3M'])

    # Fill na with blanks to keep DF consistent
    amb['BENE_ID'] = amb['BENE_ID'].fillna('')
    amb['MSIS_ID'] = amb['MSIS_ID'].fillna('')

    #-----------------Keep those at least 91 days in Medicaid----------------------#

    #---------Codes to count number of days in first month---------#

    # Convert columns to floats (the range up to 16 to account for those who had service date oct-dec)
    for i in range(1,16):
        amb['EL_DAYS_EL_CNT_{}'.format(i)] = amb['EL_DAYS_EL_CNT_{}'.format(i)].astype('float')

    # Find the end of the month from service begin date
    amb['EndOfMonth'] =  dd.to_datetime(amb['SRVC_BGN_DT']) + MonthEnd(1)

    # Find number of days from service begin date to end of month
    amb['Days_Until_End_Month'] = amb['EndOfMonth'] - amb['SRVC_BGN_DT']

    # Convert from days/timedelta to integer
    amb['Days_Until_End_Month'] = amb['Days_Until_End_Month'].dt.days.astype('int64')

    # Create column for days enrolled for that month based on service begin date
    amb['days_enrolled'] = ''
    for i in range(1,13):
        amb['days_enrolled'] = amb['days_enrolled'].mask((amb['SRVC_BGN_DT'].dt.month==i), amb['EL_DAYS_EL_CNT_{}'.format(i)])

    # Convert to float
    amb['days_enrolled'] = amb['days_enrolled'].astype('float')

    # Filter only those with days enrolled more than days until end of month (i.e. for the first month, individual needs to be enrolled more than the time since they took the amb ride for the first month)
    amb = amb[amb['days_enrolled']>=amb['Days_Until_End_Month']]

    #---Codes to count number of days enrolled in Medicaid in the next months---#

    # Create new column to account for the subsequent months after initial month
    amb['days_enrolled_after_three_months'] = ''

    # For next months: Add subsequent 3 months for number of days enrolled and put into new column
    for i in range(1,13):
        amb['days_enrolled_after_three_months'] = amb['days_enrolled_after_three_months'].mask((amb['SRVC_BGN_DT'].dt.month==i), amb['EL_DAYS_EL_CNT_{}'.format(i+1)] + \
                                                                    amb['EL_DAYS_EL_CNT_{}'.format(i+2)] + amb['EL_DAYS_EL_CNT_{}'.format(i+3)])

    # Convert to float
    amb['days_enrolled_after_three_months'] = amb['days_enrolled_after_three_months'].astype('float')

    #---Codes to filter individuals with at least 91 days in Medicaid---#

    # Add to see if individuals enrolled at least 91 days
    amb['total_enrolled_after_4_months'] = amb['days_enrolled_after_three_months'] + amb['Days_Until_End_Month']

    # Filter based on if individuals with service date from Jan-Dec have at least 91 days in medicaid
    amb = amb[(amb['total_enrolled_after_4_months'] > 90)]

    # Clean DF before exporting
    amb = amb.drop(['EL_DAYS_EL_CNT_{}'.format(i) for i in range(1,16)] +
                           ['EndOfMonth','Days_Until_End_Month','days_enrolled','days_enrolled_after_three_months','total_enrolled_after_4_months'],axis=1)

    # Create DF for missing_beneid and notmissing_beneid
    amb_missingid = amb[(amb['BENE_ID']=='')]
    amb_notmissingid = amb[(amb['BENE_ID']!='')]

    # Recover Data
    del amb

    #---Import and Match with Mileage---#

    # Specify Columns for mileage
    columns_mi = ['BENE_ID','MSIS_ID','STATE_CD','SRVC_BGN_DT','QTY_SRVC_UNITS']

    # Read in Mileage
    mileage = dd.read_parquet(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicaid_ffs_mc/{year}/mileage/{state}/',engine='fastparquet',columns=columns_mi)

    # Convert to Datetime
    mileage['SRVC_BGN_DT'] = dd.to_datetime(mileage['SRVC_BGN_DT'])

    # Create column to count number matched
    mileage['ind_for_mi_match'] = 1

    # Add column of consecutive numbers. Need this to drop additional duplicates due to input errors
    amb_missingid = amb_missingid.reset_index(drop=True)
    amb_notmissingid = amb_notmissingid.reset_index(drop=True)
    amb_missingid['for_drop_dup'] = amb_missingid.reset_index().index
    amb_notmissingid['for_drop_dup'] = amb_notmissingid.reset_index().index

    # Merge the mileage info with the ambulance claims
    amb_missingid_mi = dd.merge(mileage,amb_missingid, on=['MSIS_ID','STATE_CD','SRVC_BGN_DT'],suffixes=['_MI','_AMB'],how='right')
    amb_notmissingid_mi = dd.merge(mileage,amb_notmissingid,on=['BENE_ID','SRVC_BGN_DT'],suffixes=['_MI','_AMB'],how='right')

    # Recover Memory
    del amb_missingid
    del amb_notmissingid
    del mileage

    # Drop all duplicates due to input errors
    amb_missingid_mi = amb_missingid_mi.drop_duplicates(subset=['MSIS_ID','STATE_CD','SRVC_BGN_DT','for_drop_dup'], keep = 'last')
    amb_notmissingid_mi = amb_notmissingid_mi.drop_duplicates(subset=['BENE_ID','SRVC_BGN_DT','for_drop_dup'], keep = 'last')

    # Concat data together
    amb_mi = dd.concat([amb_missingid_mi,amb_notmissingid_mi],axis=0)

    # Recover Memory
    del amb_missingid_mi
    del amb_notmissingid_mi

    # Read out Data. Due to differences in column names from merging, we separated the claims with missing BENE_ID's from those with BENE_ID's
    amb_mi.to_parquet(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicaid_ffs_mc/{year}/{mcaid_payment_type}_merged_amb_mileage/{state}/',
                                compression='gzip', engine='fastparquet')

#________________________________________Run Defined functions_________________________________________________________#

# Specify years
years=[2011,2012,2013,2014]

# Specify States
states_11=['AL','AK','AZ','AR','CA','CO','CT','DC','DE','FL','GA','HI','IL','IN','IA','KS','KY','LA','ME','MD','MA','MI',
           'MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT',
           'VT','VA','WA','WV','WI','WY']
states_12=['AL','AK','AZ','AR','CA','CO','CT','DC','DE','FL','GA','HI','ID','IL','IN','IA','KS','KY','LA','ME','MD','MA',
           'MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX',
           'UT','VT','VA','WA','WV','WI','WY']
states_13=['AZ','AR','CA','CT','GA','HI','ID','IN','IA','LA','MA','MI','MN','MS','MO','NJ','NY','OH','OK','OR','PA','SD',
           'TN','UT','VT','WA','WV','WY']
states_14=['CA','GA','ID','IA','LA','MI','MN','MS','MO','NJ','PA','SD','TN','UT','VT','WV','WY']

# Create loop
for y in years:

    # 2011
    if y in [2011]:

            # States available in 2011
            for s in states_11:

                # Run for FFS
                amb_match_mileage(y, s, 'ffs')

                # Run for MC/Encounter
                amb_match_mileage(y, s, 'mc')

    # 2012
    if y in [2012]:

            # States available in 2012
            for s in states_12:

                # Run for FFS
                amb_match_mileage(y, s, 'ffs')

                # Run for MC/Encounter
                amb_match_mileage(y, s, 'mc')

    # 2013
    if y in [2013]:

            # States available in 2013
            for s in states_13:

                # Run for FFS
                amb_match_mileage(y, s, 'ffs')

                # Run for MC/Encounter
                amb_match_mileage(y, s, 'mc')

    # 2014
    if y in [2014]:

            # States available in 2014
            for s in states_14:

                # Run for FFS
                amb_match_mileage(y, s, 'ffs')

                # Run for MC/Encounter
                amb_match_mileage(y, s, 'mc')

######################################## MATCH IP AND OP WITH AMB CLAIMS ###############################################
# The following script links the exported ambulance claims with hospital claims. We linked with IP same day, next day, #
# and the following day first then repeat the process with OP. For individuals who had ambulance rides on December 31, #
# we made sure to link the following year. We linked within states; this means that the denominator does not include   #
# individuals who lived in one state but transported to a hospital in another state.                                   #
#######################################################################################################################

#___________________________________________Define function____________________________________________________________#

# Define function to link ambulance to hospital claims
def amb_match_hos(year,state,mcaid_payment_type,list_state_available_following_year):

    #---Import Amb---#

    # Specify columns needed
    columns_amb = ['BENE_ID','MSIS_ID','STATE_CD','SRVC_END_DT','EL_DOB','EL_DOD','EL_SEX_CD','EL_RACE_ETHNCY_CD']

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
    columns_ip=['BENE_ID','MSIS_ID','STATE_CD','SRVC_BGN_DT','PATIENT_STATUS_CD'] +\
               ['DIAG_CD_{}'.format(i) for i in range(1,10)]

    if (year in [2011,2012,2013]) & (state in list_state_available_following_year): # For individuals who had ambulance rides on December 31, we made sure to link the following year.

        # Read in IP same year
        ip_sameyear = dd.read_parquet(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicaid_ffs_mc/{year}/ip/{state}/', engine='fastparquet', columns=columns_ip)

        # Read in IP the following year. Need this since patients may be admitted on Jan 1st of the next year.
        ip_nextyear = dd.read_parquet(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicaid_ffs_mc/{year+1}/ip/{state}/', engine='fastparquet', columns=columns_ip)

        # Convert to nextyear IP to datetime to filter only January 1st individuals
        ip_nextyear['SRVC_BGN_DT'] = dd.to_datetime(ip_nextyear['SRVC_BGN_DT'])

        # Keep only individuals who were admitted on Jan 1st or 2nd
        ip_nextyear = ip_nextyear[(ip_nextyear['SRVC_BGN_DT'].dt.month==1)&((ip_nextyear['SRVC_BGN_DT'].dt.day==1)|(ip_nextyear['SRVC_BGN_DT'].dt.day==2))]

        # Concat ip_sameyear and ip_nextyear
        ip = dd.concat([ip_sameyear,ip_nextyear],axis=0)

        # Delete DFs to recover memory
        del ip_nextyear
        del ip_sameyear

    else:  # Only for 2014 since we don't have 2015

        # Read in IP same year
        ip = dd.read_parquet(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicaid_ffs_mc/{year}/ip/{state}/', engine='fastparquet', columns=columns_ip)

    #---Match Amb with IP (same day)---#

    # Convert all to datetime before merging
    ip['SRVC_BGN_DT'] = dd.to_datetime(ip['SRVC_BGN_DT'])
    amb_missingid['SRVC_END_DT'] = dd.to_datetime(amb_missingid['SRVC_END_DT'])
    amb_notmissingid['SRVC_END_DT'] = dd.to_datetime(amb_notmissingid['SRVC_END_DT'])

    # Add columns of one's in IP and OP DF to eventually drop unmatched
    ip['ind_for_hos_match'] = 1

    # Create indicator for IP vs OP (used to check data later regarding trauma cases)
    ip['ip_ind'] = 1

    # Label cases where same day was found
    amb_missingid['which_day_matched'] = 'matched_same_day'
    amb_notmissingid['which_day_matched'] = 'matched_same_day'

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
    merge_with_ip_missing = merge_with_ip_missing.drop_duplicates(subset=['MSIS_ID','STATE_CD','SRVC_BGN_DT','for_drop_dup'], keep = 'last')
    merge_w_ip_notmissing = merge_w_ip_notmissing.drop_duplicates(subset=['BENE_ID','SRVC_BGN_DT','for_drop_dup'], keep = 'last')

    # Create DF of those not matched with IP
    hos_missing_unmatched = merge_with_ip_missing[merge_with_ip_missing['ind_for_hos_match'].isna()]
    hos_notmissing_unmatched = merge_w_ip_notmissing[merge_w_ip_notmissing['ind_for_hos_match'].isna()]

    # Clean unmatched dataset to remerge after adding +1
    hos_missing_unmatched = hos_missing_unmatched.drop(['BENE_ID_HOS', 'PATIENT_STATUS_CD', 'DIAG_CD_1',
                                                        'DIAG_CD_2', 'DIAG_CD_3', 'DIAG_CD_4', 'DIAG_CD_5','DIAG_CD_6', 'DIAG_CD_7', 'DIAG_CD_8',
                                                        'DIAG_CD_9', 'ind_for_hos_match', 'ip_ind','SRVC_BGN_DT','which_day_matched', 'for_drop_dup'],axis=1)
    hos_notmissing_unmatched = hos_notmissing_unmatched.drop(['MSIS_ID_HOS', 'STATE_CD_HOS', 'PATIENT_STATUS_CD', 'DIAG_CD_1',
                                                        'DIAG_CD_2', 'DIAG_CD_3', 'DIAG_CD_4', 'DIAG_CD_5','DIAG_CD_6', 'DIAG_CD_7', 'DIAG_CD_8',
                                                        'DIAG_CD_9', 'ind_for_hos_match', 'ip_ind','SRVC_BGN_DT','which_day_matched', 'for_drop_dup'],axis=1)
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

    # Label cases where day +1 was found
    hos_missing_unmatched['which_day_matched'] = 'match_day_plusone'
    hos_notmissing_unmatched['which_day_matched'] = 'match_day_plusone'

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
    merge_with_ip_missing_plus1 = merge_with_ip_missing_plus1.drop_duplicates(subset=['MSIS_ID','STATE_CD','SRVC_BGN_DT','for_drop_dup'], keep = 'last')
    merge_w_ip_notmissing_plus1 = merge_w_ip_notmissing_plus1.drop_duplicates(subset=['BENE_ID','SRVC_BGN_DT','for_drop_dup'], keep = 'last')

    # Create DF of those not matched with IP
    hos_missing_unmatched_plus1 = merge_with_ip_missing_plus1[merge_with_ip_missing_plus1['ind_for_hos_match'].isna()]
    hos_notmissing_unmatched_plus1 = merge_w_ip_notmissing_plus1[merge_w_ip_notmissing_plus1['ind_for_hos_match'].isna()]

    # Clean unmatched dataset
    hos_missing_unmatched_plus1 = hos_missing_unmatched_plus1.drop(['BENE_ID_HOS', 'PATIENT_STATUS_CD', 'DIAG_CD_1',
                                                        'DIAG_CD_2', 'DIAG_CD_3', 'DIAG_CD_4', 'DIAG_CD_5','DIAG_CD_6', 'DIAG_CD_7', 'DIAG_CD_8',
                                                        'DIAG_CD_9', 'ind_for_hos_match', 'ip_ind','SRVC_BGN_DT','SRVC_END_DT_PLUSONE','which_day_matched', 'for_drop_dup'],axis=1)
    hos_notmissing_unmatched_plus1 = hos_notmissing_unmatched_plus1.drop(['MSIS_ID_HOS', 'STATE_CD_HOS', 'PATIENT_STATUS_CD', 'DIAG_CD_1',
                                                        'DIAG_CD_2', 'DIAG_CD_3', 'DIAG_CD_4', 'DIAG_CD_5','DIAG_CD_6', 'DIAG_CD_7', 'DIAG_CD_8',
                                                        'DIAG_CD_9', 'ind_for_hos_match', 'ip_ind','SRVC_BGN_DT','SRVC_END_DT_PLUSONE','which_day_matched', 'for_drop_dup'],axis=1)
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

    # Label cases where day +1 was found
    hos_missing_unmatched_plus1['which_day_matched'] = 'match_day_plustwo'
    hos_notmissing_unmatched_plus1['which_day_matched'] = 'match_day_plustwo'

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
    merge_with_ip_missing_plus2 = merge_with_ip_missing_plus2.drop_duplicates(subset=['MSIS_ID','STATE_CD','SRVC_END_DT','for_drop_dup'], keep = 'last')
    merge_w_ip_notmissing_plus2 = merge_w_ip_notmissing_plus2.drop_duplicates(subset=['BENE_ID','SRVC_END_DT','for_drop_dup'], keep = 'last')

    # Create DF of those not matched with IP
    hos_missing_unmatched_plus2 = merge_with_ip_missing_plus2[merge_with_ip_missing_plus2['ind_for_hos_match'].isna()]
    hos_notmissing_unmatched_plus2 = merge_w_ip_notmissing_plus2[merge_w_ip_notmissing_plus2['ind_for_hos_match'].isna()]

    # Clean unmatched dataset
    hos_missing_unmatched_plus2 = hos_missing_unmatched_plus2.drop(['BENE_ID_HOS', 'PATIENT_STATUS_CD', 'DIAG_CD_1',
                                                        'DIAG_CD_2', 'DIAG_CD_3', 'DIAG_CD_4', 'DIAG_CD_5','DIAG_CD_6', 'DIAG_CD_7', 'DIAG_CD_8',
                                                        'DIAG_CD_9', 'ind_for_hos_match', 'ip_ind','SRVC_BGN_DT','SRVC_END_DT_PLUSTWO','which_day_matched', 'for_drop_dup'],axis=1)
    hos_notmissing_unmatched_plus2 = hos_notmissing_unmatched_plus2.drop(['MSIS_ID_HOS', 'STATE_CD_HOS', 'PATIENT_STATUS_CD', 'DIAG_CD_1',
                                                        'DIAG_CD_2', 'DIAG_CD_3', 'DIAG_CD_4', 'DIAG_CD_5','DIAG_CD_6', 'DIAG_CD_7', 'DIAG_CD_8',
                                                        'DIAG_CD_9', 'ind_for_hos_match', 'ip_ind','SRVC_BGN_DT','SRVC_END_DT_PLUSTWO','which_day_matched', 'for_drop_dup'],axis=1)
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
    columns_op=['BENE_ID','MSIS_ID','STATE_CD','SRVC_BGN_DT'] + ['DIAG_CD_{}'.format(i) for i in range(1,3)]

    if (year in [2011,2012,2013]) & (state in list_state_available_following_year):

        # Read in OP same year
        op_sameyear = dd.read_parquet(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicaid_ffs_mc/{year}/op_subset/{state}/',
            engine='fastparquet', columns=columns_op)

        # Read in OP the following year. Need this since patients may be admitted on Jan 1st of the next year.
        op_nextyear = dd.read_parquet(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicaid_ffs_mc/{year+1}/op_subset/{state}/',
            engine='fastparquet', columns=columns_op)

        # Convert to nextyear OP to datetime to filter only January 1st individuals
        op_nextyear['SRVC_BGN_DT'] = dd.to_datetime(op_nextyear['SRVC_BGN_DT'])

        # Keep only individuals who were admitted on Jan 1st/2nd
        op_nextyear = op_nextyear[(op_nextyear['SRVC_BGN_DT'].dt.month == 1) & ((op_nextyear['SRVC_BGN_DT'].dt.day == 1) | (op_nextyear['SRVC_BGN_DT'].dt.day == 2))]

        # Concat op_sameyear and op_nextyear
        op = dd.concat([op_sameyear, op_nextyear], axis=0)

        # Delete DFs to recover memory
        del op_nextyear
        del op_sameyear

    else: # Only for 2014 since we don't have 2015

        # Read in OP same year
        op = dd.read_parquet(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicaid_ffs_mc/{year}/op_subset/{state}/',
            engine='fastparquet', columns=columns_op)

    #---Match Amb with OP (same day)---#

    # Convert all to datetime before merging
    op['SRVC_BGN_DT'] = dd.to_datetime(op['SRVC_BGN_DT'])

    # Add columns of one's in OP and OP DF to filter out unmatched
    op['ind_for_hos_match'] = 1

    # Create indicator for IP vs OP (used to check data later regarding trauma cases)
    op['ip_ind'] = 0

    # Label cases where same day was found
    hos_missing_unmatched_plus2['which_day_matched'] = 'matched_same_day'
    hos_notmissing_unmatched_plus2['which_day_matched'] = 'matched_same_day'

    # Add column of consecutive numbers. Needed to drop additional duplicates in OP due to input errors
    hos_missing_unmatched_plus2 = hos_missing_unmatched_plus2.reset_index(drop=True)
    hos_notmissing_unmatched_plus2 = hos_notmissing_unmatched_plus2.reset_index(drop=True)
    hos_missing_unmatched_plus2['for_drop_dup'] = hos_missing_unmatched_plus2.reset_index().index
    hos_notmissing_unmatched_plus2['for_drop_dup'] = hos_notmissing_unmatched_plus2.reset_index().index

    # First, merge with OP
    merge_with_op_missing = dd.merge(op,hos_missing_unmatched_plus2, left_on=['MSIS_ID','STATE_CD','SRVC_BGN_DT'],right_on=['MSIS_ID','STATE_CD','SRVC_END_DT'],suffixes=['_HOS','_AMB'], how = 'right')
    merge_w_op_notmissing = dd.merge(op,hos_notmissing_unmatched_plus2, left_on=['BENE_ID','SRVC_BGN_DT'],right_on=['BENE_ID','SRVC_END_DT'],suffixes=['_HOS','_AMB'], how = 'right')

    # Recover Memory
    del hos_missing_unmatched_plus2
    del hos_notmissing_unmatched_plus2

    # Drop all duplicates due to input errors
    merge_with_op_missing = merge_with_op_missing.drop_duplicates(subset=['MSIS_ID','STATE_CD','SRVC_BGN_DT','for_drop_dup'], keep = 'first')
    merge_w_op_notmissing = merge_w_op_notmissing.drop_duplicates(subset=['BENE_ID','SRVC_BGN_DT','for_drop_dup'], keep = 'first')

    # Create DF of those not matched with OP
    hos_missing_unmatched = merge_with_op_missing[merge_with_op_missing['ind_for_hos_match'].isna()]
    hos_notmissing_unmatched = merge_w_op_notmissing[merge_w_op_notmissing['ind_for_hos_match'].isna()]

    # Clean unmatched dataset
    hos_missing_unmatched = hos_missing_unmatched.drop(['BENE_ID_HOS', 'DIAG_CD_1',
                                                        'DIAG_CD_2', 'ind_for_hos_match', 'ip_ind','SRVC_BGN_DT','which_day_matched', 'for_drop_dup'],axis=1)
    hos_notmissing_unmatched = hos_notmissing_unmatched.drop(['MSIS_ID_HOS', 'STATE_CD_HOS', 'DIAG_CD_1',
                                                        'DIAG_CD_2', 'ind_for_hos_match', 'ip_ind','SRVC_BGN_DT','which_day_matched', 'for_drop_dup'],axis=1)
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

    # Label cases where day +1 was found
    hos_missing_unmatched['which_day_matched'] = 'match_day_plusone'
    hos_notmissing_unmatched['which_day_matched'] = 'match_day_plusone'

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
    merge_with_op_missing_plus1 = merge_with_op_missing_plus1.drop_duplicates(subset=['MSIS_ID','STATE_CD','SRVC_BGN_DT','for_drop_dup'], keep = 'first')
    merge_w_op_notmissing_plus1 = merge_w_op_notmissing_plus1.drop_duplicates(subset=['BENE_ID','SRVC_BGN_DT','for_drop_dup'], keep = 'first')

    # Create DF of those not matched with OP
    hos_missing_unmatched_plus1 = merge_with_op_missing_plus1[merge_with_op_missing_plus1['ind_for_hos_match'].isna()]
    hos_notmissing_unmatched_plus1 = merge_w_op_notmissing_plus1[merge_w_op_notmissing_plus1['ind_for_hos_match'].isna()]

    # Clean unmatched dataset
    hos_missing_unmatched_plus1 = hos_missing_unmatched_plus1.drop(['BENE_ID_HOS', 'DIAG_CD_1',
                                                        'DIAG_CD_2', 'ind_for_hos_match', 'ip_ind','SRVC_BGN_DT','SRVC_END_DT_PLUSONE','which_day_matched', 'for_drop_dup'],axis=1)
    hos_notmissing_unmatched_plus1 = hos_notmissing_unmatched_plus1.drop(['MSIS_ID_HOS', 'STATE_CD_HOS', 'DIAG_CD_1',
                                                        'DIAG_CD_2', 'ind_for_hos_match', 'ip_ind','SRVC_BGN_DT','SRVC_END_DT_PLUSONE','which_day_matched', 'for_drop_dup'],axis=1)
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

    # Label cases where day +1 was found
    hos_missing_unmatched_plus1['which_day_matched'] = 'match_day_plustwo'
    hos_notmissing_unmatched_plus1['which_day_matched'] = 'match_day_plustwo'

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
    merge_with_op_missing_plus2 = merge_with_op_missing_plus2.drop_duplicates(subset=['MSIS_ID','STATE_CD','SRVC_BGN_DT','for_drop_dup'], keep = 'first')
    merge_w_op_notmissing_plus2 = merge_w_op_notmissing_plus2.drop_duplicates(subset=['BENE_ID','SRVC_BGN_DT','for_drop_dup'], keep = 'first')

    # Create DF of those not matched with OP
    hos_missing_unmatched_plus2 = merge_with_op_missing_plus2[merge_with_op_missing_plus2['ind_for_hos_match'].isna()]
    hos_notmissing_unmatched_plus2 = merge_w_op_notmissing_plus2[merge_w_op_notmissing_plus2['ind_for_hos_match'].isna()]

    # Clean unmatched dataset
    hos_missing_unmatched_plus2 = hos_missing_unmatched_plus2.drop(['BENE_ID_HOS', 'DIAG_CD_1','ind_for_hos_match','ip_ind',
                                                        'DIAG_CD_2','SRVC_BGN_DT','SRVC_END_DT_PLUSTWO','which_day_matched', 'for_drop_dup'],axis=1)
    hos_notmissing_unmatched_plus2 = hos_notmissing_unmatched_plus2.drop(['MSIS_ID_HOS', 'STATE_CD_HOS', 'DIAG_CD_1','ind_for_hos_match','ip_ind',
                                                        'DIAG_CD_2','SRVC_BGN_DT','SRVC_END_DT_PLUSTWO','which_day_matched', 'for_drop_dup'],axis=1)
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

# Specify years
years=[2011,2012,2013,2014]

# Specify States
states_11=['AL','AK','AZ','AR','CA','CO','CT','DC','DE','FL','GA','HI','IL','IN','IA','KS','KY','LA','ME','MD','MA','MI',
           'MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT',
           'VT','VA','WA','WV','WI','WY']
states_12=['AL','AK','AZ','AR','CA','CO','CT','DC','DE','FL','GA','HI','ID','IL','IN','IA','KS','KY','LA','ME','MD','MA',
           'MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX',
           'UT','VT','VA','WA','WV','WI','WY']
states_13=['AZ','AR','CA','CT','GA','HI','ID','IN','IA','LA','MA','MI','MN','MS','MO','NJ','NY','OH','OK','OR','PA','SD',
           'TN','UT','VT','WA','WV','WY']
states_14=['CA','GA','ID','IA','LA','MI','MN','MS','MO','NJ','PA','SD','TN','UT','VT','WV','WY']

# Specify the states available the following year
state_11_available_in_12=['AL','AK','AZ','AR','CA','CO','CT','DC','DE','FL','GA','HI','IL','IN','IA','KS','KY','LA','ME','MD','MA','MI',
                          'MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT',
                          'VT','VA','WA','WV','WI','WY']
state_12_available_in_13=['AZ','AR','CA','CT','GA','HI','ID','IN','IA','LA','MA','MI','MN','MS','MO','NJ','NY','OH','OK',
                          'OR','PA','SD','TN','UT','VT','WA','WV','WY']
state_13_available_in_14=['CA','GA','ID','IA','LA','MI','MN','MS','MO','NJ','PA','SD','TN','UT','VT','WV','WY']

# Create loop
for y in years:

    # 2011
    if y in [2011]:

            # States available in 2011
            for s in states_11:

                # Run for FFS
                amb_match_hos(y,s,'ffs',state_11_available_in_12)

                # Run for MC/Encounter
                amb_match_hos(y,s,'mc',state_11_available_in_12)

    # 2012
    if y in [2012]:

            # States available in 2012
            for s in states_12:

                # Run for FFS
                amb_match_hos(y,s,'ffs',state_12_available_in_13)

                # Run for MC/Encounter
                amb_match_hos(y,s,'mc',state_12_available_in_13)

    # 2013
    if y in [2013]:

            # States available in 2013
            for s in states_13:

                # Run for FFS
                amb_match_hos(y,s,'ffs',state_13_available_in_14)

                # Run for MC/Encounter
                amb_match_hos(y,s,'mc',state_13_available_in_14)

    # 2014
    if y in [2014]:

            # States available in 2014
            for s in states_14:

                # Run for FFS
                amb_match_hos(y,s,'ffs',['filler']) # no 2015 so I just added a filler

                # Run for MC/Encounter
                amb_match_hos(y,s,'mc',['filler']) # no 2015 so I just added a filler

####################### KEEP THOSE AT LEAST 91 CONSECUTIVE DAYS IN MCAID FROM THE AMB-HOS MERGE ########################
# Here, I imported all ambulance claims that were matched and not matched between ambulance and hospital claims and    #
# remove those with less than 91 consecutive days in MCAID and did not have a death at discharge status.               #                                                                                                 #
########################################################################################################################

#___________________________________________Define function____________________________________________________________#

# Define function to link ambulance to hospital claims
def keep_at_least_ninetyone_days(year,state,mcaid_payment_type,list_state_available_following_year):

    # Specify columns to use
    columns_amb = ['MSIS_ID', 'STATE_CD', 'PATIENT_STATUS_CD', 'DIAG_CD_1', 'DIAG_CD_2', 'DIAG_CD_3', 'DIAG_CD_4',
                   'DIAG_CD_5', 'DIAG_CD_6', 'DIAG_CD_7', 'DIAG_CD_8', 'DIAG_CD_9', 'ind_for_hos_match', 'ip_ind',
                   'BENE_ID','SRVC_END_DT', 'EL_DOB', 'EL_SEX_CD', 'EL_RACE_ETHNCY_CD', 'which_day_matched']

    # Read in Data
    amb_hos_concat = dd.read_parquet(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicaid_ffs_mc/{year}/{mcaid_payment_type}_merged_amb_hos_claims/{state}/',
                                     compression='gzip', engine='fastparquet', columns=columns_amb)

    # Fill all na's in ID's with blanks
    amb_hos_concat['BENE_ID'] = amb_hos_concat['BENE_ID'].fillna('')
    amb_hos_concat['MSIS_ID'] = amb_hos_concat['MSIS_ID'].fillna('')

    # Convert all to datetime
    amb_hos_concat['SRVC_END_DT'] = dd.to_datetime(amb_hos_concat['SRVC_END_DT'])

    # ------------ Merge PS to remove those with less than 91 consecutive days in medicaid --------------#

    if (year in [2011,2012,2013]) & (state in list_state_available_following_year): # Only for 2011-2013 since they have data for next year

        # Define Columns
        columns_ps = ['BENE_ID', 'MSIS_ID', 'STATE_CD'] + ['EL_DAYS_EL_CNT_{}'.format(i) for i in range(1, 13)]
        columns_ps_next3m = ['BENE_ID', 'MSIS_ID', 'STATE_CD'] + ['EL_DAYS_EL_CNT_{}'.format(i) for i in range(1, 4)]

        # Read in data using dask
        ps = dd.read_parquet(f'/mnt/data/medicaid-max/data/{year}/ps/parquet/{state}/', engine='fastparquet', columns=columns_ps)
        ps_next3m = dd.read_parquet(f'/mnt/data/medicaid-max/data/{year+1}/ps/parquet/{state}/', engine='fastparquet', columns=columns_ps_next3m)

        # Rename columns for ps next 3 months (fixed el_dod here also)
        ps_next3m = ps_next3m.rename(columns={'EL_DAYS_EL_CNT_1': 'EL_DAYS_EL_CNT_13', 'EL_DAYS_EL_CNT_2': 'EL_DAYS_EL_CNT_14',
                     'EL_DAYS_EL_CNT_3': 'EL_DAYS_EL_CNT_15', 'EL_DOD': 'EL_DOD_PS_NEXT3M'})

        # Separate into missing bene_id vs not missing bene_id to merge with PS
        amb_missingbeneid = amb_hos_concat[amb_hos_concat['BENE_ID'] == '']
        amb_notmissingbeneid = amb_hos_concat[amb_hos_concat['BENE_ID'] != '']

        # Recover Memory
        del amb_hos_concat

        # Merge current year PS with amb
        amb_missingbeneid_ps = dd.merge(amb_missingbeneid, ps, on=['MSIS_ID', 'STATE_CD'], suffixes=['_AMB', '_PS'],
                                        how='left')
        amb_notmissingbeneid_ps = dd.merge(amb_notmissingbeneid, ps, on=['BENE_ID'], suffixes=['_AMB', '_PS'], how='left')

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

        # Merge PS again but for the next three months
        amb_missingbeneid_ps = dd.merge(amb_missingbeneid_ps, ps_next3m, on=['MSIS_ID', 'STATE_CD'],
                                        suffixes=['_AMB', '_PS'], how='left')
        amb_notmissingbeneid_ps = dd.merge(amb_notmissingbeneid_ps, ps_next3m, on=['BENE_ID'], suffixes=['_AMB', '_PS'],
                                           how='left')

        # Clean DF
        amb_missingbeneid_ps = amb_missingbeneid_ps.drop(['BENE_ID_PS'], axis=1)
        amb_missingbeneid_ps = amb_missingbeneid_ps.rename(columns={'BENE_ID_AMB': 'BENE_ID'})
        amb_notmissingbeneid_ps = amb_notmissingbeneid_ps.drop(['MSIS_ID_PS', 'STATE_CD_PS'], axis=1)
        amb_notmissingbeneid_ps = amb_notmissingbeneid_ps.rename(
            columns={'STATE_CD_AMB': 'STATE_CD', 'MSIS_ID_AMB': 'MSIS_ID'})

        # Recover memory
        del ps_next3m

        # Concat
        amb_hos_concat = dd.concat([amb_missingbeneid_ps, amb_notmissingbeneid_ps], axis=0)

        # Fill in all na's with zero
        for i in range(1, 16):
            amb_hos_concat['EL_DAYS_EL_CNT_{}'.format(i)] = amb_hos_concat['EL_DAYS_EL_CNT_{}'.format(i)].fillna('0')

    else: # only for 2014 since we do not have data for next year (i.e. no 2015)

        # Define Columns
        columns_ps = ['BENE_ID', 'MSIS_ID', 'STATE_CD'] + ['EL_DAYS_EL_CNT_{}'.format(i) for i in range(1, 13)]

        # Read in data using dask
        ps = dd.read_parquet(f'/mnt/data/medicaid-max/data/{year}/ps/parquet/{state}/', engine='fastparquet', columns=columns_ps)

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

        # Add columns for States that do not have data for the following year (i.e. we do not have 2015 data so we cannot create the next 3 months columns for the following year)
        amb_missingbeneid_ps['EL_DAYS_EL_CNT_13'] = '0'
        amb_missingbeneid_ps['EL_DAYS_EL_CNT_14'] = '0'
        amb_missingbeneid_ps['EL_DAYS_EL_CNT_15'] = '0'
        amb_notmissingbeneid_ps['EL_DAYS_EL_CNT_13'] = '0'
        amb_notmissingbeneid_ps['EL_DAYS_EL_CNT_14'] = '0'
        amb_notmissingbeneid_ps['EL_DAYS_EL_CNT_15'] = '0'
        amb_missingbeneid_ps['EL_DOD_PS_NEXT3M'] = pd.NaT
        amb_notmissingbeneid_ps['EL_DOD_PS_NEXT3M'] = pd.NaT

        # Concat
        amb_hos_concat = dd.concat([amb_missingbeneid_ps, amb_notmissingbeneid_ps], axis=0)

    # Fill in all na's with zero
    for i in range(1, 16):
        amb_hos_concat['EL_DAYS_EL_CNT_{}'.format(i)] = amb_hos_concat['EL_DAYS_EL_CNT_{}'.format(i)].fillna('0')

    # -----------------Keep those at least 91 days in Medicaid----------------------#

    # ---------Codes to count number of days in first month---------#

    # Convert all to datetime
    amb_hos_concat['SRVC_END_DT'] = dd.to_datetime(amb_hos_concat['SRVC_END_DT'])

    # Remove any claims that have nan in service begin date
    amb_hos_concat = amb_hos_concat[~amb_hos_concat['SRVC_END_DT'].isna()]

    # Convert columns to floats (the range up to 16 to account for those who had service date oct-dec)
    for i in range(1, 16):
        amb_hos_concat['EL_DAYS_EL_CNT_{}'.format(i)] = amb_hos_concat['EL_DAYS_EL_CNT_{}'.format(i)].astype('float')

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
            (amb_hos_concat['SRVC_END_DT'].dt.month == i), amb_hos_concat['EL_DAYS_EL_CNT_{}'.format(i)])

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
            (amb_hos_concat['SRVC_END_DT'].dt.month == i), amb_hos_concat['EL_DAYS_EL_CNT_{}'.format(i + 1)] + \
            amb_hos_concat['EL_DAYS_EL_CNT_{}'.format(i + 2)] + amb_hos_concat['EL_DAYS_EL_CNT_{}'.format(i + 3)])

    # Convert to float
    amb_hos_concat['days_enrolled_after_three_months'] = amb_hos_concat['days_enrolled_after_three_months'].astype(
        'float')

    # ---Codes to filter individuals with at least 91 days in Medicaid---#

    # Add to see if individuals enrolled at least 91 days
    amb_hos_concat['total_enrolled_after_4_months'] = amb_hos_concat['days_enrolled_after_three_months'] + \
                                                      amb_hos_concat['Days_Until_End_Month']

    # Filter based on if individuals with service date from Jan-Dec have at least 91 days in medicaid or are dead within the 90 days
    amb_hos_concat = amb_hos_concat[
        (amb_hos_concat['total_enrolled_after_4_months'] > 90) | (amb_hos_concat['PATIENT_STATUS_CD'] == '20') | (
                    amb_hos_concat['PATIENT_STATUS_CD'] == '40') |
        (amb_hos_concat['PATIENT_STATUS_CD'] == '41') | (amb_hos_concat['PATIENT_STATUS_CD'] == '42')]

    # Clean DF before exporting
    amb_hos_concat = amb_hos_concat.drop(['EL_DAYS_EL_CNT_{}'.format(i) for i in range(1, 16)] +
                                         ['EndOfMonth', 'Days_Until_End_Month', 'days_enrolled',
                                          'days_enrolled_after_three_months', 'total_enrolled_after_4_months'], axis=1)

    # Read Out
    amb_hos_concat.to_parquet(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicaid_ffs_mc/{year}/{mcaid_payment_type}_merged_amb_hos_claims_ninetyonedays/{state}/',
                              compression='gzip', engine='fastparquet')

#________________________________________Run Defined functions_________________________________________________________#

# Specify years
years=[2011,2012,2013,2014]

# Specify States
states_11=['AL','AK','AZ','AR','CA','CO','CT','DC','DE','FL','GA','HI','IL','IN','IA','KS','KY','LA','ME','MD','MA','MI',
           'MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT',
           'VT','VA','WA','WV','WI','WY']
states_12=['AL','AK','AZ','AR','CA','CO','CT','DC','DE','FL','GA','HI','ID','IL','IN','IA','KS','KY','LA','ME','MD','MA',
           'MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX',
           'UT','VT','VA','WA','WV','WI','WY']
states_13=['AZ','AR','CA','CT','GA','HI','ID','IN','IA','LA','MA','MI','MN','MS','MO','NJ','NY','OH','OK','OR','PA','SD',
           'TN','UT','VT','WA','WV','WY']
states_14=['CA','GA','ID','IA','LA','MI','MN','MS','MO','NJ','PA','SD','TN','UT','VT','WV','WY']

# Specify the states available the following year
state_11_available_in_12=['AL','AK','AZ','AR','CA','CO','CT','DC','DE','FL','GA','HI','IL','IN','IA','KS','KY','LA','ME','MD','MA','MI',
                          'MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT',
                          'VT','VA','WA','WV','WI','WY']
state_12_available_in_13=['AZ','AR','CA','CT','GA','HI','ID','IN','IA','LA','MA','MI','MN','MS','MO','NJ','NY','OH','OK',
                          'OR','PA','SD','TN','UT','VT','WA','WV','WY']
state_13_available_in_14=['CA','GA','ID','IA','LA','MI','MN','MS','MO','NJ','PA','SD','TN','UT','VT','WV','WY']

# Create loop
for y in years:

    # 2011
    if y in [2011]:

            # States available in 2011
            for s in states_11:

                # Run for FFS
                keep_at_least_ninetyone_days(y,s,'ffs',state_11_available_in_12)

                # Run for MC/Encounter
                keep_at_least_ninetyone_days(y,s,'mc',state_11_available_in_12)

    # 2012
    if y in [2012]:

            # States available in 2012
            for s in states_12:

                # Run for FFS
                keep_at_least_ninetyone_days(y,s,'ffs',state_12_available_in_13)

                # Run for MC/Encounter
                keep_at_least_ninetyone_days(y,s,'mc',state_12_available_in_13)

    # 2013
    if y in [2013]:

            # States available in 2013
            for s in states_13:

                # Run for FFS
                keep_at_least_ninetyone_days(y,s,'ffs',state_13_available_in_14)

                # Run for MC/Encounter
                keep_at_least_ninetyone_days(y,s,'mc',state_13_available_in_14)

    # 2014
    if y in [2014]:

            # States available in 2014
            for s in states_14:

                # Run for FFS
                keep_at_least_ninetyone_days(y,s,'ffs',['filler'])

                # Run for MC/Encounter
                keep_at_least_ninetyone_days(y,s,'mc',['filler'])





