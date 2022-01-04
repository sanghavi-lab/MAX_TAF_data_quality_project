#----------------------------------------------------------------------------------------------------------------------#
# Project: Medicaid Data Quality Project
# Authors: Jessy Nguyen and Nadia Ghazali
# Last Updated: August 12, 2021
# Description: This script will export Medicaid TAF's inpatient claims and, from the other-therapy (OT) file, the
#              mileage, outpatient, and ambulance claims for 2016.
#----------------------------------------------------------------------------------------------------------------------#

################################################ IMPORT MODULES ########################################################

# Read in relevant libraries
import pandas as pd
import dask.dataframe as dd
import numpy as np
from datetime import datetime, timedelta

################################################ MODULE FOR CLUSTER ####################################################

# Read in libraries to use cluster
from dask.distributed import Client
client = Client('[insert_ip_address_for_cluster]')

####################################### CREATE TAF DATA FOR MILEAGE INFORMATION ########################################

#________________________________________________ Define Functions ____________________________________________________#

# Define a function to export mileage information from the OT file.
def export_mileage(year,state):

    # Specify relevant line columns
    line_cols = ['BENE_ID','MSIS_ID','STATE_CD','CLM_ID','LINE_SRVC_BGN_DT','LINE_PRCDR_CD','ACTL_SRVC_QTY']

    # Read in OTL file
    ot_line = dd.read_parquet(f'/mnt/data2/medicaid/{year}/taf/TAFOTL/parquet/{state}/', engine='fastparquet', columns=line_cols)

    # Fill nas with empty strings
    ot_line = ot_line.fillna('')

    # Convert every column to string
    ot_line = ot_line.astype(str)

    # Keep only Mileage
    mileage_cd = ['A0425', 'X0034','A0390', 'A0380'] # X0034 is for California
    col_hcpcs = ['LINE_PRCDR_CD']
    mileage = ot_line.loc[ot_line[col_hcpcs].isin(mileage_cd).any(1)]

    # Recover RAM
    del ot_line

    # Read out Data for mileage
    mileage.to_parquet(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicaid_ffs_mc/{year}/mileage/{state}/',compression='gzip', engine='fastparquet')

#____________________________________________ Run Defined Functions ___________________________________________________#

# Specify the year
years = [2016]

# Specify the states available
states_16=['AL','AK','AZ','AR','CA','CO','CT','DC','DE','FL','GA','HI','ID','IL','IN','IA','KS','KY','LA','ME','MD','MA',
           'MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX',
           'UT','VT','VA','WA','WV','WI','WY']

# Create loop for each year (here we are only exporting 2016 for now)
for y in years:

    # Create loop for 2016 available states
    for s in states_16:

        # Use function defined above to export mileage by year and state
        export_mileage(y, s)

####################################### CREATE TAF DATA FOR INPATIENT STAYS ############################################
# The following will construct unique IP stays from the claims provided in the TAF IP file                             #
########################################################################################################################

#________________________________________________ Define Function _____________________________________________________#

def taf_ip_construct_stays(year, state):

    # Specify relevant IP header and line columns
    ip_header_cols = ['BENE_ID','MSIS_ID','STATE_CD','ADMSN_DT','DSCHRG_DT','PTNT_DSCHRG_STUS_CD','BLG_PRVDR_ID',
                      'CLM_TYPE_CD','CROSSOVER_CLM_IND'] + [f'DGNS_CD_{i}' for i in range(1, 13)]

    # Read in IP header file
    ip_head = pd.read_parquet(f'/mnt/data2/medicaid/{year}/taf/TAFIPH/parquet/{state}', engine='fastparquet',columns=ip_header_cols)

    # Replace na's in ID columns with empty strings and then convert to str
    col_to_fillna =  ['BENE_ID','MSIS_ID','STATE_CD','PTNT_DSCHRG_STUS_CD','BLG_PRVDR_ID'] + [f'DGNS_CD_{i}' for i in range(1, 13)]
    ip_head[col_to_fillna] = ip_head[col_to_fillna].fillna('')
    ip_head[col_to_fillna] = ip_head[col_to_fillna].astype(str)

    # Convert date columns to datetime and fill na's to fix inconsistent dtype error
    ip_head['ADMSN_DT'] = ip_head['ADMSN_DT'].fillna(pd.NaT)
    ip_head['ADMSN_DT'] = ip_head['ADMSN_DT'].mask(ip_head['ADMSN_DT']=='',pd.NaT)
    ip_head['ADMSN_DT'] = pd.to_datetime(ip_head['ADMSN_DT'], errors = 'coerce')
    ip_head['DSCHRG_DT'] = ip_head['DSCHRG_DT'].fillna(pd.NaT)
    ip_head['DSCHRG_DT'] = ip_head['DSCHRG_DT'].mask(ip_head['DSCHRG_DT']=='',pd.NaT)
    ip_head['DSCHRG_DT'] = pd.to_datetime(ip_head['DSCHRG_DT'], errors = 'coerce')

    # Create BENE_MSIS column. Contains BENE_ID, or MSIS_ID if BENE_ID is not available. BENE_MSIS_col will be used as a column for matching.
    ip_head['BENE_MSIS'] = ip_head['BENE_ID']
    ip_head['BENE_MSIS'] = ip_head['BENE_MSIS'].mask((ip_head['BENE_MSIS'] == '')|(ip_head['BENE_MSIS'].isna()), ip_head['MSIS_ID'])

    # Sort dataframe by billing BENE_ID, provider ID, admission date, and discharge date (ascending)
    ip_sort = ip_head.sort_values(by=['BENE_MSIS', 'BLG_PRVDR_ID', 'ADMSN_DT', 'DSCHRG_DT'], ascending=[True,True,True,True])

    # Create a STAY_START indicator, where a 1 indicates the first claim in a stay and 0s indicate all other stays.
    ip_sort['STAY_START'] = 1
    ip_sort['STAY_START'] = ip_sort['STAY_START'].mask(((ip_sort['DSCHRG_DT'].shift(1) + timedelta(1)) >= ip_sort['ADMSN_DT'])&
                                                       (ip_sort['BENE_MSIS'].shift(1)==ip_sort['BENE_MSIS'])&
                                                       (ip_sort['BLG_PRVDR_ID'].shift(1)==ip_sort['BLG_PRVDR_ID']), 0)

    # Create a new column STAY_ID, which is a cumulative sum of STAY_START. Since STAY_START only equals 1 at the beginning of a new stay, the sum in STAY_ID will be the same for claims within the same stay. I.e. all claims within a single stay will have the same STAY_ID
    ip_sort['STAY_ID'] = ip_sort['STAY_START'].cumsum()

    # Create column to count number of diagnosis codes in each claim
    diag_col = [f'DGNS_CD_{i}' for i in range(1, 13)]  # Specify 12 diagnosis claims
    ip_sort[diag_col] = ip_sort[diag_col].replace('', np.nan)  # Change empty strings to nans
    ip_sort['num_diag_codes'] = ip_sort[diag_col].count(1)  # Create new column to count diagnosis codes
    ip_sort[diag_col] = ip_sort[diag_col].fillna('')  # Fill nan's with empty strings

    # Create a separate df with the max number of diagnosis codes per stay
    max_dgns = ip_sort.sort_values(by=['num_diag_codes'], ascending=True)
    max_dgns = max_dgns.drop(['ADMSN_DT', 'DSCHRG_DT'],axis=1)  # Drop date columns because we only need diagnosis code info
    max_dgns = max_dgns.drop_duplicates(subset=['STAY_ID'], keep='last')  # Drop duplicates

    # Create a groupby DF with min and max service dates. Need this to obtain the correct start date and end date of unique stays.
    min_max_dates = ip_sort.groupby(by=['BENE_MSIS', 'STAY_ID']).agg({'ADMSN_DT': ['min'], 'DSCHRG_DT': ['max']}).reset_index()
    min_max_dates.columns = ['BENE_MSIS', 'STAY_ID', 'ADMSN_DT_min', 'DSCHRG_DT_max']

    # Merge max_dgns and min_max_dates dataframe to get one row per stay with full stay dates and max diagnosis codes. Merge on STAY_ID
    ip_stay = pd.merge(max_dgns,min_max_dates, how='right',suffixes=['_DGNS','_DATES'], on=['STAY_ID'])

    # Clean DF
    ip_stay = ip_stay.rename(columns={'ADMSN_DT_min': 'ADMSN_DT', 'DSCHRG_DT_max': 'DSCHRG_DT'})
    ip_stay = ip_stay.drop(['BENE_MSIS_DATES','BENE_MSIS_DGNS','DSCHRG_DT','BLG_PRVDR_ID','STAY_START', 'STAY_ID', 'num_diag_codes',],axis=1)

    # Recover RAM by deleting lingering DF's
    del ip_sort
    del min_max_dates
    del max_dgns

    # Count the number of diagnosis codes
    diag_col = [f'DGNS_CD_{i}' for i in range(1, 13)]  # Define diagnosis columns
    ip_stay[diag_col] = ip_stay[diag_col].replace('', np.nan)  # Replace empty strings to count number of diagnosis codes
    ip_stay['num_of_diag_codes'] = ip_stay[diag_col].count(1)  # Count diagnosis codes
    ip_stay[diag_col] = ip_stay[diag_col].fillna('')  # Fill nan's with empty strings

    # Split DF into those with missing bene_id and those with bene_id
    ip_missingid = ip_stay[ip_stay['BENE_ID'] == '']
    ip_notmissingid = ip_stay[ip_stay['BENE_ID'] != '']

    # Sort each partition in ascending order
    ip_missingid = ip_missingid.sort_values(by=['num_of_diag_codes'], ascending=True)
    ip_notmissingid = ip_notmissingid.sort_values(by=['num_of_diag_codes'], ascending=True)

    # Drop duplicated rows by keeping last (i.e. keep the most dx information)
    ip_missingid = ip_missingid.drop_duplicates(subset=['MSIS_ID', 'STATE_CD', 'ADMSN_DT'], keep='last')
    ip_notmissingid = ip_notmissingid.drop_duplicates(subset=['BENE_ID', 'ADMSN_DT'], keep='last')

    # Concat and clean the DFs
    ip = pd.concat([ip_missingid, ip_notmissingid], axis=0)
    ip['BENE_ID'] = ip['BENE_ID'].fillna('')
    ip['MSIS_ID'] = ip['MSIS_ID'].fillna('')
    ip['STATE_CD'] = ip['STATE_CD'].fillna('')

    # Export to parquet
    ip.to_csv(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicaid_ffs_mc/{year}/ip_stay/{state}.csv',index=False,index_label=False)

#___________________________________________ Run Function _____________________________________________________________#

# Specify the years
years = [2016]

# Specify the states available
states_16=['AL','AK','AZ','AR','CA','CO','CT','DC','DE','FL','GA','HI','ID','IL','IN','IA','KS','KY','LA','ME','MD','MA',
           'MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX',
           'UT','VT','VA','WA','WV','WI','WY']

# Create loop for each year (here we are only exporting 2016 for now)
for y in years:

    # Create loop for 2016 available states
    for s in states_16:

        # Use function defined above to export ip by year and state
        taf_ip_construct_stays(y, s)

##################################### CREATE TAF DATA FOR AMBULANCE CLAIMS #############################################
# This script will remove the ambulance claims that are duals, children, and those with multiple rides in one day.     #
# Those who were not in MCAID for at least 90 days will be removed later right before calculating percents for table   #
# 3 because we need those within 90 days for the last data quality indicator.                                          #
########################################################################################################################

#________________________________________________ Define Function _____________________________________________________#

# Define function to export ambulance claims
def export_amb_w_data(year,state):

    # Specify columns
    columns_otl = ['BENE_ID','MSIS_ID','STATE_CD','CLM_ID','LINE_PRCDR_CD','LINE_SRVC_BGN_DT','LINE_SRVC_END_DT'] + [f'LINE_PRCDR_MDFR_CD_{i}' for i in range(1,5)]
    columns_oth = ['CLM_ID','CLM_TYPE_CD','CROSSOVER_CLM_IND']
    columns_ps = ['BENE_ID', 'MSIS_ID', 'STATE_CD', 'BIRTH_DT', 'DEATH_DT','SEX_CD', 'RACE_ETHNCTY_CD'] + [f'DUAL_ELGBL_CD_{m:02}' for m in range(1, 13)] + [
                 f'MDCD_ENRLMT_DAYS_{m:02}' for m in range(1, 13)] +['MC_PLAN_TYPE_CD_01']

    # Read in data using dask
    ot_line = dd.read_parquet(f'/mnt/labshares/sanghavi-lab/Manu/medicaid-data-extraction/data/{year}/TAFOTL/parquet/{state}/', engine='fastparquet', columns=columns_otl)
    ot_head = dd.read_parquet(f'/mnt/labshares/sanghavi-lab/Rhys/medicaid_data_extraction/{year}/TAFOTH/parquet/{state}/', engine='fastparquet', columns=columns_oth) # use this for now since it's not corrupted

    # Fill nas with empty strings and convert to string
    col_to_str_ot = ['BENE_ID','MSIS_ID','STATE_CD','CLM_ID','LINE_PRCDR_CD'] + [f'LINE_PRCDR_MDFR_CD_{i}' for i in range(1,5)]
    ot_line[col_to_str_ot] = ot_line[col_to_str_ot].fillna('')
    ot_line[col_to_str_ot] = ot_line[col_to_str_ot].astype(str)
    ot_head= ot_head.fillna('')
    ot_head = ot_head.astype(str)

    # Keep emergency ambulance rides to hospitals
    col_hcpcs = ['LINE_PRCDR_CD']
    ambulance_cd = ['A0427', 'A0429', 'A0433', 'X0030']
    amb = ot_line.loc[ot_line[col_hcpcs].isin(ambulance_cd).any(1)]

    # Recover RAM
    del ot_line

    # Change column names
    amb = amb.rename(columns={'LINE_SRVC_BGN_DT':'SRVC_BGN_DT','LINE_SRVC_END_DT':'SRVC_END_DT'})

    # Merge ot head with line to obtain claim type code
    amb_clm_type = dd.merge(ot_head,amb,on=['CLM_ID'],how='right')

    # Recover RAM
    del amb
    del ot_head

    # Read in Personal Summary file
    ps = dd.read_parquet(f'/mnt/labshares/sanghavi-lab/Rhys/medicaid_data_extraction/{year}/TAFDEBSE/parquet/{state}/', engine='fastparquet',
                        columns=columns_ps)

    # Count for the month of Jan for the number of 00's in this column. Will use this variable "count" later when removing dually eligible. See bottom notes on why I manually assigned CA a number greater than 0
    if state in ['CA']:
        count = 10 # CA too large to do the sum().compute() command. Instead, I already manually checked there was at least one 00 prior to running this code so I gave an arbitrary count above 10.
    else:
        count = ps['DUAL_ELGBL_CD_01'].str.contains('00').sum().compute() # I will ultimately use this count variable to determine if I can use the 12 DUAL_ELGBL_CD columns to remove duals. If states have at least one 00, then I can use the columns. Some states do not have an 00 so I will need to use another column.

    # Fill na's with empty strings and convert to string
    col_to_str_ps = ['BENE_ID', 'MSIS_ID', 'STATE_CD','SEX_CD', 'RACE_ETHNCTY_CD'] + [f'DUAL_ELGBL_CD_{m:02}' for m in range(1, 13)] + [
                 f'MDCD_ENRLMT_DAYS_{m:02}' for m in range(1, 13)]
    ps[col_to_str_ps] = ps[col_to_str_ps].fillna('')
    ps[col_to_str_ps] = ps[col_to_str_ps].astype(str)

    # Separate into missing bene_id vs not missing bene_id to merge with PS
    amb_missingbeneid = amb_clm_type[amb_clm_type['BENE_ID'] == '']
    amb_notmissingbeneid = amb_clm_type[amb_clm_type['BENE_ID'] != '']

    # Recover memory
    del amb_clm_type

    # Merge current year PS with ambulance
    amb_missingbeneid_ps_final = dd.merge(amb_missingbeneid, ps, on=['MSIS_ID', 'STATE_CD'], suffixes=['_AMB', '_PS'],
                                    how='left')
    amb_notmissingbeneid_ps_final = dd.merge(amb_notmissingbeneid, ps, on=['BENE_ID'], suffixes=['_AMB', '_PS'],
                                       how='left')

    # Recover memory
    del ps
    del amb_missingbeneid
    del amb_notmissingbeneid

    # Clean DF
    amb_missingbeneid_ps_final = amb_missingbeneid_ps_final.drop(['BENE_ID_PS'], axis=1)
    amb_missingbeneid_ps_final = amb_missingbeneid_ps_final.rename(columns={'BENE_ID_AMB': 'BENE_ID'})
    amb_notmissingbeneid_ps_final = amb_notmissingbeneid_ps_final.drop(['MSIS_ID_PS', 'STATE_CD_PS'], axis=1)
    amb_notmissingbeneid_ps_final = amb_notmissingbeneid_ps_final.rename(columns={'STATE_CD_AMB': 'STATE_CD', 'MSIS_ID_AMB': 'MSIS_ID'})

    # Add columns for states that do not have data for the following year (i.e. data for next 3 months). We do not have Medicaid 2017 (next year), so I will add columns in place.
    amb_missingbeneid_ps_final['EL_DAYS_EL_CNT_13'] = '0'
    amb_missingbeneid_ps_final['EL_DAYS_EL_CNT_14'] = '0'
    amb_missingbeneid_ps_final['EL_DAYS_EL_CNT_15'] = '0'
    amb_notmissingbeneid_ps_final['EL_DAYS_EL_CNT_13'] = '0'
    amb_notmissingbeneid_ps_final['EL_DAYS_EL_CNT_14'] = '0'
    amb_notmissingbeneid_ps_final['EL_DAYS_EL_CNT_15'] = '0'
    amb_missingbeneid_ps_final['EL_DOD_PS_NEXT3M'] = pd.NaT
    amb_notmissingbeneid_ps_final['EL_DOD_PS_NEXT3M'] = pd.NaT

    #-----------------Drop Beneficiaries with multiple rides in one day------------------------#

    # Convert all date columns to datetime
    amb_missingbeneid_ps_final['SRVC_BGN_DT']=dd.to_datetime(amb_missingbeneid_ps_final['SRVC_BGN_DT'], errors = 'coerce')
    amb_notmissingbeneid_ps_final['SRVC_BGN_DT']=dd.to_datetime(amb_notmissingbeneid_ps_final['SRVC_BGN_DT'], errors = 'coerce')

    # Create new column to count
    amb_missingbeneid_ps_final['NUM_OF_RIDES'] = 1
    amb_notmissingbeneid_ps_final['NUM_OF_RIDES'] = 1

    # Group by to see if there were multiple trips in one day for one unique claim
    amb_missingbeneid_multirides = amb_missingbeneid_ps_final.groupby(['MSIS_ID','STATE_CD','SRVC_BGN_DT'])['NUM_OF_RIDES'].sum().to_frame().reset_index()
    amb_notmissingbeneid_multirides = amb_notmissingbeneid_ps_final.groupby(['BENE_ID','SRVC_BGN_DT'])['NUM_OF_RIDES'].sum().to_frame().reset_index()

    # Create conditional column where 1 = multiple rides and 0 = only a single ride in a day
    amb_missingbeneid_multirides['MULT_RIDE_IND'] = 0
    amb_notmissingbeneid_multirides['MULT_RIDE_IND'] = 0
    amb_missingbeneid_multirides['MULT_RIDE_IND'] = amb_missingbeneid_multirides['MULT_RIDE_IND'].mask(amb_missingbeneid_multirides['NUM_OF_RIDES'] > 1, 1)
    amb_notmissingbeneid_multirides['MULT_RIDE_IND'] = amb_notmissingbeneid_multirides['MULT_RIDE_IND'].mask(amb_notmissingbeneid_multirides['NUM_OF_RIDES'] > 1, 1)

    # Merge dataset back to original ambulance claims in order to filter out those with multiple rides
    amb_missingbeneid_merge = dd.merge(amb_missingbeneid_ps_final,amb_missingbeneid_multirides, on=['MSIS_ID','STATE_CD','SRVC_BGN_DT'], suffixes=['_original','_multi'] , how = 'left')
    amb_notmissingbeneid_merge = dd.merge(amb_notmissingbeneid_ps_final,amb_notmissingbeneid_multirides, on=['BENE_ID','SRVC_BGN_DT'], suffixes=['_original','_multi'] , how = 'left')

    # Recover memory
    del amb_missingbeneid_ps_final
    del amb_missingbeneid_multirides
    del amb_notmissingbeneid_ps_final
    del amb_notmissingbeneid_multirides

    # Keep those with only 1 ride per day
    amb_missingbeneid_oneride = amb_missingbeneid_merge[amb_missingbeneid_merge['MULT_RIDE_IND']==0]
    amb_notmissingbeneid_oneride = amb_notmissingbeneid_merge[amb_notmissingbeneid_merge['MULT_RIDE_IND']==0]

    # Recover memory
    del amb_missingbeneid_merge
    del amb_notmissingbeneid_merge

    # Clean Data
    amb_missingbeneid_oneride = amb_missingbeneid_oneride.drop(['NUM_OF_RIDES_original','NUM_OF_RIDES_multi','MULT_RIDE_IND'],axis=1)
    amb_notmissingbeneid_oneride = amb_notmissingbeneid_oneride.drop(['NUM_OF_RIDES_original','NUM_OF_RIDES_multi','MULT_RIDE_IND'],axis=1)

    #------------------------------Drop Children 17 and below------------------------------------#

    # Convert all to datetime
    amb_missingbeneid_oneride['SRVC_BGN_DT']=dd.to_datetime(amb_missingbeneid_oneride['SRVC_BGN_DT'], errors = 'coerce')
    amb_missingbeneid_oneride['BIRTH_DT']=dd.to_datetime(amb_missingbeneid_oneride['BIRTH_DT'], errors = 'coerce')
    amb_notmissingbeneid_oneride['SRVC_BGN_DT']=dd.to_datetime(amb_notmissingbeneid_oneride['SRVC_BGN_DT'], errors = 'coerce')
    amb_notmissingbeneid_oneride['BIRTH_DT']=dd.to_datetime(amb_notmissingbeneid_oneride['BIRTH_DT'], errors = 'coerce')

    # Find age in days
    amb_missingbeneid_oneride['age_in_days'] = amb_missingbeneid_oneride['SRVC_BGN_DT'] - amb_missingbeneid_oneride['BIRTH_DT']
    amb_notmissingbeneid_oneride['age_in_days'] = amb_notmissingbeneid_oneride['SRVC_BGN_DT'] - amb_notmissingbeneid_oneride['BIRTH_DT']

    # Convert column to integer
    amb_missingbeneid_oneride['age_in_days'] = amb_missingbeneid_oneride['age_in_days'].dt.days.astype('float')
    amb_notmissingbeneid_oneride['age_in_days'] = amb_notmissingbeneid_oneride['age_in_days'].dt.days.astype('float')

    # Convert age to years
    amb_missingbeneid_oneride['age_in_years'] = amb_missingbeneid_oneride['age_in_days']/365
    amb_notmissingbeneid_oneride['age_in_years'] = amb_notmissingbeneid_oneride['age_in_days']/365

    # Keep beneficiaries at least 18 years of age
    amb_missingbeneid_oneride = amb_missingbeneid_oneride[(amb_missingbeneid_oneride['age_in_years']>=18)]
    amb_notmissingbeneid_oneride = amb_notmissingbeneid_oneride[(amb_notmissingbeneid_oneride['age_in_years']>=18)]

    # Clean Data
    amb_missingbeneid_oneride = amb_missingbeneid_oneride.drop(['age_in_days'],axis=1)
    amb_notmissingbeneid_oneride = amb_notmissingbeneid_oneride.drop(['age_in_days'],axis=1)

    #--------------------------------Keep nonduals--------------------------------#
    # I am going to use both dual eligibility code and the crossover code since   #
    # some states do not have 00 in DUAL_ELGBL_CD columns                         #
    ###############################################################################

    # Convert all dates to YYYY-MM-DD format
    amb_missingbeneid_oneride['SRVC_BGN_DT']=dd.to_datetime(amb_missingbeneid_oneride['SRVC_BGN_DT'], errors = 'coerce')
    amb_notmissingbeneid_oneride['SRVC_BGN_DT']=dd.to_datetime(amb_notmissingbeneid_oneride['SRVC_BGN_DT'], errors = 'coerce')

    # Keep non duals
    if (count>0): # If the number of rows containing 00 in January is at least 1, then use the DUAL_ELGBL_CD columns. Otherwise, we will need to use the CROSSOVER_CLM_IND columns (see below)
        amb_ps_missingbeneid_oneride_nd = amb_missingbeneid_oneride[(amb_missingbeneid_oneride['SRVC_BGN_DT'].dt.month==1) & (amb_missingbeneid_oneride['DUAL_ELGBL_CD_01']=='00') |
                                                                   (amb_missingbeneid_oneride['SRVC_BGN_DT'].dt.month==2) & (amb_missingbeneid_oneride['DUAL_ELGBL_CD_02']=='00') |
                                                                   (amb_missingbeneid_oneride['SRVC_BGN_DT'].dt.month==3) & (amb_missingbeneid_oneride['DUAL_ELGBL_CD_03']=='00') |
                                                                   (amb_missingbeneid_oneride['SRVC_BGN_DT'].dt.month==4) & (amb_missingbeneid_oneride['DUAL_ELGBL_CD_04']=='00') |
                                                                   (amb_missingbeneid_oneride['SRVC_BGN_DT'].dt.month==5) & (amb_missingbeneid_oneride['DUAL_ELGBL_CD_05']=='00') |
                                                                   (amb_missingbeneid_oneride['SRVC_BGN_DT'].dt.month==6) & (amb_missingbeneid_oneride['DUAL_ELGBL_CD_06']=='00') |
                                                                   (amb_missingbeneid_oneride['SRVC_BGN_DT'].dt.month==7) & (amb_missingbeneid_oneride['DUAL_ELGBL_CD_07']=='00') |
                                                                   (amb_missingbeneid_oneride['SRVC_BGN_DT'].dt.month==8) & (amb_missingbeneid_oneride['DUAL_ELGBL_CD_08']=='00') |
                                                                   (amb_missingbeneid_oneride['SRVC_BGN_DT'].dt.month==9) & (amb_missingbeneid_oneride['DUAL_ELGBL_CD_09']=='00') |
                                                                   (amb_missingbeneid_oneride['SRVC_BGN_DT'].dt.month==10) & (amb_missingbeneid_oneride['DUAL_ELGBL_CD_10']=='00') |
                                                                   (amb_missingbeneid_oneride['SRVC_BGN_DT'].dt.month==11) & (amb_missingbeneid_oneride['DUAL_ELGBL_CD_11']=='00') |
                                                                   (amb_missingbeneid_oneride['SRVC_BGN_DT'].dt.month==12) & (amb_missingbeneid_oneride['DUAL_ELGBL_CD_12']=='00')]
        amb_ps_notmissingbeneid_oneride_nd = amb_notmissingbeneid_oneride[(amb_notmissingbeneid_oneride['SRVC_BGN_DT'].dt.month==1) & (amb_notmissingbeneid_oneride['DUAL_ELGBL_CD_01']=='00') |
                                                                         (amb_notmissingbeneid_oneride['SRVC_BGN_DT'].dt.month==2) & (amb_notmissingbeneid_oneride['DUAL_ELGBL_CD_02']=='00') |
                                                                         (amb_notmissingbeneid_oneride['SRVC_BGN_DT'].dt.month==3) & (amb_notmissingbeneid_oneride['DUAL_ELGBL_CD_03']=='00') |
                                                                         (amb_notmissingbeneid_oneride['SRVC_BGN_DT'].dt.month==4) & (amb_notmissingbeneid_oneride['DUAL_ELGBL_CD_04']=='00') |
                                                                         (amb_notmissingbeneid_oneride['SRVC_BGN_DT'].dt.month==5) & (amb_notmissingbeneid_oneride['DUAL_ELGBL_CD_05']=='00') |
                                                                         (amb_notmissingbeneid_oneride['SRVC_BGN_DT'].dt.month==6) & (amb_notmissingbeneid_oneride['DUAL_ELGBL_CD_06']=='00') |
                                                                         (amb_notmissingbeneid_oneride['SRVC_BGN_DT'].dt.month==7) & (amb_notmissingbeneid_oneride['DUAL_ELGBL_CD_07']=='00') |
                                                                         (amb_notmissingbeneid_oneride['SRVC_BGN_DT'].dt.month==8) & (amb_notmissingbeneid_oneride['DUAL_ELGBL_CD_08']=='00') |
                                                                         (amb_notmissingbeneid_oneride['SRVC_BGN_DT'].dt.month==9) & (amb_notmissingbeneid_oneride['DUAL_ELGBL_CD_09']=='00') |
                                                                         (amb_notmissingbeneid_oneride['SRVC_BGN_DT'].dt.month==10) & (amb_notmissingbeneid_oneride['DUAL_ELGBL_CD_10']=='00') |
                                                                         (amb_notmissingbeneid_oneride['SRVC_BGN_DT'].dt.month==11) & (amb_notmissingbeneid_oneride['DUAL_ELGBL_CD_11']=='00') |
                                                                         (amb_notmissingbeneid_oneride['SRVC_BGN_DT'].dt.month==12) & (amb_notmissingbeneid_oneride['DUAL_ELGBL_CD_12']=='00')]
    else: # If DUAL_ELGBL_CD does not contain 00, then we must use the crossover claim indicator instead as recommended by CMS technical guide
        amb_ps_missingbeneid_oneride_nd = amb_missingbeneid_oneride[amb_missingbeneid_oneride['CROSSOVER_CLM_IND'] == '0']
        amb_ps_notmissingbeneid_oneride_nd = amb_notmissingbeneid_oneride[amb_notmissingbeneid_oneride['CROSSOVER_CLM_IND'] == '0']
        print('State where DUAL_ELGBL_CD does not contain 00: ',f'{state}')

    # Recover Memory
    del amb_missingbeneid_oneride
    del amb_notmissingbeneid_oneride

    # Clean DF
    amb_ps_missingbeneid_oneride_nd = amb_ps_missingbeneid_oneride_nd.drop([f'DUAL_ELGBL_CD_{m:02}' for m in range(1, 13)]+['CROSSOVER_CLM_IND'],axis=1)
    amb_ps_notmissingbeneid_oneride_nd = amb_ps_notmissingbeneid_oneride_nd.drop([f'DUAL_ELGBL_CD_{m:02}' for m in range(1, 13)]+['CROSSOVER_CLM_IND'],axis=1)

    # Concat missing bene_id and not missing bene_id
    amb_ps_concat_oneride_nd = dd.concat([amb_ps_missingbeneid_oneride_nd,amb_ps_notmissingbeneid_oneride_nd],axis=0)

    # Recover Memory
    del amb_ps_missingbeneid_oneride_nd
    del amb_ps_notmissingbeneid_oneride_nd

    #-------------------------------Filter FFS vs Encounter-------------------------------------#

    # Specify list (as recommended by CMS TAF Technical guide) (FFS or Capitated only...NO SUPPLEMENTAL PAYMENT indicators were used)
    ffs = ['1','A','U']
    encounter = ['2','3','B','C','V','W']

    # Separate FFS and Encounter
    amb_ps_concat_oneride_nd_ffs = amb_ps_concat_oneride_nd[amb_ps_concat_oneride_nd['CLM_TYPE_CD'].isin(ffs)]
    amb_ps_concat_oneride_nd_e = amb_ps_concat_oneride_nd[amb_ps_concat_oneride_nd['CLM_TYPE_CD'].isin(encounter)]

    # Recover memory
    del amb_ps_concat_oneride_nd

    # Clean DF
    amb_ps_concat_oneride_nd_ffs = amb_ps_concat_oneride_nd_ffs.drop(['CLM_TYPE_CD'],axis=1)
    amb_ps_concat_oneride_nd_e = amb_ps_concat_oneride_nd_e.drop(['CLM_TYPE_CD'],axis=1)

    #-------------------------------Read out Data---------------------------------#

    # Read out data
    amb_ps_concat_oneride_nd_ffs.to_parquet(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicaid_ffs_mc/{year}/amb_ffs/{state}/', compression='gzip', engine='fastparquet')
    amb_ps_concat_oneride_nd_e.to_parquet(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicaid_ffs_mc/{year}/amb_mc/{state}/', compression='gzip', engine='fastparquet')

#____________________________________________ Run Defined Functions ___________________________________________________#

# Specify the year 2016
years = [2016]

# Specify the states available
states_16=['AL','AK','AZ','AR','CA','CO','CT','DC','DE','FL','GA','HI','ID','IL','IN','IA','KS','KY','LA','ME','MD','MA',
           'MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX',
           'UT','VT','VA','WA','WV','WI','WY']

# Create loop for each year (here we are only exporting 2016 for now)
for y in years:

    # Create loop for 2016 available states
    for s in states_16:

        # Use  function defined above to export amb by year and state
        export_amb_w_data(y,s)

#################################### Creating TAF Data for CA Outpatient Claims ########################################
# CA is too large and takes up too much RAM so I will need to export op for CA first                                   #
########################################################################################################################

# Define columns
columns_op = ['BENE_ID','MSIS_ID', 'STATE_CD', 'SRVC_BGN_DT','POS_CD'] + [f'DGNS_CD_{i}' for i in range(1, 3)]

# Read in head file using dask for CA
op = dd.read_parquet(f'/mnt/labshares/sanghavi-lab/Rhys/medicaid_data_extraction/2016/TAFOTH/parquet/CA/', engine='fastparquet',columns=columns_op) # use folder in Rhys for now since data2 is corrupted

# Fill nas with empty strings and convert to str
col_to_str_ot = ['BENE_ID','MSIS_ID', 'STATE_CD','POS_CD'] + [f'DGNS_CD_{i}' for i in range(1, 3)]
op[col_to_str_ot] = op[col_to_str_ot].fillna('')
op[col_to_str_ot] = op[col_to_str_ot].astype(str)

# Keep outpatient
op = op.loc[(op['POS_CD'] == '22') | (op['POS_CD'] == '23'), :]

# Clean df
op = op.drop(['POS_CD'],axis=1)

# Convert date columns to datetime
op['SRVC_BGN_DT'] = dd.to_datetime(op['SRVC_BGN_DT'], errors = 'coerce')

# Count the number of diagnosis codes
diag_col = ['DGNS_CD_{}'.format(i) for i in range(1, 3)]  # Define diagnosis columns
op[diag_col] = op[diag_col].replace('', np.nan)  # Replace empty strings to count number of diagnosis codes
op['num_of_diag_codes'] = op[diag_col].count(1)  # Count diagnosis codes
op[diag_col] = op[diag_col].fillna('')  # Fill nan's with empty strings

# Set index
op = op.set_index('num_of_diag_codes')

# Drop duplicated rows by keeping last using MSIS (i.e. keep the most information)
op = op.drop_duplicates(subset=['MSIS_ID', 'STATE_CD', 'SRVC_BGN_DT'], keep='last')

# Reset index and drop the num_of_diag_codes
op = op.reset_index(drop=True)

# CA is too large. Need to read out then read back in.
op.to_parquet(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicaid_ffs_mc/2016/op/CA_all_op/',
              compression='gzip', engine='fastparquet')

###################################### Creating TAF Data for Outpatient Claims #########################################
# Goal is to create a subset of op claims that matched with amb claims.                                                #                                                                                                      #
########################################################################################################################

#________________________________________________ Define Function _____________________________________________________#

# Define function to keep the op claims that matched with amb claims and export the file
def export_subset_op(year,state):

    # Read in data
    columns_op = ['BENE_ID','MSIS_ID', 'STATE_CD', 'SRVC_BGN_DT','POS_CD','CLM_TYPE_CD','CROSSOVER_CLM_IND'] + [f'DGNS_CD_{i}' for i in range(1, 3)]
    op = dd.read_parquet(f'/mnt/labshares/sanghavi-lab/Rhys/medicaid_data_extraction/{year}/TAFOTH/parquet/{state}/', engine='fastparquet',columns=columns_op) # use folder in Rhys for now since data2 is corrupted

    # Fill nas with empty strings and convert to str
    col_to_str_ot = ['BENE_ID','MSIS_ID', 'STATE_CD','POS_CD'] + [f'DGNS_CD_{i}' for i in range(1, 3)]
    op[col_to_str_ot] = op[col_to_str_ot].fillna('')
    op[col_to_str_ot] = op[col_to_str_ot].astype(str)

    # Keep outpatients
    op = op.loc[(op['POS_CD'] == '22') | (op['POS_CD'] == '23'), :]

    # Clean df
    op = op.drop(['POS_CD'],axis=1)

    # Convert date columns to datetime
    op['SRVC_BGN_DT'] = dd.to_datetime(op['SRVC_BGN_DT'], errors = 'coerce')

    # Count the number of diagnosis codes
    diag_col = ['DGNS_CD_{}'.format(i) for i in range(1, 3)]  # Define diagnosis columns
    op[diag_col] = op[diag_col].replace('', np.nan)  # Replace empty strings to count number of diagnosis codes
    op['num_of_diag_codes'] = op[diag_col].count(1)  # Count diagnosis codes
    op[diag_col] = op[diag_col].fillna('')  # Fill nan's with empty strings

    # Split into missing and not missing bene_id for op
    op_missingid = op[op['BENE_ID']=='']
    op_notmissingid = op[op['BENE_ID']!='']

    # Delete df to recover memory
    del op

    # Sort num of diag codes in order by setting index first
    op_missingid = op_missingid.set_index('num_of_diag_codes')
    op_notmissingid = op_notmissingid.set_index('num_of_diag_codes')

    # Drop duplicated rows by keeping last (i.e. keep the most information)
    op_missingid = op_missingid.drop_duplicates(subset=['MSIS_ID', 'STATE_CD', 'SRVC_BGN_DT'], keep='last')
    op_notmissingid = op_notmissingid.drop_duplicates(subset=['BENE_ID', 'SRVC_BGN_DT'], keep='last')

    # Reset index and drop the num_of_diag_codes
    op_missingid = op_missingid.reset_index(drop=True)
    op_notmissingid = op_notmissingid.reset_index(drop=True)

    # Read in ffs and mc ambulance claims with only bene_id and msis_id/state_cd
    amb_ffs = dd.read_parquet(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicaid_ffs_mc/{year}/amb_ffs/{state}/', engine='fastparquet',columns=['BENE_ID','MSIS_ID','STATE_CD'])
    amb_mc = dd.read_parquet(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicaid_ffs_mc/{year}/amb_mc/{state}/', engine='fastparquet',columns=['BENE_ID','MSIS_ID','STATE_CD'])

    # Concat
    amb = dd.concat([amb_ffs,amb_mc],axis=0)

    # Recover RAM
    del amb_ffs
    del amb_mc

    # Merge amb with op to keep only op with amb rides. Makes the DF smaller to work with.
    op_missingid_merge = dd.merge(op_missingid,amb,on=['MSIS_ID','STATE_CD'],how='inner',suffixes=['_op','_amb'])
    op_notmissingid_merge = dd.merge(op_notmissingid,amb,on=['BENE_ID'],how='inner',suffixes=['_op','_amb'])

    # Delete df to recover memory
    del amb
    del op_missingid
    del op_notmissingid

    # Clean df
    op_missingid_merge=op_missingid_merge.drop(['BENE_ID_amb'],axis=1)
    op_missingid_merge=op_missingid_merge.rename(columns={'BENE_ID_op': 'BENE_ID'})
    op_notmissingid_merge=op_notmissingid_merge.drop(['MSIS_ID_amb','STATE_CD_amb'],axis=1)
    op_notmissingid_merge=op_notmissingid_merge.rename(columns={'MSIS_ID_op': 'MSIS_ID','STATE_CD_op':'STATE_CD'})

    # Concat
    op = dd.concat([op_missingid_merge, op_notmissingid_merge], axis=0)

    # Delete df to recover memory
    del op_missingid_merge
    del op_notmissingid_merge

    # Clean DFs
    op['BENE_ID'] = op['BENE_ID'].fillna('')
    op['MSIS_ID'] = op['MSIS_ID'].fillna('')
    op['STATE_CD'] = op['STATE_CD'].fillna('')

    # Read out Data for OP
    op.to_parquet(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicaid_ffs_mc/{year}/op/{state}/',compression='gzip', engine='fastparquet')

#____________________________________________ Run Defined Functions ___________________________________________________#

# Specify the year
years = [2016]

# Specify the states available
states_16=['AL','AK','AZ','AR','CO','CT','DC','DE','FL','GA','HI','ID','IL','IN','IA','KS','KY','LA','ME','MD','MA',
           'MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX',
           'UT','VT','VA','WA','WV','WI','WY'] # CA op has already been exported.

# Create loop for each year (here we are only exporting 2016 for now)
for y in years:

    # Create loop for 2016 available states
    for s in states_16:

        # Use function defined above to export a subset of op by year and state
        export_subset_op(y,s)




