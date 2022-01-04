#----------------------------------------------------------------------------------------------------------------------#
# Project: Medicaid Data Quality Project
# Authors: Jessy Nguyen
# Last Updated: August 12, 2021
# Description: This script will export Medicaid MAX's inpatient claims and, from the other-therapy (OT) file, the
#              mileage, outpatient, and ambulance claims for 2011 to 2014.
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

####################################### CREATE MAX DATA FOR MILEAGE INFORMATION ########################################
# Note that each year has different number of states                                                                   #
########################################################################################################################

#________________________________________________ Define Function _____________________________________________________#

# Define a function to export mileage information from the OT file.
def export_mileage(year,state):

    # Specify relevant columns
    columns_mileage = ['BENE_ID', 'MSIS_ID', 'STATE_CD', 'SRVC_BGN_DT', 'PRCDR_CD_SYS', 'PRCDR_CD','PRCDR_SRVC_MDFR_CD',
                       'QTY_SRVC_UNITS', 'SRVC_END_DT']

    # Read in data using dask
    ot = dd.read_parquet(f'/mnt/data/medicaid-max/data/{year}/ot/parquet/{state}/', engine='fastparquet',columns=columns_mileage)

    # Keep only Mileage
    mileage_cd = ['A0425', 'X0034','A0390', 'A0380'] # X0034 is for California
    col_hcpcs = ['PRCDR_CD']
    mileage = ot.loc[ot[col_hcpcs].isin(mileage_cd).any(1)]

    # Del Df to recover RAM
    del ot

    # Read out Data for mileage
    mileage.to_parquet(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicaid_ffs_mc/{year}/mileage/{state}/',compression='gzip', engine='fastparquet')

#____________________________________________ Run Defined Function ____________________________________________________#

# Specify the years
years = [2011,2012,2013,2014]

# Specify the states available for each year
states_11=['AL','AK','AZ','AR','CA','CO','CT','DC','DE','FL','GA','HI','IL','IN','IA','KS','KY','LA','ME','MD','MA','MI',
           'MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT',
           'VT','VA','WA','WV','WI','WY']
states_12=['AL','AK','AZ','AR','CA','CO','CT','DC','DE','FL','GA','HI','ID','IL','IN','IA','KS','KY','LA','ME','MD','MA',
           'MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX',
           'UT','VT','VA','WA','WV','WI','WY']
states_13=['AZ','AR','CA','CT','GA','HI','ID','IN','IA','LA','MA','MI','MN','MS','MO','NJ','NY','OH','OK','OR','PA','SD',
           'TN','UT','VT','WA','WV','WY']
states_14=['CA','GA','ID','IA','LA','MI','MN','MS','MO','NJ','PA','SD','TN','UT','VT','WV','WY']

# Create loop for each year
for y in years:

    # Use if/then since each year does not contain the same number of states
    if y in [2011]:

        # Create loop for 2011 available states
        for s in states_11:

            # Use function defined above to export mileage by year and state
            export_mileage(y, s)

    elif y in [2012]:

        # Create loop for 2012 available states
        for s in states_12:

            # Use function defined above to export mileage by year and state
            export_mileage(y, s)

    elif y in [2013]:

        # Create loop for 2013 available states
        for s in states_13:

            # Use function defined above to export mileage by year and state
            export_mileage(y, s)

    elif y in [2014]:

        # Create loop for 2014 available states
        for s in states_14:

            # Use function defined above to export mileage by year and state
            export_mileage(y, s)

####################################### CREATE MAX DATA FOR INPATIENT CLAIMS ###########################################
# Note that each year has different number of states                                                                   #
########################################################################################################################

#________________________________________________ Define Function _____________________________________________________#

# Define a function to export ip information from the IP file.
def export_ip(year,state):

    # Specify relevant columns
    columns_ip=['BENE_ID','MSIS_ID','STATE_CD','SRVC_BGN_DT','PATIENT_STATUS_CD'] +[f'DIAG_CD_{i}' for i in range(1,10)]

    # Read in data using dask
    ip = dd.read_parquet(f'/mnt/data/medicaid-max/data/{year}/ip/parquet/{state}/', engine='fastparquet', columns=columns_ip)

    # Count the number of diagnosis codes for each row
    diag_col = ['DIAG_CD_{}'.format(i) for i in range(1, 10)]  # Define diagnosis columns
    ip[diag_col] = ip[diag_col].replace('', np.nan)  # Replace empty strings to count number of diagnosis codes
    ip['num_of_diag_codes'] = ip[diag_col].count(1)  # Count diagnosis codes
    ip[diag_col] = ip[diag_col].fillna('')  # Fill nan's with empty strings

    # Split DF into those with missing bene_id and those with bene_id
    ip_missingid = ip[ip['BENE_ID'] == '']
    ip_notmissingid = ip[ip['BENE_ID'] != '']

    # Sort each partition in ascending order
    ip_missingid = ip_missingid.map_partitions(lambda x: x.sort_values(by=['num_of_diag_codes'], ascending=True))
    ip_notmissingid = ip_notmissingid.map_partitions(lambda x: x.sort_values(by=['num_of_diag_codes'], ascending=True))

    # Drop duplicated rows by keeping last (i.e. keep the most information)
    ip_missingid = ip_missingid.drop_duplicates(subset=['MSIS_ID', 'STATE_CD', 'SRVC_BGN_DT'], keep='last')
    ip_notmissingid = ip_notmissingid.drop_duplicates(subset=['BENE_ID', 'SRVC_BGN_DT'], keep='last')

    # Concat and clean the DFs
    ip = dd.concat([ip_missingid, ip_notmissingid], axis=0)
    ip['BENE_ID'] = ip['BENE_ID'].fillna('')
    ip['MSIS_ID'] = ip['MSIS_ID'].fillna('')
    ip['STATE_CD'] = ip['STATE_CD'].fillna('')

    # Read out Data for IP
    ip.to_parquet(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicaid_ffs_mc/{year}/ip/{state}/', compression='gzip', engine='fastparquet')

#____________________________________________ Run Defined Function ____________________________________________________#

# Specify the years
years = [2011,2012,2013,2014]

# Specify the states available for each year
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

    # Use if/then since each year does not contain the same number of states
    if y in [2011]:

        # Create loop for 2011 available states
        for s in states_11:

            # Use function defined above to export ip by year and state
            export_ip(y,s)

    elif y in [2012]:

        # Create loop for 2012 available states
        for s in states_12:

            # Use function defined above to export ip by year and state
            export_ip(y,s)

    elif y in [2013]:

        # Create loop for 2013 available states
        for s in states_13:

            # Use function defined above to export ip by year and state
            export_ip(y,s)

    elif y in [2014]:

        # Create loop for 2014 available states
        for s in states_14:

            # Use function defined above to export ip by year and state
            export_ip(y,s)

##################################### CREATE MAX DATA FOR AMBULANCE CLAIMS #############################################
# This script will remove the ambulance claims that are duals, children, and those with multiple rides in one day.     #
# Note that each year has different number of states. Those who were not in mcaid for at least 90 days will be removed #
# later right before calculating percents for table 2 because we need those within 90 days for last data quality       #
# indicator.                                                                                                           #
########################################################################################################################

#________________________________________________ Define Function _____________________________________________________#

# Define first function to export ambulance claims
def export_amb_w_data(year,state,list_state_available_following_year):

    if (year in [2011,2012,2013]) & (state in list_state_available_following_year): # We need a list of states if they are available in the following year since we do not have all states for 2011, 2013, and 2014.

        # Specify columns
        columns_ot = ['BENE_ID','MSIS_ID','STATE_CD','SRVC_BGN_DT','SRVC_END_DT','PRCDR_CD','PRCDR_SRVC_MDFR_CD','MAX_TOS','PRCDR_CD_SYS']+['DIAG_CD_{}'.format(i) for i in range(1,3)]
        columns_ps = ['BENE_ID','MSIS_ID','STATE_CD','EL_DOB','EL_DOD','EL_AGE_GRP_CD','EL_RSDNC_CNTY_CD_LTST','EL_RSDNC_ZIP_CD_LTST',
                      'EL_SEX_CD','EL_RACE_ETHNCY_CD']+['EL_MDCR_DUAL_MO_{}'.format(i) for i in range(1,13)] + ['EL_PHP_TYPE_1_{}'.format(i) for i in range(1,13)] + \
                     ['EL_DAYS_EL_CNT_{}'.format(i) for i in range(1,13)]
        columns_ps_next3m = ['BENE_ID','MSIS_ID','STATE_CD','EL_DOD'] + ['EL_DAYS_EL_CNT_{}'.format(i) for i in range(1,4)]

        # Read in data using dask
        ot = dd.read_parquet(f'/mnt/data/medicaid-max/data/{year}/ot/parquet/{state}/', engine='fastparquet', columns=columns_ot)
        ps = dd.read_parquet(f'/mnt/data/medicaid-max/data/{year}/ps/parquet/{state}/', engine='fastparquet', columns=columns_ps)
        ps_next3m = dd.read_parquet(f'/mnt/data/medicaid-max/data/{year+1}/ps/parquet/{state}/', engine='fastparquet', columns=columns_ps_next3m)

        # Rename columns for ps following 3 months (next year)
        ps_next3m = ps_next3m.rename(columns={'EL_DAYS_EL_CNT_1':'EL_DAYS_EL_CNT_13','EL_DAYS_EL_CNT_2':'EL_DAYS_EL_CNT_14',
                                                  'EL_DAYS_EL_CNT_3':'EL_DAYS_EL_CNT_15','EL_DOD':'EL_DOD_PS_NEXT3M'})

        # Filter DF for emergency ambulance rides
        col_hcpcs = ['PRCDR_CD']
        ambulance_cd = ['A0427', 'A0429', 'A0433', 'X0030']
        amb = ot.loc[ot[col_hcpcs].isin(ambulance_cd).any(1)]

        # Del DF to recover memory
        del ot

        # Separate into missing bene_id vs not missing bene_id to merge with PS
        amb_missingbeneid = amb[amb['BENE_ID']=='']
        amb_notmissingbeneid = amb[amb['BENE_ID']!='']

        # Recover memory
        del amb

        # Merge current year PS with amb
        amb_missingbeneid_ps = dd.merge(amb_missingbeneid,ps,on=['MSIS_ID','STATE_CD'],suffixes=['_AMB','_PS'],how='inner')
        amb_notmissingbeneid_ps = dd.merge(amb_notmissingbeneid,ps,on=['BENE_ID'],suffixes=['_AMB','_PS'],how='inner')

        # Recover memory
        del ps
        del amb_missingbeneid
        del amb_notmissingbeneid

        # Clean DF
        amb_missingbeneid_ps = amb_missingbeneid_ps.drop(['BENE_ID_PS'],axis=1)
        amb_missingbeneid_ps = amb_missingbeneid_ps.rename(columns={'BENE_ID_AMB':'BENE_ID'})
        amb_notmissingbeneid_ps = amb_notmissingbeneid_ps.drop(['MSIS_ID_PS','STATE_CD_PS'],axis=1)
        amb_notmissingbeneid_ps = amb_notmissingbeneid_ps.rename(columns={'STATE_CD_AMB':'STATE_CD','MSIS_ID_AMB':'MSIS_ID'})

        # Merge PS again but with the next three months
        amb_missingbeneid_ps_final = dd.merge(amb_missingbeneid_ps,ps_next3m,on=['MSIS_ID','STATE_CD'],suffixes=['_AMB','_PS'],how='left')
        amb_notmissingbeneid_ps_final = dd.merge(amb_notmissingbeneid_ps,ps_next3m,on=['BENE_ID'],suffixes=['_AMB','_PS'],how='left')

        # Recover memory
        del amb_missingbeneid_ps
        del amb_notmissingbeneid_ps
        del ps_next3m

        # Clean DF by dropping columns
        amb_missingbeneid_ps_final = amb_missingbeneid_ps_final.drop(['BENE_ID_PS'],axis=1)
        amb_notmissingbeneid_ps_final = amb_notmissingbeneid_ps_final.drop(['MSIS_ID_PS','STATE_CD_PS'],axis=1)

        # Rename columns
        amb_missingbeneid_ps_final = amb_missingbeneid_ps_final.rename(columns={'BENE_ID_AMB':'BENE_ID'})
        amb_notmissingbeneid_ps_final = amb_notmissingbeneid_ps_final.rename(columns={'STATE_CD_AMB':'STATE_CD','MSIS_ID_AMB':'MSIS_ID'})

    else:

        # Specify columns
        columns_ot = ['BENE_ID', 'MSIS_ID', 'STATE_CD', 'SRVC_BGN_DT', 'SRVC_END_DT', 'PRCDR_CD', 'PRCDR_SRVC_MDFR_CD',
                      'MAX_TOS', 'PRCDR_CD_SYS'] + ['DIAG_CD_{}'.format(i) for i in range(1, 3)]
        columns_ps = ['BENE_ID', 'MSIS_ID', 'STATE_CD', 'EL_DOB', 'EL_DOD', 'EL_AGE_GRP_CD', 'EL_RSDNC_CNTY_CD_LTST',
                      'EL_RSDNC_ZIP_CD_LTST', 'EL_SEX_CD', 'EL_RACE_ETHNCY_CD'] + ['EL_MDCR_DUAL_MO_{}'.format(i) for i in range(1, 13)] + [
                      'EL_PHP_TYPE_1_{}'.format(i) for i in range(1, 13)] + ['EL_DAYS_EL_CNT_{}'.format(i) for i in range(1, 13)]

        # Read in data using dask
        ot = dd.read_parquet(f'/mnt/data/medicaid-max/data/{year}/ot/parquet/{state}/', engine='fastparquet', columns=columns_ot)
        ps = dd.read_parquet(f'/mnt/data/medicaid-max/data/{year}/ps/parquet/{state}/', engine='fastparquet', columns=columns_ps)

        # Keep emergency ambulance rides
        col_hcpcs = ['PRCDR_CD']
        ambulance_cd = ['A0427', 'A0429', 'A0433', 'X0030']
        amb = ot.loc[ot[col_hcpcs].isin(ambulance_cd).any(1)]

        # Del Df to recover memory
        del ot

        # Separate into missing bene_id vs not missing bene_id to merge with PS
        amb_missingbeneid = amb[amb['BENE_ID'] == '']
        amb_notmissingbeneid = amb[amb['BENE_ID'] != '']

        # Recover memory
        del amb

        # Merge current year PS with amb
        amb_missingbeneid_ps_final = dd.merge(amb_missingbeneid, ps, on=['MSIS_ID', 'STATE_CD'], suffixes=['_AMB', '_PS'],how='left')
        amb_notmissingbeneid_ps_final = dd.merge(amb_notmissingbeneid, ps, on=['BENE_ID'], suffixes=['_AMB', '_PS'],how='left')

        # Recover memory
        del ps
        del amb_missingbeneid
        del amb_notmissingbeneid

        # Clean DF
        amb_missingbeneid_ps_final = amb_missingbeneid_ps_final.drop(['BENE_ID_PS'], axis=1)
        amb_missingbeneid_ps_final = amb_missingbeneid_ps_final.rename(columns={'BENE_ID_AMB': 'BENE_ID'})
        amb_notmissingbeneid_ps_final = amb_notmissingbeneid_ps_final.drop(['MSIS_ID_PS', 'STATE_CD_PS'], axis=1)
        amb_notmissingbeneid_ps_final = amb_notmissingbeneid_ps_final.rename(
            columns={'STATE_CD_AMB': 'STATE_CD', 'MSIS_ID_AMB': 'MSIS_ID'})

        # Add columns for States that do not have data for the following year (i.e. 2014 does not have data for next 3 months because we don't have 2015 Medicaid data)
        amb_missingbeneid_ps_final['EL_DAYS_EL_CNT_13'] = '0'
        amb_missingbeneid_ps_final['EL_DAYS_EL_CNT_14'] = '0'
        amb_missingbeneid_ps_final['EL_DAYS_EL_CNT_15'] = '0'
        amb_notmissingbeneid_ps_final['EL_DAYS_EL_CNT_13'] = '0'
        amb_notmissingbeneid_ps_final['EL_DAYS_EL_CNT_14'] = '0'
        amb_notmissingbeneid_ps_final['EL_DAYS_EL_CNT_15'] = '0'
        amb_missingbeneid_ps_final['EL_DOD_PS_NEXT3M'] = pd.NaT
        amb_notmissingbeneid_ps_final['EL_DOD_PS_NEXT3M'] = pd.NaT

    #-----------------Drop Beneficiaries with multiple rides in one day------------------------#

    # Convert all to datetime
    amb_missingbeneid_ps_final['SRVC_BGN_DT']=dd.to_datetime(amb_missingbeneid_ps_final['SRVC_BGN_DT'])
    amb_notmissingbeneid_ps_final['SRVC_BGN_DT']=dd.to_datetime(amb_notmissingbeneid_ps_final['SRVC_BGN_DT'])

    # Create new column to count number of rides in one day
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

    # Filter those with only 1 ride per day (i.e. keep if indicator is zero)
    amb_missingbeneid_oneride = amb_missingbeneid_merge[amb_missingbeneid_merge['MULT_RIDE_IND']==0]
    amb_notmissingbeneid_oneride = amb_notmissingbeneid_merge[amb_notmissingbeneid_merge['MULT_RIDE_IND']==0]

    # Recover memory
    del amb_missingbeneid_merge
    del amb_notmissingbeneid_merge

    # Clean Data
    amb_missingbeneid_oneride = amb_missingbeneid_oneride.drop(['NUM_OF_RIDES_original','NUM_OF_RIDES_multi','MULT_RIDE_IND'],axis=1)
    amb_notmissingbeneid_oneride = amb_notmissingbeneid_oneride.drop(['NUM_OF_RIDES_original','NUM_OF_RIDES_multi','MULT_RIDE_IND'],axis=1)

    #------------------------------Drop Children 17 and below------------------------------------#

    # Convert all date columns to datetime
    amb_missingbeneid_oneride['SRVC_BGN_DT']=dd.to_datetime(amb_missingbeneid_oneride['SRVC_BGN_DT'])
    amb_missingbeneid_oneride['EL_DOB']=dd.to_datetime(amb_missingbeneid_oneride['EL_DOB'])
    amb_notmissingbeneid_oneride['SRVC_BGN_DT']=dd.to_datetime(amb_notmissingbeneid_oneride['SRVC_BGN_DT'])
    amb_notmissingbeneid_oneride['EL_DOB']=dd.to_datetime(amb_notmissingbeneid_oneride['EL_DOB'])

    # Find age in days
    amb_missingbeneid_oneride['age_in_days'] = amb_missingbeneid_oneride['SRVC_BGN_DT'] - amb_missingbeneid_oneride['EL_DOB']
    amb_notmissingbeneid_oneride['age_in_days'] = amb_notmissingbeneid_oneride['SRVC_BGN_DT'] - amb_notmissingbeneid_oneride['EL_DOB']

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

    # Convert all dates to YYYY-MM-DD format
    amb_missingbeneid_oneride['SRVC_BGN_DT']=dd.to_datetime(amb_missingbeneid_oneride['SRVC_BGN_DT'])
    amb_notmissingbeneid_oneride['SRVC_BGN_DT']=dd.to_datetime(amb_notmissingbeneid_oneride['SRVC_BGN_DT'])

    # Keep non duals
    amb_ps_missingbeneid_oneride_nd = amb_missingbeneid_oneride[(amb_missingbeneid_oneride['SRVC_BGN_DT'].dt.month==1) & (amb_missingbeneid_oneride['EL_MDCR_DUAL_MO_1']=='00') |
                                                               (amb_missingbeneid_oneride['SRVC_BGN_DT'].dt.month==2) & (amb_missingbeneid_oneride['EL_MDCR_DUAL_MO_2']=='00') |
                                                               (amb_missingbeneid_oneride['SRVC_BGN_DT'].dt.month==3) & (amb_missingbeneid_oneride['EL_MDCR_DUAL_MO_3']=='00') |
                                                               (amb_missingbeneid_oneride['SRVC_BGN_DT'].dt.month==4) & (amb_missingbeneid_oneride['EL_MDCR_DUAL_MO_4']=='00') |
                                                               (amb_missingbeneid_oneride['SRVC_BGN_DT'].dt.month==5) & (amb_missingbeneid_oneride['EL_MDCR_DUAL_MO_5']=='00') |
                                                               (amb_missingbeneid_oneride['SRVC_BGN_DT'].dt.month==6) & (amb_missingbeneid_oneride['EL_MDCR_DUAL_MO_6']=='00') |
                                                               (amb_missingbeneid_oneride['SRVC_BGN_DT'].dt.month==7) & (amb_missingbeneid_oneride['EL_MDCR_DUAL_MO_7']=='00') |
                                                               (amb_missingbeneid_oneride['SRVC_BGN_DT'].dt.month==8) & (amb_missingbeneid_oneride['EL_MDCR_DUAL_MO_8']=='00') |
                                                               (amb_missingbeneid_oneride['SRVC_BGN_DT'].dt.month==9) & (amb_missingbeneid_oneride['EL_MDCR_DUAL_MO_9']=='00') |
                                                               (amb_missingbeneid_oneride['SRVC_BGN_DT'].dt.month==10) & (amb_missingbeneid_oneride['EL_MDCR_DUAL_MO_10']=='00') |
                                                               (amb_missingbeneid_oneride['SRVC_BGN_DT'].dt.month==11) & (amb_missingbeneid_oneride['EL_MDCR_DUAL_MO_11']=='00') |
                                                               (amb_missingbeneid_oneride['SRVC_BGN_DT'].dt.month==12) & (amb_missingbeneid_oneride['EL_MDCR_DUAL_MO_12']=='00')]
    amb_ps_notmissingbeneid_oneride_nd = amb_notmissingbeneid_oneride[(amb_notmissingbeneid_oneride['SRVC_BGN_DT'].dt.month==1) & (amb_notmissingbeneid_oneride['EL_MDCR_DUAL_MO_1']=='00') |
                                                                     (amb_notmissingbeneid_oneride['SRVC_BGN_DT'].dt.month==2) & (amb_notmissingbeneid_oneride['EL_MDCR_DUAL_MO_2']=='00') |
                                                                     (amb_notmissingbeneid_oneride['SRVC_BGN_DT'].dt.month==3) & (amb_notmissingbeneid_oneride['EL_MDCR_DUAL_MO_3']=='00') |
                                                                     (amb_notmissingbeneid_oneride['SRVC_BGN_DT'].dt.month==4) & (amb_notmissingbeneid_oneride['EL_MDCR_DUAL_MO_4']=='00') |
                                                                     (amb_notmissingbeneid_oneride['SRVC_BGN_DT'].dt.month==5) & (amb_notmissingbeneid_oneride['EL_MDCR_DUAL_MO_5']=='00') |
                                                                     (amb_notmissingbeneid_oneride['SRVC_BGN_DT'].dt.month==6) & (amb_notmissingbeneid_oneride['EL_MDCR_DUAL_MO_6']=='00') |
                                                                     (amb_notmissingbeneid_oneride['SRVC_BGN_DT'].dt.month==7) & (amb_notmissingbeneid_oneride['EL_MDCR_DUAL_MO_7']=='00') |
                                                                     (amb_notmissingbeneid_oneride['SRVC_BGN_DT'].dt.month==8) & (amb_notmissingbeneid_oneride['EL_MDCR_DUAL_MO_8']=='00') |
                                                                     (amb_notmissingbeneid_oneride['SRVC_BGN_DT'].dt.month==9) & (amb_notmissingbeneid_oneride['EL_MDCR_DUAL_MO_9']=='00') |
                                                                     (amb_notmissingbeneid_oneride['SRVC_BGN_DT'].dt.month==10) & (amb_notmissingbeneid_oneride['EL_MDCR_DUAL_MO_10']=='00') |
                                                                     (amb_notmissingbeneid_oneride['SRVC_BGN_DT'].dt.month==11) & (amb_notmissingbeneid_oneride['EL_MDCR_DUAL_MO_11']=='00') |
                                                                     (amb_notmissingbeneid_oneride['SRVC_BGN_DT'].dt.month==12) & (amb_notmissingbeneid_oneride['EL_MDCR_DUAL_MO_12']=='00')]

    # Recover Memory
    del amb_missingbeneid_oneride
    del amb_notmissingbeneid_oneride

    # Clean DF
    amb_ps_missingbeneid_oneride_nd = amb_ps_missingbeneid_oneride_nd.drop(['EL_MDCR_DUAL_MO_{}'.format(i) for i in range(1,13)],axis=1)
    amb_ps_notmissingbeneid_oneride_nd = amb_ps_notmissingbeneid_oneride_nd.drop(['EL_MDCR_DUAL_MO_{}'.format(i) for i in range(1,13)],axis=1)

    # Concat missing bene_id and not missing bene_id
    amb_ps_concat_oneride_nd = dd.concat([amb_ps_missingbeneid_oneride_nd,amb_ps_notmissingbeneid_oneride_nd],axis=0)

    # Recover Memory
    del amb_ps_missingbeneid_oneride_nd
    del amb_ps_notmissingbeneid_oneride_nd

    #-------------------------------Filter FFS vs Encounter-------------------------------------#

    # Specify list
    ffs = ['02','2','03','3','07','7', '88']
    e = ['01','1','04','4','05','5','06','6','08','8']

    # Convert all dates to YYYY-MM-DD format
    amb_ps_concat_oneride_nd['SRVC_BGN_DT']=dd.to_datetime(amb_ps_concat_oneride_nd['SRVC_BGN_DT'])

    # Separate FFS and Encounter data
    amb_ps_concat_oneride_nd_ffs = amb_ps_concat_oneride_nd[(amb_ps_concat_oneride_nd['SRVC_BGN_DT'].dt.month==1) & (amb_ps_concat_oneride_nd['EL_PHP_TYPE_1_1'].isin(ffs)) |
                                                          (amb_ps_concat_oneride_nd['SRVC_BGN_DT'].dt.month==2) & (amb_ps_concat_oneride_nd['EL_PHP_TYPE_1_2'].isin(ffs)) |
                                                          (amb_ps_concat_oneride_nd['SRVC_BGN_DT'].dt.month==3) & (amb_ps_concat_oneride_nd['EL_PHP_TYPE_1_3'].isin(ffs)) |
                                                          (amb_ps_concat_oneride_nd['SRVC_BGN_DT'].dt.month==4) & (amb_ps_concat_oneride_nd['EL_PHP_TYPE_1_4'].isin(ffs)) |
                                                          (amb_ps_concat_oneride_nd['SRVC_BGN_DT'].dt.month==5) & (amb_ps_concat_oneride_nd['EL_PHP_TYPE_1_5'].isin(ffs)) |
                                                          (amb_ps_concat_oneride_nd['SRVC_BGN_DT'].dt.month==6) & (amb_ps_concat_oneride_nd['EL_PHP_TYPE_1_6'].isin(ffs)) |
                                                          (amb_ps_concat_oneride_nd['SRVC_BGN_DT'].dt.month==7) & (amb_ps_concat_oneride_nd['EL_PHP_TYPE_1_7'].isin(ffs)) |
                                                          (amb_ps_concat_oneride_nd['SRVC_BGN_DT'].dt.month==8) & (amb_ps_concat_oneride_nd['EL_PHP_TYPE_1_8'].isin(ffs)) |
                                                          (amb_ps_concat_oneride_nd['SRVC_BGN_DT'].dt.month==9) & (amb_ps_concat_oneride_nd['EL_PHP_TYPE_1_9'].isin(ffs)) |
                                                          (amb_ps_concat_oneride_nd['SRVC_BGN_DT'].dt.month==10) & (amb_ps_concat_oneride_nd['EL_PHP_TYPE_1_10'].isin(ffs)) |
                                                          (amb_ps_concat_oneride_nd['SRVC_BGN_DT'].dt.month==11) & (amb_ps_concat_oneride_nd['EL_PHP_TYPE_1_11'].isin(ffs)) |
                                                          (amb_ps_concat_oneride_nd['SRVC_BGN_DT'].dt.month==12) & (amb_ps_concat_oneride_nd['EL_PHP_TYPE_1_12'].isin(ffs))]
    amb_ps_concat_oneride_nd_e = amb_ps_concat_oneride_nd[(amb_ps_concat_oneride_nd['SRVC_BGN_DT'].dt.month==1) & (amb_ps_concat_oneride_nd['EL_PHP_TYPE_1_1'].isin(e)) |
                                                          (amb_ps_concat_oneride_nd['SRVC_BGN_DT'].dt.month==2) & (amb_ps_concat_oneride_nd['EL_PHP_TYPE_1_2'].isin(e)) |
                                                          (amb_ps_concat_oneride_nd['SRVC_BGN_DT'].dt.month==3) & (amb_ps_concat_oneride_nd['EL_PHP_TYPE_1_3'].isin(e)) |
                                                          (amb_ps_concat_oneride_nd['SRVC_BGN_DT'].dt.month==4) & (amb_ps_concat_oneride_nd['EL_PHP_TYPE_1_4'].isin(e)) |
                                                          (amb_ps_concat_oneride_nd['SRVC_BGN_DT'].dt.month==5) & (amb_ps_concat_oneride_nd['EL_PHP_TYPE_1_5'].isin(e)) |
                                                          (amb_ps_concat_oneride_nd['SRVC_BGN_DT'].dt.month==6) & (amb_ps_concat_oneride_nd['EL_PHP_TYPE_1_6'].isin(e)) |
                                                          (amb_ps_concat_oneride_nd['SRVC_BGN_DT'].dt.month==7) & (amb_ps_concat_oneride_nd['EL_PHP_TYPE_1_7'].isin(e)) |
                                                          (amb_ps_concat_oneride_nd['SRVC_BGN_DT'].dt.month==8) & (amb_ps_concat_oneride_nd['EL_PHP_TYPE_1_8'].isin(e)) |
                                                          (amb_ps_concat_oneride_nd['SRVC_BGN_DT'].dt.month==9) & (amb_ps_concat_oneride_nd['EL_PHP_TYPE_1_9'].isin(e)) |
                                                          (amb_ps_concat_oneride_nd['SRVC_BGN_DT'].dt.month==10) & (amb_ps_concat_oneride_nd['EL_PHP_TYPE_1_10'].isin(e)) |
                                                          (amb_ps_concat_oneride_nd['SRVC_BGN_DT'].dt.month==11) & (amb_ps_concat_oneride_nd['EL_PHP_TYPE_1_11'].isin(e)) |
                                                          (amb_ps_concat_oneride_nd['SRVC_BGN_DT'].dt.month==12) & (amb_ps_concat_oneride_nd['EL_PHP_TYPE_1_12'].isin(e))]

    # Recover memory
    del amb_ps_concat_oneride_nd

    # Clean DF
    amb_ps_concat_oneride_nd_ffs = amb_ps_concat_oneride_nd_ffs.drop(['EL_PHP_TYPE_1_{}'.format(i) for i in range(1,13)],axis=1)
    amb_ps_concat_oneride_nd_e = amb_ps_concat_oneride_nd_e.drop(['EL_PHP_TYPE_1_{}'.format(i) for i in range(1,13)],axis=1)

    #-------------------------------Read out Data---------------------------------#

    # Read out final ambulance data
    amb_ps_concat_oneride_nd_ffs.to_parquet(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicaid_ffs_mc/{year}/amb_ffs/{state}/', compression='gzip', engine='fastparquet')
    amb_ps_concat_oneride_nd_e.to_parquet(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicaid_ffs_mc/{year}/amb_mc/{state}/', compression='gzip', engine='fastparquet')

#____________________________________________ Run Defined Functions ___________________________________________________#

# Specify the years
years = [2011,2012,2013,2014]

# Specify the states available for each year
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

    # Use if/then since each year does not contain the same number of states
    if y in [2011]:

        # Create loop for 2011 available states
        for s in states_11:

            # Use function defined above to export amb by year and state
            export_amb_w_data(y,s,state_11_available_in_12)

    elif y in [2012]:

        # Create loop for 2012 available states
        for s in states_12:

            # Use function defined above to export amb by year and state
            export_amb_w_data(y,s,state_12_available_in_13)

    elif y in [2013]:

        # Create loop for 2013 available states
        for s in states_13:

            # Use function defined above to export amb by year and state
            export_amb_w_data(y,s,state_13_available_in_14)

    elif y in [2014]:

        # Create loop for 2014 available states
        for s in states_14:

            # Use function defined above to export amb by year and state
            export_amb_w_data(y,s,['filler']) # I listed a random filler since 2014 does NOT have data for the following year.

###################################### Creating MAX Data for Outpatient Claims #########################################
# Note that each year has different number of states. Goal is to create a subset of op claims that matched with amb    #
# claims.                                                                                                              #
########################################################################################################################

#________________________________________________ Define Function _____________________________________________________#

# Define function to keep the op claims that matched with amb claims and export the file
def export_subset_op(year,state):

    # Specify relevant columns
    columns_op = ['BENE_ID', 'MSIS_ID', 'STATE_CD', 'SRVC_BGN_DT', 'MAX_TOS', 'PLC_OF_SRVC_CD', 'SRVC_END_DT'] + [
        'DIAG_CD_{}'.format(i) for i in range(1, 3)]

    # Read in data using dask
    ot = dd.read_parquet(f'/mnt/data/medicaid-max/data/{year}/ot/parquet/{state}/', engine='fastparquet',
                            columns=columns_op)

    # Keep outpatient
    op = ot.loc[(ot['MAX_TOS'] == '11') | (ot['PLC_OF_SRVC_CD'] == '22') | (ot['PLC_OF_SRVC_CD'] == '23'), :]

    # Del Df to recover memory
    del ot

    # Read in ffs and mc ambulance claims with only bene_id and msis_id/state_cd
    amb_ffs = dd.read_parquet(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicaid_ffs_mc/{year}/amb_ffs/{state}/', engine='fastparquet',
                              columns=['BENE_ID','MSIS_ID','STATE_CD'])
    amb_e = dd.read_parquet(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicaid_ffs_mc/{year}/amb_mc/{state}/', engine='fastparquet',
                              columns=['BENE_ID','MSIS_ID','STATE_CD'])

    # Concat amb claims
    amb = dd.concat([amb_ffs,amb_e],axis=0)

    # Delete df to recover memory
    del amb_ffs
    del amb_e

    # Split into missing and not missing bene_id for op
    op_missingid = op[op['BENE_ID']=='']
    op_notmissingid = op[op['BENE_ID']!='']

    # Delete df to recover memory
    del op

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

    # Concat df
    op = dd.concat([op_missingid_merge,op_notmissingid_merge],axis=0)

    # Delete df to recover memory
    del op_missingid_merge
    del op_notmissingid_merge

    # Count the number of diagnosis codes
    diag_col = ['DIAG_CD_{}'.format(i) for i in range(1, 3)]  # Define diagnosis columns
    op[diag_col] = op[diag_col].replace('', np.nan)  # Replace empty strings to count number of diagnosis codes
    op['num_of_diag_codes'] = op[diag_col].count(1)  # Count diagnosis codes
    op[diag_col] = op[diag_col].fillna('')  # Fill nan's with empty strings

    # Split the DF into those missing bene_id and those with bene_id
    op_missingid = op[op['BENE_ID'] == '']
    op_notmissingid = op[op['BENE_ID'] != '']

    # Sort each partition in ascending order
    op_missingid = op_missingid.map_partitions(lambda x: x.sort_values(by=['num_of_diag_codes'], ascending=True))
    op_notmissingid = op_notmissingid.map_partitions(lambda x: x.sort_values(by=['num_of_diag_codes'], ascending=True))

    # Drop duplicated rows by keeping last (i.e. keep the most information)
    op_missingid = op_missingid.drop_duplicates(subset=['MSIS_ID', 'STATE_CD', 'SRVC_BGN_DT'], keep='last')
    op_notmissingid = op_notmissingid.drop_duplicates(subset=['BENE_ID', 'SRVC_BGN_DT'], keep='last')

    # Concat and clean DFs
    op = dd.concat([op_missingid, op_notmissingid], axis=0)
    op['BENE_ID'] = op['BENE_ID'].fillna('')
    op['MSIS_ID'] = op['MSIS_ID'].fillna('')
    op['STATE_CD'] = op['STATE_CD'].fillna('')

    # Read out Data for OP
    op.to_parquet(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicaid_ffs_mc/{year}/op_subset/{state}/',compression='gzip', engine='fastparquet')

#____________________________________________ Run Defined Functions ___________________________________________________#

# Specify the years
years = [2011,2012,2013,2014]

# Specify the states available for each year
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

    # Use if/then since each year does not contain the same number of states
    if y in [2011]:

        # Create loop for 2011 available states
        for s in states_11:

            # Use function defined above to export a subset of op by year and state
            export_subset_op(y,s)

    elif y in [2012]:

        # Create loop for 2012 available states
        for s in states_12:

            # Use function defined above to export a subset of op by year and state
            export_subset_op(y,s)

    elif y in [2013]:

        # Create loop for 2013 available states
        for s in states_13:

            # Use function defined above to export a subset of op by year and state
            export_subset_op(y,s)

    elif y in [2014]:

        # Create loop for 2014 available states
        for s in states_14:

            # Use function defined above to export a subset of op by year and state
            export_subset_op(y,s)








