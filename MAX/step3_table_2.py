#----------------------------------------------------------------------------------------------------------------------#
# Project: Medicaid Data Quality Project
# Authors: Jessy Nguyen
# Last Updated: August 12, 2021
# Description: The goal of this script is to calculate the proportions for each indicator for exhibit 2 by each state for
#              Medicaid and Medicare. The table layout was created using Excel but the numbers were generated here. Also,
#              the numbers were converted to percents in Excel. Some Medicare data quality indicators were exported as
#              a table first for easier transfer to to Excel to form exhibit 2. Note that some Medicaid files containing
#              the numerator and denominator were exported as csv in order to conduct t-tests for exhibit 3 (table 3).
#              We simply used the statsmodels.stats.proportion.proportions_ztest package to conduct the significance
#              test using the exported csv files containing the numerator and denominators (not included in these codes).
#----------------------------------------------------------------------------------------------------------------------#

################################################# IMPORT PACKAGES ######################################################

# Read in relevant libraries
import dask.dataframe as dd
from datetime import datetime, timedelta
import pandas as pd
from pandas.tseries.offsets import MonthEnd
import numpy as np
from itertools import chain

################################################# MODULE FOR CLUSTER ###################################################

# Read in libraries to use cluster
from dask.distributed import Client
client = Client('[insert_ip_address_for_cluster]')

######################### CRITERIA 1 (MOD): PROPORTION OF MOD CODE IN AMB CLAIMS #######################################
# This script for the MOD criteria calculates the proportion of FFS/Encounter/Medicare ambulance claims that are not   #
# missing or correctly labeled pickup and drop-off modifier codes. Since we are missing states in some of the years,   #
# we used an if/then conditional script to aid in reading in the files. We also dropped individuals who were not in    #
# MCAID for at least 91 consecutive days since the date of service.                                                    #
########################################################################################################################

#__________________________________Calculate proportion for Medicaid (MOD)_____________________________________________#

#------------------------------------Define function for Medicaid------------------------------------------------------#

# Define function to calculate proportion of ambulance claims with mods
def calc_mod_criteria(state,mcaid_payment_type):

    # Specify States available for 2011-2014
    states_11_14 = ['CA', 'GA', 'IA', 'LA', 'MI', 'MN', 'MO', 'MS', 'NJ', 'PA', 'SD', 'TN', 'UT', 'VT', 'WV', 'WY']

    # Specify States available for 2011-2013
    states_11_13 = ['AR', 'AZ', 'CT', 'HI', 'IN', 'MA', 'NY', 'OH', 'OK', 'OR', 'WA']

    # Specify States available for 2011-2012
    states_11_12 = ['AL', 'AK', 'CO', 'DC', 'DE', 'FL', 'IL', 'KS', 'KY', 'ME', 'MD', 'MT', 'NE', 'NV', 'NH', 'NM',
                    'NC', 'ND', 'RI', 'SC', 'TX', 'VA', 'WI']

    # Specify States available for 2012-2014
    states_12_14 = ['ID']

    #---Read in Amb Claims---#

    # Define Columns
    amb_columns = ['PRCDR_SRVC_MDFR_CD', 'SRVC_BGN_DT'] + ['EL_DAYS_EL_CNT_{}'.format(i) for i in range(1, 16)]

    # Define empty dictionary
    df_dict = {}

    # Create empty list to concat
    df_list = []

    # Due to missing states in some years, we use if/then statement
    if state in states_11_14:

        # Define years. Also will use as "key" in dictionary
        years=[2011,2012,2013,2014]

        # Read in each year and store in dictionary
        for y in years:
            df_dict[y] = dd.read_parquet(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicaid_ffs_mc/{y}/amb_{mcaid_payment_type}/{state}/', engine='fastparquet',columns=amb_columns)

            # Append df from dictionary to list
            df_list.append(df_dict[y])

    elif state in states_11_13:

        # Define years. Also will use as "key" in dictionary
        years=[2011,2012,2013]

        # Read in each year and store in dictionary
        for y in years:
            df_dict[y] = dd.read_parquet(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicaid_ffs_mc/{y}/amb_{mcaid_payment_type}/{state}/', engine='fastparquet',columns=amb_columns)

            # Append df from dictionary to list
            df_list.append(df_dict[y])

    elif state in states_11_12:

        # Define years. Also will use as "key" in dictionary
        years=[2011,2012]

        # Read in each year and store in dictionary
        for y in years:
            df_dict[y] = dd.read_parquet(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicaid_ffs_mc/{y}/amb_{mcaid_payment_type}/{state}/', engine='fastparquet',columns=amb_columns)

            # Append df from dictionary to list
            df_list.append(df_dict[y])

    elif state in states_12_14:

        # Define years. Also will use as "key" in dictionary
        years=[2012,2013,2014]

        # Read in each year and store in dictionary
        for y in years:
            df_dict[y] = dd.read_parquet(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicaid_ffs_mc/{y}/amb_{mcaid_payment_type}/{state}/', engine='fastparquet',columns=amb_columns)

            # Append df from dictionary to list
            df_list.append(df_dict[y])

    # Concat all FFS ambulance claims
    amb=dd.concat(df_list,axis=0)

    #---------------------Filter Those at least 91 days in Medicaid-------------------------#

    #---Codes to count number of days in first month---#

    # Convert all to datetime
    amb['SRVC_BGN_DT'] = dd.to_datetime(amb['SRVC_BGN_DT'])

    # Remove any claims that have nan in service begin date
    amb = amb[~amb['SRVC_BGN_DT'].isna()]

    # Convert columns to floats (the range up to 16 to account for those who had service date oct-dec)
    for i in range(1,16):
        amb['EL_DAYS_EL_CNT_{}'.format(i)] = amb['EL_DAYS_EL_CNT_{}'.format(i)].astype('float')

    # Find the end of the month from service begin date
    amb['EndOfMonth'] =  dd.to_datetime(amb['SRVC_BGN_DT']) + MonthEnd(1)

    # Find number of days from service begin date to end of month
    amb['Days_Until_End_Month'] = amb['EndOfMonth'] - amb['SRVC_BGN_DT']

    # Convert from days/timedelta to float
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

    # convert to float
    amb['days_enrolled_after_three_months'] = amb['days_enrolled_after_three_months'].astype('float')

    #---Codes to filter individuals with at least 91 days in Medicaid---#

    # Add to see if individuals enrolled at least 91 days
    amb['total_enrolled_after_4_months'] = amb['days_enrolled_after_three_months'] + amb['Days_Until_End_Month']

    # Filter based on if individuals with service date from Jan-Dec have at least 91 days in medicaid
    amb = amb[(amb['total_enrolled_after_4_months'] > 90)]

    # Clean DF
    amb = amb.drop(['EL_DAYS_EL_CNT_{}'.format(i) for i in range(1,16)] +
                           ['EndOfMonth','Days_Until_End_Month','days_enrolled','days_enrolled_after_three_months','total_enrolled_after_4_months'],axis=1)

    #---Calculate numbers---#

    # Convert all NA's to blanks
    amb['PRCDR_SRVC_MDFR_CD'] = amb['PRCDR_SRVC_MDFR_CD'].fillna('')

    # Define all possible ambulance modifiers
    mod=['GY','QL','GM','GA','GZ','QM','QN','DD','DE','DG','DH','DI','DJ','DN','DP','DR','DS','DX','ED','EE','EG','EH','EI',
         'EJ','EN','EP','ER','ES','EX','GD','GE','GG','GH','GI','GJ','GN','GP','GR','GS','GX','HD','HE','HG','HH','HI','HJ',
         'HN','HP','HR','HS','HX','ID','IE','IG','IH','II','IJ','IN','IP','IR','IS','IX','JD','JE','JG','JH','JI','JJ','JN',
         'JP','JR','JS','JX','ND','NE','NG','NH','NI','NJ','NN','NP','NR','NS','NX','PD','PE','PG','PH','PI','PJ','PN','PP',
         'PR','PS','PX','RD','RE','RG','RH','RI','RJ','RN','RP','RR','RS','RX','SD','SE','SG','SH','SI','SJ','SN','SP','SR',
         'SS','SX','XD','XE','XG','XH','XI','XJ','XN','XP','XR','XS','XX']

    # Create indicator variable if there is a pickup/dropoff code
    amb['mod_ind']=0
    amb['mod_ind']=amb['mod_ind'].mask((amb['PRCDR_SRVC_MDFR_CD'].isin(mod)),1)

    # Calculate and store numbers in variables
    total = amb.shape[0].compute()
    num_notmissingmod = amb['mod_ind'].sum().compute()

    # Recover memory
    del amb

    # Calculate proportion of claims in each state with amb modifiers
    print(state, f'{mcaid_payment_type} Crit 1 Proportion of claims not missing modifiers \n')
    if (total > 10) & (num_notmissingmod > 10): # Prevent violation of cell supression policy
        print(num_notmissingmod/total)

        # append numerator and denominator to the empty list defined below if not nan (for t-test between MAX and TAF)
        if mcaid_payment_type in ['ffs']:
            max_ffs_num_mod.append(num_notmissingmod)
            max_ffs_denom_mod.append(total)
        elif mcaid_payment_type in ['mc']:
            max_mc_num_mod.append(num_notmissingmod)
            max_mc_denom_mod.append(total)

    else:
        print('nan')

        # append numerator and denominator to the empty list defined below if nan (for t-test between MAX and TAF)
        if mcaid_payment_type in ['ffs']:
            max_ffs_num_mod.append(np.nan)
            max_ffs_denom_mod.append(np.nan)
        elif mcaid_payment_type in ['mc']:
            max_mc_num_mod.append(np.nan)
            max_mc_denom_mod.append(np.nan)

#-----------------------------------Run Defined function for Medicaid--------------------------------------------------#

# Specify all States
all_states=['AL','AK','AZ','AR','CA','CO','CT','DC','DE','FL','GA','HI','ID','IL','IN','IA','KS','KY','LA','ME','MD','MA',
            'MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX',
            'UT','VT','VA','WA','WV','WI','WY']

# Create empty list to store numbers (for t-test between MAX and TAF)
max_ffs_num_mod = []
max_mc_num_mod = []
max_ffs_denom_mod = []
max_mc_denom_mod = []

# Create loop over each state
for s in all_states:

    # Run function for FFS
    calc_mod_criteria(s, 'ffs')

    # Run function for MC
    calc_mod_criteria(s, 'mc')

# Create empty dictionary. Will be used to create a DF (for t-test between MAX and TAF)
df_dict_mod = {}

# Append each list (and the states) above to dictionary (for t-test between MAX and TAF)
df_dict_mod['max_ffs_num_mod'] = max_ffs_num_mod
df_dict_mod['max_mc_num_mod'] = max_mc_num_mod
df_dict_mod['max_ffs_denom_mod'] = max_ffs_denom_mod
df_dict_mod['max_mc_denom_mod'] = max_mc_denom_mod
df_dict_mod['state'] = all_states

# Create Dataframe from dictionary (for t-test between MAX and TAF)
df_mod = pd.DataFrame.from_dict(df_dict_mod)

# Calculate proportion to double check with exhibit (for t-test between MAX and TAF)
df_mod['prop_ffs'] = df_mod['max_ffs_num_mod']/df_mod['max_ffs_denom_mod']
df_mod['prop_mc'] = df_mod['max_mc_num_mod']/df_mod['max_mc_denom_mod']

# Read out DF that was created from dictionary (for t-test between MAX and TAF)
df_mod.to_csv('/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicaid_ffs_mc/df_for_t_test_max_vs_taf/df_mod_max.csv',index=False,index_label=False)

#__________________________________Calculate proportion for Medicare (MOD)_____________________________________________#

# Create empty dictionary
df_dict={}

# Specify years
years = [2011,2012,2013,2014]

# Append df to dictionary
for y in years:
    df_dict[y]=dd.read_parquet(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicare_ab/{y}/amb_ab/', engine='fastparquet',
                                  columns=['PRVDR_STATE_CD','HCPCS_1ST_MDFR_CD','HCPCS_2ND_MDFR_CD'])

# Create empty list to concat all df's
df_list=[]

# Append df to list
for y in years:
    df_list.append(df_dict[y])

# Concat
amb_11_14 = dd.concat(df_list,axis=0)

# Recover memory
del df_dict
del df_list

# Convert all NA's to blanks
amb_11_14['HCPCS_1ST_MDFR_CD'] = amb_11_14['HCPCS_1ST_MDFR_CD'].fillna('')
amb_11_14['HCPCS_2ND_MDFR_CD'] = amb_11_14['HCPCS_2ND_MDFR_CD'].fillna('')

# Create column to calculate denominator (total ambulance claims)
amb_11_14['num_total_amb_claims']=1

# Define all possible ambulance modifiers
mod = ['GY', 'QL', 'GM', 'GA', 'GZ', 'QM', 'QN', 'DD', 'DE', 'DG', 'DH', 'DI', 'DJ', 'DN', 'DP', 'DR', 'DS', 'DX', 'ED',
       'EE', 'EG', 'EH', 'EI', 'EJ', 'EN', 'EP', 'ER', 'ES', 'EX', 'GD', 'GE', 'GG', 'GH', 'GI', 'GJ', 'GN', 'GP', 'GR',
       'GS', 'GX', 'HD', 'HE', 'HG', 'HH', 'HI', 'HJ', 'HN', 'HP', 'HR', 'HS', 'HX', 'ID', 'IE', 'IG', 'IH', 'II', 'IJ',
       'IN', 'IP', 'IR', 'IS', 'IX', 'JD', 'JE', 'JG', 'JH', 'JI', 'JJ', 'JN', 'JP', 'JR', 'JS', 'JX', 'ND', 'NE', 'NG',
       'NH', 'NI', 'NJ', 'NN', 'NP', 'NR', 'NS', 'NX', 'PD', 'PE', 'PG', 'PH', 'PI', 'PJ', 'PN', 'PP', 'PR', 'PS', 'PX',
       'RD', 'RE', 'RG', 'RH', 'RI', 'RJ', 'RN', 'RP', 'RR', 'RS', 'RX', 'SD', 'SE', 'SG', 'SH', 'SI', 'SJ', 'SN', 'SP',
       'SR', 'SS', 'SX', 'XD', 'XE', 'XG', 'XH', 'XI', 'XJ', 'XN', 'XP', 'XR', 'XS', 'XX']

# Create indicator variable if there is a pickup/dropoff code
amb_11_14['num_with_mod'] = 0
amb_11_14['num_with_mod'] = amb_11_14['num_with_mod'].mask(((amb_11_14['HCPCS_1ST_MDFR_CD'].isin(mod))|(amb_11_14['HCPCS_2ND_MDFR_CD'].isin(mod))), 1)

#-----------------Calculate Numbers-------------------------#

# Group by states to calc the numerator (number with modifier) and denominator (total ambulance claims)
proportion_amb_11_14 = amb_11_14.groupby(['PRVDR_STATE_CD'])[['num_total_amb_claims','num_with_mod']].sum().reset_index()

# Create a new column calculating the proportion
proportion_amb_11_14['proportion_w_mod'] = proportion_amb_11_14['num_with_mod']/proportion_amb_11_14['num_total_amb_claims']

# Rename states
proportion_amb_11_14['PRVDR_STATE_CD'] = proportion_amb_11_14['PRVDR_STATE_CD'].replace(
             {'01':'AL','02':'AK','03':'AZ','04':'AR','05':'CA','06':'CO','07':'CT','08':'DE','09':'DC','10':'FL','11':'GA',
              '12':'HI','13':'ID','14':'IL','15':'IN','16':'IA','17':'KS','18':'KY','19':'LA','20':'ME','21':'MD','22':'MA',
              '23':'MI','24':'MN','25':'MS','26':'MO','27':'MT','28':'NE','29':'NV','30':'NH','31':'NJ','32':'NM','33':'NY',
              '34':'NC','35':'ND','36':'OH','37':'OK','38':'OR','39':'PA','41':'RI','42':'SC','43':'SD','44':'TN','45':'TX',
              '46':'UT','47':'VT','49':'VA','50':'WA','51':'WV','52':'WI','53':'WY'})

# View results
print(proportion_amb_11_14.head(60))

# Read out to CSV to easily transfer results over to excel
proportion_amb_11_14.to_csv('/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicare_ab/table_two_mcare_results/MOD.csv')

######################## CRITERIA 2 (MIL): AMB CLAIMS MATCHED WITH MILEAGE INFO ########################################
# This script calculates the proportion of ambulance that are matched with Mileage information. This calculates by     #
# state for all the years. FFS and Encounter are both calculated. We've already dropped individuals who were not in    #
# Medicaid for at least 91 days when merging with mileage information so I do not need to run that script here.        #
########################################################################################################################

#----------------------------------------Define function for Medicaid--------------------------------------------------#

# Define function to calculate proportion amb claims matched with mileage info
def calc_amb_match_mi_criteria(state,mcaid_payment_type):

    # Specify States available for 2011-2014
    states_11_14 = ['CA', 'GA', 'IA', 'LA', 'MI', 'MN', 'MO', 'MS', 'NJ', 'PA', 'SD', 'TN', 'UT', 'VT', 'WV', 'WY']

    # Specify States available for 2011-2013
    states_11_13 = ['AR', 'AZ', 'CT', 'HI', 'IN', 'MA', 'NY', 'OH', 'OK', 'OR', 'WA']

    # Specify States available for 2011-2012
    states_11_12 = ['AL', 'AK', 'CO', 'DC', 'DE', 'FL', 'IL', 'KS', 'KY', 'ME', 'MD', 'MT', 'NE', 'NV', 'NH', 'NM',
                    'NC', 'ND', 'RI', 'SC', 'TX', 'VA', 'WI']

    # Specify States available for 2012-2014
    states_12_14 = ['ID']

    # Due to missing states in some years, we use if/then statement
    if state in states_11_14:

        # Specify dictionary
        df_dict_11_14={}

        # Append 2011-2014 file paths to list
        years=[2011,2012,2013,2014]
        for y in years:
            df_dict_11_14[y] = pd.read_parquet(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicaid_ffs_mc/{y}/{mcaid_payment_type}_merged_amb_mileage/{state}/',engine='fastparquet',columns=['ind_for_mi_match','QTY_SRVC_UNITS'])

        # Create empty list to concat
        df_list_11_14=[]

        # Append DF dictionary to list
        for y in years:
            df_list_11_14.append(df_dict_11_14[y])

        # Concat DF
        amb_mi = pd.concat(df_list_11_14,axis=0)

    elif state in states_11_13:

        # Specify dictionary
        df_dict_11_13={}

        # Append 2011-2013 file paths to list
        years=[2011,2012,2013]
        for y in years:
            df_dict_11_13[y] = pd.read_parquet(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicaid_ffs_mc/{y}/{mcaid_payment_type}_merged_amb_mileage/{state}/',engine='fastparquet',columns=['ind_for_mi_match','QTY_SRVC_UNITS'])

        # Create empty list to concat
        df_list_11_13=[]

        # Append DF dictionary to list
        for y in years:
            df_list_11_13.append(df_dict_11_13[y])

        # Concat DF
        amb_mi = pd.concat(df_list_11_13,axis=0)

    elif state in states_11_12:

        # Specify dictionary
        df_dict_11_12={}

        # Append 2011-2012 file paths to list
        years=[2011,2012]
        for y in years:
            df_dict_11_12[y] = pd.read_parquet(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicaid_ffs_mc/{y}/{mcaid_payment_type}_merged_amb_mileage/{state}/',engine='fastparquet',columns=['ind_for_mi_match','QTY_SRVC_UNITS'])

        # Create empty list to concat
        df_list_11_12=[]

        # Append DF dictionary to list
        for y in years:
            df_list_11_12.append(df_dict_11_12[y])

        # Concat DF
        amb_mi = pd.concat(df_list_11_12,axis=0)

    # Due to missing states in some years, we use if/then statement
    elif state in states_12_14:

        # Specify dictionary
        df_dict_12_14={}

        # Append 2012-2014 file paths to list
        years=[2012,2013,2014]
        for y in years:
            df_dict_12_14[y] = pd.read_parquet(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicaid_ffs_mc/{y}/{mcaid_payment_type}_merged_amb_mileage/{state}/',engine='fastparquet',columns=['ind_for_mi_match','QTY_SRVC_UNITS'])

        # Create empty list to concat
        df_list_12_14=[]

        # Append DF dictionary to list
        for y in years:
            df_list_12_14.append(df_dict_12_14[y])

        # Concat DF
        amb_mi = pd.concat(df_list_12_14,axis=0)

    # Create indicator that reflects the claims has a match with amb claims, does not have an na, does not have a blank, or does not have all zero's
    amb_mi['mi_ind'] = 1
    amb_mi['mi_ind'] = amb_mi['mi_ind'].mask((amb_mi['ind_for_mi_match'].isna())|(amb_mi['QTY_SRVC_UNITS']=='')|
                                                     (amb_mi['QTY_SRVC_UNITS'].isna()),0)

    # Create a DF to check if states only reported all 0's, na's, or blanks. States that report only 0's, na's, or blanks are not correctly reporting mileage
    df_zero_na_blanks = amb_mi[(amb_mi['QTY_SRVC_UNITS']=='0')|(amb_mi['QTY_SRVC_UNITS']=='')|(amb_mi['QTY_SRVC_UNITS'].isna())]

    #------------Calculate Numbers------------#
    # Calculate and store numbers in variables
    total_amb = amb_mi.shape[0]
    num_matched = amb_mi['mi_ind'].sum()

    # Calculate numbers
    print(state,f'{mcaid_payment_type}','Crit 2 Proportion matched btwn mileage and amb claims \n')
    if ((total_amb == df_zero_na_blanks.shape[0]) & (total_amb!=0) & (amb_mi['ind_for_mi_match'].sum()>0)): # Need all of these expressions to ensure that we account for states who only reported 0/blank/na
        print('contained only all zeros/missing/na')

        # append numerator and denominator to the empty list defined below if nan (for t-test between MAX and TAF)
        if mcaid_payment_type in ['ffs']:
            max_ffs_num_mil.append(np.nan)
            max_ffs_denom_mil.append(np.nan)
        elif mcaid_payment_type in ['mc']:
            max_mc_num_mil.append(np.nan)
            max_mc_denom_mil.append(np.nan)

    elif ((total_amb > 10) & (num_matched > 10)) | ((num_matched==0) & (total_amb > 10)): # Prevent violation of cell supression policy
        print((num_matched/total_amb))

        # append numerator and denominator to the empty list defined below if not nan (for t-test between MAX and TAF)
        if mcaid_payment_type in ['ffs']:
            max_ffs_num_mil.append(num_matched)
            max_ffs_denom_mil.append(total_amb)
        elif mcaid_payment_type in ['mc']:
            max_mc_num_mil.append(num_matched)
            max_mc_denom_mil.append(total_amb)

    else:
        print('nan')

        # append numerator and denominator to the empty list defined below if nan (for t-test between MAX and TAF)
        if mcaid_payment_type in ['ffs']:
            max_ffs_num_mil.append(np.nan)
            max_ffs_denom_mil.append(np.nan)
        elif mcaid_payment_type in ['mc']:
            max_mc_num_mil.append(np.nan)
            max_mc_denom_mil.append(np.nan)

#-----------------------------------Run Defined function for Medicaid-------------------------------------------------#

# Specify all States
all_states=['AL','AK','AZ','AR','CA','CO','CT','DC','DE','FL','GA','HI','ID','IL','IN','IA','KS','KY','LA','ME','MD','MA',
            'MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX',
            'UT','VT','VA','WA','WV','WI','WY']

# Create empty list to store numbers (for t-test between MAX and TAF)
max_ffs_num_mil = []
max_mc_num_mil = []
max_ffs_denom_mil = []
max_mc_denom_mil = []

for s in all_states:

    # Run function for FFS
    calc_amb_match_mi_criteria(s, 'ffs')

    # Run function for MC
    calc_amb_match_mi_criteria(s, 'mc')

# Create empty dictionary. Will be used to create a DF (for t-test between MAX and TAF)
df_dict_mil = {}

# Append each list (and the states) above to dictionary (for t-test between MAX and TAF)
df_dict_mil['max_ffs_num_mil'] = max_ffs_num_mil
df_dict_mil['max_mc_num_mil'] = max_mc_num_mil
df_dict_mil['max_ffs_denom_mil'] = max_ffs_denom_mil
df_dict_mil['max_mc_denom_mil'] = max_mc_denom_mil
df_dict_mil['state'] = all_states

# Create Dataframe from dictionary (for t-test between MAX and TAF)
df_mil = pd.DataFrame.from_dict(df_dict_mil)

# Calculate proportion to double check with exhibit (for t-test between MAX and TAF)
df_mil['prop_ffs'] = df_mil['max_ffs_num_mil']/df_mil['max_ffs_denom_mil']
df_mil['prop_mc'] = df_mil['max_mc_num_mil']/df_mil['max_mc_denom_mil']

# Read out DF that was created from dictionary (for t-test between MAX and TAF)
df_mil.to_csv('/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicaid_ffs_mc/df_for_t_test_max_vs_taf/df_mil_max.csv',index=False,index_label=False)

#_______________________________________________Medicare (MIL)_________________________________________________________#

# Specify years
years=[2011,2012,2013,2014]

#---Read in Ambulance matched with Mileage---#

# Define empty dictionary to store DFs
dict_df_amb_mi={}

# Read in 11-14 amb matched with mileage data into df dictionary
for y in years:
    dict_df_amb_mi[y] = dd.read_parquet(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicare_ab/{y}/ab_merged_amb_mileage/',
                                        engine='fastparquet',columns=['ind_for_mi_match','PRVDR_STATE_CD','CARR_LINE_MTUS_CNT'])

# Define empty list to concat all data
list_df_amb_mi=[]

# Append df from dictionary to list
for y in years:
    list_df_amb_mi.append(dict_df_amb_mi[y])

# Concat all data
mcare_amb_mi = dd.concat(list_df_amb_mi,axis=0)

# Delete DF to recover
del dict_df_amb_mi
del list_df_amb_mi

# Create indicator that reflects the claims has a match with amb claims, does not have an na, does not have a blank, or does not have all zero's
mcare_amb_mi['mi_ind'] = 1
mcare_amb_mi['mi_ind'] = mcare_amb_mi['mi_ind'].mask((mcare_amb_mi['ind_for_mi_match'].isna())|(mcare_amb_mi['CARR_LINE_MTUS_CNT'].isna()),0)

#-----------------Calculate Numbers-------------------------#

# Create dictionary to convert code to states
state_dict = {'01':'AL','02':'AK','03':'AZ','04':'AR','05':'CA','06':'CO','07':'CT','08':'DE','09':'DC','10':'FL','11':'GA',
              '12':'HI','13':'ID','14':'IL','15':'IN','16':'IA','17':'KS','18':'KY','19':'LA','20':'ME','21':'MD','22':'MA',
              '23':'MI','24':'MN','25':'MS','26':'MO','27':'MT','28':'NE','29':'NV','30':'NH','31':'NJ','32':'NM','33':'NY',
              '34':'NC','35':'ND','36':'OH','37':'OK','38':'OR','39':'PA','41':'RI','42':'SC','43':'SD','44':'TN','45':'TX',
              '46':'UT','47':'VT','49':'VA','50':'WA','51':'WV','52':'WI','53':'WY'}

for s in state_dict:

    # Obtain one state at a time
    mcare_amb_mi_perstate = mcare_amb_mi[mcare_amb_mi['PRVDR_STATE_CD']==s]

    # Create a DF to check if states only reported all 0's, na's, or blanks. States that report only 0's, na's, or blanks are not correctly reporting mileage
    df_zero_na_blanks = mcare_amb_mi_perstate[(mcare_amb_mi_perstate['ind_for_mi_match'].isna()) | (mcare_amb_mi_perstate['CARR_LINE_MTUS_CNT'].isna())]

    # Specify State
    print('MCARE', state_dict[s])

    # Store numbers in variables
    total_amb = mcare_amb_mi_perstate.shape[0].compute()
    num_matched = mcare_amb_mi_perstate['mi_ind'].sum().compute()

    # Calculate numbers
    print(state_dict[s],'Medicare Crit 2 Proportion matched btwn mileage and amb claims \n')
    if ((total_amb == df_zero_na_blanks.shape[0].compute()) & (total_amb!=0) & (mcare_amb_mi_perstate['ind_for_mi_match'].sum().compute()>0)): # Need all of these expressions to ensure that we account for states who only reported 0/blank/na
        print('contained only all zeros/missing/na')
    elif (total_amb > 10) & (num_matched > 10) | ((num_matched==0) & (total_amb > 10)): # Prevent violation of cell supression policy
        print((num_matched/total_amb))
    else:
        print('nan')

######################## CRITERIA 3 (HOSP): AMB CLAIMS MATCHED WITH HOS CLAIMS #########################################
# This script calculates the proportion of ambulance that are matched with hospital claims. This calculates by state   #
# for all the years. FFS and Encounter are both calculated. I kept individuals who have passed away or were in         #
# Medicaid for at least 91 consecutive days since the date of ambulance ride. In other words, the difference between   #
# the ambulance claims denominator in criteria 3 and the denominators in criteria 1 (MOD) and criteria 2 (MIL) is the  #
# addition of those who have died within the 90 days. We needed to add in the individuals who have expired before 90   #
# days because we needed the death information.                                                                        #
########################################################################################################################

#-----------------------------------------Define function for Medicaid-------------------------------------------------#

# Define function to calculate proportion amb claims matched with hos
def calc_amb_match_hos_criteria(state,mcaid_payment_type):

    # Specify States available for 2011-2014
    states_11_14 = ['CA', 'GA', 'IA', 'LA', 'MI', 'MN', 'MO', 'MS', 'NJ', 'PA', 'SD', 'TN', 'UT', 'VT', 'WV', 'WY']

    # Specify States available for 2011-2013
    states_11_13 = ['AR', 'AZ', 'CT', 'HI', 'IN', 'MA', 'NY', 'OH', 'OK', 'OR', 'WA']

    # Specify States available for 2011-2012
    states_11_12 = ['AL', 'AK', 'CO', 'DC', 'DE', 'FL', 'IL', 'KS', 'KY', 'ME', 'MD', 'MT', 'NE', 'NV', 'NH', 'NM',
                    'NC', 'ND', 'RI', 'SC', 'TX', 'VA', 'WI']

    # Specify States available for 2012-2014
    states_12_14 = ['ID']

    # Due to missing states in some years, we use if/then statement
    if state in states_11_14:

        # Specify empty list for pathways
        file_paths=[]

        # Append 2011-2014 file paths to list
        years=[2011,2012,2013,2014]
        for y in years:
            file_paths.append(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicaid_ffs_mc/{y}/{mcaid_payment_type}_merged_amb_hos_claims_ninetyonedays/{state}/')

    elif state in states_11_13:

        # Specify empty list for pathways
        file_paths=[]

        # Append 2011-2013 file paths to list
        years=[2011,2012,2013]
        for y in years:
            file_paths.append(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicaid_ffs_mc/{y}/{mcaid_payment_type}_merged_amb_hos_claims_ninetyonedays/{state}/')

    elif state in states_11_12:

        # Specify empty list for pathways
        file_paths=[]

        # Append 2011-2012 file paths to list
        years=[2011,2012]
        for y in years:
            file_paths.append(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicaid_ffs_mc/{y}/{mcaid_payment_type}_merged_amb_hos_claims_ninetyonedays/{state}/')

    elif state in states_12_14:

        # Specify empty list for pathways
        file_paths=[]

        # Append 2012-2014 file paths to list
        years=[2012,2013,2014]
        for y in years:
            file_paths.append(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicaid_ffs_mc/{y}/{mcaid_payment_type}_merged_amb_hos_claims_ninetyonedays/{state}/')


    # Specify relevant column to calculate proportion matched
    columns_amb = ['ind_for_hos_match']

    # Read in
    amb_hos = dd.read_parquet(file_paths, engine='fastparquet',columns=columns_amb)

    #-----------------Calculate Numbers-------------------------#

    # Calculate denominator and numerators and store them in variables
    total_amb = amb_hos.shape[0].compute()
    num_matched_hos = amb_hos['ind_for_hos_match'].sum().compute()

    # Specify state and payment type
    print(state,f'{mcaid_payment_type} Crit 3 MCAID Proportion of amb claims match with hos \n')

    # Proportion match
    if (total_amb > 10) & (num_matched_hos > 10) | ((num_matched_hos==0) & (total_amb > 10)): # Prevent violation of cell supression policy
        print((num_matched_hos/total_amb))

        # append numerator and denominator to the empty list defined below if not nan (for t-test between MAX and TAF)
        if mcaid_payment_type in ['ffs']:
            max_ffs_num_hosp.append(num_matched_hos)
            max_ffs_denom_hosp.append(total_amb)
        elif mcaid_payment_type in ['mc']:
            max_mc_num_hosp.append(num_matched_hos)
            max_mc_denom_hosp.append(total_amb)

    else:
        print('nan')

        # append numerator and denominator to the empty list defined below if nan (for t-test between MAX and TAF)
        if mcaid_payment_type in ['ffs']:
            max_ffs_num_hosp.append(np.nan)
            max_ffs_denom_hosp.append(np.nan)
        elif mcaid_payment_type in ['mc']:
            max_mc_num_hosp.append(np.nan)
            max_mc_denom_hosp.append(np.nan)

#------------------------------------Run Defined function for Medicaid-------------------------------------------------#

# Specify all States
all_states=['AL','AK','AZ','AR','CA','CO','CT','DC','DE','FL','GA','HI','ID','IL','IN','IA','KS','KY','LA','ME','MD','MA',
            'MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX',
            'UT','VT','VA','WA','WV','WI','WY']

# Create empty list to store numbers (for t-test between MAX and TAF)
max_ffs_num_hosp = []
max_mc_num_hosp = []
max_ffs_denom_hosp = []
max_mc_denom_hosp = []

for s in all_states:

    # Run function for FFS
    calc_amb_match_hos_criteria(s, 'ffs')

    # Run function for MC
    calc_amb_match_hos_criteria(s, 'mc')

# Create empty dictionary. Will be used to create a DF (for t-test between MAX and TAF)
df_dict_hosp = {}

# Append each list (and the states) above to dictionary (for t-test between MAX and TAF)
df_dict_hosp['max_ffs_num_hosp'] = max_ffs_num_hosp
df_dict_hosp['max_mc_num_hosp'] = max_mc_num_hosp
df_dict_hosp['max_ffs_denom_hosp'] = max_ffs_denom_hosp
df_dict_hosp['max_mc_denom_hosp'] = max_mc_denom_hosp
df_dict_hosp['state'] = all_states

# Create Dataframe from dictionary (for t-test between MAX and TAF)
df_hosp = pd.DataFrame.from_dict(df_dict_hosp)

# Calculate proportion to double check with exhibit (for t-test between MAX and TAF)
df_hosp['prop_ffs'] = df_hosp['max_ffs_num_hosp']/df_hosp['max_ffs_denom_hosp']
df_hosp['prop_mc'] = df_hosp['max_mc_num_hosp']/df_hosp['max_mc_denom_hosp']

# Read out DF that was created from dictionary (for t-test between MAX and TAF)
df_hosp.to_csv('/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicaid_ffs_mc/df_for_t_test_max_vs_taf/df_hosp_max.csv',index=False,index_label=False)

#____________________________________________Medicare (HOSP)___________________________________________________________#
# This script will calculate the proportion of ambulance claims that matched with hospital claims. Since we matched on #
# IP then OP, we read in each claims that were matched to IP and OP and not matched separately for 11-14. We then      #
# concatenated all of the dataframe before calculating the proportions per state.                                      #
########################################################################################################################

# Specify years
years=[2011,2012,2013,2014]

#---Read in Ambulance matched with OP---#

# Define empty dictionary to store DFs
dict_df_op={}

# Read in 11-14 amb matched with op data into df dictionary
for y in years:
    dict_df_op[y] = dd.read_parquet(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicare_ab/{y}/merged_amb_hos_claims/op_merged_amb/',
                               engine='fastparquet',columns=['STATE_CODE','PRVDR_STATE_CD','ind_for_hos_match','HCPCS_1ST_MDFR_CD','HCPCS_2ND_MDFR_CD'])

# Define empty list to concat all op data
list_df_op=[]

# Append df from dictionary to list
for y in years:
    list_df_op.append(dict_df_op[y])

# Concat all data
mcare_amb_op = dd.concat(list_df_op,axis=0)

# Delete DF to recover
del dict_df_op
del list_df_op

# Keep only claims that did NOT transport patients across state
mcare_amb_op = mcare_amb_op[mcare_amb_op['STATE_CODE']==mcare_amb_op['PRVDR_STATE_CD']]

# Clean dataframe before concatenating
mcare_amb_op = mcare_amb_op.drop(['STATE_CODE'],axis=1)
mcare_amb_op = mcare_amb_op.rename(columns={'PRVDR_STATE_CD':'STATE_CD'})

# Create column to calculate denominator
mcare_amb_op['num_total_amb_claims']=1

#---Read in Ambulance matched with IP---#

# Define empty dictionary to store DFs
dict_df_ip={}

# Read in 11-14 amb matched with ip data into df dictionary
for y in years:
    dict_df_ip[y] = dd.read_parquet(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicare_ab/{y}/merged_amb_hos_claims/ip_merged_amb/',
                               engine='fastparquet',columns=['BENE_RSDNC_SSA_STATE_CD','ind_for_hos_match','HCPCS_1ST_MDFR_CD','HCPCS_2ND_MDFR_CD'])

# Define empty list to concat all ip data
list_df_ip=[]

# Append df from dictionary to list
for y in years:
    list_df_ip.append(dict_df_ip[y])

# Concat all data
mcare_amb_ip = dd.concat(list_df_ip,axis=0)

# Delete DF to recover
del dict_df_ip
del list_df_ip

# Create column to calculate denominator
mcare_amb_ip['num_total_amb_claims']=1

# Clean dataframe before concatenating
mcare_amb_ip = mcare_amb_ip.rename(columns={'BENE_RSDNC_SSA_STATE_CD':'STATE_CD'})

#---Read in Ambulance notmatched with hospital---#

# Define empty dictionary to store DFs
dict_df_notmatched={}

# Read in amb not matched
for y in years:
    dict_df_notmatched[y] = dd.read_parquet(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicare_ab/{y}/merged_amb_hos_claims/amb_claims_notmatched/',
                               engine='fastparquet',columns=['STATE_CODE','HCPCS_1ST_MDFR_CD','HCPCS_2ND_MDFR_CD'])

# Define empty list to concat all ip data
list_df_notmatched=[]

# Append df from dictionary to list
for y in years:
    list_df_notmatched.append(dict_df_notmatched[y])

# Concat all data
mcare_amb_notmatched = dd.concat(list_df_notmatched,axis=0)

# Delete DF to recover
del dict_df_notmatched
del list_df_notmatched

# Clean dataframe before concatenating
mcare_amb_notmatched = mcare_amb_notmatched.rename(columns={'STATE_CODE':'STATE_CD'})

# Create column to calculate numerator and denominator
mcare_amb_notmatched['ind_for_hos_match']=0
mcare_amb_notmatched['num_total_amb_claims']=1

#---Concat all above data (amb matched with ip, op, and those not matched)---#

# Concat all 2011-2014 matched and not matched data
matched_df = dd.concat([mcare_amb_op,mcare_amb_ip,mcare_amb_notmatched],axis=0)

# Delete dataframes to recover memory
del mcare_amb_op
del mcare_amb_ip
del mcare_amb_notmatched

#-----------------Calculate Numbers-------------------------#

# Group by states to calc the numerator (number of ecodes per state) and denominator (total trauma cases per state)
proportion_matched_df = matched_df.groupby(['STATE_CD'])[['ind_for_hos_match','num_total_amb_claims']].sum().reset_index()

# Create a new column calculating the proportion
proportion_matched_df['proportion_matched'] = proportion_matched_df['ind_for_hos_match']/proportion_matched_df['num_total_amb_claims']

# Rename states
proportion_matched_df['STATE_CD'] = proportion_matched_df['STATE_CD'].replace({'01':'AL','02':'AK','03':'AZ','04':'AR','05':'CA','06':'CO','07':'CT','08':'DE','09':'DC','10':'FL','11':'GA',
              '12':'HI','13':'ID','14':'IL','15':'IN','16':'IA','17':'KS','18':'KY','19':'LA','20':'ME','21':'MD','22':'MA',
              '23':'MI','24':'MN','25':'MS','26':'MO','27':'MT','28':'NE','29':'NV','30':'NH','31':'NJ','32':'NM','33':'NY',
              '34':'NC','35':'ND','36':'OH','37':'OK','38':'OR','39':'PA','41':'RI','42':'SC','43':'SD','44':'TN','45':'TX',
              '46':'UT','47':'VT','49':'VA','50':'WA','51':'WV','52':'WI','53':'WY'})

# View results
print(proportion_matched_df.head(60))

# Read out to CSV to easily transfer results over to excel
proportion_matched_df.to_csv('/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicare_ab/table_two_mcare_results/HOSP.csv')

######### CRITERIA 4 (ECD-IP): TRAUMA CASES IN AMB CLAIMS MATCHED WITH IP CLAIMS CONTAINING AT LEAST ONE E-CODE ########
# This script calculates the proportion of ambulance claims that are matched with ip claims with at least one e-code   #
# conditional on trauma codes.                                                                                         #
########################################################################################################################

#-----------------------------------------Define function for Medicaid-------------------------------------------------#

# Define function to calculate proportion with at least one e-code
def calc_prop_e_code_ip_criteria(state,mcaid_payment_type):

    # Specify States available for 2011-2014
    states_11_14 = ['CA', 'GA', 'IA', 'LA', 'MI', 'MN', 'MO', 'MS', 'NJ', 'PA', 'SD', 'TN', 'UT', 'VT', 'WV', 'WY']

    # Specify States available for 2011-2013
    states_11_13 = ['AR', 'AZ', 'CT', 'HI', 'IN', 'MA', 'NY', 'OH', 'OK', 'OR', 'WA']

    # Specify States available for 2011-2012
    states_11_12 = ['AL', 'AK', 'CO', 'DC', 'DE', 'FL', 'IL', 'KS', 'KY', 'ME', 'MD', 'MT', 'NE', 'NV', 'NH', 'NM',
                    'NC', 'ND', 'RI', 'SC', 'TX', 'VA', 'WI']

    # Specify States available for 2012-2014
    states_12_14 = ['ID']

    # Due to missing states in some years, we use if/then statement
    if state in states_11_14:

        # Specify empty list for pathways
        file_paths=[]

        # Append 2011-2014 file paths to list
        years=[2011,2012,2013,2014]
        for y in years:
            file_paths.append(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicaid_ffs_mc/{y}/{mcaid_payment_type}_merged_amb_hos_claims_ninetyonedays/{state}/')

    elif state in states_11_13:

        # Specify empty list for pathways
        file_paths=[]

        # Append 2011-2013 file paths to list
        years=[2011,2012,2013]
        for y in years:
            file_paths.append(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicaid_ffs_mc/{y}/{mcaid_payment_type}_merged_amb_hos_claims_ninetyonedays/{state}/')

    elif state in states_11_12:

        # Specify empty list for pathways
        file_paths=[]

        # Append 2011-2012 file paths to list
        years=[2011,2012]
        for y in years:
            file_paths.append(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicaid_ffs_mc/{y}/{mcaid_payment_type}_merged_amb_hos_claims_ninetyonedays/{state}/')

    elif state in states_12_14:

        # Specify empty list for pathways
        file_paths=[]

        # Append 2012-2014 file paths to list
        years=[2012,2013,2014]
        for y in years:
            file_paths.append(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicaid_ffs_mc/{y}/{mcaid_payment_type}_merged_amb_hos_claims_ninetyonedays/{state}/')

    # Specify relevant column to calculate proportion matched
    columns_amb = ['DIAG_CD_1', 'DIAG_CD_2', 'DIAG_CD_3', 'DIAG_CD_4', 'DIAG_CD_5', 'DIAG_CD_6', 'DIAG_CD_7',
                   'DIAG_CD_8', 'DIAG_CD_9','ip_ind']

    # Read in
    amb_hos = dd.read_parquet(file_paths, engine='fastparquet',columns=columns_amb)

    # Convert all NA's to blanks
    for i in range(1,10):
        amb_hos['DIAG_CD_{}'.format(i)] = amb_hos['DIAG_CD_{}'.format(i)].fillna('')

    # Keep only ip claims
    amb_ip = amb_hos[amb_hos['ip_ind']==1]

    # Delete DF to recover memory
    del amb_hos

    #--- Keep trauma ---#

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
    amb_ip[diag_ecode_col] = amb_ip[diag_ecode_col].astype(str)

    # First, we filter based on lst_include_codes, while ignoring lst_ignore_codes.
    amb_ip_trauma = amb_ip.loc[(amb_ip[diag_first_three_cols].applymap(lambda x: x.startswith(tuple(lst_include_codes)) & (~x.startswith(tuple(lst_ignore_codes)))).any(axis='columns'))]

    # Recover Memory
    del amb_ip

    # Second, we obtain our final subset by excluding (with the "~" sign) the claims using the Ecodes defined above (ecode_to_remove)
    amb_ip_trauma = amb_ip_trauma.loc[~(amb_ip_trauma[diag_ecode_col].applymap(lambda x: x.startswith(tuple(ecode_to_remove))).any(axis='columns'))]

    #---Calculate claims with trauma code with at least one e-code---#

    # Convert all to uppercase
    for i in diag_ecode_col:
        amb_ip_trauma[i] = amb_ip_trauma[i].str.upper()

    # Identify if they reported at least one e code
    amb_ip_trauma['ecode_ind'] = amb_ip_trauma[diag_ecode_col].applymap(lambda x: 'E' in x).any(1).astype(int)

    # Generate numbers and store in variables
    num_atleast_one_ecode = amb_ip_trauma['ecode_ind'].sum().compute()
    num_trauma_cases = amb_ip_trauma.shape[0].compute()

    # Print numbers
    print(state, f'{mcaid_payment_type} Proportion of claims with at least one ecode over all trauma cases (ip) \n')
    if (num_trauma_cases > 10) & (num_atleast_one_ecode > 10) | ((num_atleast_one_ecode==0) & (num_trauma_cases > 10)): # Prevent violation of cell supression policy
        print(num_atleast_one_ecode/num_trauma_cases)

        # append numerator and denominator to the empty list defined below if not nan (for t-test between MAX and TAF)
        if mcaid_payment_type in ['ffs']:
            max_ffs_num_eip.append(num_atleast_one_ecode)
            max_ffs_denom_eip.append(num_trauma_cases)
        elif mcaid_payment_type in ['mc']:
            max_mc_num_eip.append(num_atleast_one_ecode)
            max_mc_denom_eip.append(num_trauma_cases)

    else:
        print('nan')

        # append numerator and denominator to the empty list defined below if nan (for t-test between MAX and TAF)
        if mcaid_payment_type in ['ffs']:
            max_ffs_num_eip.append(np.nan)
            max_ffs_denom_eip.append(np.nan)
        elif mcaid_payment_type in ['mc']:
            max_mc_num_eip.append(np.nan)
            max_mc_denom_eip.append(np.nan)

#------------------------------------Run Defined function for Medicaid-------------------------------------------------#

# Specify all States
all_states=['AL','AK','AZ','AR','CA','CO','CT','DC','DE','FL','GA','HI','ID','IL','IN','IA','KS','KY','LA','ME','MD','MA',
            'MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX',
            'UT','VT','VA','WA','WV','WI','WY']

# Create empty list to store numbers (for t-test between MAX and TAF)
max_ffs_num_eip = []
max_mc_num_eip = []
max_ffs_denom_eip = []
max_mc_denom_eip = []

for s in all_states:

    # Run function for FFS
    calc_prop_e_code_ip_criteria(s, 'ffs')

    # Run function for MC
    calc_prop_e_code_ip_criteria(s, 'mc')

# Create empty dictionary. Will be used to create a DF (for t-test between MAX and TAF)
df_dict_eip = {}

# Append each list (and the states) above to dictionary (for t-test between MAX and TAF)
df_dict_eip['max_ffs_num_eip'] = max_ffs_num_eip
df_dict_eip['max_mc_num_eip'] = max_mc_num_eip
df_dict_eip['max_ffs_denom_eip'] = max_ffs_denom_eip
df_dict_eip['max_mc_denom_eip'] = max_mc_denom_eip
df_dict_eip['state'] = all_states

# Create Dataframe from dictionary (for t-test between MAX and TAF)
df_eip = pd.DataFrame.from_dict(df_dict_eip)

# Calculate proportion to double check with exhibit (for t-test between MAX and TAF)
df_eip['prop_ffs'] = df_eip['max_ffs_num_eip']/df_eip['max_ffs_denom_eip']
df_eip['prop_mc'] = df_eip['max_mc_num_eip']/df_eip['max_mc_denom_eip']

# Read out DF that was created from dictionary (for t-test between MAX and TAF)
df_eip.to_csv('/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicaid_ffs_mc/df_for_t_test_max_vs_taf/df_eip_max.csv',index=False,index_label=False)

#___________________________________________Medicare (ECD-IP)__________________________________________________________#
# This script aims to calculate the proportion of claims with a trauma code that have at least one e-code. We read in  #
# each claims that were matched to IP for 11-14. We filtered out the trauma based on the HCUP definition then          #
# calculated the proportions per state.                                                                                #
########################################################################################################################

# Specify years
years=[2011,2012,2013,2014]

#---Read in Ambulance matched with IP---#

# Define empty dictionary to store DFs
dict_df_ip={}

# Read in 11-14 amb matched with ip data into df dictionary
for y in years:
    dict_df_ip[y] = dd.read_parquet(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicare_ab/{y}/merged_amb_hos_claims/ip_merged_amb/',
                               engine='fastparquet',columns=['BENE_RSDNC_SSA_STATE_CD']+['ADMTG_DGNS_CD'] + ['DGNS_{}_CD'.format(i) for i in range(1, 26)] +
                               ['DGNS_E_{}_CD'.format(k) for k in range(1, 13)])

# Define empty list to concat all ip data
list_df_ip=[]

# Append df from dictionary to list
for y in years:
    list_df_ip.append(dict_df_ip[y])

# Concat all data
mcare_amb_ip = dd.concat(list_df_ip,axis=0)

# Delete DF to recover
del dict_df_ip
del list_df_ip

#--- Keep trauma ---#

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

# Define list of first three diagnosis columns (first and second columns may be duplicated according the ResDAC so I examined the first 4 columns)
diag_first_three_cols = ['ADMTG_DGNS_CD'] + ['DGNS_{}_CD'.format(i) for i in range(1, 4)]

# Convert all diagnosis and ecodes columns to string
mcare_amb_ip[diag_ecode_col] = mcare_amb_ip[diag_ecode_col].astype(str)

# First, we filter based on lst_include_codes, while ignoring lst_ignore_codes.
mcare_amb_ip_trauma = mcare_amb_ip.loc[(mcare_amb_ip[diag_first_three_cols].applymap(lambda x: x.startswith(tuple(lst_include_codes)) & (~x.startswith(tuple(lst_ignore_codes)))).any(axis='columns'))]

# Recover Memory
del mcare_amb_ip

# Second, we obtain our final subset by excluding (with the "~" sign) the claims using the Ecodes defined above (ecode_to_remove)
mcare_amb_ip_trauma = mcare_amb_ip_trauma.loc[~(mcare_amb_ip_trauma[diag_ecode_col].applymap(lambda x: x.startswith(tuple(ecode_to_remove))).any(axis='columns'))]

# Convert all to uppercase
for i in diag_ecode_col:
    mcare_amb_ip_trauma[i] = mcare_amb_ip_trauma[i].str.upper()

# Identify if they reported at least one e code
mcare_amb_ip_trauma['ecode_ind'] = mcare_amb_ip_trauma[diag_ecode_col].applymap(lambda x: 'E' in x).any(1).astype(
        int)

#-----------------Calculate Numbers-------------------------#

# Create another columns to calculate the denominator
mcare_amb_ip_trauma['total_trauma_cases']=1

# Group by states to calc the numerator (number of ecodes per state) and denominator (total trauma cases per state)
proportion_ecode_df = mcare_amb_ip_trauma.groupby(['BENE_RSDNC_SSA_STATE_CD'])[['ecode_ind','total_trauma_cases']].sum().reset_index()

# Create a new column calculating the proportion
proportion_ecode_df['proportion_ecode'] = proportion_ecode_df['ecode_ind']/proportion_ecode_df['total_trauma_cases']

# Rename states
proportion_ecode_df['BENE_RSDNC_SSA_STATE_CD'] = proportion_ecode_df['BENE_RSDNC_SSA_STATE_CD'].replace({'01':'AL','02':'AK','03':'AZ','04':'AR','05':'CA','06':'CO','07':'CT','08':'DE','09':'DC','10':'FL','11':'GA',
              '12':'HI','13':'ID','14':'IL','15':'IN','16':'IA','17':'KS','18':'KY','19':'LA','20':'ME','21':'MD','22':'MA',
              '23':'MI','24':'MN','25':'MS','26':'MO','27':'MT','28':'NE','29':'NV','30':'NH','31':'NJ','32':'NM','33':'NY',
              '34':'NC','35':'ND','36':'OH','37':'OK','38':'OR','39':'PA','41':'RI','42':'SC','43':'SD','44':'TN','45':'TX',
              '46':'UT','47':'VT','49':'VA','50':'WA','51':'WV','52':'WI','53':'WY'})

# View results
print(proportion_ecode_df.head(60))

# Read out results
proportion_ecode_df.to_csv('/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicare_ab/table_two_mcare_results/ECD-IP.csv')

######### CRITERIA 5 (ECD-OP): TRAUMA CASES IN AMB CLAIMS MATCHED WITH OP CLAIMS CONTAINING AT LEAST ONE E-CODE ########
# This script calculates the proportion of ambulance that are matched with op claims with at least one e-code          #
# conditional on trauma codes.                                                                                         #
########################################################################################################################

#-----------------------------------------Define function for Medicaid-------------------------------------------------#

# Define function to calculate proportion with at least one e-code
def calc_prop_e_code_op_criteria(state,mcaid_payment_type):

    # Specify States available for 2011-2014
    states_11_14 = ['CA', 'GA', 'IA', 'LA', 'MI', 'MN', 'MO', 'MS', 'NJ', 'PA', 'SD', 'TN', 'UT', 'VT', 'WV', 'WY']

    # Specify States available for 2011-2013
    states_11_13 = ['AR', 'AZ', 'CT', 'HI', 'IN', 'MA', 'NY', 'OH', 'OK', 'OR', 'WA']

    # Specify States available for 2011-2012
    states_11_12 = ['AL', 'AK', 'CO', 'DC', 'DE', 'FL', 'IL', 'KS', 'KY', 'ME', 'MD', 'MT', 'NE', 'NV', 'NH', 'NM',
                    'NC', 'ND', 'RI', 'SC', 'TX', 'VA', 'WI']

    # Specify States available for 2012-2014
    states_12_14 = ['ID']

    # Due to missing states in some years, we use if/then statement
    if state in states_11_14:

        # Specify empty list for pathways
        file_paths=[]

        # Append 2011-2014 file paths to list
        years=[2011,2012,2013,2014]
        for y in years:
            file_paths.append(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicaid_ffs_mc/{y}/{mcaid_payment_type}_merged_amb_hos_claims_ninetyonedays/{state}/')

    elif state in states_11_13:

        # Specify empty list for pathways
        file_paths=[]

        # Append 2011-2013 file paths to list
        years=[2011,2012,2013]
        for y in years:
            file_paths.append(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicaid_ffs_mc/{y}/{mcaid_payment_type}_merged_amb_hos_claims_ninetyonedays/{state}/')

    elif state in states_11_12:

        # Specify empty list for pathways
        file_paths=[]

        # Append 2011-2012 file paths to list
        years=[2011,2012]
        for y in years:
            file_paths.append(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicaid_ffs_mc/{y}/{mcaid_payment_type}_merged_amb_hos_claims_ninetyonedays/{state}/')

    elif state in states_12_14:

        # Specify empty list for pathways
        file_paths=[]

        # Append 2012-2014 file paths to list
        years=[2012,2013,2014]
        for y in years:
            file_paths.append(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicaid_ffs_mc/{y}/{mcaid_payment_type}_merged_amb_hos_claims_ninetyonedays/{state}/')

    # Specify relevant column to calculate proportion matched
    columns_amb = ['DIAG_CD_1', 'DIAG_CD_2','ip_ind']

    # Read in
    amb_hos = dd.read_parquet(file_paths, engine='fastparquet',columns=columns_amb)

    # Convert all NA's to blanks
    for i in range(1,3):
        amb_hos['DIAG_CD_{}'.format(i)] = amb_hos['DIAG_CD_{}'.format(i)] .fillna('')

    # Keep op claims
    amb_op = amb_hos[amb_hos['ip_ind']==0]

    # Delete DF to recover memory
    del amb_hos

    #--- Keep trauma ---#

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
    diag_ecode_col = ['DIAG_CD_{}'.format(i) for i in range(1, 3)]

    # Define list of first two diagnosis columns
    diag_first_two_cols = ['DIAG_CD_{}'.format(i) for i in range(1, 3)]

    # Convert all diagnosis and ecodes columns to string
    amb_op[diag_ecode_col] = amb_op[diag_ecode_col].astype(str)

    # First, we filter based on lst_include_codes, while ignoring lst_ignore_codes.
    amb_op_trauma = amb_op.loc[(amb_op[diag_first_two_cols].applymap(lambda x: x.startswith(tuple(lst_include_codes)) & (~x.startswith(tuple(lst_ignore_codes)))).any(axis='columns'))]

    # Recover Memory
    del amb_op

    # Second, we obtain our final subset by excluding (with the "~" sign) the claims using the Ecodes defined above (ecode_to_remove)
    amb_op_trauma = amb_op_trauma.loc[~(amb_op_trauma[diag_ecode_col].applymap(lambda x: x.startswith(tuple(ecode_to_remove))).any(axis='columns'))]

    #---Calculate claims with trauma code with at least one e-code---#

    # Convert all to uppercase
    for i in diag_ecode_col:
        amb_op_trauma[i] = amb_op_trauma[i].str.upper()

    # Identify if they reported at least one e code
    amb_op_trauma['ecode_ind'] = amb_op_trauma[diag_ecode_col].applymap(lambda x: 'E' in x).any(1).astype(int)

    # Generate numbers and store in variables
    num_atleast_one_ecode = amb_op_trauma['ecode_ind'].sum().compute()
    num_trauma_cases = amb_op_trauma.shape[0].compute()

    # Print numbers
    print(state, f'{mcaid_payment_type} Proportion of claims with at least one ecode over all trauma cases (op) \n')
    if (num_trauma_cases > 10) & (num_atleast_one_ecode > 10) | ((num_atleast_one_ecode==0) & (num_trauma_cases > 10)): # Prevent violation of cell supression policy
        print(num_atleast_one_ecode/num_trauma_cases)

        # append numerator and denominator to the empty list defined below if not nan (for t-test between MAX and TAF)
        if mcaid_payment_type in ['ffs']:
            max_ffs_num_eop.append(num_atleast_one_ecode)
            max_ffs_denom_eop.append(num_trauma_cases)
        elif mcaid_payment_type in ['mc']:
            max_mc_num_eop.append(num_atleast_one_ecode)
            max_mc_denom_eop.append(num_trauma_cases)

    else:
        print('nan')

        # append numerator and denominator to the empty list defined below if nan (for t-test between MAX and TAF)
        if mcaid_payment_type in ['ffs']:
            max_ffs_num_eop.append(np.nan)
            max_ffs_denom_eop.append(np.nan)
        elif mcaid_payment_type in ['mc']:
            max_mc_num_eop.append(np.nan)
            max_mc_denom_eop.append(np.nan)

#------------------------------------Run Defined function for Medicaid-------------------------------------------------#

# Specify all States
all_states=['AL','AK','AZ','AR','CA','CO','CT','DC','DE','FL','GA','HI','ID','IL','IN','IA','KS','KY','LA','ME','MD','MA',
            'MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX',
            'UT','VT','VA','WA','WV','WI','WY']

# Create empty list to store numbers (for t-test between MAX and TAF)
max_ffs_num_eop = []
max_mc_num_eop = []
max_ffs_denom_eop = []
max_mc_denom_eop = []

for s in all_states:

    # Run function for FFS
    calc_prop_e_code_op_criteria(s, 'ffs')

    # Run function for MC
    calc_prop_e_code_op_criteria(s, 'mc')

# Create empty dictionary. Will be used to create a DF (for t-test between MAX and TAF)
df_dict_eop = {}

# Append each list (and the states) above to dictionary (for t-test between MAX and TAF)
df_dict_eop['max_ffs_num_eop'] = max_ffs_num_eop
df_dict_eop['max_mc_num_eop'] = max_mc_num_eop
df_dict_eop['max_ffs_denom_eop'] = max_ffs_denom_eop
df_dict_eop['max_mc_denom_eop'] = max_mc_denom_eop
df_dict_eop['state'] = all_states

# Create Dataframe from dictionary (for t-test between MAX and TAF)
df_eop = pd.DataFrame.from_dict(df_dict_eop)

# Calculate proportion to double check with exhibit (for t-test between MAX and TAF)
df_eop['prop_ffs'] = df_eop['max_ffs_num_eop']/df_eop['max_ffs_denom_eop']
df_eop['prop_mc'] = df_eop['max_mc_num_eop']/df_eop['max_mc_denom_eop']

# Read out DF that was created from dictionary (for t-test between MAX and TAF)
df_eop.to_csv('/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicaid_ffs_mc/df_for_t_test_max_vs_taf/df_eop_max.csv',index=False,index_label=False)

#__________________________________________Medicare (ECD-OP)___________________________________________________________#
# This script aims to calculate the proportion of ambulance claims matched with op claims with a trauma code that have #
# at least one e-code. We read in each claims that were matched to OP for 11-14. We filtered out the trauma based on   #
# the HCUP definition then calculated the proportions per state.                                                       #
########################################################################################################################

# Specify years
years=[2011,2012,2013,2014]

#---Read in Ambulance matched with OP---#

# Define empty dictionary to store DFs
dict_df_op={}

# Read in 11-14 amb matched with op data into df dictionary
for y in years:
    dict_df_op[y] = dd.read_parquet(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicare_ab/{y}/merged_amb_hos_claims/op_merged_amb/',
                               engine='fastparquet',columns=['STATE_CODE','PRVDR_STATE_CD']+['PRNCPAL_DGNS_CD', 'FST_DGNS_E_CD'] +
                               ['ICD_DGNS_CD{}'.format(i) for i in range(1, 26)] + ['ICD_DGNS_E_CD{}'.format(j) for j in range(1, 13)])

# Define empty list to concat all op data
list_df_op=[]

# Append df from dictionary to list
for y in years:
    list_df_op.append(dict_df_op[y])

# Concat all data
mcare_amb_op = dd.concat(list_df_op,axis=0)

# Delete DF to recover
del dict_df_op
del list_df_op

# Keep only claims that did NOT transport patients across state. Only for op
mcare_amb_op_same_state = mcare_amb_op[mcare_amb_op['STATE_CODE']==mcare_amb_op['PRVDR_STATE_CD']]

# Delete dataframe to recover memory
del mcare_amb_op

# Clean dataframe before concatenating
mcare_amb_op_same_state = mcare_amb_op_same_state.drop(['STATE_CODE'],axis=1)

#--- Keep trauma ---#

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
diag_ecode_col = ['PRNCPAL_DGNS_CD', 'FST_DGNS_E_CD'] + ['ICD_DGNS_CD{}'.format(i) for i in range(1, 26)] + ['ICD_DGNS_E_CD{}'.format(j) for j in range(1, 13)]

# Define list of first three diagnosis columns (first and second columns may be duplicated according the ResDAC so I examined the first 4 columns)
diag_first_three_cols = ['PRNCPAL_DGNS_CD'] + ['ICD_DGNS_CD{}'.format(i) for i in range(1, 4)]

# Convert all diagnosis and ecodes columns to string
mcare_amb_op_same_state[diag_ecode_col] = mcare_amb_op_same_state[diag_ecode_col].astype(str)

# First, we filter based on lst_include_codes, while ignoring lst_ignore_codes.
mcare_amb_op_same_state_trauma = mcare_amb_op_same_state.loc[(mcare_amb_op_same_state[diag_first_three_cols].applymap(lambda x: x.startswith(tuple(lst_include_codes)) & (~x.startswith(tuple(lst_ignore_codes)))).any(axis='columns'))]

# Recover Memory
del mcare_amb_op_same_state

# Second, we obtain our final subset by excluding (with the "~" sign) the claims using the Ecodes defined above (ecode_to_remove)
mcare_amb_op_same_state_trauma = mcare_amb_op_same_state_trauma.loc[~(mcare_amb_op_same_state_trauma[diag_ecode_col].applymap(lambda x: x.startswith(tuple(ecode_to_remove))).any(axis='columns'))]

# Convert all to uppercase
for i in diag_ecode_col:
    mcare_amb_op_same_state_trauma[i] = mcare_amb_op_same_state_trauma[i].str.upper()

# Identify if they reported at least one e code
mcare_amb_op_same_state_trauma['ecode_ind'] = mcare_amb_op_same_state_trauma[diag_ecode_col].applymap(lambda x: 'E' in x).any(1).astype(int)

#-----------------Calculate Numbers-------------------------#

# Create another columns to calculate the denominator
mcare_amb_op_same_state_trauma['total_trauma_cases']=1

# Group by states to calc the numerator (number of ecodes per state) and denominator (total trauma cases per state)
proportion_ecode_df = mcare_amb_op_same_state_trauma.groupby(['PRVDR_STATE_CD'])[['ecode_ind','total_trauma_cases']].sum().reset_index()

# Create a new column calculating the proportion
proportion_ecode_df['proportion_ecode'] = proportion_ecode_df['ecode_ind']/proportion_ecode_df['total_trauma_cases']

# Rename states
proportion_ecode_df['PRVDR_STATE_CD'] = proportion_ecode_df['PRVDR_STATE_CD'].replace({'01':'AL','02':'AK','03':'AZ','04':'AR','05':'CA','06':'CO','07':'CT','08':'DE','09':'DC','10':'FL','11':'GA',
              '12':'HI','13':'ID','14':'IL','15':'IN','16':'IA','17':'KS','18':'KY','19':'LA','20':'ME','21':'MD','22':'MA',
              '23':'MI','24':'MN','25':'MS','26':'MO','27':'MT','28':'NE','29':'NV','30':'NH','31':'NJ','32':'NM','33':'NY',
              '34':'NC','35':'ND','36':'OH','37':'OK','38':'OR','39':'PA','41':'RI','42':'SC','43':'SD','44':'TN','45':'TX',
              '46':'UT','47':'VT','49':'VA','50':'WA','51':'WV','52':'WI','53':'WY'})

# View results
print(proportion_ecode_df.head(60))

# Read out results
proportion_ecode_df.to_csv('/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicare_ab/table_two_mcare_results/ECD-OP.csv')














