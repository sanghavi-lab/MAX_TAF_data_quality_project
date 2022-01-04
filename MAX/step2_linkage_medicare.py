#----------------------------------------------------------------------------------------------------------------------#
# Project: Medicaid Data Quality Project
# Authors: Jessy Nguyen
# Last Updated: August 12, 2021
# Description: The goal of this script is to link (1) ambulance claims with mileage information and (2) ambulance
#              claims with hospital claims for each year for Medicare.
#----------------------------------------------------------------------------------------------------------------------#

############################################### IMPORT MODULES #########################################################

# Read in relevant libraries
from datetime import datetime, timedelta
import pandas as pd
import dask.dataframe as dd

############################################# MODULE FOR CLUSTER #######################################################

# Read in libraries to use cluster
from dask.distributed import Client
client = Client('[insert_ip_address_for_cluster]')

######################################### MATCH MILEAGE WITH AMB CLAIMS ################################################
# The following script links the exported ambulance claims with mileage information claims. We do not need to          #
# eliminate rides who traveled out of state since we are using the provider state code from the ambulance claims. We   #
# did not use modifiers when matching since we dropped all individuals with multiple ambulance rides in one day.       #
########################################################################################################################

# Define years
years=[2011,2012,2013,2014]

for y in years:

    #___Read in Mileage info___#

    # Specify columns for Mileage DF
    columns_mi = ['CLM_ID','HCPCS_CD','CLM_THRU_DT','LINE_1ST_EXPNS_DT','LINE_LAST_EXPNS_DT',
                  'LINE_PRCSG_IND_CD','CARR_LINE_MTUS_CNT']

    # Read in carrier line data for the particular year
    df_BCARRL = dd.read_csv(f'/mnt/data/medicare-share/data/{y}/BCARRL/csv/bcarrier_line_k.csv',usecols=columns_mi,sep=',',
                            engine='c', dtype='object', na_filter=False, skipinitialspace=True, low_memory=False)

    # Keep mileage
    mileage_cd = ['A0425']
    payment_allowed_cd = ['A']
    mileage = df_BCARRL.loc[(df_BCARRL['HCPCS_CD'].isin(mileage_cd)) & (df_BCARRL['LINE_PRCSG_IND_CD'].isin(payment_allowed_cd))]

    # Recover memory
    del df_BCARRL

    # Clean DF
    mileage = mileage.drop(['LINE_PRCSG_IND_CD','HCPCS_CD'],axis=1)

    # Convert all to datetime
    mileage['CLM_THRU_DT'] = dd.to_datetime(mileage['CLM_THRU_DT'])
    mileage['LINE_1ST_EXPNS_DT'] = dd.to_datetime(mileage['LINE_1ST_EXPNS_DT'])
    mileage['LINE_LAST_EXPNS_DT'] = dd.to_datetime(mileage['LINE_LAST_EXPNS_DT'])

    # Create column to count number matched
    mileage['ind_for_mi_match'] = 1

    # Convert string to floats
    mileage['CARR_LINE_MTUS_CNT'] = mileage['CARR_LINE_MTUS_CNT'].astype(float)

    #___Read in Amb claims___#

    # Specify columns
    columns_amb = ['CLM_ID','PRVDR_STATE_CD','CLM_THRU_DT','LINE_1ST_EXPNS_DT','LINE_LAST_EXPNS_DT']

    # Read in ambulance claims with only BENE_ID and CLM_THRU_DT
    amb = dd.read_parquet(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicare_ab/{y}/amb_ab/',
                          engine='fastparquet',columns=columns_amb)

    # Remove CLM_ID duplicates
    amb = amb.drop_duplicates(subset=['CLM_ID'], keep = 'last')

    # Add column of consecutive numbers. Needed to drop additional duplicates.
    amb = amb.reset_index(drop=True)
    amb['for_drop_dup'] = amb.reset_index().index

    # Convert all to datetime
    amb['CLM_THRU_DT'] = dd.to_datetime(amb['CLM_THRU_DT'])
    amb['LINE_1ST_EXPNS_DT'] = dd.to_datetime(amb['LINE_1ST_EXPNS_DT'])
    amb['LINE_LAST_EXPNS_DT'] = dd.to_datetime(amb['LINE_LAST_EXPNS_DT'])

    # Merge Amb with Mileage
    merge_amb_mi = dd.merge(amb,mileage,on=['CLM_ID','CLM_THRU_DT','LINE_1ST_EXPNS_DT','LINE_LAST_EXPNS_DT'], how='left')

    # Recover Memory
    del amb
    del mileage

    # Drop all duplicates
    merge_amb_mi = merge_amb_mi.drop_duplicates(subset=['CLM_ID','CLM_THRU_DT','LINE_1ST_EXPNS_DT','LINE_LAST_EXPNS_DT',
                                                        'for_drop_dup'], keep = 'last')

    # Export to parquet
    merge_amb_mi.to_parquet(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicare_ab/{y}/ab_merged_amb_mileage/',
                            compression='gzip', engine='fastparquet')

######################################### MATCH IP AND OP WITH AMB CLAIMS ##############################################
# The following script links the exported ambulance claims with hospital claims. We linked with IP same day, next day, #
# and the following day first then repeat the process with OP. For individuals who had ambulance rides on December 31, #
# we made sure to link the following year. We will be sure to eliminate individuals who were transported out of state  #
# before calculating the proportion matched.                                                                           #
########################################################################################################################

#_______________Matched with IP same day, plus one, plus two________________#

# Specify Years
years=[2011,2012,2013,2014]

# Loop for matching with IP
for y in years:

    #___Import Amb___#

    # Specify columns
    columns_amb = ['CLM_ID','BENE_ID','CLM_THRU_DT','CLM_FROM_DT','STATE_CODE', 'BENE_BIRTH_DT', 'SEX_IDENT_CD',
                   'RTI_RACE_CD','HCPCS_1ST_MDFR_CD','HCPCS_2ND_MDFR_CD','VALID_DEATH_DT_SW','BENE_DEATH_DT']

    # Read in ambulance claims with only BENE_ID and CLM_THRU_DT. Note: CLM_THRU_DT was converted to datetime already when exported.
    amb = dd.read_parquet(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicare_ab/{y}/amb_ab/',engine='fastparquet',columns=columns_amb)

    # Remove CLM_ID duplicates
    amb = amb.drop_duplicates(subset=['CLM_ID'], keep = 'last')

    # Clean DF
    amb = amb.drop(['CLM_ID'],axis=1)

    #___Import IP___#

    # Specify years with available data in the following year
    years_with_next_year_data = [2011,2012,2013]

    # Specify columns
    columns_ip = ['BENE_ID','ADMSN_DT','DSCHRG_DT', 'BENE_RSDNC_SSA_STATE_CD', 'BENE_DSCHRG_STUS_CD','ADMTG_DGNS_CD'] + \
                 ['DGNS_{}_CD'.format(i) for i in range(1, 26)] + ['DGNS_E_{}_CD'.format(k) for k in range(1, 13)]

    # Loop to read in IP data for the following year and concat with data from same year.
    if y in years_with_next_year_data:

        # Read in IP in the same year. Raw IP is store in trauma center folder
        ip_sameyear = dd.read_parquet(f'/mnt/labshares/sanghavi-lab/Jessy/data/trauma_center_project/ip/{y}/parquet/',
                                      engine='fastparquet',columns=columns_ip)

        # Read in IP the following year. Need this since patients may be admitted on Jan 1st of the next year.
        ip_nextyear = dd.read_parquet(f'/mnt/labshares/sanghavi-lab/Jessy/data/trauma_center_project/ip/{y+1}/parquet/',
                                          engine='fastparquet',columns=columns_ip)

        # Keep only individuals who were admitted on Jan 1st/2nd
        ip_nextyear = ip_nextyear[(ip_nextyear['ADMSN_DT'].dt.month==1)&((ip_nextyear['ADMSN_DT'].dt.day==1)|(ip_nextyear['ADMSN_DT'].dt.day==2))]

        # Concat ip_sameyear and ip_nextyear
        ip = dd.concat([ip_sameyear,ip_nextyear],axis=0)

        # Delete DFs to recover memory
        del ip_nextyear
        del ip_sameyear

    else:

        # Read in IP claims
        ip = dd.read_parquet(f'/mnt/labshares/sanghavi-lab/Jessy/data/trauma_center_project/ip/{y}/parquet/',
                             engine='fastparquet',columns=columns_ip)

    # Create indicator for ip
    ip['ip_ind'] = 1

    # Add columns of one's in IP DF to filter out unmatched
    ip['ind_for_hos_match'] = 1

    # Create label matched on first day
    ip['which_day_matched_on'] = 'matched_same_day'

    #______________Match with IP Same Day__________________#

    # Add column of consecutive numbers. Needed to drop additional duplicates in IP
    amb = amb.reset_index(drop=True)
    amb['for_drop_dup'] = amb.reset_index().index

    # Merged amb with ip. Keep all of ambulance claims
    amb_merge_ip =  dd.merge(ip,amb,left_on=['BENE_ID','ADMSN_DT'],right_on=['BENE_ID','CLM_THRU_DT'],
                             suffixes=['_ip','_amb'],how='right')

    # Recover memory
    del amb

    # Drop all duplicates due to input errors
    amb_merge_ip = amb_merge_ip.drop_duplicates(subset=['BENE_ID','CLM_THRU_DT','for_drop_dup'], keep = 'last')

    # Filter out those that matched on same day
    amb_merge_ip_matched = amb_merge_ip[amb_merge_ip['ind_for_hos_match']==1]

    # Filter out those that did not match on same day
    amb_merge_ip_notmatched = amb_merge_ip[amb_merge_ip['ind_for_hos_match'].isna()]

    # Recover Memory
    del amb_merge_ip

    # Clean DF
    amb_merge_ip_matched = amb_merge_ip_matched.drop(['for_drop_dup'],axis=1)
    amb_merge_ip_notmatched = amb_merge_ip_notmatched.drop(['ADMSN_DT','DSCHRG_DT', 'BENE_RSDNC_SSA_STATE_CD', 'BENE_DSCHRG_STUS_CD',
                    'ADMTG_DGNS_CD'] + ['DGNS_{}_CD'.format(i) for i in range(1, 26)] + ['DGNS_E_{}_CD'.format(k) for k in range(1, 13)]+
                    ['ip_ind', 'which_day_matched_on','ind_for_hos_match','for_drop_dup'],axis=1)

    #______________Match with IP Plus One__________________#

    # Add one day to amb date
    amb_merge_ip_notmatched['CLM_THRU_DT_PLUS_ONE'] = amb_merge_ip_notmatched['CLM_THRU_DT'] + timedelta(days=1)

    # Create label matched on first day
    ip['which_day_matched_on'] = 'matched_plus_one'

    # Add column of consecutive numbers. Needed to drop additional duplicates in IP
    amb_merge_ip_notmatched = amb_merge_ip_notmatched.reset_index(drop=True)
    amb_merge_ip_notmatched['for_drop_dup'] = amb_merge_ip_notmatched.reset_index().index

    # Merged amb with ip. Keep all of ambulance claims
    amb_merge_ip_plus_one = dd.merge(ip,amb_merge_ip_notmatched,left_on=['BENE_ID','ADMSN_DT'],right_on=['BENE_ID','CLM_THRU_DT_PLUS_ONE'],
                                     suffixes=['_ip','_amb'],how='right')

    # Recover memory
    del amb_merge_ip_notmatched

    # Drop all duplicates due to input errors
    amb_merge_ip_plus_one = amb_merge_ip_plus_one.drop_duplicates(subset=['BENE_ID','ADMSN_DT','for_drop_dup'], keep = 'last')

    # Filter out those that matched
    amb_merge_ip_matched_plus_one = amb_merge_ip_plus_one[amb_merge_ip_plus_one['ind_for_hos_match']==1]

    # Filter out those that did not match
    amb_merge_ip_notmatched_plus_one = amb_merge_ip_plus_one[amb_merge_ip_plus_one['ind_for_hos_match'].isna()]

    # Recover Memory
    del amb_merge_ip_plus_one

    # Clean DF
    amb_merge_ip_matched_plus_one = amb_merge_ip_matched_plus_one.drop(['CLM_THRU_DT_PLUS_ONE','for_drop_dup'],axis=1)
    amb_merge_ip_notmatched_plus_one = amb_merge_ip_notmatched_plus_one.drop(['ADMSN_DT','DSCHRG_DT','CLM_THRU_DT_PLUS_ONE', 'BENE_RSDNC_SSA_STATE_CD', 'BENE_DSCHRG_STUS_CD',
                    'ADMTG_DGNS_CD'] + ['DGNS_{}_CD'.format(i) for i in range(1, 26)] + ['DGNS_E_{}_CD'.format(k) for k in range(1, 13)]+
                    ['ip_ind', 'which_day_matched_on','ind_for_hos_match','for_drop_dup'],axis=1)

    #______________Match with IP Plus Two__________________#

    # Add two days to amb date
    amb_merge_ip_notmatched_plus_one['CLM_THRU_DT_PLUS_TWO'] = amb_merge_ip_notmatched_plus_one['CLM_THRU_DT'] + timedelta(days=2)

    # Create label matched on second day
    ip['which_day_matched_on'] = 'matched_plus_two'

    # Add column of consecutive numbers. Needed to drop additional duplicates in IP
    amb_merge_ip_notmatched_plus_one = amb_merge_ip_notmatched_plus_one.reset_index(drop=True)
    amb_merge_ip_notmatched_plus_one['for_drop_dup'] = amb_merge_ip_notmatched_plus_one.reset_index().index

    # Merged amb with ip. Keep all of ambulance claims
    amb_merge_ip_plus_two =  dd.merge(ip,amb_merge_ip_notmatched_plus_one,left_on=['BENE_ID','ADMSN_DT'],right_on=['BENE_ID','CLM_THRU_DT_PLUS_TWO'],
                                      suffixes=['_ip','_amb'],how='right')

    # Recover memory
    del amb_merge_ip_notmatched_plus_one

    # Drop all duplicates due to input errors
    amb_merge_ip_plus_two = amb_merge_ip_plus_two.drop_duplicates(subset=['BENE_ID','ADMSN_DT','for_drop_dup'], keep = 'last')

    # Filter out those that matched
    amb_merge_ip_matched_plus_two = amb_merge_ip_plus_two[amb_merge_ip_plus_two['ind_for_hos_match']==1]

    # Filter out those that did not match
    amb_merge_ip_notmatched_plus_two = amb_merge_ip_plus_two[amb_merge_ip_plus_two['ind_for_hos_match'].isna()]

    # Recover Memory
    del amb_merge_ip_plus_two

    # Clean DF
    amb_merge_ip_matched_plus_two = amb_merge_ip_matched_plus_two.drop(['CLM_THRU_DT_PLUS_TWO','for_drop_dup'],axis=1)
    amb_merge_ip_notmatched_plus_two = amb_merge_ip_notmatched_plus_two.drop(['ADMSN_DT','DSCHRG_DT','CLM_THRU_DT_PLUS_TWO', 'BENE_RSDNC_SSA_STATE_CD', 'BENE_DSCHRG_STUS_CD',
                    'ADMTG_DGNS_CD'] + ['DGNS_{}_CD'.format(i) for i in range(1, 26)] + ['DGNS_E_{}_CD'.format(k) for k in range(1, 13)]+
                    ['ip_ind', 'which_day_matched_on','ind_for_hos_match','for_drop_dup'],axis=1)

    #------Concat and export matched and not matched------#

    # Concat all matched claims
    ip_matched_concat = dd.concat([amb_merge_ip_matched,amb_merge_ip_matched_plus_one,amb_merge_ip_matched_plus_two],axis=0)

    # Export matched and not matched claims
    ip_matched_concat.to_parquet(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicare_ab/{y}/merged_amb_hos_claims/ip_merged_amb/',
                                 compression='gzip', engine='fastparquet')
    amb_merge_ip_notmatched_plus_two.to_parquet(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicare_ab/{y}/merged_amb_hos_claims/claims_notmatched_with_ip/',
                                                compression='gzip', engine='fastparquet')

#_______________Matched with OP same day, plus one, plus two________________#

# Specify Years
years=[2011,2012,2013,2014]

# Loop for matching with OP
for y in years:

    #___Import Amb___#

    # Specify columns
    columns_amb = ['BENE_ID','CLM_THRU_DT','STATE_CODE', 'BENE_BIRTH_DT', 'SEX_IDENT_CD', 'RTI_RACE_CD','HCPCS_1ST_MDFR_CD','HCPCS_2ND_MDFR_CD']

    # Read in ambulance claims that were NOT matched with OP
    amb = dd.read_parquet(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicare_ab/{y}/merged_amb_hos_claims/claims_notmatched_with_ip/',
                          engine='fastparquet',columns=columns_amb)

    #___Import OP___#

    # Specify years with available data in the following year
    years_with_next_year_data = [2011,2012,2013]

    # Specify columns
    columns_op = ['BENE_ID','CLM_FROM_DT','PRVDR_STATE_CD','PTNT_DSCHRG_STUS_CD', 'PRNCPAL_DGNS_CD', 'FST_DGNS_E_CD'] + \
                  ['ICD_DGNS_CD{}'.format(i) for i in range(1, 26)] + ['ICD_DGNS_E_CD{}'.format(j) for j in range(1, 13)]

    # Loop to read in IP data for the following year and concat with data from same year.
    if y in years_with_next_year_data:

        # Read in IP in the same year. Raw IP is store in trauma center folder
        op_sameyear = dd.read_parquet(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicare_ab/{y}/op_subset/',
                                      engine='fastparquet',columns=columns_op)

        # Read in IP the following year. Need this since patients may be admitted on Jan 1st of the next year.
        op_nextyear = dd.read_parquet(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicare_ab/{y+1}/op_subset/',
                                          engine='fastparquet',columns=columns_op)

        # Keep only individuals who were admitted on Jan 1st/2nd
        op_nextyear = op_nextyear[(op_nextyear['CLM_FROM_DT'].dt.month==1)&((op_nextyear['CLM_FROM_DT'].dt.day==1)|(op_nextyear['CLM_FROM_DT'].dt.day==2))]

        # Concat ip_sameyear and ip_nextyear
        op = dd.concat([op_sameyear,op_nextyear],axis=0)

        # Delete DFs to recover memory
        del op_sameyear
        del op_nextyear

    else:

        # Read in OP claims
        op = dd.read_parquet(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicare_ab/{y}/op_subset/',
                                      engine='fastparquet',columns=columns_op)

    # Create indicator for op
    op['ip_ind'] = 0

    #______________Match with OP Same Day__________________#

    # Create label matched on first day
    op['which_day_matched_on'] = 'matched_same_day'

    # Add columns of one's to filter out unmatched
    op['ind_for_hos_match'] = 1

    # Add column of consecutive numbers. Needed to drop additional duplicates
    amb = amb.reset_index(drop=True)
    amb['for_drop_dup'] = amb.reset_index().index

    # Merged amb with op. Keep all of ambulance claims
    amb_merge_op = dd.merge(op,amb,left_on=['BENE_ID','CLM_FROM_DT'],right_on=['BENE_ID','CLM_THRU_DT'],suffixes=['_op','_amb'],how='right')

    # Recover memory
    del amb

    # Drop all duplicates
    amb_merge_op = amb_merge_op.drop_duplicates(subset=['BENE_ID','CLM_THRU_DT','for_drop_dup'], keep = 'last')

    # Filter out those that matched on same day
    amb_merge_op_matched = amb_merge_op[amb_merge_op['ind_for_hos_match']==1]

    # Filter out those that did not match on same day
    amb_merge_op_notmatched = amb_merge_op[amb_merge_op['ind_for_hos_match'].isna()]

    # Recover Memory
    del amb_merge_op

    # Clean DF
    amb_merge_op_matched = amb_merge_op_matched.drop(['for_drop_dup'],axis=1)
    amb_merge_op_notmatched = amb_merge_op_notmatched.drop(['CLM_FROM_DT','PRVDR_STATE_CD','PTNT_DSCHRG_STUS_CD', 'PRNCPAL_DGNS_CD', 'FST_DGNS_E_CD'] + \
                  ['ICD_DGNS_CD{}'.format(i) for i in range(1, 26)] + ['ICD_DGNS_E_CD{}'.format(j) for j in range(1, 13)]+
                    ['ip_ind', 'which_day_matched_on','ind_for_hos_match','for_drop_dup'],axis=1)

    #______________Match with OP Plus One__________________#

    # Add one day to CLM_THRU_DT Date
    amb_merge_op_notmatched['CLM_THRU_DT_PLUS_ONE'] = amb_merge_op_notmatched['CLM_THRU_DT'] + timedelta(days=1)

    # Create label matched
    op['which_day_matched_on'] = 'matched_plus_one'

    # Add column of consecutive numbers. Needed to drop additional duplicates in OP
    amb_merge_op_notmatched = amb_merge_op_notmatched.reset_index(drop=True)
    amb_merge_op_notmatched['for_drop_dup'] = amb_merge_op_notmatched.reset_index().index

    # Merged amb with op. Keep all of ambulance claims
    amb_merge_op_plus_one = dd.merge(op,amb_merge_op_notmatched,left_on=['BENE_ID','CLM_FROM_DT'],right_on=['BENE_ID','CLM_THRU_DT_PLUS_ONE'],
                                      suffixes=['_op','_amb'],how='right')

    # Recover memory
    del amb_merge_op_notmatched

    # Drop all duplicates
    amb_merge_op_plus_one = amb_merge_op_plus_one.drop_duplicates(subset=['BENE_ID','CLM_FROM_DT','for_drop_dup'], keep = 'last')

    # Filter out those that matched on same day
    amb_merge_op_matched_plus_one = amb_merge_op_plus_one[amb_merge_op_plus_one['ind_for_hos_match']==1]

    # Filter out those that did not match on same day
    amb_merge_op_notmatched_plus_one = amb_merge_op_plus_one[amb_merge_op_plus_one['ind_for_hos_match'].isna()]

    # Recover Memory
    del amb_merge_op_plus_one

    # Clean DF
    amb_merge_op_matched_plus_one = amb_merge_op_matched_plus_one.drop(['CLM_THRU_DT_PLUS_ONE','for_drop_dup'],axis=1)
    amb_merge_op_notmatched_plus_one = amb_merge_op_notmatched_plus_one.drop(['CLM_THRU_DT_PLUS_ONE','CLM_FROM_DT','PRVDR_STATE_CD','PTNT_DSCHRG_STUS_CD', 'PRNCPAL_DGNS_CD', 'FST_DGNS_E_CD'] + \
                  ['ICD_DGNS_CD{}'.format(i) for i in range(1, 26)] + ['ICD_DGNS_E_CD{}'.format(j) for j in range(1, 13)]+
                    ['ip_ind', 'which_day_matched_on','ind_for_hos_match','for_drop_dup'],axis=1)

    #______________Match with OP Plus Two__________________#

    # Add two days to CLM_THRU_DT Date
    amb_merge_op_notmatched_plus_one['CLM_THRU_DT_PLUS_TWO'] = amb_merge_op_notmatched_plus_one['CLM_THRU_DT'] + timedelta(days=2)

    # Create label matched on first day
    op['which_day_matched_on'] = 'matched_plus_two'

    # Add column of consecutive numbers. Needed to drop additional duplicates
    amb_merge_op_notmatched_plus_one = amb_merge_op_notmatched_plus_one.reset_index(drop=True)
    amb_merge_op_notmatched_plus_one['for_drop_dup'] = amb_merge_op_notmatched_plus_one.reset_index().index

    # Merged amb with op. Keep all of ambulance claims
    amb_merge_op_plus_two =  dd.merge(op,amb_merge_op_notmatched_plus_one,left_on=['BENE_ID','CLM_FROM_DT'],
                                      right_on=['BENE_ID','CLM_THRU_DT_PLUS_TWO'],suffixes=['_op','_amb'],how='right')

    # Recover memory
    del amb_merge_op_notmatched_plus_one

    # Drop all duplicates due to input errors
    amb_merge_op_plus_two = amb_merge_op_plus_two.drop_duplicates(subset=['BENE_ID','CLM_FROM_DT','for_drop_dup'], keep = 'last')

    # Filter out those that matched
    amb_merge_op_matched_plus_two = amb_merge_op_plus_two[amb_merge_op_plus_two['ind_for_hos_match']==1]

    # Filter out those that did not match
    amb_merge_op_notmatched_plus_two = amb_merge_op_plus_two[amb_merge_op_plus_two['ind_for_hos_match'].isna()]

    # Recover Memory
    del amb_merge_op_plus_two

    # Clean DF
    amb_merge_op_matched_plus_two = amb_merge_op_matched_plus_two.drop(['CLM_THRU_DT_PLUS_TWO','for_drop_dup'],axis=1)
    amb_merge_op_notmatched_plus_two = amb_merge_op_notmatched_plus_two.drop(['CLM_THRU_DT_PLUS_TWO','CLM_FROM_DT','PRVDR_STATE_CD','PTNT_DSCHRG_STUS_CD', 'PRNCPAL_DGNS_CD', 'FST_DGNS_E_CD'] + \
                  ['ICD_DGNS_CD{}'.format(i) for i in range(1, 26)] + ['ICD_DGNS_E_CD{}'.format(j) for j in range(1, 13)]+
                    ['ip_ind', 'which_day_matched_on','ind_for_hos_match','for_drop_dup'],axis=1)

    #--------Concat amb matched with op----------#

    # Concat all matched claims
    op_matched_concat = dd.concat([amb_merge_op_matched,amb_merge_op_matched_plus_one,amb_merge_op_matched_plus_two],axis=0)

    # Export matched and not matched claims
    op_matched_concat.to_parquet(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicare_ab/{y}/merged_amb_hos_claims/op_merged_amb/',
                                 compression='gzip', engine='fastparquet')
    amb_merge_op_notmatched_plus_two.to_parquet(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicare_ab/{y}/merged_amb_hos_claims/amb_claims_notmatched/',
                                                compression='gzip', engine='fastparquet')

















