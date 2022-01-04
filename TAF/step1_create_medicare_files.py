#----------------------------------------------------------------------------------------------------------------------#
# Project: Medicaid Data Quality Project
# Authors: Jessy Nguyen
# Last Updated: August 12, 2021
# Description: This script exports the inpatient, ambulance, and outpatient claims for 2016. Mileage information will be
#              extracted from the carrier when we eventually link the ambulance claims with mileage.
#----------------------------------------------------------------------------------------------------------------------#

################################################# IMPORT MODULES #######################################################

# Read in relevant libraries
from datetime import datetime, timedelta
import pandas as pd
from pandas.tseries.offsets import MonthEnd
import dask.dataframe as dd

############################################## MODULE FOR CLUSTER ######################################################

# Read in libraries to use cluster. Be sure to specify the correct ip address
from dask.distributed import Client
client = Client('[insert_ip_address_for_cluster]')

###################################### CREATE INPATIENT DATA FROM MEDICARE #############################################
# MedPAR has both skilled nursing facilities and IP claims. This script filters only IP claims in order to match with  #
# the ambulance claims.                                                                                                #
########################################################################################################################

# Specify years to output files
years=[2016]

# Loop function for each year
for y in years:

    # Define columns from MedPAR
    medpar_columns = ['BENE_ID','ADMSN_DT', 'BENE_AGE_CNT', 'BENE_SEX_CD','BENE_RSDNC_SSA_STATE_CD','MEDPAR_ID',
                      'BENE_RSDNC_SSA_CNTY_CD','BENE_MLG_CNTCT_ZIP_CD','PRVDR_NUM','ER_CHRG_AMT','DSCHRG_DT',
                      'BENE_PRMRY_PYR_AMT', 'ORG_NPI_NUM', 'IP_ADMSN_TYPE_CD', 'BENE_DSCHRG_STUS_CD','ICU_IND_CD',
                      'BENE_IP_DDCTBL_AMT', 'BENE_PTA_COINSRNC_AMT', 'BLOOD_PT_FRNSH_QTY', 'DRG_CD', 'DRG_OUTLIER_STAY_CD',
                      'SRC_IP_ADMSN_CD', 'UTLZTN_DAY_CNT', 'BENE_BLOOD_DDCTBL_AMT', 'ADMTG_DGNS_CD','SS_LS_SNF_IND_CD','GHO_PD_CD'] + \
                     ['DGNS_{}_CD'.format(i) for i in range(1, 26)] + \
                     ['POA_DGNS_{}_IND_CD'.format(j) for j in range(1, 26)] + \
                     ['DGNS_E_{}_CD'.format(k) for k in range(1, 13)] + ['POA_DGNS_E_{}_IND_CD'.format(l) for l in range(1, 13)]

    # Read in data from MedPAR
    medpar_df = dd.read_csv(f'/mnt/data/medicare-share/data/{y}/MEDPAR/csv/medpar_{y}.csv',usecols=medpar_columns,sep=',', engine='c',
                            dtype='object', na_filter=False, skipinitialspace=True, low_memory=False)

    # Keep only IP claims (i.e. excludes nursing homes)
    ip_df = medpar_df[(medpar_df['SS_LS_SNF_IND_CD']!='N')]

    # Delete medpar_df
    del medpar_df

    # Convert to datetime
    ip_df['ADMSN_DT'] = dd.to_datetime(ip_df['ADMSN_DT'])

    # Read out IP data
    ip_df.to_parquet(f'/mnt/labshares/sanghavi-lab/Jessy/data/trauma_center_project/ip/{y}/parquet/', compression='gzip', engine='fastparquet') # Save in trauma center project folder for now so I won't have to create duplicates for the ip claims

############################################# Create Ambulance Claims ##################################################
# This script will remove the ambulance claims with multiple rides in one day and keep those who are in Medicare Parts #
# A and B.                                                                                                             #
########################################################################################################################

# Specify years to output files (here we are only using 2016 for now)
years=[2016]

# Loop function to loop over the years defined above (here we are only using 2016 for now)
for y in years:

    # Identify all columns needed in the carrier line file
    columns_BCARRL = ['BENE_ID','CLM_ID','PRVDR_STATE_CD','CLM_THRU_DT','HCPCS_CD','HCPCS_1ST_MDFR_CD',
                      'HCPCS_2ND_MDFR_CD','LINE_1ST_EXPNS_DT','LINE_LAST_EXPNS_DT','LINE_PRCSG_IND_CD']

    # Read in carrier line and header data for the particular year
    df_BCARRL = dd.read_csv(f'/mnt/data/medicare-share/data/{y}/BCARRL/csv/bcarrier_line_k.csv',usecols=columns_BCARRL,sep=',',
                            engine='c', dtype='object', na_filter=False, skipinitialspace=True, low_memory=False)

    # Keep emergency ambulance rides
    em_ambulance_cd = ['A0427', 'A0429', 'A0433']
    df_BCARRL = df_BCARRL.loc[(df_BCARRL['HCPCS_CD'].isin(em_ambulance_cd)) & (df_BCARRL['LINE_PRCSG_IND_CD']=='A')]

    # Read in BCARRB to obtain claim from date and denial code
    df_BCARRB = dd.read_csv(f'/mnt/data/medicare-share/data/{y}/BCARRB/csv/bcarrier_claims_k.csv',usecols=['CLM_ID','CLM_FROM_DT','CARR_CLM_PMT_DNL_CD'],sep=',',
                            engine='c', dtype='object', na_filter=False, skipinitialspace=True, low_memory=False)

    # Merge BCARRL with BCARRB using claim ID
    df_BCARRL_BCARRB = dd.merge(df_BCARRL,df_BCARRB,on=['CLM_ID'],how='inner')

    # Delete dataframes
    del df_BCARRL
    del df_BCARRB

    # Keep claims where claim was NOT denied
    df_BCARRL_BCARRB = df_BCARRL_BCARRB[df_BCARRL_BCARRB['CARR_CLM_PMT_DNL_CD'] != '0']

    # Identify all columns needed in the MBSF
    columns_MBSF = ['BENE_ID','STATE_CODE','BENE_BIRTH_DT','SEX_IDENT_CD','VALID_DEATH_DT_SW','BENE_DEATH_DT','RTI_RACE_CD'] +\
                   ['MDCR_ENTLMT_BUYIN_IND_0{}'.format(i) for i in range(1,10)] + ['MDCR_ENTLMT_BUYIN_IND_{}'.format(i) for i in range(10,13)]

    # Read in MBSF
    df_MBSF = dd.read_csv(f'/mnt/data/medicare-share/data/{y}/MBSFABCD/csv/mbsf_abcd_summary.csv',sep=',', engine='c',
                          dtype='object', na_filter=False,skipinitialspace=True, low_memory=False,usecols=columns_MBSF)

    # Merge Personal Summary with Carrier file
    carrier_ps_merge = dd.merge(df_BCARRL_BCARRB,df_MBSF,on=['BENE_ID'],how='left')

    # Recover memory
    del df_BCARRL_BCARRB
    del df_MBSF

    # Convert claim thru date to datetime
    carrier_ps_merge['CLM_THRU_DT'] = dd.to_datetime(carrier_ps_merge['CLM_THRU_DT'])

    # Filter out Part A and Part B
    carrier_ps_merge = carrier_ps_merge[((carrier_ps_merge['CLM_THRU_DT'].dt.month==1) & (carrier_ps_merge['MDCR_ENTLMT_BUYIN_IND_01']=='3')|(carrier_ps_merge['MDCR_ENTLMT_BUYIN_IND_01']=='C')) |
                                        ((carrier_ps_merge['CLM_THRU_DT'].dt.month==2) & (carrier_ps_merge['MDCR_ENTLMT_BUYIN_IND_02']=='3')|(carrier_ps_merge['MDCR_ENTLMT_BUYIN_IND_02']=='C')) |
                                        ((carrier_ps_merge['CLM_THRU_DT'].dt.month==3) & (carrier_ps_merge['MDCR_ENTLMT_BUYIN_IND_03']=='3')|(carrier_ps_merge['MDCR_ENTLMT_BUYIN_IND_03']=='C')) |
                                        ((carrier_ps_merge['CLM_THRU_DT'].dt.month==4) & (carrier_ps_merge['MDCR_ENTLMT_BUYIN_IND_04']=='3')|(carrier_ps_merge['MDCR_ENTLMT_BUYIN_IND_04']=='C')) |
                                        ((carrier_ps_merge['CLM_THRU_DT'].dt.month==5) & (carrier_ps_merge['MDCR_ENTLMT_BUYIN_IND_05']=='3')|(carrier_ps_merge['MDCR_ENTLMT_BUYIN_IND_05']=='C')) |
                                        ((carrier_ps_merge['CLM_THRU_DT'].dt.month==6) & (carrier_ps_merge['MDCR_ENTLMT_BUYIN_IND_06']=='3')|(carrier_ps_merge['MDCR_ENTLMT_BUYIN_IND_06']=='C')) |
                                        ((carrier_ps_merge['CLM_THRU_DT'].dt.month==7) & (carrier_ps_merge['MDCR_ENTLMT_BUYIN_IND_07']=='3')|(carrier_ps_merge['MDCR_ENTLMT_BUYIN_IND_07']=='C')) |
                                        ((carrier_ps_merge['CLM_THRU_DT'].dt.month==8) & (carrier_ps_merge['MDCR_ENTLMT_BUYIN_IND_08']=='3')|(carrier_ps_merge['MDCR_ENTLMT_BUYIN_IND_08']=='C')) |
                                        ((carrier_ps_merge['CLM_THRU_DT'].dt.month==9) & (carrier_ps_merge['MDCR_ENTLMT_BUYIN_IND_09']=='3')|(carrier_ps_merge['MDCR_ENTLMT_BUYIN_IND_09']=='C')) |
                                        ((carrier_ps_merge['CLM_THRU_DT'].dt.month==10) & (carrier_ps_merge['MDCR_ENTLMT_BUYIN_IND_10']=='3')|(carrier_ps_merge['MDCR_ENTLMT_BUYIN_IND_10']=='C')) |
                                        ((carrier_ps_merge['CLM_THRU_DT'].dt.month==11) & (carrier_ps_merge['MDCR_ENTLMT_BUYIN_IND_11']=='3')|(carrier_ps_merge['MDCR_ENTLMT_BUYIN_IND_11']=='C')) |
                                        ((carrier_ps_merge['CLM_THRU_DT'].dt.month==12) & (carrier_ps_merge['MDCR_ENTLMT_BUYIN_IND_12']=='3')|(carrier_ps_merge['MDCR_ENTLMT_BUYIN_IND_12']=='C'))]

    # Clean DF
    carrier_ps_merge = carrier_ps_merge.drop(['MDCR_ENTLMT_BUYIN_IND_0{}'.format(i) for i in range(1,10)],axis=1)
    carrier_ps_merge = carrier_ps_merge.drop(['MDCR_ENTLMT_BUYIN_IND_{}'.format(i) for i in range(10,13)],axis=1)

    #___________________________Drop Beneficiaries with multiple rides in one day______________________________________#

    # Create new column to count
    carrier_ps_merge['NUM_OF_RIDES'] = 1

    # Group by to see if there were multiple trips in one day for one beneficiary
    carrier_ps_merge_multirides = carrier_ps_merge.groupby(['BENE_ID','CLM_THRU_DT'])['NUM_OF_RIDES'].sum().to_frame().reset_index()

    # Merge dataset back to original ambulance claims in order to filter out those with multiple rides
    amb_merge_multirides = dd.merge(carrier_ps_merge,carrier_ps_merge_multirides, on=['BENE_ID','CLM_THRU_DT'],
                                    suffixes=['_original','_multi'] ,how = 'left')

    # Delete dataframes
    del carrier_ps_merge_multirides
    del carrier_ps_merge

    # Filter those with only 1 ride per day using NUM_OF_RIDES column from the groupby dataframe
    amb_oneride = amb_merge_multirides[amb_merge_multirides['NUM_OF_RIDES_multi']==1]

    # Recover memory
    del amb_merge_multirides

    # Clean Data
    amb_oneride = amb_oneride.drop(['NUM_OF_RIDES_original','NUM_OF_RIDES_multi'],axis=1)

    # Read out Ambulance claims
    amb_oneride.to_parquet(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicare_ab/{y}/amb_ab/', compression='gzip', engine='fastparquet')

#################################### Creating Medicare Data for subset of OP ###########################################
# This script creates a subset of OP claims that matched with ambulance claims. This is necessary since Medicare OP is #
# very large.                                                                                                          #
########################################################################################################################

# Specify years (only using 2016 for now)
years=[2016]

# Define columns for OPB file
columns_opb = ['BENE_ID', 'CLM_ID', 'CLM_FROM_DT','CLM_THRU_DT', 'PRVDR_STATE_CD','PTNT_DSCHRG_STUS_CD', 'PRNCPAL_DGNS_CD', 'FST_DGNS_E_CD'] + \
              ['ICD_DGNS_CD{}'.format(i) for i in range(1, 26)] + ['ICD_DGNS_E_CD{}'.format(j) for j in range(1, 13)]

# Loop function to loop over years defined above (only using 2016 for now)
for y in years:

    # Read in OP claims
    opb = dd.read_csv(f'/mnt/data/medicare-share/data/{y}/OPB/csv/outpatient_base_claims_k.csv', usecols=columns_opb,sep=',',
                      engine='c', dtype='object', na_filter=False, skipinitialspace=True, low_memory=False)

    # Convert to datetime
    opb['CLM_FROM_DT'] = dd.to_datetime(opb['CLM_FROM_DT'])
    opb['CLM_THRU_DT'] = dd.to_datetime(opb['CLM_THRU_DT'])

    # Specify only BENE_IDs for ambulance claims
    columns_amb = ['BENE_ID']

    # Read in ambulance claims
    amb = dd.read_parquet(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicare_ab/{y}/amb_ab/',engine='fastparquet',columns=columns_amb)

    # Merge amb with opb to keep only op with amb rides
    opb_merge = dd.merge(opb,amb,on=['BENE_ID'],how='inner')

    # Read out file to parquet
    opb_merge.to_parquet(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicare_ab/{y}/op_subset/', compression='gzip', engine='fastparquet')







