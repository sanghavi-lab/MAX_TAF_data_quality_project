#----------------------------------------------------------------------------------------------------------------------#
# Project: Medicaid Data Quality Project
# Author: Jessy Nguyen
# Last Updated: August 13, 2021
# Description: The goal of this script is to create figures 1 and 2 using only the ambulance claims that merged with IP
#              record for both MAX and TAF
#----------------------------------------------------------------------------------------------------------------------#

################################################ IMPORT PACKAGES #######################################################

# Read in relevant libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import Affine2D # to move error bars
import matplotlib.transforms as transforms

################################## NISS VS DEATH RATE GRAPH FOR MEDICAID ###############################################
# Includes both MAX and TAF on one graph. Fig 1 is FFS and Fig 2 is managed care                                       #
########################################################################################################################

#___________________________________________Define function____________________________________________________________#

# Define function to graph the predicted death data vs reported death data
def graph_predicted_vs_reported(mcaid_payment_type,linear_or_logit):

    # Define columns
    cols_taf = ['death_ind_discharge', 'AGE', 'niss','niss_6_to_5', 'SEX_CD', 'RACE_ETHNCTY_CD', 'STATE_CD','BENE_ID','MSIS_ID']
    cols_max = ['death_ind_discharge', 'AGE', 'niss','niss_6_to_5', 'EL_RACE_ETHNCY_CD', 'EL_SEX_CD', 'STATE_CD','BENE_ID','MSIS_ID']
    model_cols = ['_m1', '_margin', '_ci_ub', '_ci_lb']

    # Read in data
    medicaid_reported_death_taf = pd.read_csv(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicaid_ffs_mc/data_for_prediction_model/icdpicr/taf_{mcaid_payment_type}_from_icdpicr_w_niss_six_to_five.csv',usecols=cols_taf,dtype=str)
    medicaid_predicted_death_model_taf = pd.read_stata(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicaid_ffs_mc/data_for_prediction_model/icdpicr/taf_predicted_death_medicaid_{mcaid_payment_type}_{linear_or_logit}_model.dta',columns=model_cols)
    medicaid_reported_death_max = pd.read_csv(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicaid_ffs_mc/data_for_prediction_model/icdpicr/{mcaid_payment_type}_from_icdpicr_w_niss_six_to_five.csv',usecols=cols_max,dtype=str)
    medicaid_predicted_death_model_max = pd.read_stata(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicaid_ffs_mc/data_for_prediction_model/icdpicr/predicted_death_medicaid_{mcaid_payment_type}_{linear_or_logit}_model.dta',columns=model_cols)

    # Change column names
    medicaid_reported_death_max = medicaid_reported_death_max.rename(columns={'EL_RACE_ETHNCY_CD':'RACE_ETHNCTY_CD','EL_SEX_CD':'SEX_CD'})

    # Rename niss columns
    medicaid_reported_death_taf = medicaid_reported_death_taf.drop(['niss'],axis=1) # this is the niss where 6 means an niss of 75. Drop this column.
    medicaid_reported_death_taf = medicaid_reported_death_taf.rename(columns={'niss_6_to_5': 'niss'}) # Instead, use the column were 6 was converted to a 5.
    medicaid_reported_death_max = medicaid_reported_death_max.drop(['niss'],axis=1) # this is the niss where 6 means an niss of 75. Drop this column.
    medicaid_reported_death_max = medicaid_reported_death_max.rename(columns={'niss_6_to_5': 'niss'}) # Instead, use the column were 6 was converted to a 5.

    # Convert numeric to floats
    num = ['AGE','niss','death_ind_discharge']
    for n in num:
        medicaid_reported_death_taf[f'{n}'] = medicaid_reported_death_taf[f'{n}'].astype('float')
        medicaid_reported_death_max[f'{n}'] = medicaid_reported_death_max[f'{n}'].astype('float')

    # Convert character columns to string
    string = ['RACE_ETHNCTY_CD','SEX_CD','STATE_CD']
    for s in string:
        medicaid_reported_death_taf[f'{s}'] = medicaid_reported_death_taf[f'{s}'].astype('str')
        medicaid_reported_death_max[f'{s}'] = medicaid_reported_death_max[f'{s}'].astype('str')

    # Removed observations where niss is not within 1-75
    medicaid_reported_death_taf = medicaid_reported_death_taf[(medicaid_reported_death_taf['niss'] >= 1) & (medicaid_reported_death_taf['niss'] <= 75)]
    medicaid_reported_death_max = medicaid_reported_death_max[(medicaid_reported_death_max['niss'] >= 1) & (medicaid_reported_death_max['niss'] <= 75)]

    # Remove unknown race and sex
    medicaid_reported_death_taf = medicaid_reported_death_taf[~(medicaid_reported_death_taf['RACE_ETHNCTY_CD'] == '')]
    medicaid_reported_death_taf = medicaid_reported_death_taf[~(medicaid_reported_death_taf['SEX_CD'] == '')]
    medicaid_reported_death_max = medicaid_reported_death_max[~(medicaid_reported_death_max['RACE_ETHNCTY_CD'] == '9')]
    medicaid_reported_death_max = medicaid_reported_death_max[~(medicaid_reported_death_max['SEX_CD'] == '0')]

    # Keep AGE between 50 and 64
    medicaid_reported_death_taf = medicaid_reported_death_taf[(medicaid_reported_death_taf['AGE'] >= 50) & (medicaid_reported_death_taf['AGE'] < 65)]
    medicaid_reported_death_max = medicaid_reported_death_max[(medicaid_reported_death_max['AGE'] >= 50) & (medicaid_reported_death_max['AGE'] < 65)]

    # Create NISS bins for reported data
    medicaid_reported_death_taf['niss_bins'] = np.where(((medicaid_reported_death_taf['niss'] > 0) & (medicaid_reported_death_taf['niss'] <= 8)), '1-8',
                                           np.where(((medicaid_reported_death_taf['niss'] > 8) & (medicaid_reported_death_taf['niss'] <= 15)), '9-15',
                                           np.where(((medicaid_reported_death_taf['niss'] > 15) & (medicaid_reported_death_taf['niss'] <= 24)),'16-24',
                                           np.where(((medicaid_reported_death_taf['niss'] > 24) & (medicaid_reported_death_taf['niss'] <= 40)), '25-40',
                                           np.where(((medicaid_reported_death_taf['niss'] > 40)), '41+', 0)))))
    medicaid_reported_death_max['niss_bins'] = np.where(((medicaid_reported_death_max['niss'] > 0) & (medicaid_reported_death_max['niss'] <= 8)), '1-8',
                                           np.where(((medicaid_reported_death_max['niss'] > 8) & (medicaid_reported_death_max['niss'] <= 15)), '9-15',
                                           np.where(((medicaid_reported_death_max['niss'] > 15) & (medicaid_reported_death_max['niss'] <= 24)),'16-24',
                                           np.where(((medicaid_reported_death_max['niss'] > 24) & (medicaid_reported_death_max['niss'] <= 40)), '25-40',
                                           np.where(((medicaid_reported_death_max['niss'] > 40)), '41+', 0)))))

    #--- Set up Raw Data ---#

    # Find average death rate in each bin for raw data.
    medicaid_reported_death_taf = medicaid_reported_death_taf.groupby(['niss_bins'])['death_ind_discharge'].mean().to_frame().reset_index()
    medicaid_reported_death_max = medicaid_reported_death_max.groupby(['niss_bins'])['death_ind_discharge'].mean().to_frame().reset_index()

    # View in order to observe the distance between the lower bound of the confidence interval and the reported level
    print(mcaid_payment_type,linear_or_logit)
    print(medicaid_reported_death_taf)
    print(medicaid_predicted_death_model_taf)
    print(medicaid_reported_death_max)
    print(medicaid_predicted_death_model_max)
        # Specifically, we observe potential underreporting by 8.3 percentage points in MAX and 10.3 percentage points in TAF for the cases with the most severe injuries (41+).

    # Create column of labels
    medicaid_reported_death_taf['mcaid_type'] = 'TAF'
    medicaid_reported_death_max['mcaid_type'] = 'MAX'
    medicaid_predicted_death_model_taf['mcaid_type'] = 'TAF'
    medicaid_predicted_death_model_max['mcaid_type'] = 'MAX'

    # Concat together reported averages
    mcaid_concat_taf_max = pd.concat([medicaid_reported_death_max,medicaid_reported_death_taf],axis=0)

    # Concat data prediction together
    prediction_concat_taf_max = pd.concat([medicaid_predicted_death_model_max,medicaid_predicted_death_model_taf],axis=0)

    #--- Set up error bars ---#

    # Calc Error bars length (need to half it since the function to create CI will double the length)
    medicaid_predicted_death_model_taf['bar_length'] = (medicaid_predicted_death_model_taf['_ci_ub'] - medicaid_predicted_death_model_taf['_ci_lb'])/2
    medicaid_predicted_death_model_max['bar_length'] = (medicaid_predicted_death_model_max['_ci_ub'] - medicaid_predicted_death_model_max['_ci_lb'])/2

    # Create list from series in order to plot the error bars (half length)
    taf_error_bar_length_list = medicaid_predicted_death_model_taf['bar_length'].tolist()
    max_error_bar_length_list = medicaid_predicted_death_model_max['bar_length'].tolist()

    # Create list from series in order to plot the error bars (midpoint length)
    taf_error_bar_mdpt_list = medicaid_predicted_death_model_taf['_margin'].tolist()  # Note that the midpoint needs to be the demeaned one
    max_error_bar_mdpt_list = medicaid_predicted_death_model_max['_margin'].tolist()  # Note that the midpoint needs to be the demeaned one

    #--- Graph model ---#

    # Medicaid: Bar Plot
    ax = sns.barplot(x='niss_bins', y='death_ind_discharge', data=mcaid_concat_taf_max, hue='mcaid_type',palette=['royalblue','indianred'],
                zorder=0, order=['1-8', '9-15', '16-24', '25-40', '41+'])

    if linear_or_logit in ['linear']:

        # Medicaid: Prediction points
        sns.pointplot(x='_m1', y='_margin', data=prediction_concat_taf_max, hue='mcaid_type',palette=['blue','red'], zorder=0,
                      order=['1-8', '9-15', '16-24', '25-40', '41+'], join=False, scale=0.9,dodge=0.4,ax=ax)

    if linear_or_logit in ['logit']:

        # Medicaid: Prediction points
        sns.pointplot(x='_m1', y='_margin', data=prediction_concat_taf_max, hue='mcaid_type',palette=['blueviolet','orangered'], zorder=2,
                      order=['1-8', '9-15', '16-24', '25-40', '41+'], markers=['^','^'], join=False, scale=0.9,dodge=0.4,ax=ax)

    # Create transformations to move error bars to align with points
    trans_MAX = Affine2D().translate(-0.2, 0.0) + ax.transData
    trans_TAF = Affine2D().translate(0.2, 0.0) + ax.transData

    if linear_or_logit in ['linear']:

        # Plot error bars
        x = [0, 1, 2, 3, 4]  # the index starts at 0 so since there are ten categorical variables, need to start at 0 to 9
        plt.errorbar(x, taf_error_bar_mdpt_list, yerr=taf_error_bar_length_list, fmt=',', color='red',transform=trans_TAF,capsize=3,ax=ax)
        plt.errorbar(x, max_error_bar_mdpt_list, yerr=max_error_bar_length_list, fmt=',', color='blue',transform=trans_MAX,capsize=3,ax=ax)

    if linear_or_logit in ['logit']:

        # Plot error bars
        x = [0, 1, 2, 3, 4]  # the index starts at 0 so since there are ten categorical variables, need to start at 0 to 9
        plt.errorbar(x, taf_error_bar_mdpt_list, yerr=taf_error_bar_length_list, fmt=',', color='orangered',transform=trans_TAF,capsize=3)
        plt.errorbar(x, max_error_bar_mdpt_list, yerr=max_error_bar_length_list, fmt=',', color='blueviolet',transform=trans_MAX,capsize=3)

    # Labels
    ax.set_xlabel('Injury severity score', fontsize=9)
    ax.set_ylabel('Proportion of cases reported dead on hospital \ndischarge status', fontsize=9)

    # Remove figure's self-created legends. Will use a custom legend instead.
    ax.get_legend().remove()

    if linear_or_logit in ['linear']:
        legend_elements = [Patch(facecolor='royalblue', edgecolor='royalblue', label='Reported mortality (MAX)'),
                           Patch(facecolor='indianred', edgecolor='indianred', label='Reported mortality (TAF)'),
                           Line2D([0], [0], marker='o', color='w', label=f'Predicted mortality (MAX)',markerfacecolor='blue', markersize=10),
                           Line2D([0], [0], marker='o', color='w', label=f'Predicted mortality (TAF)',markerfacecolor='red', markersize=10)]
    if linear_or_logit in ['logit']:
        legend_elements = [Patch(facecolor='royalblue', edgecolor='royalblue', label='Reported mortality (MAX)'),
                           Patch(facecolor='indianred', edgecolor='indianred', label='Reported mortality (TAF)'),
                           Line2D([0], [0], marker='^', color='w', label=f'Predicted mortality (MAX)',markerfacecolor='blueviolet', markersize=10),
                           Line2D([0], [0], marker='^', color='w', label=f'Predicted mortality (TAF)',markerfacecolor='orangered', markersize=10)]
    plt.legend(handles=legend_elements, fontsize=8, loc='upper left')
    ax.set_ylim(0, 0.5)
    ax.spines['top'].set_visible(False)
    plt.savefig(f'/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/figures/{mcaid_payment_type}_{linear_or_logit}_predicted.pdf',bbox_inches='tight')  # Save figure
    plt.close()  # Deletes top graph

#________________________________________Run Defined function__________________________________________________________#

# Run defined function for ffs and mc (here we decided on logit)
graph_predicted_vs_reported('ffs','logit')
graph_predicted_vs_reported('mc','logit')








