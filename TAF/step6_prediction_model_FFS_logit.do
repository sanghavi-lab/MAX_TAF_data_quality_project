*------------------------------------------------------------------------------*
* Project: Medicaid Data Quality Project
* Author: Jessy Nguyen
* Last Updated: August 13, 2021
* Description: This script will perform an out-of-sample estimation to predict the death rate in Medicaid FFS using a logit prediction model that was trained in Medicare data.
*------------------------------------------------------------------------------*

*----- First, generate model using Medicare data ---*

* Set Working Directory
cd "/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicare_ab/data_for_prediction_model/icdpicr/"

* Import Medicare data from directory specifying columns
use death_ind_discharge AGE niss_6_to_5 RTI_RACE_CD SEX_IDENT_CD BENE_RSDNC_SSA_STATE_CD using medicare_16_from_icdpicr_w_niss_six_to_five.dta, clear

* Use niss where ais 6 is converted to ais 5
ren niss_6_to_5 niss /* Change name of relevant niss variable */

* Remove AGE 74 or greater to keep age group between MCARE and MCAID as close as possible
keep if AGE<74

* Remove observations with NISS >=76 or 0
drop if niss>=76 | niss == 0

* Remove unknown race and sex
drop if RTI_RACE_CD == "0"
drop if SEX_IDENT_CD == "0"

* Creating Bands for niss
gen niss_bands = .
replace niss_bands = 1 if niss>0 & niss<=8
replace niss_bands = 2 if niss>8 & niss<=15
replace niss_bands = 3 if niss>15 & niss<=24
replace niss_bands = 4 if niss>24 & niss<=40
replace niss_bands = 5 if niss>40

* Label the niss categorical variable
label define niss_label 1 "1-8" 2 "9-15" 3 "16-24" 4 "25-40" 5 "41+"
label values niss_bands niss_label

* Destring and Rename variables
destring RTI_RACE_CD, generate(RACE)
destring SEX_IDENT_CD, generate(SEX)
destring BENE_RSDNC_SSA_STATE_CD, generate(STATE)

* Retain only U.S. States
drop if STATE==40 | STATE==48 | STATE>=54

* Label the variables that were destringed
label define RACE_label 1 "White" 2 "Black" 3 "Other" 4 "Asian/PI" 5 "Hispanic" 6 "Native Americans/Alaskan Native"
label values RACE RACE_label
label define SEX_label 1 "M" 2 "F"
label values SEX SEX_label
label define STATE_label 1 "AL" 2 "AK" 3 "AZ" 4 "AR" 5 "CA" 6 "CO" 7 "CT" 8 "DE" 9 "DC" 10 "FL" 11 "GA" 12 "HI" 13 "ID" 14 "IL" 15 "IN" 16 "IA" 17 "KS" 18 "KY" 19 "LA" 20 "ME" 21 "MD" 22 "MA" 23 "MI" 24 "MN" 25 "MS" 26 "MO" 27 "MT" 28 "NE" 29 "NV" 30 "NH" 31 "NJ" 32 "NM" 33 "NY" 34 "NC" 35 "ND" 36 "OH" 37 "OK" 38 "OR" 39 "PA" 41 "RI" 42 "SC" 43 "SD" 44 "TN" 45 "TX" 46 "UT" 47 "VT"49 "VA" 50 "WA" 51 "WV" 52 "WI" 53 "WY"
label values STATE STATE_label

* Run logit model (no year FE because I only use 2016)
logit death_ind_discharge i.niss_bands i.RACE i.SEX i.STATE c.AGE

*----- Second, read in Medicaid data to predict ---*

* Import Medicaid Data FFS
cd "/mnt/labshares/sanghavi-lab/Nadia/data/data_quality_project/medicaid_ffs_mc/data_for_prediction_model/icdpicr/"
use death_ind_discharge AGE niss_6_to_5 RACE_ETHNCTY_CD SEX_CD STATE_CD using taf_ffs_from_icdpicr_w_niss_six_to_five.dta, clear

* Use niss where ais 6 is converted to ais 5
ren niss_6_to_5 niss /* Change name of relevant niss variable */

* Remove observations with NISS >=76 or 0
drop if niss>=76 | niss == 0

* Remove unknown race and sex
drop if RACE_ETHNCTY_CD == ""
drop if SEX_CD == ""

* Remove AGE less than 50 to keep age group between MCARE and MCAID as close as possible
keep if AGE>=50&AGE<65

* Creating Bands for niss
gen niss_bands = .
replace niss_bands = 1 if niss>0 & niss<=8
replace niss_bands = 2 if niss>8 & niss<=15
replace niss_bands = 3 if niss>15 & niss<=24
replace niss_bands = 4 if niss>24 & niss<=40
replace niss_bands = 5 if niss>40

* Label the niss categorical variable
label define niss_label 1 "1-8" 2 "9-15" 3 "16-24" 4 "25-40" 5 "41+"
label values niss_bands niss_label

* Change values in sex variable so it's consistent with Medicare
gen SEX = .
replace SEX=1 if SEX_CD=="M"
replace SEX=2 if SEX_CD=="F"

* Change values in RACE variable so it's consistent with Medicare
gen RACE = .
replace RACE=1 if RACE_ETHNCTY_CD== "1"
replace RACE=2 if RACE_ETHNCTY_CD== "2"
replace RACE=3 if RACE_ETHNCTY_CD== "6"
replace RACE=4 if RACE_ETHNCTY_CD== "3" | RACE_ETHNCTY_CD== "5"
replace RACE=5 if RACE_ETHNCTY_CD== "7"
replace RACE=6 if RACE_ETHNCTY_CD== "4"

* Change values in STATE_CD so it's consistent with Medicare
gen STATE = .
replace STATE = 1 if STATE_CD== "AL"
replace STATE = 2 if STATE_CD== "AK"
replace STATE = 3 if STATE_CD== "AZ"
replace STATE = 4 if STATE_CD== "AR"
replace STATE = 5 if STATE_CD== "CA"
replace STATE = 6 if STATE_CD== "CO"
replace STATE = 7 if STATE_CD== "CT"
replace STATE = 8 if STATE_CD== "DE"
replace STATE = 9 if STATE_CD== "DC"
replace STATE = 10 if STATE_CD== "FL"
replace STATE = 11 if STATE_CD== "GA"
replace STATE = 12 if STATE_CD== "HI"
replace STATE = 13 if STATE_CD== "ID"
replace STATE = 14 if STATE_CD== "IL"
replace STATE = 15 if STATE_CD== "IN"
replace STATE = 16 if STATE_CD== "IA"
replace STATE = 17 if STATE_CD== "KS"
replace STATE = 18 if STATE_CD== "KY"
replace STATE = 19 if STATE_CD== "LA"
replace STATE = 20 if STATE_CD== "ME"
replace STATE = 21 if STATE_CD== "MD"
replace STATE = 22 if STATE_CD== "MA"
replace STATE = 23 if STATE_CD== "MI"
replace STATE = 24 if STATE_CD== "MN"
replace STATE = 25 if STATE_CD== "MS"
replace STATE = 26 if STATE_CD== "MO"
replace STATE = 27 if STATE_CD== "MT"
replace STATE = 28 if STATE_CD== "NE"
replace STATE = 29 if STATE_CD== "NV"
replace STATE = 30 if STATE_CD== "NH"
replace STATE = 31 if STATE_CD== "NJ"
replace STATE = 32 if STATE_CD== "NM"
replace STATE = 33 if STATE_CD== "NY"
replace STATE = 34 if STATE_CD== "NC"
replace STATE = 35 if STATE_CD== "ND"
replace STATE = 36 if STATE_CD== "OH"
replace STATE = 37 if STATE_CD== "OK"
replace STATE = 38 if STATE_CD== "OR"
replace STATE = 39 if STATE_CD== "PA"
replace STATE = 41 if STATE_CD== "RI"
replace STATE = 42 if STATE_CD== "SC"
replace STATE = 43 if STATE_CD== "SD"
replace STATE = 44 if STATE_CD== "TN"
replace STATE = 45 if STATE_CD== "TX"
replace STATE = 46 if STATE_CD== "UT"
replace STATE = 47 if STATE_CD== "VT"
replace STATE = 49 if STATE_CD== "VA"
replace STATE = 50 if STATE_CD== "WA"
replace STATE = 51 if STATE_CD== "WV"
replace STATE = 52 if STATE_CD== "WI"
replace STATE = 53 if STATE_CD== "WY"

* Predict but using margins to categorize them into niss_bands AND Save results
margins niss_bands, noesample saving(taf_predicted_death_medicaid_ffs_logit_model, replace)


