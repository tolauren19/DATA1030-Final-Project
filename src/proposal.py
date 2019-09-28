import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

df = pd.read_stata('tedsd_puf_2017_edit.dta')

change_educ = df[['REGION', 'EMPLOY', 'EDUC']].values
imputer = IterativeImputer(missing_values = -9.0, estimator=RandomForestRegressor(), random_state=0)
educ_impute = imputer.fit_transform(change_educ)
df[['REGION', 'EMPLOY', 'EDUC']] = educ_impute
df[['EMPLOY', 'EDUC']] = (df[['EMPLOY', 'EDUC']].astype(int)).astype(float)
print(df[['REGION', 'EMPLOY', 'EDUC']][40:60])

serv_recode = {1:1, 2:1, 3:2, 4:2, 5:2, 6:3, 7:3, 8:3}
df['SERVICES'] = df['SERVICES'].replace(serv_recode)

drugs_recode = {1: 1, 2:2, 3:3, 4:4, 5:5, 6:5, 7:5, 8:8, 9:8, 10:6, 11:6, 12:8, 13:7, 14:8, 15:8, 16:8, 17:8, 18:8, 19:8}
df[['SUB1', 'SUB2', 'SUB3']] = df[['SUB1', 'SUB2', 'SUB3']].replace(drugs_recode)

ethnic_recode = {1:1, 2:1, 3:1, 4:0, 5:1}
df['ETHNIC'] = df['ETHNIC'].replace(ethnic_recode)

reasons_recode = {2:0, 3:0, 4:0, 5:0, 6:0, 7:0}
df['REASON'] = df['REASON'].replace(reasons_recode)

df['PSYPROB'] = df['PSYPROB'].replace({2:0})
df['METHUSE'] = df['METHUSE'].replace({2:0})
df['PREG'] = df['PREG'].replace({-9:0, 2:0})


df = df[~df.eq(-9).any(1)] #remove missing values after replacing 
print(df.shape)

onehot_cols = df[['SERVICES', 'REGION', 'PSOURCE', 'GENDER', 'LIVARAG', 'RACE', 
				'SUB1', 'SUB2', 'SUB3', 'ALCDRUG', 'EMPLOY', 'MARSTAT', 'ROUTE1']]

onehot_enc = OneHotEncoder(sparse=False)
onehot_enc.fit_transform(onehot_cols)


gen = ['MALE', 'FEMALE'] 
race = ['ALASKA_NAT', 'AM_INDIAN', 'API', 'BLACK', 'WHITE', 'ASIAN', 'ONE_R', 'TWO_R', 'HAWAIIAN']
servs = ['DETOX_A', 'REHAB_RES_A', 'AMBUL_A']
livarr = ['HOMELESS', 'DEP_LIV', 'INDEP_LIV']
sources = ['INDIV', 'ALC_DRUG_CAREPROV', 'OTHER_CAREPROV', 'SCHOOL', 'EMPLOYER', 'COMM_REF', 'CRIM_JUST']
subs1 = ['ALC1', 'COC1', 'MARIJ1', 'HEROIN_OPS1', 'METH1', 'BENZ1', 'OTHER1']
subs2 = ['NONE2', 'ALC2', 'COC2', 'MARIJ2', 'HEROIN_OPS2', 'METH2', 'BENZ2', 'OTHER2']
subs3 = ['NONE3', 'ALC3', 'COC3', 'MARIJ3', 'HEROIN_OPS3', 'METH3', 'BENZ3', 'OTHER3']
regions = ['US_TERR', 'NE', 'MW', 'SOUTH', 'WEST']
ad_ind = ['ALC_ONLY', 'DRUGS_ONLY', 'ALC_DRUGS']
jobs = ['FT', 'PT', 'UNEMP', 'NLF']
marr = ['NEVER_MAR', 'MARRIED', 'SEPARATED', 'DIVORCED_WIDOWED']
routes1 = ['ORAL1', 'SMOK1', 'INHAL1', 'INJ1', 'OTHER_ROUTE1']

new_onehot_cols = servs + regions + sources + gen + livarr + race + subs1 + subs2 + subs3 + ad_ind + jobs + marr + routes1

newdf = pd.DataFrame(onehot_enc.transform(onehot_cols), columns = new_onehot_cols)

ord_cols = df[['EDUC', 'LOS', 'ARRESTS', 'ARRESTS_D', 'FREQ1', 'FRSTUSE1']]
newdf[['EDUC', 'LOS', 'ARRESTS', 'ARRESTS_D', 'FREQ1', 'FRSTUSE1']] = ord_cols
newdf[['METHUSE', 'ETHNIC', 'PREG', 'REASON']] = df[['METHUSE', 'ETHNIC', 'PREG', 'REASON']]

# ord_enc = OrdinalEncoder(categories='auto')
# neword_cols = ord_enc.fit_transform(ord_cols)
# newdf['LOS'] = ord_enc.transform(df[['LOS']])

newdf = newdf.dropna()
newdf.index = range(len(newdf))
print(newdf.shape)

newdf.to_csv("clean_teds2017.csv")





