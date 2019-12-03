import pandas as pd
import numpy as np
import sklearn as sk
import math as ma
import scipy.stats as st
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
import warnings

warnings.simplefilter(action='ignore', category=Warning)
df = pd.read_stata('tedsd_puf_2017_edited.dta')
df=df.take(np.random.permutation(len(df))[:100000])
#df = df[df['DSMCRIT']==5 && df['SERVICES'] > 5]
df.drop(columns=['DSMCRIT', 'DISYR', 'CASEID', 'CBSA2010', 'DETNLF', 'DETNLF_D', 
	'DIVISION', 'STFIPS', 'MARSTAT', 'PRIMPAY', 'IDU', 'DETCRIM', 'ROUTE2', 
	'ROUTE3', 'FREQ2', 'FREQ3', 'FREQ1_D', 'FREQ2_D', 'FREQ3_D', 'HLTHINS', 'EMPLOY_D', 'SERVICES', 'SERVICES_D'], inplace=True)

df.replace(to_replace = -9, value = np.NaN, inplace = True)
df.dropna(subset=['NOPRIOR', 'RACE', 'ETHNIC', 'ARRESTS', 'EDUC', 'EMPLOY', 
	'PSOURCE', 'METHUSE', 'GENDER', 'LIVARAG', 'FRSTUSE1', 'ROUTE1', 'FREQ1', 'SUB1'], inplace=True)

df.index = range(len(df))


#print(df.head(50))
def checks_input_mcar_tests(data):
    """ Checks whether the input parameter of class McarTests is correct
            Parameters
            ----------
            data:
                The input of McarTests specified as 'data'
            Returns
            -------
            bool
                True if input is correct
            """

    if not isinstance(data, pd.DataFrame):
        print("Error: Data should be a Pandas DataFrame")
        return False

    if not any(data.dtypes.values == np.float):
        if not any(data.dtypes.values == np.int):
            print("Error: Dataset cannot contain other value types than floats and/or integers")
            return False

    if not data.isnull().values.any():
        print("Error: No NaN's in given data")
        return False

    return True


def mcar_test(data):
    """ Implementation of Little's MCAR test
    Parameters
    ----------
    data: Pandas DataFrame
        An incomplete dataset with samples as index and variables as columns
    Returns
    -------
    p_value: Float
        This value is the outcome of a chi-square statistical test, testing whether the null hypothesis
        'the missingness mechanism of the incomplete dataset is MCAR' can be rejected.
    """

    if not checks_input_mcar_tests(data):
        raise Exception("Input not correct")

    dataset = data.copy()
    vars = dataset.dtypes.index.values
    n_var = dataset.shape[1]

    # mean and covariance estimates
    # ideally, this is done with a maximum likelihood estimator
    gmean = dataset.mean()
    gcov = dataset.cov()

    # set up missing data patterns
    r = 1 * dataset.isnull()
    mdp = np.dot(r, list(map(lambda x: ma.pow(2, x), range(n_var))))
    sorted_mdp = sorted(np.unique(mdp))
    n_pat = len(sorted_mdp)
    correct_mdp = list(map(lambda x: sorted_mdp.index(x), mdp))
    dataset['mdp'] = pd.Series(correct_mdp, index=dataset.index)

    # calculate statistic and df
    pj = 0
    d2 = 0
    for i in range(n_pat):
        dataset_temp = dataset.loc[dataset['mdp'] == i, vars]
        select_vars = ~dataset_temp.isnull().any()
        pj += np.sum(select_vars)
        select_vars = vars[select_vars]
        means = dataset_temp[select_vars].mean() - gmean[select_vars]
        select_cov = gcov.loc[select_vars, select_vars]
        mj = len(dataset_temp)
        parta = np.dot(means.T, np.linalg.solve(select_cov, np.identity(select_cov.shape[1])))
        d2 += mj * (np.dot(parta, means))

    df = pj - n_var

    # perform test and save output
    p_value = 1 - st.chi2.cdf(d2, df)

    return p_value

drugs_recode = {1:1, 2:2, 3:3, 4:4, 5:5, 6:5, 7:5, 8:8, 9:8, 10:6, 11:6, 12:8, 13:7, 14:8, 15:8, 16:8, 17:8, 18:8, 19:8}
df[['SUB1', 'SUB2', 'SUB3', 'SUB1_D', 'SUB2_D', 'SUB3_D']] = df[['SUB1', 'SUB2', 'SUB3', 'SUB1_D', 'SUB2_D', 'SUB3_D']].replace(drugs_recode)

ethnic_recode = {1:1, 2:1, 3:1, 4:0, 5:1}
df['ETHNIC'] = df['ETHNIC'].replace(ethnic_recode)

reasons_recode = {2:0, 3:0, 4:0,5:0, 6:0, 7:0}
df['REASON'] = df['REASON'].replace(reasons_recode)

tdf['METHUSE'] = df['METHUSE'].replace({2:0})
df['PREG'] = df['PREG'].replace({np.NaN:0, 2:0})
df['GENDER'] = df['GENDER'].replace({2:0})
df['VET'] = df['VET'].replace({2:0})
df['PSYPROB'] = df['PSYPROB'].replace({2:0}


# nan_cols = df.columns[df.isnull().any()].tolist()
# drop_cols = []
# for i in range(len(nan_cols)):
#     if mcar_test(df[['REASON', nan_cols[i]]]) > 0.05:
#         drop_cols.append(nan_cols[i])

# print(drop_cols)

X = df.loc[:, df.columns != 'REASON'].values
cols = df.loc[:, df.columns != 'REASON'].columns.tolist()
imputer = IterativeImputer(missing_values = np.NaN, estimator=RandomForestRegressor(), random_state=2)
imputed_vals = imputer.fit_transform(X)
df[cols] = imputed_vals #multivariate imputer for education 
df[cols] = (df[cols].astype(int)).astype(float)

onehot_cols = df[['REGION', 'PSOURCE', 'LIVARAG', 'LIVARAG_D', 'RACE', 
				'SUB1', 'SUB2', 'SUB3', 'SUB1_D', 'SUB2_D', 'SUB3_D', 'ALCDRUG', 'EMPLOY', 'ROUTE1', 'PRIMINC']]

onehot_enc = OneHotEncoder(sparse=False)
onehot_enc.fit_transform(onehot_cols)

race = ['ALASKA_NAT', 'AM_INDIAN', 'API', 'BLACK', 'WHITE', 'ASIAN', 'ONE_R', 'TWO_R', 'HAWAIIAN']
livarr = ['HOMELESS', 'DEP_LIV', 'INDEP_LIV']
livarr_d = ['HOMELESS_D', 'DEP_LIV_D', 'INDEP_LIV_D']
sources = ['INDIV', 'ALC_DRUG_CAREPROV', 'OTHER_CAREPROV', 'SCHOOL', 'EMPLOYER', 'COMM_REF', 'CRIM_JUST']
subs1 = ['ALC1', 'COC1', 'MARIJ1', 'HEROIN_OPS1', 'METH1', 'BENZ1', 'OTHER1']
subs2 = ['NONE2', 'ALC2', 'COC2', 'MARIJ2', 'HEROIN_OPS2', 'METH2', 'BENZ2', 'OTHER2']
subs3 = ['NONE3', 'ALC3', 'COC3', 'MARIJ3', 'HEROIN_OPS3', 'METH3', 'BENZ3', 'OTHER3']
subs1_d = ['NONE1_D', 'ALC1_D', 'COC1_D', 'MARIJ1_D', 'HEROIN_OPS1_D', 'METH1_D', 'BENZ1_D', 'OTHER1_D']
subs2_d = ['NONE2_D', 'ALC2_D', 'COC2_D', 'MARIJ2_D', 'HEROIN_OPS2_D', 'METH2_D', 'BENZ2_D', 'OTHER2_D']
subs3_d = ['NONE3_D', 'ALC3_D', 'COC3_D', 'MARIJ3_D', 'HEROIN_OPS3_D', 'METH3_D', 'BENZ3_D', 'OTHER3_D']
regions = ['US_TERR', 'NE', 'MW', 'SOUTH', 'WEST']
ad_ind = ['ALC_ONLY', 'DRUGS_ONLY', 'ALC_DRUGS']
jobs = ['FT', 'PT', 'UNEMP', 'NLF']
routes1 = ['ORAL1', 'SMOK1', 'INHAL1', 'INJ1', 'OTHER_ROUTE1']
inc = ['WAGE', 'PUB_ASSIST', 'RET_DIS', 'OTHER', 'NONE']

new_onehot_cols = regions + sources + livarr + livarr_d + race + subs1 + subs2 + subs3 + subs1_d + subs2_d + subs3_d + ad_ind + jobs + routes1 + inc

newdf = pd.DataFrame(onehot_enc.transform(onehot_cols), columns = new_onehot_cols)

clean_cols = ['METHUSE', 'ETHNIC', 'PREG', 'REASON', 'GENDER', 'VET', 'NOPRIOR', 'PSYPROB']
ord_cols = ['EDUC', 'LOS', 'ARRESTS', 'ARRESTS_D', 'FREQ1', 'FRSTUSE1', 'DAYWAIT', 'FREQ_ATND_SELF_HELP', 'FREQ_ATND_SELF_HELP_D']
newdf[['EDUC', 'LOS', 'ARRESTS', 'ARRESTS_D', 'FREQ1', 'FRSTUSE1', 'DAYWAIT', 'SELFHELP_P', 'SELFHELP_D']] = df[ord_cols]
newdf[clean_cols] = df[clean_cols]

# # ord_enc = OrdinalEncoder(categories='auto')
# # neword_cols = ord_enc.fit_transform(ord_cols)
# # newdf['LOS'] = ord_enc.transform(df[['LOS']])

# # newdf = newdf.dropna()
# # newdf.index = range(len(newdf))
print(newdf.head(50))
#print(newdf[['EDUC', 'LOS', 'ARRESTS', 'ARRESTS_D', 'FREQ1', 'FRSTUSE1', 'PRIMINC', 'DAYWAIT', 'SELFHELP_P', 'SELFHELP_D']])

newdf.to_csv("processed_teds2017.csv")


# run f test regression, pick top 50 features
# make plots of first 10 



