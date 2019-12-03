import numpy as np
import pandas as pd
import math 
import scipy.stats as st
import warnings
import sklearn as sk
import matplotlib.pyplot as plt
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.feature_selection import SelectKBest


warnings.simplefilter(action='ignore', category=Warning)
df = pd.read_csv('processed_teds2017.csv')


y = df['REASON'].values # the target variable
df.drop(columns=['REASON'],inplace=True)
X = df.values # the feature matrix
feature_names = df.columns # feature names

# ft, pv = f_classif(X, y)
# ftlist = [(feature_names[i], pv[i]) for i in range(len(feature_names))]
# f_feats = sorted(ftlist, key=lambda tup: tup[1])[:30]
# print(f_feats)

# mi = mutual_info_classif(X, y)
# selector = SelectKBest(mutual_info_classif, k=30)
# X_f = selector.fit_transform(X,y)
# m_feats = feature_names[selector.get_support()]
# print(m_feats)

f_pairs = [('NE', 0.0), ('CRIM_JUST', 0.0), ('NONE1_D', 0.0), ('HEROIN_OPS1_D', 0.0), ('LOS', 0.0), 
('SELFHELP_D', 0.0), ('FREQ1', 9.985759241745626e-275), ('NONE2_D', 2.5625124168248653e-252), 
('METHUSE', 1.494641463076019e-182), ('INDIV', 9.136359370209462e-151), ('SOUTH', 1.2094055356712664e-137), 
('MW', 4.86751995029476e-105), ('SELFHELP_P', 6.173784754764213e-104), ('COC2_D', 3.3612320370508764e-78), 
('NONE3_D', 2.250224858696344e-66), ('DEP_LIV', 1.6148861674470944e-58), ('NONE2', 7.793345774050984e-50), 
('NONE', 5.75967905800418e-49), ('WEST', 1.133440462844911e-40), ('FT', 1.6529414286664293e-37), 
('UNEMP', 1.4471426736830556e-36), ('OTHER_CAREPROV', 4.935142846206232e-31), 
('INDEP_LIV', 5.0400078991284144e-29), ('INDEP_LIV_D', 1.571576364415799e-27), 
('BLACK', 2.2501246485946119e-26), ('GENDER', 1.3214700862398605e-25), 
('RET_DIS', 1.1132099389900259e-23), ('MARIJ2_D', 1.1022274574707944e-20), ('WAGE', 1.2178578303948547e-19), 
('PUB_ASSIST', 1.5295633805842423e-19)]
f_output = [f[0] for f in f_pairs]


m_output = ['NE', 'MW', 'SOUTH', 'INDIV', 'CRIM_JUST', 'INDEP_LIV', 'INDEP_LIV_D',
       'WHITE', 'HEROIN_OPS1', 'BENZ1', 'NONE2', 'NONE3', 'NONE1_D',
       'HEROIN_OPS1_D', 'NONE2_D', 'COC2_D', 'NONE3_D', 'DRUGS_ONLY', 'UNEMP',
       'INJ1', 'NONE', 'EDUC', 'LOS', 'FREQ1', 'SELFHELP_P', 'SELFHELP_D',
       'METHUSE', 'GENDER', 'NOPRIOR', 'PSYPROB']


col_intersect = list(set(m_output).intersection(set(f_output)))
#print(len(col_intersect), col_intersect)



def cat_plotter(cols):
    df = pd.read_csv('processed_teds2017.csv')
    label = 'REASON' 
    for item in cols:
        count_matrix = df.groupby([item, label]).size().unstack()
        count_matrix_norm = count_matrix.div(count_matrix.sum(axis=1),axis=0)
        count_matrix_norm.plot(kind='bar', stacked=True)
        plt.xlabel(item)
        plt.ylabel('Fraction of Points in Group')
        plt.title('Bar Plot of %s by Reason for Discharge' % item)
        plt.savefig('catfeature_%s.png' % item)
        plt.show()

def cont_plotter(cols):
    df = pd.read_csv('processed_teds2017.csv')
    label = 'REASON'
    for item in cols:
        df[[item,label]].boxplot(by=label)
        plt.xlabel(label)
        plt.ylabel(item)
        plt.title('Boxplot of Reason for Discharge by %s' % item)
        plt.suptitle("")
        plt.savefig('contfeature_%s.png' % item)
        plt.show()


# cat_plotter(['NONE1_D', 'COC2_D', 'MW', 'INDEP_LIV', 'SOUTH', 
# 'INDEP_LIV_D', 'HEROIN_OPS1_D', 'INDIV', 'NONE3_D', 'METHUSE', 'NONE2_D', 
# 'NONE', 'UNEMP', 'NONE2', 'CRIM_JUST', 'NE', 'GENDER'])

#cont_plotter(['FREQ1', 'LOS', 'SELFHELP_P', 'SELFHELP_D'])

newdf = pd.read_csv('processed_teds2017.csv')

cm = newdf[['REASON'] + col_intersect].corr()

plt.figure(figsize=(10,10))
plt.matshow(cm,vmin=-1,vmax=1,cmap='seismic',fignum=0)
plt.colorbar(label='corr. coeff.')
plt.xticks(np.arange(cm.shape[0]),['REASON'] + list(col_intersect),rotation=90)
plt.yticks(np.arange(cm.shape[0]),['REASON'] + list(col_intersect))
plt.title('Top Feature Correlation Matrix', y = 1.18)
plt.savefig('corrmatrix.png')
plt.show()
