# Scenario
# A fast-food chain plans to add a new item to its menu. However, they are still undecided between three possible
# marketing campaigns for promoting the new product.
# In order to determine which promotion has the greatest effect on sales,the new item is introduced at locations in
# several randomly selected markets. A different promotion is used at each location,and the weekly sales of
# the new item are recorded for the first four weeks.


# Goal
# Evaluate A/B testing results and decide which marketing strategy works the best

# Variables
# MarketID: unique identifier for market
# MarketSize: size of market area by sales
# LocationID: unique identifier for store location
# AgeOfStore: age of store in years
# Promotion: one of three promotions that were tested
# week: one of four weeks when the promotions were run
# SalesInThousands: sales amount for a specific LocationID, Promotion, and week

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
!pip install statsmodels
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
    pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.multicomp import MultiComparison

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


df = pd.read_csv("datasets/WA_Marketing-Campaign.csv")
df.head(10)
df.shape
df.info()
df.describe().T
df.isnull().sum()


df["MarketSize"].value_counts()
df["AgeOfStore"].value_counts()
df["Promotion"].value_counts()


pd.pivot_table(df, values='MarketID', index=["Promotion"],
                       columns=['MarketSize'], aggfunc="sum")
df.groupby("Promotion").agg({"SalesInThousands": ["mean", "count", "sum"]})
df.groupby(["MarketSize","Promotion"]).agg({"SalesInThousands":["count", "mean", "sum"]})
######################################################
# AB Testing (Bağımsız İki Örneklem T Testi)
######################################################

# 1. Hipotezleri Kur
# H0: M1 = M2 = M3 -> Promosyonların satışları arasında fark yoktur.
# H1: M1 != M2 != M3 -> promosyon satışları arasında fark vardır.
# # 2. Varsayım Kontrolü
#   - 1. Normallik Varsayımı
#   - 2. Varyans Homojenliği

test_stat, pvalue = shapiro(df.loc[df["Promotion"], "SalesInThousands"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# p-value < ise 0.05'ten HO RED. kararı idi.

# görüyoruz ki test istatistiğinin değeri 0,7644 p değeri 0,000 dolayısı ile p değeri<0,05 olduğundan H0 red edilir.
# Normallik varsayımı sağlanmadığı için Parametrik olmayan Testlerden Mann whitney U testi ile devam edicez.
# yinede varyansların homojenliğini de kontrol edelim.

############################
# Varyans Homojenligi Varsayımı
############################
# levene testi ile varyansların homojenliği test edilir.
test_stat, pvalue = levene(df.loc[df['Promotion'] == 1, "SalesInThousands"],
                           df.loc[df['Promotion'] == 2, "SalesInThousands"],
                           df.loc[df['Promotion'] == 3, "SalesInThousands"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# p değeri = 0,2818 >0,05 dolayısı ike H0 red edilemez yani varyanslar homojendir.


# parametrik olmayan testlerden mann whitney U testi ile:
test_stat, pvalue = mannwhitneyu(df.loc[df['Promotion'] == 1, "SalesInThousands"],
                           df.loc[df['Promotion'] == 2, "SalesInThousands"],
                           df.loc[df['Promotion'] == 3, "SalesInThousands"])
print("pvalue: ", "%.3f" % pvalue)

# Let's see pvalue more mathematical form to understand better
test_stat, pvalue = kruskal(df.loc[df['Promotion'] == 1, "SalesInThousands"],
                            df.loc[df['Promotion'] == 2, "SalesInThousands"],
                            df.loc[df['Promotion'] == 3, "SalesInThousands"])

print("pvalue: ", "%.3f" % pvalue)

# p_value > 0.05 is rejected. There is a statistically significant difference between Promotions