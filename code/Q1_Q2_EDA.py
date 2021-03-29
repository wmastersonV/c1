import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pickle
import math as m

# Q1: EDA
dat0 = pd.read_json("./transactions.txt", lines=True)

print('starting EDA')
def eda_raw(dat):
    #EDA: Manual investigation. See pandas profiling outputs for a better  visualizations

    # summary statistics for continuous vars
    print('\n\n', dat.availableMoney.describe())
    print('\n\n', dat.currentBalance.describe())
    print('\n\n', dat.transactionAmount.describe())

    # summary statistics for categorical vars
    print("\n", dat.isFraud.value_counts())
    print("\n",dat.acqCountry.value_counts())
    print("\n",dat.cardPresent.value_counts())
    print("\n",dat.creditLimit.value_counts())
    print("\n",dat.echoBuffer.value_counts())
    print("\n",dat.expirationDateKeyInMatch.value_counts())
    print("\n",dat.merchantCategoryCode.value_counts())
    print("\n",dat.merchantCity.value_counts())
    print("\n",dat.merchantName.value_counts())
    print("\n",dat.merchantState.value_counts())
    print("\n",dat.posConditionCode.value_counts())
    print("\n",dat.posOnPremises.value_counts())
    print("\n",dat.recurringAuthInd.value_counts())
    print("\n",dat.transactionType.value_counts())
    print("\n unique customers: ", len(list(set(dat.customerId))))
    print("\n unique accounts: ", len(list(set(dat.accountNumber))))
    print("\n unique cards: ", len(list(set(dat.cardLast4Digits))))
    print(dat.describe(include = [np.number]).to_string())
    print(dat.describe(include = ['object']).to_string())
    return(dat)

dat = eda_raw(dat0)

print('drop_missing_cols')
def drop_missing_cols(dat):
    drop_cols = []
    #columns with missing data
    for c in dat.columns:
        if len(dat[c].value_counts()) == 1:
            print(c)
            drop_cols = drop_cols + [c]
    dat_out = dat.drop(columns = drop_cols)
    return(dat_out)
dat = drop_missing_cols(dat)


# Outlier analysis
# Account number 380680241, customer ID 380680241 has large number of transactions (4.2%)
print('drop_outliers')
def drop_outliers(dat):
    print(dat['accountNumber'].value_counts() / dat.shape[0])
    print(dat['customerId'].value_counts() / dat.shape[0])
    dat = dat[(dat['accountNumber'] != 380680241) & (dat['customerId'] != 380680241) ]

    for c in dat[['availableMoney', 'transactionAmount', 'creditLimit', 'currentBalance']].columns:
        col = dat[c]
        outlier_index = np.abs(col - col.mean()) / col.std() > 5
        num_outliers = sum(outlier_index)
        if num_outliers > 0:
            print(f'dropping ', {num_outliers} ,' outliers from column: ', {c})
            dat = dat[~outlier_index]
    return(dat)
dat = drop_outliers(dat)

print('drop_blanks')
def drop_blanks(dat):
    #drops records with missing values like ''
    for c in dat.select_dtypes(exclude=['int16', 'int32', 'int64', 'float16', 'float32', 'float64']).columns:
        if sum(dat[c] == '') > 0:
            print(dat[c].value_counts())
            print(f'dropping ', sum(dat[c] == ''), ' records')
            dat = dat[dat[c] != '']
    return(dat)
dat = drop_blanks(dat)
# posConditionCode, posEntryMode, merchantCountryCode, acqCountry have blanks

print('transform_merchant_name')
def transform_merchant_name(dat):
    # transform merchant name: remove order number.
    print(dat['merchantName'].value_counts())
    x = dat['merchantName'].str.split('#', expand = True)
    dat['merchantName'] = x[0].str.strip()
    print(dat['merchantName'].value_counts())
    return dat
dat = transform_merchant_name(dat)
# EDA merchant fraud pct
fraud_by_merchant = dat.groupby('merchantName').apply(np.mean)['isFraud']

print(f'top 20 merchants fraud percent:')
print(fraud_by_merchant.sort_values(ascending = False)[1:20])

print('filter_date')
def filter_date(dat):
    dat['accountOpenDate'] = pd.to_datetime(dat['accountOpenDate'])
    dat['dateOfLastAddressChange'] = pd.to_datetime(dat['dateOfLastAddressChange'])
    dat['transactionDateTime'] = pd.to_datetime(dat['transactionDateTime'])
    dat = dat[dat['accountOpenDate'] > '2002-05-13']
    dat = dat[dat['dateOfLastAddressChange'] > '2002-11-05']
    return(dat)
dat = filter_date(dat)

print('after removing records, about 1.4% fraud. This is OK')
# dropped  52,457 records, or 7% of the dataset
print(str(dat0.shape[0] - dat.shape[0]))
print(str(dat.shape[0] / dat0.shape[0]))

print('saving to pickle file for future modeling')
with open('./data/dat_transformed.pickle', 'wb') as handle:
    pickle.dump(dat, handle, protocol=pickle.HIGHEST_PROTOCOL)
dat.to_csv("./dat_qc_filter.csv", index= False)

print(f'Q2: TransactionAmount')
# low correlation of transaction amount and fraud
np.corrcoef(dat['isFraud'], dat['transactionAmount'])
# investigate structure of dataset

# histogram of transactionAmount, filter out zero values
print('\n\n', dat.transactionAmount[dat.transactionAmount > 0].describe())
rng = np.random.RandomState(10)
# remove address verification and transactions with 0
x = dat.transactionAmount
a = np.hstack((rng.normal(size=100), rng.normal(loc=5, scale=2, size=100)))
plt.hist(x.values, bins='auto')
plt.title("histogram of transactionAmount")
plt.show()

# Apply lognormal transform
log_x = np.log(dat.transactionAmount + 1)
# log_x = np.log(dat.transactionAmount[dat.transactionAmount > 0] )
a = np.hstack((rng.normal(size=100), rng.normal(loc=5, scale=2, size=100)))
plt.hist(log_x.values, bins='auto')  # arguments are passed to np.histogram
plt.title("log transform of transactionAmount ")
plt.show()
k2, p = stats.normaltest(log_x)
# doesnt pass statistical normality test
if p < 1e-3:
    print('can reject null hypothesis that X comes from normal distribution')
else:
    print('cannot reject null hypothesis that X comes from normal distribution')



