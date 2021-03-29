import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pickle
import math as m

dat = pd.read_json("./transactions.txt", lines=True)

# Preprocess dataset
dat['date_transaction'] = pd.to_datetime(dat['transactionDateTime']).dt.date
dat['datetime_transaction'] = pd.to_datetime(dat['transactionDateTime'])
dat = dat.sort_values(["customerId", "accountNumber", "datetime_transaction"],
                      ascending = (False, False, True)).reset_index(drop=True)

def transform_merchant_name(dat):
    # transform merchant name: remove order number.
    print(dat['merchantName'].value_counts())
    x = dat['merchantName'].str.split('#', expand = True)
    dat['merchantName'] = x[0].str.strip()
    print(dat['merchantName'].value_counts())
    return dat
# remove the transaction ID from the merchant
dat = transform_merchant_name(dat)
dat['date_key'] = dat['date_transaction'].apply(lambda x: str(x))

# EDA
dat[dat['transactionType'] == 'REVERSAL']['customerId'].value_counts()
dat[dat['customerId'] == 882815134][['transactionType', 'datetime_transaction', 'transactionType','transactionAmount',
                                     'merchantName','isFraud','accountNumber']].to_csv("./Q3_reversal_example.csv")
# After manual inspection, it looks like reversals can sometimes happen very quickly, so we need to take this into account
# when counting the number of multiswipes. Most of the time reversals happen a couple of days later.

# number of reversed transactions
print(f'total reversals:', sum(dat['transactionType'] == 'REVERSAL'), ', total amount: $',
      sum(dat['transactionAmount'][dat['transactionType'] == 'REVERSAL']))

multiswipe_dict = {}
multiswipe_amount_dict = {}
# multiswipe logic:
#    if a transaction is not a reversal, and has the same customerID,  amount, account numbermerchantName,
#    and occurs on the same day

for index, row in dat.iterrows():
    key = str(row['customerId']) + '_' + str(row['transactionAmount']) + '_' + str(row['merchantName']) \
          + '_' + str(row['accountNumber'])  + '_' + str(row['date_key'] )
    if index % 10000 == 0:
        print(f'proccessed row:', index)
    if row['transactionType'] != 'REVERSAL':
        try:
            # transaction already exists and is not a reversal
            multiswipe_dict[key] += 1
            multiswipe_amount_dict[key] += row['transactionAmount']
        except:
            # first instance of the transaction
            multiswipe_dict[key] = 0
            multiswipe_amount_dict[key] = 0

# number of multiswipe transactions
print(f'total multi-swipes:', sum(multiswipe_dict.values()), ', total amount: $', sum(multiswipe_amount_dict.values()))
# total multi-swipes: 7900 , total amount: $ 1103357.7500000035