# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 10:26:10 2023

@author: 31553
"""

import pandas as pd
# Read all tables in hypnograms
url = 'path' # HTML report
df_list = pd.read_html(url, encoding='utf-8')

df = pd.concat(df_list, ignore_index=True)

# extract the sleep stage label and corresponding number
df.drop(df.index[0:15], inplace=True)

df1 = df[~df.isin(['LAmp'])].dropna(axis=0).iloc[:, 20:22] # # locate the specific character and only keep the last two column

df2 = df1.reset_index(drop=True) # rearrange index

#create the duration column
tf=[30]*len(df2)
s = pd.Series(tf)
s1 =pd.DataFrame(s)

c=pd.concat([df2,s1],axis=1,ignore_index=True,sort=True,join='outer')

c.to_csv("D:\\test.csv",header=0,index=False) # save the data





