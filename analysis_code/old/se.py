# -*- coding: utf-8 -*-
"""
anphy sleep
"""
import pandas as pd
import plotly_express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots  #
import plotly.io as pio
pio.renderers.default = 'svg'
pio.renderers.default = 'browser'


## This is used to plot the sleep efficiency distribution
#read the data
df1 = pd.read_csv('path')
data=df1.iloc[:,5:9]
print(data)
fig = px.treemap(
    df1,
    path=[px.Constant("Sleep efficiency distribution(Size represents TST time(min))"),"Subjects ID","TST"], # path 
    values='TST',  # size of figure depending on the TST values
    color="SE",  # color
    color_continuous_scale="RdBu",  # color setting
)
fig.update_layout(uniformtext=dict(minsize=14),
                  margin=dict(t=40,l=25,r=25,b=30))

fig.update_traces(textfont=dict(size=18))

fig.update_layout(margin=dict(t=40,l=20,r=25,b=30))   

fig.show()