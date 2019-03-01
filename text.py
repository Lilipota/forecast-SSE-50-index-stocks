# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 19:51:25 2019

@author: Jebs
"""

import pandas as pd
from fbprophet import Prophet
# 读入数据集
df = pd.read_csv('text.csv')


# 拟合普通的模型
m = Prophet(changepoint_prior_scale=0.01)
m.fit(df)
#通过Prophet.make_future_dataframe 来将未来的日期扩展指定的天数
future = m.make_future_dataframe(periods=1000) 
future.tail()
#predict方法将会对每一行future得到一个预测值（称为 yhat ）与置信区间
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
# 展示预测结果
m.plot(forecast);
#查看预测的年度季节性和周季节性
m.plot_components(forecast);

from fbprophet.diagnostics import cross_validation
df_cv = cross_validation(m, initial='1500 days', period='500 days', horizon = '300 days')
df_cv.head()

from fbprophet.diagnostics import performance_metrics
df_p = performance_metrics(df_cv)
df_p.head()

from fbprophet.plot import plot_cross_validation_metric
fig = plot_cross_validation_metric(df_cv, metric='mape')


