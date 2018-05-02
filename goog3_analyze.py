from __future__ import division
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('/Users/jonas/Downloads/simple_goog3.csv')
df = df.astype(float)
df.fillna(0, inplace=True)

dd = df.loc[:, df.columns != 'maxbid']
#dd.loc[:, dd.columns != 'max'].max(axis=1)
# getting the maxs
df['max_'] = dd.loc[:, dd.columns != 'max'].max(axis=1)
df['max_boost'] = dd.loc[:, dd.columns != 'max'].idxmax(axis=1)
m = df[['maxbid', 'max_boost', 'max_']]

m = m.sort_values('maxbid')
m.reset_index(inplace=True, drop=True)
m['max_boost'] = m['max_boost'].apply(pd.to_numeric)
#m = m[m.maxbid <= 2000]
fig, ax1 = plt.subplots()
#ax1.bar('maxbid', 'max_boost', data=m, color='orange', width=8)
ax1.stackplot('maxbid', 'max_boost', data=m, color='orange')
#ax1.plot('maxbid', 'max_boost', data=m, color='blue')
#ax1.scatter('maxbid', 'max_boost', data=m, color='blue')

ax2 = ax1.twinx()
ax2.plot('maxbid', 'max_', data=m, color='blue', alpha=0.5, linewidth=3.0,
         label = 'max_ecpm')
ax2.grid()
ax2.set_ylabel('net_effective_cpm')
ax2.legend(loc='lower right')
ax1.set_xlabel('maxbid')
ax1.set_ylabel('boost %')
ax1.legend()

plt.show()

'''
X = df['dfp_logs Product'] / 100
imps = df['Ad Exchange.1']

log_imps = df['log_rev_adx'] = np.log(df['Ad Exchange.1'])
df['slope_adx'] = (df['dfp_logs Product'] * 1.75) / 100
df['test2'] = df['slope_adx'] * ((100 - df['log_rev_adx']) / 100)
df['test3'] = df['dfp_logs Product'] * (1.75 - ((7 - df['log_rev_adx']) / 10))
df['test4'] = df['dfp_logs Product'] * 1.75 + (df['dfp_logs Product']**1/2 * (df['log_rev_adx'] / 10))
df['test4'] = df['dfp_logs Product'] * 1.75 + (df['dfp_logs Product']**1/2 * (df['log_rev_adx'] / 10))

polyfit = np.polyfit(X, df['Ad Exchange'], deg=2)#, w=imps)
a = polyfit[0]
b = polyfit[1]
c = polyfit[2]
df['poly2'] = a * X**2 + b * X + c

polyfit = np.polyfit(X, df['Ad Exchange'], deg=3)#, w=imps)
a = polyfit[0]
b = polyfit[1]
c = polyfit[2]
d = polyfit[3]
df['poly3'] = a * X**3 + b * X**2 + c * X + d

polyfit = np.polyfit(X, df['Ad Exchange'], deg=4)#, w=imps)
a = polyfit[0]
b = polyfit[1]
c = polyfit[2]
d = polyfit[3]
e = polyfit[4]
df['poly4_fit'] = a * X**4 + b * X**3 + c * X**2 + d * X + e

#df['poly4X'] = a * X**4 + b * X**3 + (c-0.003) * X**2 + d * X + e
df['poly4_mod1'] = a * X**4 + (b-0.00005) * X**3 + c * X**2 + d * X + e
df['poly4_mod2'] = a * X**4 + (b-0.0001) * X**3 + c * X**2 + d * X + e
df['poly4_mod3'] = a * X**4 + (b-0.00015) * X**3 + c * X**2 + d * X + e
df['poly4_mod4'] = a * X**4 + (b-0.0002) * X**3 + c * X**2 + d * X + e
#df = df[185:]
print(log_imps)
print(polyfit)
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
#ax3 = ax2.twinx()
ax1.plot('dfp_logs Product', 'Ad Exchange', data=df, color='darkgreen', linewidth=3)
ax1.plot('dfp_logs Product', 'Exchange Bidding', data=df, color='limegreen')
ax1.plot('dfp_logs Product', 'Ad Server', data=df, color='blue')
#ax1.plot('dfp_logs Product', 'poly2', data=df, color='thistle', alpha=0.7)
ax1.plot('dfp_logs Product', 'poly3', data=df, color='thistle', alpha=0.7)
ax1.plot('dfp_logs Product', 'poly4_fit', data=df, color='fuchsia', alpha=0.7)
ax1.plot('dfp_logs Product', 'poly4_mod1', data=df, color='purple', alpha=0.7)
ax1.plot('dfp_logs Product', 'poly4_mod2', data=df, color='violet', alpha=0.7)
ax1.plot('dfp_logs Product', 'poly4_mod3', data=df, color='navy', alpha=0.7)
ax1.plot('dfp_logs Product', 'poly4_mod4', data=df, color='black', alpha=0.7)
ax1.grid()
#ax1.plot('dfp_logs Product', 'test2', data=df, color='orange')
ax2.bar('dfp_logs Product', 'Ad Exchange.1', data=df, color='orange', width=3, alpha=0.7)
#ax1.plot('dfp_logs Product', 'test4', data=df, color='black')
#ax2.plot('dfp_logs Product', 'log_rev_adx', data=df, color='yellow')
ax2.legend()
ax1.legend()
#ax3.legend()

plt.show()
'''
