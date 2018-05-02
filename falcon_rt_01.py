import pandas as pd
import numpy as np

from matplotlib import pyplot as plt

df = pd.DataFrame.from_csv('/Users/jonas/Code/data/data_falcon_rt_20171030.csv')

def draw_partner(type=None):
    i = 0
    for p in df['partner'].unique():
        df_p = df[df['partner'] == p][['t','ecpm']]
        green = df_p['ecpm'].sum() / 100000
        if green < 20:
            #print(p, 'is too greedy, will only cough up:', green)
            continue
        df_p['t'] = pd.to_datetime(df_p['t'])
        df_p['h'] = df_p['t'].dt.hour
        if type == 'mean':
            df_p = df_p.groupby(['h'])['ecpm'].mean() / 100000
        else:
            df_p = df_p.groupby(['h'])['ecpm'].sum()
            df_p = df_p.cumsum() / 100000 #$
        if i == 0:
            print(df_p.head(), df_p.tail())
            i+=1
        df_p.plot(label=p)
        plt.legend()

def draw(segment='partner', type=None, thresh=500):
    i = 0
    for p in df[segment].unique():
        df_p = df[df[segment] == p][['t','ecpm']]
        green = df_p['ecpm'].sum() / 100000
        if green < 500:
            #print(p, 'is too greedy, will only cough up:', green)
            continue
        df_p['t'] = pd.to_datetime(df_p['t'])
        df_p['h'] = df_p['t'].dt.hour
        df_p = df_p.groupby(['h'])['ecpm'].sum()
        if type == 'growth':
            df_p = df_p.pct_change()
        else:
            df_p = df_p.cumsum() / 100000 #$

        if i == 0:
            print(df_p.head(), df_p.tail())
            i += 1
        df_p.plot(label=p)
        plt.legend()

if __name__ == '__main__':
    draw(segment='camp', type='growth')
    plt.show()
    importp pdb; pdb.set_trace()
