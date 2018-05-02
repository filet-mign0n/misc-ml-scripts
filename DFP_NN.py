import sys
import warnings
import itertools
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest


warnings.filterwarnings('ignore')
PLOT = False
file_path = '/Users/jonas/Code/data/DFP_NN_01_rand_100K_preprocessed.csv'


df = pd.read_csv(file_path)
#df = df[df['advertiser'] == 'Pubmatic']
df = df[:5000]

cats = list(df.select_dtypes(include=['object']).columns)
feat_cols = [ cat + '_cat' for cat in cats ] + ['pv']
target = ['ecpm']
all_cols = feat_cols + target

df = df[all_cols]
df = df.drop_duplicates()

#print(df[all_cols].head(3))
#print(df.columns)

if PLOT:
    df['pv'].hist(bins=600)
    gb_spec = {'ecpm': pd.DataFrame.mean}
    m = df.groupby(['pv']).agg(gb_spec).reset_index()
    fig, ax1 = plt.subplots()
    ax1.bar('pv', 'ecpm', data=m)
    plt.legend()
    plt.show()

msk = np.random.rand(len(df)) < 0.8
train = df[msk]
test = df[~msk]
print(len(train), len(test))

# remove outliers
clf = IsolationForest(max_samples = 100, random_state = 42)
clf.fit(train)
y_noano = clf.predict(train)
y_noano = pd.DataFrame(y_noano, columns = ['Top'])
train = train.iloc[y_noano[y_noano['Top'] == 1].index.values]
train.reset_index(drop = True, inplace = True)
#print(train)

# scale features
mat_train = np.matrix(train)
mat_test  = np.matrix(test)
# mat_new = np.matrix(train.drop('ecpm', axis=1)) ???
mat_y = np.array(train.ecpm).reshape((train.ecpm.shape[0],1))

prepro_y = MinMaxScaler()
prepro_y.fit(mat_y)

prepro = MinMaxScaler()
prepro.fit(mat_train)

prepro_test = MinMaxScaler()
prepro_test.fit(mat_test) # (mat_new) ???

train = pd.DataFrame(prepro.transform(mat_train), columns=all_cols)
test  = pd.DataFrame(prepro_test.transform(mat_test),columns=all_cols)

#print(train.head(4))
#print(train.head(3))

x_train, x_test, y_train, y_test = train_test_split(
    train[feat_cols], train['ecpm'], test_size=0.33, random_state=42)
y_train = pd.DataFrame(y_train, columns=target)

training_set = pd.DataFrame(x_train, columns=feat_cols)\
                    .merge(y_train, left_index=True, right_index=True)

y_test = pd.DataFrame(y_test, columns=target)
testing_set = pd.DataFrame(x_test, columns=feat_cols)\
                    .merge(y_test, left_index=True, right_index=True)

testing_set.reset_index(drop=True, inplace=True)

col_train = list(train.columns)
training_sub = training_set[col_train]

#print('tain set')
#print(training_set)
#print('test set')
#print(testing_set)
#sys.exit(0)

# Neural Net
feature_cols = [tf.contrib.layers.real_valued_column(k) for k in feat_cols]
tf.logging.set_verbosity(tf.logging.ERROR)

# 5 hidden layers with repsectly 200, 100, 50, 25 and 12 units and activation func is  Relu
regressor = tf.contrib.learn.DNNRegressor(
    feature_columns=feature_cols, activation_fn=tf.nn.relu, hidden_units=[200, 100, 50, 25, 12])
    #optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.1))

def input_fn(data_set, pred=False):
    feature_cols = {k: tf.constant(data_set[k].values) for k in feat_cols}
    if pred == False:
        labels = tf.constant(data_set['ecpm'].values)
        return feature_cols, labels
    if pred == True:
        return feature_cols

# Deep Neural Network Regressor with the training set which contain the data split by train test split
regressor.fit(input_fn=lambda: input_fn(training_set), steps=2000)

# Evaluation on the test set created by train_test_split
ev = regressor.evaluate(input_fn=lambda: input_fn(testing_set), steps=1)

loss_score1 = ev["loss"]
print("Final Loss on the testing set: {0:f}".format(loss_score1))

# predictions
y = regressor.predict(input_fn=lambda: input_fn(testing_set))
predictions = list(itertools.islice(y, testing_set.shape[0]))
predictions = pd.DataFrame(
    #np.array(predictions).reshape(len(predictions),1),
    prepro_y.inverse_transform(np.array(predictions).reshape(len(predictions),1)),
    columns=['Prediction'])

reality = pd.DataFrame(prepro.inverse_transform(testing_set), # testing_set
                       columns=col_train).ecpm

for i, j in zip(predictions.values, reality.values):
    print('[DFP_logs ECPM] real:', j, 'pred:', i[0])

r2 = r2_score(predictions.values, reality.values)
mse = mean_squared_error(predictions.values, reality.values)

print('r2:', r2)
print('mse:', mse)


matplotlib.rc('xtick', labelsize=10)
matplotlib.rc('ytick', labelsize=10)
fig, ax = plt.subplots(figsize=(10, 5))
plt.style.use('ggplot')
plt.plot(predictions.values, reality.values, 'ro')
plt.xlabel('Predictions', fontsize = 20)
plt.ylabel('Reality', fontsize = 20)
plt.title('Predictions x Reality on dataset Test', fontsize=10)
ax.plot([0, 10], [0, 10], 'k--', lw=4)
#ax.plot([reality.min(), reality.max()], [reality.min(), reality.max()], 'k--', lw=4)
plt.show()

sys.exit(0)

"""
X = df.as_matrix(columns=feat_cols)
Y = df.as_matrix(columns=target)

# 80% train, 20% test
Xtrn, Xtest, Ytrn, Ytest = train_test_split(X, Y, test_size=0.2, shuffle=False, stratify=None)

#print('Xtrn\n', Xtrn, '\nXtest\n', Xtest, '\nYtrn\n', Ytrn, '\nYtest\n', Ytest)
"""
