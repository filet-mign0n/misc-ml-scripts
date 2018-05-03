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

sess = tf.Session()
saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=2)
saver.save(sess, 'my_test_model') # global_step=step,write_meta_graph=False)
#saves a model every 2 hours and maximum 4 latest models are saved.
saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=2)



df = pd.read_csv(file_path)
#df = df[df['advertiser'] == 'Rubicon']
#rows = 20000
#df = df[:rows]

cats = list(df.select_dtypes(include=['object']).columns)
feat_cols = [ cat + '_cat' for cat in cats ] + ['pv']
target = ['ecpm']
all_cols = feat_cols + target
df = df.drop_duplicates()

df_num = df[['pv', 'ecpm']]
df_cat = df.select_dtypes(include=['object'])
df = df_num.merge(df_cat, left_index=True, right_index=True)
print(list(df.columns))

#msk = np.random.rand(len(df)) < 0.8
#train = df[msk]
#test = df[~msk]
#print(len(train), len(test))

# remove outliers
clf = IsolationForest(max_samples = 100, random_state = 42)
clf.fit(df_num)
y_noano = clf.predict(df_num)
y_noano = pd.DataFrame(y_noano, columns = ['Top'])
df_num = df_num.iloc[y_noano[y_noano['Top'] == 1].index.values]
df_num.reset_index(drop=True, inplace=True)

df_cat = df_cat.iloc[y_noano[y_noano['Top'] == 1].index.values]
df_cat.reset_index(drop=True, inplace=True)

df = df.iloc[y_noano[y_noano['Top'] == 1].index.values]
df.reset_index(drop=True, inplace=True)

# scale features
col_df = list(df.columns)
col_num = list(df_num.columns)
col_num_bis = list(df_num.columns)
col_cat = list(df_cat.columns)
col_num_bis.remove('ecpm')

mat_train = np.matrix(df_num)
mat_y = np.array(df['ecpm'])

prepro_y = MinMaxScaler()
prepro_y.fit(mat_y.reshape(len(df['ecpm']),1))

prepro = MinMaxScaler()
prepro.fit(mat_train)

#df_num_scale = pd.DataFrame(prepro.transform(mat_train), columns=col_df)
#print(mat_train.shape, col_num)

df[col_num] = pd.DataFrame(prepro.transform(mat_train),columns=col_num)

COLUMNS = col_num
FEATURES = col_num_bis
FEATURES_CAT = col_cat
LABEL = 'ecpm'

engineered_features = []

for cat in FEATURES_CAT:
    sparse_column = tf.contrib.layers.sparse_column_with_hash_bucket(cat, hash_bucket_size=160)

    engineered_features.append(tf.contrib.layers.embedding_column(sparse_id_column=sparse_column, dimension=16, combiner='sum'))

training_set = df[FEATURES + FEATURES_CAT]
prediction_set = df[LABEL]
x_train, x_test, y_train, y_test = train_test_split(
    training_set[FEATURES + FEATURES_CAT], prediction_set, test_size=0.2, random_state=42
)
y_train = pd.DataFrame(y_train, columns = [LABEL])
y_test = pd.DataFrame(y_test, columns=[LABEL])

training_set = pd.DataFrame(x_train, columns = FEATURES+FEATURES_CAT)\
                .merge(y_train, left_index=True, right_index=True)
testing_set = pd.DataFrame(x_test, columns=FEATURES+FEATURES_CAT)\
                .merge(y_test, left_index=True, right_index=True)

training_set[FEATURES_CAT] = training_set[FEATURES_CAT].applymap(str)
testing_set[FEATURES_CAT] = testing_set[FEATURES_CAT].applymap(str)

def input_fn_new(data_set, training=True):
    continuous_cols = {k: tf.constant(data_set[k].values) for k in FEATURES}
    categorical_cols = {
        k: tf.SparseTensor(
            indices=[[i, 0] for i in range(data_set[k].size)],
            values=data_set[k].values,
            dense_shape=[data_set[k].size, 1]
        ) for k in FEATURES_CAT
    }
    feature_cols = dict(list(continuous_cols.items()) + list(categorical_cols.items()))

    if training == True:
        label = tf.constant(data_set[LABEL].values)
        return feature_cols, label

    return feature_cols

#categorical_cols = {
#    k: tf.SparseTensor(
#        indices=[[i, 0] for i in range(training_set[k].size)],
#        values=training_set[k].values,
#        dense_shape=[training_set[k].size, 1]
#    ) for k in FEATURES_CAT
#}

# 5 hidden layers with Relu
relu200 = tf.contrib.learn.DNNRegressor(feature_columns=engineered_features,
                                          activation_fn=tf.nn.relu,
                                          hidden_units=[200, 100, 50, 25, 12])
# 5 hidden layers with Relu
relu24 = tf.contrib.learn.DNNRegressor(feature_columns=engineered_features,
                                          activation_fn=tf.nn.relu,
                                          hidden_units=[24, 12, 6, 3])
# Shallow Network
shallow = tf.contrib.learn.DNNRegressor(feature_columns=engineered_features,
                                          activation_fn=tf.nn.relu,
                                          hidden_units=[1000])
# Leaky Relu
def leaky_relu(x): return tf.nn.relu(x) - 0.01 * tf.nn.relu(-x)
leaky = tf.contrib.learn.DNNRegressor(feature_columns=engineered_features,
                                          activation_fn=leaky_relu,
                                          hidden_units=[200, 100, 50, 25, 12])

regressors = [
    #('ReLU 200', relu200),
    ('ReLU 24', relu24),
    #('Leaky ReLU', leaky),
    #('Shallow Network (ReLU)', shallow),
]

for name, regressor in regressors:

    import pdb; pdb.set_trace()
    regressor.fit(input_fn=lambda: input_fn_new(training_set), steps=2000)
    ev = regressor.evaluate(input_fn=lambda: input_fn_new(testing_set, training=True), steps=1)
    loss_score4 = ev['loss']
    print('\n### ', name, '###')
    print('Final Loss on the testing set: {0:f}'.format(loss_score4))

    y = regressor.predict(input_fn=lambda: input_fn_new(testing_set))
    predictions = list(itertools.islice(y, testing_set.shape[0]))
    predictions = pd.DataFrame(prepro_y.inverse_transform(
        np.array(predictions).reshape(len(predictions),1)))

    reality = pd.DataFrame(prepro.inverse_transform(testing_set[col_num]), columns=col_num)['ecpm']

    #for i, j in zip(predictions.values, reality.values):
    #    print('[DFP_logs ECPM] real:', j, 'pred:', i[0])

    r2 = r2_score(predictions.values, reality.values)
    mse = mean_squared_error(predictions.values, reality.values)
    print('r2:', r2)
    print('mse:', mse)


matplotlib.rc('xtick', labelsize=10)
matplotlib.rc('ytick', labelsize=10)
fig, ax = plt.subplots(figsize=(20, 20))
plt.style.use('ggplot')
plt.plot(predictions.values, reality.values, 'ro')
plt.xlabel('Predictions', fontsize = 20)
plt.ylabel('Reality', fontsize = 20)
plt.title('Predictions x Reality on dataset Test', fontsize=20)
ax.plot([0, 10], [0, 10], 'k--', lw=4)
#ax.plot([reality.min(), reality.max()], [reality.min(), reality.max()], 'k--', lw=4)
plt.show()

sys.exit(0)
