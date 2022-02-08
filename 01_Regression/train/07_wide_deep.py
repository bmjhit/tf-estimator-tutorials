import os, sys
import tensorflow as tf
import tensorflow.estimator
import numpy as np
import math

MODEL_NAME = 'wideDeep'
train_data_file_pattern = '../data/train*.tfrecords'
valid_data_file_pattern = '../data/valid*.tfrecords'
test_data_file_pattern = '../data/test*.tfrecords'

RESUME_TRAINING = False
PROCESS_FEATURE = True
EXTEND_FEATURE_COLUMNS = True
MULTI_THREADING = True

# 定义特征列
HEADER = ['key', 'x', 'y', 'alpha', 'beta', 'target']
HEADER_DEFAULTS = [[0], [0.0], [0.0], ['NA'], ['NA'], [0.0]]

NUMERIC_FEATURE_NAMES = ['x', 'y']
CATEGORICAL_FEATURE_NAMES_WITH_VOCABULARY = {'alpha': ['ax01', 'ax02'], 'beta': ['bx01', 'bx02']}
CATEGORICAL_FEATURE_NAMES = list(CATEGORICAL_FEATURE_NAMES_WITH_VOCABULARY.keys())

FEATURE_NAMES = NUMERIC_FEATURE_NAMES + CATEGORICAL_FEATURE_NAMES

TARGET_NAME = 'target'

UNUSED_FEATURE_NAMES = list(set(HEADER) - set(NUMERIC_FEATURE_NAMES) - {TARGET_NAME})

# print(HEADER)
# print(NUMERIC_FEATURE_NAMES)
# print(CATEGORICAL_FEATURE_NAMES)
# print(FEATURE_NAMES)
# print(UNUSED_FEATURE_NAMES)

# 定义数据输入方法

def parse_tf_example(example_proto):
    feature_spec = {}
    for feature_name in NUMERIC_FEATURE_NAMES:
        # 将每列数据转化为定长tensor， 输出的tensor的shape为(batch_size, shape)
        feature_spec[feature_name] = tf.FixedLenFeature(shape=(1), dtype=tf.float32)
    for feature_name in CATEGORICAL_FEATURE_NAMES:
        feature_spec[feature_name] = tf.FixedLenFeature(shape=(1), dtype=tf.string)
    feature_spec[TARGET_NAME] = tf.FixedLenFeature(shape=(1), dtype=tf.float32)
    # 从tf.example内将数据解析出来
    # 输出：字典格式数据，将特征映射到Tensor、SparseTensor值
    parsed_features = tf.parse_example(serialized=example_proto, features=feature_spec)
    target = parsed_features.pop(TARGET_NAME)
    return parsed_features, target

def process_features(features):
    # example
    # clipping 将数值限制在一定范围之内
    features['x'] = tf.clip_by_value(features['x'], clip_value_min=-3, clip_value_max=3)
    features['y'] = tf.clip_by_value(features['y'], clip_value_min=-3, clip_value_max=3)
    # polynomial expansion
    features['x_2'] = tf.square(features['x'])
    features['y_2'] = tf.square(features['y'])
    # nonlinearity
    features['xy'] = features['x'] * features['y']
    # custom logic
    # tf.squared_difference 计算对应元素差的平方
    features['dist_xy'] = tf.sqrt(tf.squared_difference(features['x'], features['y']))
    features['sin_x'] = tf.sin(features['x'])
    features['sin_y'] = tf.sin(features['y'])

    return features

# data pipeline input function
def tfrecords_input_fn(files_name_pattern, mode=tf.estimator.ModeKeys.EVAL, num_epochs=None, batch_size=100):
    shuffle = True if mode==tf.estimator.ModeKeys.TRAIN else False
    print('data input fn: ')
    print('================')
    print('Mode: {}'.format(mode))
    print('Input file(s): {}'.format(files_name_pattern))
    print('Batch_size: {}'.format(batch_size))
    print('Epochs: {}'.format(num_epochs))
    print('Shuffle: {}'.format(shuffle))
    print('================')
    file_names = tf.matching_files(files_name_pattern)
    dataset = tf.data.TFRecordDataset(filenames=file_names)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=2 * batch_size + 1)
    # tf.dataset: shuffle repeat batch
    # batch 会将每batch_size个数据分为一组，同时增加了一维， 若shuffle、repeat在batch操作之后则shuffle、repeat的都是batch而不是其中数据
    dataset = dataset.batch(batch_size)
    # 自定义处理函数， 将数据转化为Tensor
    dataset = dataset.map(lambda tf_example: parse_tf_example(tf_example))
    if PROCESS_FEATURE:
        dataset = dataset.map(lambda features, target: (process_features(features), target))
    dataset = dataset.repeat(num_epochs)
    iterator = dataset.make_one_shot_iterator()
    features, target = iterator.get_next()
    return features, target

# define feature columns
def extend_feature_columns(feature_columns, hparams):
    num_buckets = hparams.num_buckets
    embedding_size = hparams.embedding_size
    buckets = np.linspace(-3, 3, num_buckets).tolist()
    # 交叉特征 输出维度为指定维度的one-hot
    alpha_X_beta = tf.feature_column.crossed_column([feature_columns['alpha'], feature_columns['beta']], 4)
    # 将连续值进行分桶离散化
    x_bucketized = tf.feature_column.bucketized_column(feature_columns['x'], boundaries=buckets)
    y_bucketized = tf.feature_column.bucketized_column(feature_columns['y'], boundaries=buckets)
    x_bucketized_X_y_bucketized = tf.feature_column.crossed_column([x_bucketized, y_bucketized], num_buckets**2)
    x_bucketized_X_y_bucketized_embedded = tf.feature_column.embedding_column(x_bucketized_X_y_bucketized, embedding_size)

    feature_columns['alpha_X_beta'] = alpha_X_beta
    feature_columns['x_bucketized_X_y_bucketized'] = x_bucketized_X_y_bucketized
    feature_columns['x_bucketized_X_y_bucketized_embedded'] = x_bucketized_X_y_bucketized_embedded
    return feature_columns

# 将tensor转化为tf.feature_column
def get_feature_columns(hparams):
    CONSTRUCTED_NUMERIC_FEATURES_NAMES = ['x_2', 'y_2', 'xy', 'dist_xy', 'sin_x', 'sin_y']
    all_numeric_feature_names = NUMERIC_FEATURE_NAMES.copy()
    if PROCESS_FEATURE:
        all_numeric_feature_names += CONSTRUCTED_NUMERIC_FEATURES_NAMES
    numeric_columns = {feature_name: tf.feature_column.numeric_column(feature_name) for feature_name in all_numeric_feature_names}
    categorical_column_with_vocabulary = {item[0]: tf.feature_column.categorical_column_with_vocabulary_list(item[0], item[1])
                                                for item in CATEGORICAL_FEATURE_NAMES_WITH_VOCABULARY.items()}
    feature_columns = {}
    if numeric_columns is not None:
        feature_columns.update(numeric_columns)
    if categorical_column_with_vocabulary is not None:
        feature_columns.update(categorical_column_with_vocabulary)
    if EXTEND_FEATURE_COLUMNS:
        feature_columns = extend_feature_columns(feature_columns, hparams)
    return feature_columns

# define an Estimator Creation Function
# get wide and deep feature columns
def get_wide_deep_columns():



if __name__=='__main__':
    feature_columns = get_feature_columns(tf.contrib.training.HParams(num_buckets=5, embedding_size=3))
    print("feature columns: {}".format(feature_columns))


