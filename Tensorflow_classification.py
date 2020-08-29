#!/usr/bin/env python
# coding: utf-8

# In[4]:


from __future__ import absolute_import,division,print_function,unicode_literals
import tensorflow as tf

import pandas as pd


# In[5]:


CSV_COLUMN.NAMES = ['SepalLength','SepalWidth','PetalLength','PetalWidth','Species']
SPECIES =['Setosa','Versicolor','Virginica']


# In[7]:


train_path = tf.keras.utils.get_file("iris_training.csv","https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path = tf.keras.utils.get_file("iris_test.csv","https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
train.head()


# In[9]:


from __future__ import absolute_import,division,print_function,unicode_literals
import tensorflow as tf

import pandas as pd
CSV_COLUMN_NAMES = ['SepalLength','SepalWidth','PetalLength','PetalWidth','Species']
SPECIES =['Setosa','Versicolor','Virginica']
train_path = tf.keras.utils.get_file("iris_training.csv","https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path = tf.keras.utils.get_file("iris_test.csv","https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
train.head()


# In[10]:


train_y = train.pop('Species')
test_y = test.pop('Species')


# In[11]:


train.head()


# In[12]:


train.shape()


# In[13]:


train.shape


# In[14]:


train.describe()


# In[22]:


def input_fn(features, labels, training = True, batch_size=256):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features),labels))
    if training: 
        dataset = dataset.shuffle(1000).repeat()
    return dataset.batch(batch_size)
my_feature_columns = []
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))
    
classifier = tf.estimator.DNNClassifier(feature_columns = my_feature_columns,
                                       #two hidden layers of 30 and 10 nodes
                                       hidden_units=[30,10],
                                       # the model must choose between 3 classes
                                       n_classes=3)

classifier.train(input_fn=lambda: input_fn(train, train_y, training=True),
                steps=5000)


# In[20]:





# In[23]:


eval_result=classifier.evaluate(input_fn=lambda: input_fn(test, test_y, training=False))
print('\nTest set Accuracy: {accuracy:0.3f}\n'.format(**eval_result))


# In[25]:


def input_fn(features, batch_size=256):
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_sizes)

features = ['SepalLength','SepalWidth','PetalLength','PetalWidth']
predict = {}

print('Please Type numeric values as prompted.')
for feature in features:
    valid = True
    while valid:
        val = input(feature +": ")
        if not val.isdigit(): valid = False
            
    predict[feature] = [float(val)]
predictions = classifier.predict(input_fn = lambda: input_fn(predict))
for prod_dict in predictions:
    class_id = pred_dict['class_ids'][0]
    possibility = pred_dict['probabilities'][class_id]
    
    print('Prediction is "{}"({:.1f}%'.format(SPECIES[class_id],100*probability))


# In[ ]:





# In[29]:


def input_fn(features, batch_size=256):
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_sizes)

features = ['SepalLength','SepalWidth','PetalLength','PetalWidth']
predict = {}

print('Please Type numeric values as prompted.')
for feature in features:
    valid = True
    while valid:
        val = input(feature +": ")
        if not val.isdigit(): valid = False
            
    predict[feature] = [float(val)]
predictions = classifier.predict(input_fn = lambda: input_fn(predict))
for pred_dict in predictions:
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]
    
    print('Prediction is "{}"({:.1f}%'.format(SPECIES[class_id],100*probability))


# In[36]:


def input_fn(features, batch_sizes=256):
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_sizes)

features = ['SepalLength','SepalWidth','PetalLength','PetalWidth']
predict = {}

print('Please Type numeric values as prompted.')
for feature in features:
    valid = True
    while valid:
        val = input(feature +": ")
        if not val.isdigit(): valid = False
            
    predict[feature] = [float(val)]
 
predictions = classifier.predict(input_fn = lambda: input_fn(predict))
for pred_dict in predictions:
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]
    
    print('Prediction is "{}"({:.1f}%)'.format(SPECIES[class_id],100*probability))


# In[ ]:




