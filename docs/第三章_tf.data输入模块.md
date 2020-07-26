> 代码和其他资料在 [github](https://github.com/zhangasia/Tensorflow2)

#### 一、tf.data模块

* 数据分割

```python
import tensorflow as tf
dataset = tf.data.Dataset.from_tensor_slices([1,2,3,4,5,6,7]) #1维
dataset2 = tf.data.Dataset.from_tensor_slices([[1,2],[3,4],[5,6]]) #2维
dataset_dic = tf.data.Dataset.from_tensor_slices({'a':[1,2,3,4],'b':[6,7,8,9], 'c':[12,13,14,15]}) #字典
```

`tf.data.Dataset.from_tensor_slices()` 数据切割，并且转化为 `Tensor` 类型。

```python
dataset
for ele in dataset:
    print(ele)
```
输入：
```
<TensorSliceDataset shapes: (), types: tf.int32>
tf.Tensor(1, shape=(), dtype=int32)
tf.Tensor(2, shape=(), dtype=int32)
tf.Tensor(3, shape=(), dtype=int32)
tf.Tensor(4, shape=(), dtype=int32)
tf.Tensor(5, shape=(), dtype=int32)
tf.Tensor(6, shape=(), dtype=int32)
tf.Tensor(7, shape=(), dtype=int32)
```
```python
for ele in dataset:
    print(ele.numpy())
```
输入：
```
1
2
3
4
5
6
7
```
```python
dataset2
for ele2 in dataset2:
    print(ele2.numpy())
```
输入：
```
<TensorSliceDataset shapes: (2,), types: tf.int32>
[1 2]
[3 4]
[5 6]
```
```python
dataset_dic
for ele_dic in dataset_dic:
    print(ele_dic)
```
输入：
```
<TensorSliceDataset shapes: {a: (), b: (), c: ()}, types: {a: tf.int32, b: tf.int32, c: tf.int32}>
{'a': <tf.Tensor: shape=(), dtype=int32, numpy=1>, 'b': <tf.Tensor: shape=(), dtype=int32, numpy=6>, 'c': <tf.Tensor: shape=(), dtype=int32, numpy=12>}
{'a': <tf.Tensor: shape=(), dtype=int32, numpy=2>, 'b': <tf.Tensor: shape=(), dtype=int32, numpy=7>, 'c': <tf.Tensor: shape=(), dtype=int32, numpy=13>}
{'a': <tf.Tensor: shape=(), dtype=int32, numpy=3>, 'b': <tf.Tensor: shape=(), dtype=int32, numpy=8>, 'c': <tf.Tensor: shape=(), dtype=int32, numpy=14>}
{'a': <tf.Tensor: shape=(), dtype=int32, numpy=4>, 'b': <tf.Tensor: shape=(), dtype=int32, numpy=9>, 'c': <tf.Tensor: shape=(), dtype=int32, numpy=15>}
```

* 其他常用操作

```python
for ele_np in dataset_np.take(4): # 取出前四个
    print(ele_np)
dataset_np = dataset_np.shuffle(7) # 打乱顺序
dataset_np = dataset_np.repeat(count = 3) #重复3次，为None无限循环
dataset = dataset.map(tf.square) # 取平方
```


#### 二、手写识别实例

```python
import tensorflow as tf
(train_images,train_labels),(test_images,test_labels) = tf.keras.datasets.mnist.load_data()
train_images = train_images / 255
test_images = test_images / 255
ds_train_img = tf.data.Dataset.from_tensor_slices(train_images)
ds_train_lab = tf.data.Dataset.from_tensor_slices(train_labels)
ds_train = tf.data.Dataset.zip((ds_train_img,ds_train_lab)) # 数据合并
ds_train = ds_train.shuffle(buffer_size = 10000).repeat().batch(64)
ds_test = tf.data.Dataset.from_tensor_slices((test_images,test_labels))
ds_test = ds_test.batch(64)
model = tf.keras.Sequential([tf.keras.layers.Flatten(input_shape = (28,28)),tf.keras.layers.Dense(128,activation = 'relu'),tf.keras.layers.Dense(10,activation = 'softmax')])
model.compile(optimizer = 'adam',loss = 'sparse_categorical_crossentropy',metrics = ['accuracy'])
steps_per_epoch = train_images.shape[0] // 64 # 每个epoch的步数
model.fit(ds_train,epochs = 5,steps_per_epoch = steps_per_epoch,validation_data = ds_test,validation_steps = 10000 // 64)
```