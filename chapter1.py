import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # removes warnings and info printing
import tensorflow as tf

# INITIALIZATION METHODS  #############################################################################################

x = tf.constant([[1, 2, 3], [4, 5, 6]])

y = tf.ones((3, 3), tf.int32)
y = tf.zeros((3, 3))
y = tf.eye(3, dtype=tf.int32)  # identity matrix
y = tf.range(start=1, limit=10, delta=2)  # similar to range function

z = tf.random.normal((3, 3), mean=5, stddev=2)  # random samples from a normal distribution
z = tf.random.uniform((2, 3), minval=0, maxval=1)  # random samples from uniform distribution

a = tf.cast(x, dtype=tf.float32)  # convert datatype

# MATHEMATICAL OPERATIONS  ############################################################################################

x = tf.constant([1, 2, 3])
y = tf.constant([9, 8, 7])

z = tf.add(x, y)  # elementwise addition (same as x+y)
z = x + y  # elementwise

z = tf.subtract(x, y)
z = x - y  # elementwise subtraction

z = tf.divide(x, y)
z = x / y

z = tf.multiply(x, y)
z = x * y

z = tf.tensordot(x, y, axes=1)  # dot product
z = tf.reduce_sum(x * y, axis=0)  # equivalent to above

z = x ** 5  # elementwise exponentiaton

x = tf.random.normal((2, 3))
y = tf.random.normal((3, 4))

z = tf.matmul(x, y)  # matrix multiplication
z = x @ y  # same as above

# INDEXING  ###########################################################################################################

x = tf.constant([0, 1, 1, 2, 3, 1, 2, 3])
y = x[1:3]  # ending index is non-inclusive
y = x[::2]  # take in steps of 2
y = x[::-1]  # reverse order

indices = tf.constant([0, 3, 5])
x_ind = tf.gather(x, indices)  # extracts values pertaining to indices

x = tf.constant([[1, 2],
                 [3, 4],
                 [5, 6]])  # 3 x 2 matrix

y = x[0, :]  # row 0, all columns
y = x[0:2, :]  # row 0,1 and all columns

# RESHAPING ###########################################################################################################

x = tf.range(9)

x = tf.reshape(x, (3, 3))
x = tf.transpose(x, perm=[1, 0])  # perm determines axes for transposing

print(x)
