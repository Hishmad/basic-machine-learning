
# coding: utf-8

# # Lesson Six

# TensorFlow according to the documentation: is a powerflow open source software library released in 2015 by Google to make it wasier to design, build, and train deep learning models. At a high level, TensorFlow is a Python library that allows users to express arbitrary computation as a graph of data flows.

# In[62]:

# To plot pretty figures
get_ipython().magic('matplotlib inline')
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Common imports
import numpy as np
import os
import tensorflow as tf


# In[63]:

def reset_graph(seed=1):
    '''
    To reset some random
    '''
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


# In[64]:

# Where to save the figures
def save_data(fig_id, tight_layout=True, PROJECT_ROOT_DIR = ".", LESSON_ID = "tensorflow"):
    '''
    Saving figures
    '''
    path = os.path.join(PROJECT_ROOT_DIR, "images", LESSON_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)


# # Lets start the lesson

# In[24]:

# Resert all parameters 
reset_graph()

# Create a graph, well this does not perform ny computation, it just creates a graph.
x = tf.Variable(3, name="X")
y = tf.Variable(4, name="y")
f = x+y


# In[25]:

# Initialize the Session() instance, this is what we need to initialize the variables and evaluate f.
# The Session() takes care of placing the operations onto devices such as CPUs, GPUs or TPUs and running them. 
sess = tf.Session()
# run() pass the x variable with initializer 
sess.run(x.initializer)
# run() pass the x variable with initializer 
sess.run(y.initializer)
# run() f this will return x + y = 7
result = sess.run(f) 


# In[26]:

# And finally close the Session() to frees up resources
sess.close()


# In[27]:

# we can also use with block, using this way we dont have to invoke close()
with tf.Session() as sess:
    x.initializer.run() # this is equivalent to calling tf.get_default_session().run(x.initializer)
    y.initializer.run() # this is equivalent to calling tf.get_default_session().run(x.initializer)
    result = f.eval() # this is equivalent to calling tf.get_default.session().run(f)


# In[31]:

# Instead of manually running the initializer one for each variable, 
# we can use the global_variables_initializer()
global_init = tf.global_variables_initializer()

with tf.Session() as sess:
    global_init.run() # This will initilaize all the variables
    result = f.eval()


# In[32]:

# Another way is to use tf.InteractiveSession() it will automatically sets itself as the default session,
# so we dont need the with block but we need to invoke close() to free resources.
sess = tf.InteractiveSession()
global_init = tf.global_variables_initializer()
global_init.run()
result = f.eval()


# In[33]:

# and dont foreget to close
sess.close()


# In TensorFlow we refere the computation graph as the Construction Phase, which is typically builds a computation graph representing the Model and the algorithm to train it. Next will come the Execution Phase which will loop to evaluates a training step like one step for each mini-batch.

# In[52]:

reset_graph()

# Now, any node we create will be added to the default graph
Xx = tf.Variable(5, name="Xx")
Xx.graph is tf.get_default_graph() # this should return True


# In[53]:

# But we may want to create a temporarily graph and making it the default graph inside the with block:
graph = tf.Graph()
with graph.as_default():
    x_temp = tf.Variable(10)


# In[54]:

# This will return True
x_temp.graph is graph


# In[55]:

# This will return False
x_temp is tf.get_default_graph()


# In[56]:

# Final we better to avoid dublicate nodes, we must invoke tf.reset_default_graph()
tf.reset_default_graph()


# In[59]:

w = tf.constant(10)
x = w + 1
y = x + 2
z = x * 3

with tf.Session() as sess:
    y_val = y.eval()  # TensorFlow will automatically detects that y depends on x, which depends on w.
    z_val = z.eval()  # TensorFlow will automatically detects that z depends on x, which depends on w.


# In[60]:

# We can evaluate y and z efficiently, without evaluating w and x twice as in the code above, we can do like this:
with tf.Session() as sess:
    y_eval, z_eval = sess.run([y, z])


# Take note that in single-proces TensorFlow, multiple sessions do not share any state. But in distributed TensorFlow, variable state is stored on the servers, not in the sessions, so multiple sessions can share the same variables.

# TensorFlow operations can take any number of inputs and return any number of outputs. According to the documentations: The central unit of data in TensorFlow is the tensor. A tensor consists of a set of primitive values shaped into an array of any number of dimensions. A tensor's rank is its number of dimensions. Here are some examples of tensors:
# 

# In[61]:

3 # a rank 0 tensor; this is a scalar with shape []
[1. ,2., 3.] # a rank 1 tensor; this is a vector with shape [3]
[[1., 2., 3.], [4., 5., 6.]] # a rank 2 tensor; a matrix with shape [2, 3]
[[[1., 2., 3.]], [[7., 8., 9.]]] # a rank 3 tensor with shape [2, 1, 3]


# We can perform computations on arrays of any shape. SO the definition of theta corresponds to the Normal Equation (

# In[65]:

# The theta is a Threshold value of an artificial neuron.
reset_graph()


# In[66]:

# Get the dataset from sklearn
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
m, n = housing.data.shape


# In[68]:

housing_data_and_bias = np.c_[np.ones((m, 1)), housing.data]

# In[69]:

X = tf.constant(housing_data_and_bias, dtype=tf.float32, name="X")

# Reshape the argets y from (n,) into (n, 1) array
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")


# In[85]:

# Lets calculate the theta using TensorFlow, then later we will compare with NumPy's calculation
XT = tf.transpose(X)
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)
with tf.Session() as sess:
    theta_value = theta.eval()



# The benefit of the above code compared to computing the Normal Equation directly using NumPy is that TensorFlow will automatically run this on our GPU card if we have one.

# In[96]:

# Let's compare with NumPy
X = housing_data_and_bias
y = housing.target.reshape(-1, 1) # Reshape the argets y from (n,) into (n, 1) array
yy = housing.target
theta_numpy = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)


# In[100]:

# Let's compare with Scikit-Learn
from sklearn.linear_model import LinearRegression
lin_regs = LinearRegression()
lin_regs.fit(housing.data, housing.target.reshape(-1, 1))
theta_sklearn = np.r_[lin_regs.intercept_.reshape(-1, 1), lin_regs.coef_.T]



