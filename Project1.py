#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 0) INITIAL BASICS

# Defining the generating function
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def gen_func(x, t0=1.15, t1=0.5, t2=0.12):
    y = t0 + t1 * x + t2 * x**2
    return y


# In[2]:


# 1) Generating 15 data points randomly chosen between 0 and 10 (for x)
np.random.seed(2020802018)

x = np.random.rand(15, 1) * 10
x = pd.DataFrame(x, columns=["x"])

print(x)


# In[3]:


# 2) Adding noise to each data point (Gaussian, 0 mean and 0.15 sigma) and plot
# In Jupyter notebook, seed must be set for each cell random operation
np.random.seed(2020802018)
# Adding noise with mean zero and sigma=0.15
noise = np.random.normal(loc=0.0, scale=0.15, size=(1, 15))
# Converting noise array to Pandas dataframe
noise = pd.DataFrame(noise.transpose(), columns=["noise"])

# Arranging final noise added data
dt = x
dt["y"] = dt["x"].apply(lambda x: gen_func(x))
dt["y"] = dt["y"] + noise["noise"]

print(pd.DataFrame(dt))


# In[4]:


fig, ax = plt.subplots()
ax.scatter(dt["x"], dt["y"], color="r")
ax.set_ylabel("y")
ax.set_xlabel("x")
ax.set_title("Generated Noise Added Data")
# plt.savefig('datapoints.png')


# In[5]:


# 3) RLS Algorith
# 3.1) Model 1
# y(i) = theta_0


# Defining RLS Algorithm as a function
def rls_algo(x, y, theta, P):
    # Defining inter variable inv for P
    inv = np.linalg.inv(np.identity(1) + np.dot(np.dot(x.T, P), x))
    P = P - np.dot(np.dot(np.dot(np.dot(P, x), inv), x.T), P)
    K = np.dot(P, x)
    theta = theta + K * (y - np.dot(x.T, theta))
    return theta, K, P


x = np.array([dt["x"]]).T  # Defining x and y as np arrays
y = np.array([dt["y"]]).T


n = 1  # Number of parameters = 1 since y(i) = theta_0
theta = np.zeros((n, 1))  # Defining initial value of theta
P = np.identity(n) * 100  # Defining initial value of P(t-1)

for k in range(len(x)):
    theta, K, P = rls_algo(np.array([[1]]).T, y[k], theta, P)
    print(theta)

final_table = pd.DataFrame(index=np.arange(4), columns=np.arange(7))
final_table.columns = [
    "Name",
    "Model",
    "Theta_0",
    "Theta_1",
    "Theta_2",
    "Theta_3",
    "RLS Error",
]
final_table["Name"] = ["Model 1", "Model 2", "Model 3", "Model 4"]
final_table["Model"] = [
    "t_0",
    "t_0 + t_1*x",
    "t_0 + t_1*x + t_2*x^2",
    "t_0 + t_1*x + t_2*x^2 + t_3*x^3",
]
final_table["Theta_0"].iloc[0] = theta[0][0]

# Note that there is only one parameter
# and it is not multiplied by any input x.

y_1 = pd.DataFrame(
    pd.Series(theta[0][0] for x in range(len(dt["x"]))),
    index=np.arange(len(dt["x"])),
    columns=np.arange(1),
)
y_1 = y_1[0].astype(float)
print(y_1)

# Least squared result check
np.ones(15).T
np.dot(
    np.dot((1 / np.dot(np.ones(15).T, np.ones(15))), np.ones(15).T), dt["y"]
)


# In[6]:


# 3.2) Model 2
# y(i) = theta_0 + theta_1*x(i)

x = np.array([dt["x"]]).T  # Defining x and y as np arrays
y = np.array([dt["y"]]).T


n = 2  # Number of parameters = 1 since y(i) = theta_0
theta = np.zeros((n, 1))  # Defining initial value of theta
P = np.identity(n) * 100  # Defining initial value of P(t-1)

for k in range(len(x)):
    theta, K, P = rls_algo(np.array([[1, x[k][0]]]).T, y[k], theta, P)
    print(theta)

# Updating final_table
final_table["Theta_0"].iloc[1] = theta[0][0]
final_table["Theta_1"].iloc[1] = theta[1][0]

# Least squared result check
x_lse = np.stack((np.ones(15).T, np.array(dt["x"]).T))
x_lse = x_lse.T
np.dot(np.dot(np.linalg.inv(np.dot(x_lse.T, x_lse)), x_lse.T), dt["y"])


# In[7]:


y_2 = dt["x"].apply(
    lambda x: theta[0][0] + theta[1][0] * x
)  # Storing estimations from Model 2


# In[8]:


# 3.3) Model 3
# y(i) = theta_0 + theta_1*x(i) + theta_2*x**2(i)

x = np.array(
    [dt["x"], dt["x"].apply(lambda x: x**2)]
).T  # Defining x and y as np arrays
y = np.array([dt["y"]]).T


n = 3  # Number of parameters = 1 since y(i) = theta_0
theta = np.zeros((n, 1))  # Defining initial value of theta
P = np.identity(n) * 100  # Defining initial value of P(t-1)

for k in range(len(x)):
    theta, K, P = rls_algo(np.array([[1, x[k][0], x[k][1]]]).T, y[k], theta, P)
    print(theta)

# Updating final_table
final_table["Theta_0"].iloc[2] = theta[0][0]
final_table["Theta_1"].iloc[2] = theta[1][0]
final_table["Theta_2"].iloc[2] = theta[2][0]

# Least squared result check
x_lse = np.stack(
    (np.ones(15).T, np.array(dt["x"]).T, np.array(dt["x"] ** 2).T)
)
x_lse = x_lse.T
np.dot(np.dot(np.linalg.inv(np.dot(x_lse.T, x_lse)), x_lse.T), dt["y"])


# In[9]:


y_3 = dt["x"].apply(
    lambda x: theta[0][0] + theta[1][0] * x + theta[2][0] * x**2
)  # Storing estimations from Model 3


# In[10]:


# 3.3) Model 4
# y(i) = theta_0 + theta_1*x(i) + theta_2*x(i)**2 + theta_3*x(i)**2

x = np.array(
    [dt["x"], dt["x"].apply(lambda x: x**2), dt["x"].apply(lambda x: x**3)]
).T  # Defining x and y as np arrays
y = np.array([dt["y"]]).T


# Number of parameters = 4 since
# y(i) = theta_0 + theta_1*x(i) + theta_2*x(i)**2 + theta_3*x(i)**2
n = 4
theta = np.zeros((n, 1))  # Defining initial value of theta
P = np.identity(n) * 100  # Defining initial value of P(t-1)

for k in range(len(x)):
    theta, K, P = rls_algo(
        np.array([[1, x[k][0], x[k][1], x[k][2]]]).T, y[k], theta, P
    )
    print(theta)

# Updating final_table
final_table["Theta_0"].iloc[3] = theta[0][0]
final_table["Theta_1"].iloc[3] = theta[1][0]
final_table["Theta_2"].iloc[3] = theta[2][0]
final_table["Theta_3"].iloc[3] = theta[3][0]
print(final_table)

# Least squared result check
x_lse = np.stack(
    (
        np.ones(15).T,
        np.array(dt["x"]).T,
        np.array(dt["x"] ** 2).T,
        np.array(dt["x"] ** 3).T,
    )
)
x_lse = x_lse.T
np.dot(np.dot(np.linalg.inv(np.dot(x_lse.T, x_lse)), x_lse.T), dt["y"])


# In[11]:


y_4 = dt["x"].apply(
    lambda x: theta[0][0]
    + theta[1][0] * x
    + theta[2][0] * x**2
    + theta[3][0] * x**3
)  # Storing estimations from Model 4


# In[12]:


# 4) Calculating RLS Error for Each Model

outputs = pd.DataFrame([dt["x"], dt["y"], y_1, y_2, y_3, y_4])

outputs.index = ("x", "y", "Model_1", "Model_2", "Model_3", "Model_4")
outputs


# In[13]:


# Calculating RLS Errors
RLS_errors = pd.DataFrame(
    pd.Series(0 for x in range(4)), index=np.arange(4), columns=np.arange(1)
)
RLS_errors.columns = ["Model"]
RLS_errors["RLS"] = RLS_errors["Model"]
RLS_errors["Model"] = ["Model_1", "Model_2", "Model_3", "Model_4"]

for k in range(len(RLS_errors)):
    RLS_errors.iloc[k, 1] = (
        (outputs.iloc[1] - outputs.iloc[k + 2]) ** 2
    ).sum()

final_table["RLS Error"] = RLS_errors["RLS"]
RLS_errors


# In[14]:


# 5) The graphs of estimated curve along with the data points for each model


# Defining plot function
def plot_func(X, Y_original, Y_estimated, Model_No=1):
    fig, ax = plt.subplots()
    ax.scatter(X, Y_original, color="b", label="Actual y")
    ax.plot(np.sort(X), np.sort(Y_estimated), label="Estimated y", color="r")
    ax.set_ylabel("Actual y, Estimated y")
    ax.set_xlabel("x")
    ax.set_title("Estimated Curve vs Data Points - Model " + str(Model_No))
    plt.legend(loc="upper left")
    plt.savefig("model" + str(Model_No) + ".png", dpi=100)
    plt.show()


# Final Graphs
plot_func(dt["x"], dt["y"], y_1, Model_No=1)  # Model 1
plot_func(dt["x"], dt["y"], y_2, Model_No=2)  # Model 2
plot_func(dt["x"], dt["y"], y_3, Model_No=3)  # Model 3
plot_func(dt["x"], dt["y"], y_4, Model_No=4)  # Model 4


# In[15]:


# 6) Presenting final table with estimated p
# arameters and RLS errors for each model

print(final_table)
