# 1) Creating 4 sample image patterns of 1, 4, 7 and 9

import numpy as np
from matplotlib import pyplot as plt

one = np.array([[1,1,-1,-1,-1,1,1,1],
[1,1,-1,-1,-1,1,1,1],
[1,1,-1,-1,-1,1,1,1],
[1,1,1,-1,-1,1,1,1],
[1,1,1,-1,-1,1,1,1],
[1,1,1,-1,-1,1,1,1],
[1,1,1,-1,-1,1,1,1],
[1,1,1,-1,-1,1,1,1]])

four = np.array([[-1,-1,1,1,1,-1,-1,1],
[-1,-1,1,1,1,-1,-1,1],
[-1,-1,1,1,1,-1,-1,1],
[-1,-1,-1,-1,-1,-1,-1,1],
[-1,-1,-1,-1,-1,-1,-1,1],
[1,1,1,1,1,-1,-1,1],
[1,1,1,1,1,-1,-1,1],
[1,1,1,1,1,-1,-1,1]])

seven =np.array([[-1,-1,-1,-1,-1,-1,1,1],
[-1,-1,-1,-1,-1,-1,1,1],
[1,1,1,1,-1,-1,1,1],
[1,1,1,1,-1,-1,1,1],
[1,1,1,1,-1,-1,1,1],
[1,1,1,1,-1,-1,1,1],
[1,1,1,1,-1,-1,1,1],
[1,1,1,1,-1,-1,1,1]])

nine = np.array([[1,-1,-1,-1,-1,-1,-1,1],
[1,-1,-1,1,1,-1,-1,1],
[1,-1,-1,1,1,-1,-1,1],
[1,-1,-1,-1,-1,-1,-1,1],
[1,-1,-1,-1,-1,-1,-1,1],
[1,1,1,1,1,-1,-1,1],
[1,-1,-1,-1,-1,-1,-1,1],
[1,-1,-1,-1,-1,-1,-1,1]])


def convert_to_plot(array):            # Defining the function converting -+1 array to 0 1 array for plotting the image properly
    array_plot = np.copy(array)
    array_plot[array_plot < 0] = 0
    return array_plot


# Ploting the black and white images
plt.figure()
#subplot(r,c) provide the no. of rows and columns
f, axarr = plt.subplots(2,2, figsize=(7,7)) 

# use the created array to output multiple images
axarr[0][0].imshow(convert_to_plot(one), 'gray')
axarr[0][1].imshow(convert_to_plot(four), 'gray')
axarr[1][0].imshow(convert_to_plot(seven), 'gray')
axarr[1][1].imshow(convert_to_plot(nine), 'gray')
plt.savefig('original_patterns.png')


# 2) Converting each number to  64-element vector
one_element = one.reshape(64,1)
four_element = four.reshape(64,1)
seven_element = seven.reshape(64,1)
nine_element = nine.reshape(64,1)


# 3) Implementing Hopfield Algorithm

# Step 0: Initials
J = 64 # Dİmension of each pattern
W = np.arange(J*J).reshape(J,J) # Creating the weight matrix
input_v = np.c_[one_element, four_element, seven_element, nine_element]

# Hopfield Algorithm
# Step 1: Updating weights
for i in range(J): # Calculating Initial Weight Matrix from original inputs
    for j in range(J):
        if (i != j):
            W[i][j] = sum(input_v[i]*input_v[j])
        else:
            W[i][j] = 0
        


# In[5]:


# 4) Noise Addition
np.random.seed(2020802018)
sigma1 = 0.5
sigma2 = 0.7
sigma3 = 1

def noise_func(array_element,sigma = 1): # Defining noise addition function for J dimensional element vectors
    array_element_new = np.copy(array_element)
    for i in range(J):
        array_element_new[i] = np.sign(array_element[i] + np.random.normal(loc = 0, scale=sigma))
    return(array_element_new)

one_noisy = np.c_[noise_func(one_element, sigma1), noise_func(one_element, sigma2), noise_func(one_element, sigma3)] # Noise added array of one (3 noisy data)
four_noisy = np.c_[noise_func(four_element, sigma1), noise_func(four_element, sigma2), noise_func(four_element, sigma3)] # Noise added vector of one
seven_noisy = np.c_[noise_func(seven_element, sigma1), noise_func(seven_element, sigma2), noise_func(seven_element, sigma3)] # Noise added vector of one
nine_noisy = np.c_[noise_func(nine_element, sigma1), noise_func(nine_element, sigma2), noise_func(nine_element, sigma3)] # Noise added vector of one


# In[6]:


# Visualizing Noise Added Patterns
plt.figure()
#subplot(r,c) provide the no. of rows and columns
f, axarr = plt.subplots(4,3, figsize=(8,8)) 

# use the created array to output multiple images
for i in range(one_noisy.shape[1]):
    axarr[0][i].imshow(one_noisy[:,i].reshape(8,8), 'gray')
for i in range(four_noisy.shape[1]):
    axarr[1][i].imshow(four_noisy[:,i].reshape(8,8), 'gray')
for i in range(seven_noisy.shape[1]):
    axarr[2][i].imshow(seven_noisy[:,i].reshape(8,8), 'gray')
for i in range(nine_noisy.shape[1]):
    axarr[3][i].imshow(nine_noisy[:,i].reshape(8,8), 'gray')
plt.savefig('noise_added_patterns.png')

# 5) Iterating until convergence for each pattern and standard deviation

# Defining hopfield algorithm that stores 4 sample patterns
def hopfield_func(input_array):
    input_iter = np.copy(input_array.reshape(64,1))
    #subplot(r,c) provide the no. of rows and columns
    f, axarr = plt.subplots(8,8, figsize=(15,15))
    for i in range(J):
        input_iter[i] = np.sign(np.dot(W[i],input_iter))
        # use the created array to output multiple images
        if (i <= 7): # Row indexing for final plot
            k = 0
        elif ((i > 7) & (i <= 15)):
            k = 1
        elif ((i > 15) & (i <= 23)):
            k = 2
        elif ((i > 23) & (i <= 31)):
             k = 3
        elif ((i > 31) & (i <= 39)):
             k = 4
        elif ((i > 39) & (i <= 47)):
             k = 5
        elif ((i > 47) & (i <= 55)):
             k = 6
        else:
             k = 7       
        axarr[k][i % 8].imshow(input_iter.reshape(8,8), 'gray')
        # Adding convergence checkpoint (for each equilibrium points; one, four, seven and nine)
        if (input_iter == one_element).all() == True or (input_iter == four_element).all() == True or (input_iter == seven_element).all() == True or (input_iter == nine_element).all() == True:
            break
        else:
            continue  
    return f

one1, one2, one3 = hopfield_func(one_noisy[:,0]), hopfield_func(one_noisy[:,1]), hopfield_func(one_noisy[:,2])
four1, four2, four3 = hopfield_func(four_noisy[:,0]), hopfield_func(four_noisy[:,1]), hopfield_func(four_noisy[:,2])
seven1, seven2, seven3 = hopfield_func(seven_noisy[:,0]), hopfield_func(seven_noisy[:,1]), hopfield_func(seven_noisy[:,2])
nine1, nine2, nine3 = hopfield_func(nine_noisy[:,0]), hopfield_func(nine_noisy[:,1]), hopfield_func(nine_noisy[:,2])


# Saving final iterations' plots
final_names = np.array(['one1', 'one2', 'one3', 'four1', 'four2', 'four3', 'seven1', 'seven2', 'seven3', 'nine1', 'nine2', 'nine3'])

for i in range(len(final_names)):
    eval(final_names[i]).savefig(final_names[i] + '.png')
    
