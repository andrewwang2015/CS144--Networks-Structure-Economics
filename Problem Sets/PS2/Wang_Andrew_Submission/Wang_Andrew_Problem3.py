
# coding: utf-8

# In[2]:

# Importing necessary packages
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from numpy import cumsum


# In[3]:

G = nx.Graph()


# In[4]:

with open("facebook_edges.txt") as f:
    for line in f:
        assert(len(line.split()) == 2)
        a, b = line.split()
        G.add_edge(a,b)
        


# In[5]:

# Read in file (file is given in egelist format)
G = nx.read_edgelist("facebook_edges.txt")


# In[6]:

degrees = sorted(nx.degree(G).values())


# In[7]:

# Histogram of degrees, note the log scale on y- axis
plt.hist(degrees, bins=10, log=True)
plt.title('Degree Histogram')
plt.ylabel('Number of Nodes')
plt.xlabel('Degree')
#plt.savefig('degree_histogram.png', bbox_inches='tight')
plt.show()


# In[8]:

# Histogram of degrees, note the log scale on y- axis
plt.hist(degrees, bins=10)
plt.title('Degree Histogram')
plt.ylabel('Number of Nodes')
plt.xlabel('Degree')
#plt.savefig('degree_histogram_log.png', bbox_inches='tight')
plt.show()


# In[9]:

# Let's get the CDF of the node degrees now
cumulative_sum = np.cumsum(degrees)
sum_all_nodes = sum(degrees)
cdf = cumulative_sum/sum_all_nodes
plt.plot(cdf)
plt.title('CDF of Node Degrees')
plt.ylabel('CDF')
plt.xlabel('Degree')
#plt.savefig('degree_cdf.png', bbox_inches='tight')
plt.show()


# In[25]:

def cdf(data, xLabel, title, filename):
    data_size=len(data)

    # Set bins edges
    data_set=sorted(set(data))
    bins=np.append(data_set, data_set[-1]+1)

    # Use the histogram function to bin the data
    counts, bin_edges = np.histogram(data, bins=bins, density=False)

    counts=counts.astype(float)/data_size

    # Find the cdf
    cdf = np.cumsum(counts)

    # Plot the cdf
    plt.plot(bin_edges[0:-1], cdf)
    plt.ylim((0,1))
    plt.title(title)
    plt.ylabel("CDF")
    plt.xlabel(xLabel)
    plt.savefig(filename, bbox_inches='tight')
    #plt.grid(True)

    plt.show()
    
def ccdf(data, xLabel, title, filename):

    data_size=len(data)

    # Set bins edges
    data_set=sorted(set(data))
    bins=np.append(data_set, data_set[-1]+1)

    # Use the histogram function to bin the data
    counts, bin_edges = np.histogram(data, bins=bins, density=False)

    counts=counts.astype(float)/data_size

    # Find the ccdf
    cdf = 1-np.cumsum(counts)

    # Plot the ccdf
    plt.plot(bin_edges[0:-1], cdf)
    plt.ylim((0,1))
    plt.title(title)
    plt.ylabel("CCDF")
    plt.xlabel(xLabel)
    plt.savefig(filename, bbox_inches='tight')
    #plt.grid(True)

    plt.show()
    
def ccdf_new(lst, xLabel, title, filename):
    lst.sort()
    n = len(lst)
    p = np.arange(n) / (n-1)
    plt.ylabel("CCDF")
    plt.title(title)
    plt.xlabel(xLabel)
    plt.ylim((0, 1))
    plt.step(lst, p[::-1])
    plt.savefig(filename, bbox_inches='tight')
    plt.show()


# In[26]:

ccdf_new(degrees, "Degree", "CCDF of Node Degrees", 'test.png')


# In[92]:

cdf(degrees, 'Degree', 'CDF of Node Degrees', 'degree_cdf.png' )


# In[93]:

ccdf(degrees, 'Degree', 'CCDF of Node Degrees', 'degree_ccdf.png' )


# In[12]:

# Clustering statistics
print("The average clustering coefficient: " + str(nx.average_clustering(G)))
print("The overall clustering coefficient: " + str(nx.transitivity(G)))


# In[13]:

# Diameter statistics
print("The maximal diameter is: " + str(nx.diameter(G)))
print("The average distance is: " + str(nx.average_shortest_path_length(G)))


# In[ ]:



