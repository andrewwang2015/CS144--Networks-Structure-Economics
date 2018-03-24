
# coding: utf-8

# In[7]:

#!/usr/bin/env python
"""
A simple program that given a URL in the Caltech domain,
retrieves the web page, extracts all hyperlinks in it, counts
the number of hyperlinks, follows the extracted hyperlinks to
retrieve more pages in Caltech domain, and then repeats the process
for each successive page.
"""
import numpy as np
import urllib
import matplotlib.pyplot as plt
import fetcher3
import queue
import networkx as nx
from collections import defaultdict
import time


# In[8]:

def ccdf(lst, xLabel, title, filename):
    '''
    Plots the ccdf given a list of data, what to label the 
    X axis, what to put for graph title, and the filename
    to save it as
    '''
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


def is_in_domain(url):
    """
    Returns true if url starts with h and
    is in Caltech domain, false otherwise
    """
    return ('caltech.edu' in url and url[0] == 'h')


def is_visited(visited, url):
    """
    Returns true if url has been visited before, false otherwise
    """
    return url in visited


def return_filtered_urls(url):
    """
    Given a URL, returns a list of filtered URLs based on it being
    in the Caltech domain
    """
    new_links = fetcher3.fetch_links(url)
    if new_links is None:
        return []
    new_links_in_domain = list(filter(lambda x: is_in_domain(x), new_links))
    return new_links_in_domain


def update_dictionaries(
    fromURL,
    recipientURLsInDomain, G):
    """
    Adds an edge between URLs that are connected to one another
    """
    for i in recipientURLsInDomain:
        G.add_edge(fromURL, i)


def add_to_visited(visited, url):
    """
    Updates the visited set to avoid loops
    """
    visited.add(url)


# In[9]:

G = nx.DiGraph()
# Stores the links we have already visited as to avoid going in a cycle
visited_links = set()
num_links_visited = 0  # Tracks number of hyperlinks visited 
max_num_links = 2500  # Threshold for how many links to visit
startingURL = "http://www.caltech.edu/"
urlQueue = queue.Queue()
urlQueue.put(startingURL)


# In[10]:

while (num_links_visited < max_num_links):
        nextURL = urlQueue.get()
        if nextURL in visited_links or nextURL[:-1] in visited_links or (nextURL + "/") in visited_links:
            continue
        visited_links.add(nextURL)
        # To avoid crashing on HTTP errors
        time.sleep(1) # To avoid DOS attacks
        try:
            all_domain_links = return_filtered_urls(nextURL)
        except urllib.error.HTTPError:
            continue
        except urllib.error.URLError:
            continue
        
        update_dictionaries(nextURL, all_domain_links, G)
        for i in all_domain_links:
            urlQueue.put(i)
        num_links_visited += 1
        print(num_links_visited)


# In[12]:

# Remove nodes that haven't been fully crawled on
G.remove_nodes_from([node for node, deg in G.out_degree().items() if node not in visited_links]) 
G.remove_nodes_from([node for node, deg in G.in_degree().items() if node not in visited_links]) 


# In[19]:

# Getting outdegree (from page) and indegree (toPage) data
fromPageGraph = list(G.out_degree().values())
toPageGraph = list(G.in_degree().values())
fromPageGraph.sort()
toPageGraph.sort()


# In[ ]:

# fromPageGraphDict = G.out_degree()
# toPageGraphDict = G.in_degree()
# fromPageGraph = []
# for i in fromPageGraphDict:
#     if fromPageGraphDict[i] != 0:
#         fromPageGraph.append(fromPageGraphDict[i])
# fromPageGraph.sort()
# toPageGraph = []
# for i in toPageGraphDict:
#     if fromPageGraphDict[i] != 0:
#         toPageGraph.append(toPageGraphDict[i])
# toPageGraph.sort()
# toPageGraph[-1]


# In[20]:

# fromPageGraph = sorted(list(G.out_degree().values()))
# toPageGraph = sorted(list(G.in_degree().values()))
numBins = 35
# Histogram for number of hyperlinks per page
plt.hist(fromPageGraph, bins=numBins)
plt.title('Number of Hyperlinks From Page')
plt.ylabel('Number of Pages')
plt.xlabel('Number of Hyperlinks')
plt.savefig('histogram_per_page.png', bbox_inches='tight')
plt.show()
plt.clf()

# Histogram for number of hyperlinks to page
plt.hist(toPageGraph, bins=numBins)
plt.title('Number of Hyperlinks To Page')
plt.ylabel('Number of Pages')
plt.xlabel('Number of Hyperlinks')
plt.savefig('histogram_to_page.png', bbox_inches='tight')
plt.show()
plt.clf()

# Histogram for number of hyperlinks per page log scale
plt.hist(fromPageGraph, bins=numBins, log = True)
plt.title('Number of Hyperlinks From Page')
plt.ylabel('Number of Pages')
plt.xlabel('Number of Hyperlinks')
plt.savefig('histogram_per_page_log.png', bbox_inches='tight')
plt.show()
plt.clf()

# Histogram for number of hyperlinks to page log scale
plt.hist(toPageGraph, bins=numBins, log=True)
plt.title('Number of Hyperlinks To Page')
plt.ylabel('Number of Pages')
plt.xlabel('Number of Hyperlinks')
plt.savefig('histogram_to_page_log.png', bbox_inches='tight')
plt.show()
plt.clf()

ccdf(fromPageGraph, 'Number of Hyperlinks',
     'CCDF of Number of Hyperlinks From Page', 'graph_from_page_ccdf.png')
ccdf(toPageGraph, 'Number of Hyperlinks',
     'CCDF of Number of Hyperlinks To Page', 'graph_to_page_ccdf.png')


# In[21]:

H = G.to_undirected()
# Clustering statistics
print("The average clustering coefficient: " + str(nx.average_clustering(H)))
print("The overall clustering coefficient: " + str(nx.transitivity(H)))

# Diameter statistics
print("The maximal diameter is: " + str(nx.diameter(H)))
print("The average distance is: " + str(nx.average_shortest_path_length(H)))


# In[ ]:

# fromPageGraph = list(G.out_degree().values())
# toPageGraph = list(G.in_degree().values())
# toPage = sorted(list(numToPage.values())) 
# fromPage = sorted(list(numFromPage.values()))
# with open('fromPageValues.pkl', 'wb') as f:
#     pickle.dump(fromPage, f)
# with open('toPageValues.pkl', 'wb') as f:
#     pickle.dump(toPage, f)
# with open('toPageValuesGraph.pkl', 'wb') as f:
#     pickle.dump(toPageGraph, f)
# with open('fromPageValuesGraph.pkl', 'wb') as f:
#     pickle.dump(fromPageGraph, f)


# In[ ]:



