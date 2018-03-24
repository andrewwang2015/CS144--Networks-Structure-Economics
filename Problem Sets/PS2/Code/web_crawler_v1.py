#!/usr/bin/env python
"""
A simple program that given a URL in the Caltech domain,
retrieves the web page, extracts all hyperlinks in it, counts
the number of hyperlinks, follows the extracted hyperlinks to
retrieve more pages in Caltech domain, and then repeats the process
for each successive page.
"""
import urllib
import numpy as np
import matplotlib.pyplot as plt
from numpy import cumsum
import fetcher3
import queue
import networkx as nx
from collections import defaultdict
import pickle

def cdf(data, xLabel, title, filename):
    data_size = len(data)

    # Set bins edges
    data_set = sorted(set(data))
    bins = np.append(data_set, data_set[-1] + 1)

    # Use the histogram function to bin the data
    counts, bin_edges = np.histogram(data, bins=bins, density=False)

    counts = counts.astype(float) / data_size

    # Find the cdf
    cdf = np.cumsum(counts)

    # Plot the cdf
    plt.plot(bin_edges[0:-1], cdf)
    plt.ylim((0, 1))
    plt.title(title)
    plt.ylabel("CDF")
    plt.xlabel(xLabel)
    plt.savefig(filename, bbox_inches='tight')
    # plt.grid(True)

    plt.show()


def ccdf(data, xLabel, title, filename):

    data_size = len(data)

    # Set bins edges
    data_set = sorted(set(data))
    bins = np.append(data_set, data_set[-1] + 1)

    # Use the histogram function to bin the data
    counts, bin_edges = np.histogram(data, bins=bins, density=False)

    counts = counts.astype(float) / data_size

    # Find the ccdf
    cdf = 1 - np.cumsum(counts)

    # Plot the ccdf
    plt.plot(bin_edges[0:-1], cdf)
    plt.ylim((0, 1))
    plt.title(title)
    plt.ylabel("CCDF")
    plt.xlabel(xLabel)
    plt.savefig(filename, bbox_inches='tight')
    # plt.grid(True)

    plt.show()


def is_in_domain(url):
    """
    Returns true if url is in Caltech domain, false otherwise
    """
    return ('caltech.edu' in url and 'http' in url)


def is_visited(visited, url):
    """
    Returns true if url has been visited before, false otherwise
    """
    return url in visited


def return_filtered_urls(url):
    """
    Given a URL, returns a list of filtered URLs based on it being
    in the Caltech domain and it not having been visited before.
    """
    new_links = fetcher3.fetch_links(url)
    if new_links is None:
        return []
    new_links_in_domain = list(filter(lambda x: is_in_domain(x), new_links))
    return new_links_in_domain

def ccdf_new(lst):
    lst.sort()
    n = len(lst)
    p = np.arange(n) / (n-1)
    plt.step(lst, p[::-1])
    plt.show()

def update_dictionaries(
    fromURL,
    recipientURLsInDomain,
    adjacency,
    numFromPage,
    numToPage, G):
    """
    Updates the dictionaries corresponding to the adjacency lists,
    number of hyperlinks per page, and number of hyperlinks
    which point to each page
    """
    numFromPage[fromURL] = len(recipientURLsInDomain)
    for i in recipientURLsInDomain:
        G.add_edge(fromURL, i)
        adjacency[fromURL].add(i)
        # We do not have to have: adjacency[i].add(fromURL)
        # because we transform the graph to undirected later
        numToPage[i] += 1


def add_to_visited(visited, url):
    """
    Updates the visited set to avoid loops
    """
    visited.add(url)


def main():
    print("starting")
    G = nx.DiGraph()
    # Stores the links we have already visited as to avoid going in a cycle
    visited_links = set()
    # Maps web page to a set of hyperlinks on that page
    adjacency = defaultdict(set)
    # Maps web page to how many links are on that page
    numFromPage = defaultdict(int)
    # Maps web page to how many links point to that page
    numToPage = defaultdict(int)
    num_links_visited = 0  # Tracks number of hyperlinks visited
    max_num_links = 2000  # Threshold for how many links to visit
    startingURL = "http://www.caltech.edu/"
    urlQueue = queue.Queue()
    urlQueue.put(startingURL)
    
    while (num_links_visited < max_num_links):
        if nextURL in visited_links:
            continue
        nextURL = urlQueue.get()
        visited_links.add(nextURL)
        try:
            all_domain_links = return_filtered_urls(nextURL)
        except urllib.error.HTTPError:
            continue
        except urllib.error.URLError:
            continue
        print(len(all_domain_links))
        update_dictionaries(nextURL, all_domain_links, adjacency, numFromPage, numToPage, G)
        # for i in all_domain_links:
        #     if i not in visited_links:
        #         urlQueue.put(i)
        num_links_visited += 1
        print(num_links_visited)


    # Histogram for number of hyperlinks per page
    fromPage = sorted(list(numFromPage.values()))
    with open('fromPageValues.pkl', 'wb') as f:
        pickle.dump(fromPage, f)
    plt.hist(fromPage, bins=15)
    plt.title('Number of Hyperlinks Per Page')
    plt.ylabel('Number of Pages')
    plt.xlabel('Number of Hyperlinks')
    plt.savefig('histogram_per_page.png', bbox_inches='tight')
    plt.clf()


    # Histogram for number of hyperlinks topage
    toPage = sorted(list(numToPage.values()))
    with open('toPageValues.pkl', 'wb') as f:
        pickle.dump(toPage, f)
    plt.hist(toPage, bins=15)
    plt.title('Number of Hyperlinks To Page')
    plt.ylabel('Number of Pages')
    plt.xlabel('Number of Hyperlinks')
    plt.savefig('histogram_to_page.png', bbox_inches='tight')
    plt.clf()
    ccdf(fromPage, 'Number of Hyperlinks',
         'CCDF of Number of Hyperlinks per page', 'from_page_ccdf.png')
    ccdf(toPage, 'Number of Hyperlinks',
         'CCDF of Number of Hyperlinks to page', 'to_page_ccdf.png')
    
    

    fromPageGraph = list(G.out_degree().values())
    toPageGraph = list(G.in_degree().values())
    with open('toPageValuesGraph.pkl', 'wb') as f:
        pickle.dump(toPageGraph, f)
    with open('fromPageValuesGraph.pkl', 'wb') as f:
        pickle.dump(fromPageGraph, f)
    ccdf(fromPageGraph, 'Number of Hyperlinks',
         'CCDF of Number of Hyperlinks per page', 'graph_from_page_ccdf.png')
    ccdf(toPageGraph, 'Number of Hyperlinks',
         'CCDF of Number of Hyperlinks to page', 'graph_to_page_ccdf.png')
    #cdf(toPage, 'Number of Hyperlinks', 'CDF of Number of Hyperlinks to page', 'to_page_cdf.png')
    
    H = G.to_undirected()
    # Clustering statistics
    print("The average clustering coefficient: " + str(nx.average_clustering(H)))
    print("The overall clustering coefficient: " + str(nx.transitivity(H)))

    # Diameter statistics
    print("The maximal diameter is: " + str(nx.diameter(H)))
    print("The average distance is: " + str(nx.average_shortest_path_length(H)))
main()

