import numpy as np
import ast
from numpy.linalg import matrix_power
import itertools
import math
import random
import matplotlib.pyplot as plt
import networkx as nx

# Takes in probability matrix P and number of transition t
# Returns P^t
def transition_matrix_after_t(P, t):
	# P_t = P
	# for _ in range(t-1):
	# 	P_t = np.dot(P_t, P)

	# return P_t
	return matrix_power(P, t)

# Extracts row i from matrix P^t
# returns vector P^t_i
#def col_prob_vector(P_t, i):
#	return P_t[i,:]

# Compute probability from a community to all its adjacent vertices
# Returns vector P^t_C.
def community_to_adj(P_t, C):
	if C != []:
		N = len(P_t)
		P_t_C =	np.zeros(N)

		for j in range(N):
			total = 0 
			for i in C:
				total = total + P_t.item((i-1, j))
			P_t_C[j] = total / len(C) 
		return P_t_C 

# Convert a string of list into a tuple of lists
# '[2, 3][4]' -> ([2,3],[4])
def string_list_to_lists(s):
	return (ast.literal_eval(s[:s.index("]",2)+1]),
		ast.literal_eval(s[s.index("]")+1:]))

# Sort communities in ascending order
def sort_communities(C1, C2):
	return sorted(min(C1, C2) + max(C1, C2))

# Sort communities for query
def sort_communities_str(C1, C2):
	return (str(min(C1, C2)) + str(max(C1, C2)))

# Sort community for query
def sort_community(C1):
	return (str(sorted(C1)))

# Generate a list of community sizes
# N = Number of Vertices
# C = Number of Communities
def generate_community_list(N, C):
	return list(np.random.multinomial(N, [1/C]*C, size=1)[0])

def rand_index(P1, P2, N):
	v1 = 0
	v2 = 0
	v3 = 0

	for C1 in P1:
		v2 += len(C1)**2
		for C2 in P2:
			v1 += len(set(C1).intersection(C2))**2
	
	for C2 in P2:
		v3 += len(C2)**2
	return ( \
		(((N**2)*v1) - (v2*v3)) / \
		((((N**2)/2)*(v2+v3)) - (v2*v3))
		)

# Generate a random graph
def generate_rand_graph(num_of_vertices, vl):
	num_of_communities = math.ceil(num_of_vertices**vl)
	while True:
		G = nx.generators.community.random_partition_graph(
			generate_community_list(num_of_vertices,num_of_communities)
			, 0.9, 0.1)
		
		if nx.is_connected(G):
			break

	N = G.number_of_nodes()

	for x in range(N):
		G.add_edge(x, x)

	return G

# Plot graph G given best partition bp
def graph_plot(G, bp):
	pos = nx.spring_layout(G)
	cmap = plt.get_cmap('Pastel1')
	colors = cmap(np.linspace(0, 1, len(bp)))

	for i, C in enumerate(bp):
		nx.draw_networkx_nodes(G, pos, nodelist=C, 
			node_color=colors[i], 
			with_labels=True)
		SG = nx.subgraph(G, C)
		nx.draw_networkx_edges(G, pos, edgelist=SG.edges, 
			width=3.0, alpha=0.5)
	nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
	nx.draw_networkx_labels(G, pos)
	plt.axis('off')
	plt.show()

# Print results of random walk
def print_results(partition, Q):
	print("Max Modularity:")
	print(max(Q.values()))

	print("Best Partition: ")
	bp = partition[max(Q, key=Q.get)]
	print(bp)
	print("Number of Communities: ")
	print(len(bp))

# Plot chart given [x-axis], [y-axis]
def plot_chart(x, y, x_label, y_label):
	plt.plot(x, y, 'bs-')
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.show()
