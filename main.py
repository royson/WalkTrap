import util
import numpy as np
import networkx as nx
import itertools
import math
import time
from numpy.linalg import inv, norm

# Debug mode
DEBUG = 0;

# Modularity
# Partition -> Modularity
Q = {}

# Stores the partitions per iteration
# Partition -> Communities
partition = {}

# Stores the variances between two communities
# "Community1Community2" -> Variance
variance = {}

# Stores the probability from a community to all its adjacent nodes
# Community -> P^t_C1.
comm = {}

# Stores the current community for each vertice for fast retrieval
# Vertice -> Community
community = {}

# Reset all dictionary values
def reset_dict():
	partition.clear()
	Q.clear()
	variance.clear()
	comm.clear()
	community.clear()

# Computes Modularity for partition i in Graph G
# Includes self-loops
def compute_modularity(i, G):
	q = 0
	all_links = G.number_of_edges()

	for C in partition[i]:
		CG = nx.subgraph(G, C)
		links_in_C = CG.number_of_edges()
		links_to_C = len(G.edges(C))
		q += (links_in_C / all_links) - ((links_to_C / all_links)**2)
	Q[i] = q

# Computes Optimal Modularity for partition i in Graph G
# Access to Adjacent Matrix A bounds need to be changed
# depending on graph. For datasets that start with node 1
# such as Zachary Karate club, A[i-1,j-1],
# for other datasets that start with node 0, A[i][j]
def compute_optimal_modularity(i, G, A):
	q = 0

	all_links = G.number_of_edges()
	for C in partition[i]:
		# Get all possible pairs of vertices in the community
		pairs = [(C[i],C[j]) for i in range(len(C)) \
			for j in range(i+1, len(C))]
		for (i, j) in pairs:
			q += (A[i, j] - ((G.degree(i)*G.degree(j)) \
				/ (2*all_links)))

	Q[i] = q/(4*all_links)

# Remove old communities and insert them as a new one
def update_comm(C1, C2, C3):
	new_P_t_C = ( \
		(len(C1) * comm[util.sort_community(C1)]) + \
		(len(C2) * comm[util.sort_community(C2)]) ) / \
		(len(C1) + len(C2))

	del comm[util.sort_community(C1)]
	del comm[util.sort_community(C2)]

	comm[util.sort_community(C3)] = new_P_t_C

# Remove old variances and insert new ones
def update_variance(C1, C2, C3, C, var):
	variance.pop(util.sort_communities_str(C1, C2), None)
	variance.pop(util.sort_communities_str(C1, C), None)
	variance.pop(util.sort_communities_str(C2, C), None)
	
	variance[util.sort_communities_str(C3, C)] = var

# Choose communities based on lowest variance
def choose_communities():
	return \
		util.string_list_to_lists(min(variance, key=variance.get)) 

# Check if theorem 3 or 4 is used.
def check_compute_variance(C1, C2, C3):
	return (variance.get(util.sort_communities_str(C1, C2)) is not None and 
		variance.get(util.sort_communities_str(C1, C3)) is not None and
		variance.get(util.sort_communities_str(C2, C3)) is not None)

# (Theorem 4) Compute variance between two communities
# Assumed check_compute_variance is done before calling
def compute_variance_constant(C1, C2, C3):
	return ( \
		((len(C1) + len(C3)) * variance[util.sort_communities_str(C1, C3)]) + \
		((len(C2) + len(C3)) * variance[util.sort_communities_str(C2, C3)]) + \
		(len(C3) * variance[util.sort_communities_str(C1, C2)]) ) / \
		(len(C1) + len(C2) + len(C3))

# (Theorem 3) Compute variance between two communities 
def compute_variance_linear(N, Dd, C1, C2):
	return (((len(C1) * len(C2)) / (len(C1) + len(C2))) * 
		norm((Dd @ comm[util.sort_community(C1)]) - 
			(Dd @ comm[util.sort_community(C2)]))) / N


#np.set_printoptions(threshold=np.nan)

def main():

	# t value for WalkTrap
	t = 2

	# === Uncomment to generate random graphs ===
	# Number of vertices
	N = [100, 300, 500, 700, 900]
	
	# Number of communities = N**l
	l = [0.3, 0.42, 0.5]

	# Stores results for ri X N and time X N 
	res_ri = []
	res_time = []

	# Generate Random Graphs of {random_vertices} Vertices
	for vN in N:
		avg_time = 0.
		avg_ri = 0.
		for vl in l:
			G = util.generate_rand_graph(vN, vl)
			reset_dict()

			start_time = time.time()

			a = G.graph['partition']
			V = G.number_of_nodes()

			# Replace this with any algorithm you want to test
			# Examples: 
			bp = walktrap(G, t)
			#bp = girvan_newman(G, math.ceil(vN**vl))
			#bp = lpa_communities(G)

			ri = util.rand_index(bp, a, V)

			avg_ri += ri
			avg_time += (time.time() - start_time)
		res_ri.append(avg_ri/3)
		res_time.append(avg_time/3)

	print(res_ri)
	print(res_time)
	# Plot Evaluation Chart
	util.plot_chart(N, res_ri, "N", "R\'")
	util.plot_chart(N, res_time, "N", "Time")

	# =======

	# === Uncomment to use Karate club dataset ===
	# http://konect.uni-koblenz.de/networks/ucidata-zachary
	# G = nx.read_gml('karate.gml', label='id')
	# N = G.number_of_nodes()
	# for x in range(N):
	# 	G.add_edge(x+1, x+1)
	# walktrap(G, t)
	# =======

	# === Uncomment to use Facebook social dataset ===
	# https://snap.stanford.edu/data/egonets-Facebook.html
	# G = nx.read_adjlist('facebook_combined.txt', nodetype=int)
	# N = G.number_of_nodes()
	# for x in range(N):
	# 	G.add_edge(x, x)
	# walktrap(G, t)
	# =======

# Finds communities in a graph using LPA
def lpa_communities(G):
	res = nx.algorithms.community.asyn_lpa_communities(G)
	bp = []
	for C in res:
		bp.append(list(C))
	
	return bp

# Finds communities in a graph using the Girvan–Newman method
# Takes in Graph G and Community k
def girvan_newman(G, k):
	comp = nx.algorithms.community.girvan_newman(G)
	
	limited = itertools.takewhile(lambda c: len(c) <= k, comp)
	bp = None
	for communities in limited:
		bp = list(sorted(c) for c in communities)

	return bp 

# Start the walktrap algorithm.  
def walktrap(G, t):

	N = G.number_of_nodes()

	if DEBUG:
		print("===== Graph Edges =====")
		print(G.edges)

	# Adjacency Matrix with self-loops
	A = nx.to_numpy_matrix(G, dtype=int)

	if DEBUG:
		print("===== Adjacency Matrix =====")
		print(A)

	# Diagonal Matrix
	D = nx.laplacian_matrix(G) + A

	# D^(-1/2). For distance calculation
	Dtemp = np.diagonal(D)
	if DEBUG:
		print("===== Diagonal Matrix's Diagonals =====")
		print(Dtemp)

	Dd = np.diag(np.power(Dtemp, (-0.5)))

	# Transition Matrix P
	P = inv(D) @ A

	# Transition Matrix P^t
	P_t = util.transition_matrix_after_t(P, t)

	# Initialize Partition 1, its modularity, and community
	part = []
	for n in G.nodes:
		community[n] = [n]
		part.append([n])
	partition[1] = part
	compute_modularity(1, G)
	#compute_optimal_modularity(1, G, A)

	# Populate initial comm dictionary
	for C in part:
		comm[str(C)] = util.community_to_adj(P_t, C)

	# Populate initial variance
	for (s, d) in G.edges:
		if s != d:
			variance[util.sort_communities_str([s], [d])] = \
				compute_variance_linear(N, Dd, [s], [d])

	# Start algorithm
	for step in range(1,N):
		
		if DEBUG:
			print("Step " + str(step))

		# Choose two communities based on variance
		(C1,C2) = choose_communities()

		# Sorted communities
		C3 = util.sort_communities(C1, C2)

		# Insert new partition and its modularity 
		part = list(partition.get(step))
		
		part.remove(C1)
		part.remove(C2)
		part.append(C3)
		partition[step+1] = part
		compute_modularity(step+1, G)
		#compute_optimal_modularity(step+1, G, A)
		if DEBUG:
			print("Partition " + str(step+1))
			print(part)

		# Update comm dict by removing C1 and C2
		# and adding C3
		update_comm(C1, C2, C3)

		# Update new community for each node and
		# find all adjacent vertices in C3
		adj_vertices = set()
		for v in C3:
			community[v] = sorted(C3)
			adj_vertices |= set(G.adj[v])

		# Remove duplicates and vertices already in C3
		adj_vertices = list(adj_vertices - set(C3))

		# Find existing communities for each vertice
		adj_communities = []
		for C in adj_vertices:
			adj_communities.append(community[C])

		# Remove duplicates from adj_communities
		adj_communities = \
			list(dict((x[0], x) for x in adj_communities).values())
		# Update distance between C3 and its adjacent communities
		for C in adj_communities:
			var = 0
			if check_compute_variance(C1, C2, C):
				var = compute_variance_constant(C1, C2, C)
			else:
				var = compute_variance_linear(N, Dd, C3, C)
			update_variance(C1, C2, C3, C, var)

		if DEBUG:
			print("Variance: ")
			print(variance)

	if DEBUG:
		print("===== Partitions =====")
		print(partition)
		print("===== Modularities =====")
		print(Q)

	util.print_results(partition, Q)

	bp = partition[max(Q, key=Q.get)]

	# Uncomment for Graph Plot
	# util.graph_plot(G, bp)

	# Return best partition
	return bp

if __name__ == '__main__':
	main()