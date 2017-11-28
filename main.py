import util
import numpy as np
import networkx as nx
import itertools
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
# "Community1Community2" -> variance
variance = {}

# Stores the probability from a community to all its adjacent nodes
# Community -> P^t_C1.
comm = {}

# Stores the current community for per vertice for fast retrieval
# Vertice -> Community
community ={}

# Computes Modularity for partition i in graph G
# Includes self-loops
def compute_modularity(i, G):
	q = 0
	for C in partition[i]:
		all_links = G.number_of_edges()
		CG = nx.subgraph(G, C)
		links_in_C = CG.number_of_edges()
		links_to_C = len(G.edges(C))
		q += (links_in_C / all_links) - ((links_to_C / all_links)**2)
	Q[i] = q

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
	t = 2

	# Uncomment to generate random graphs
	# Number of vertices
	N = [100, 300, 1000, 3000, 10000]
	
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
			start_time = time.time()
			(_, ri) = walktrap(G, t)
			avg_ri += ri
			avg_time += (time.time() - start_time)
		res_ri.append(avg_ri/3)
		res_time.append(avg_time/3)

	print(res_ri)
	print(res_time)
	# Plot Evaluation Chart
	util.plot_chart(N, res_ri, "N", "R\'")
	util.plot_chart(N, res_time, "N", "Time")

	# Uncomment to use Karate club dataset
	# G = nx.read_gml('karate.gml', label='id')
	# N = G.number_of_nodes()
	# for x in range(N):
	# 	G.add_edge(x+1, x+1)
	# walktrap(G, t, 1)

	# Uncomment to use facebook social dataset
	# G = nx.read_adjlist('facebook_combined.txt', nodetype=int)
	# N = G.number_of_nodes()
	# for x in range(N):
	# 	G.add_edge(x, x)
	# walktrap(G, t, 1)


# Start the walktrap algorithm. 
# Mode=0, Random Graph Mode. Return (modularity, rand_index)
# Mode=1, Real World Data. 
def walktrap(G, t, mode=0):

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
		
		#print("Step " + str(step))

		# Choose two communities based on variance
		(C1,C2) = choose_communities()
		if DEBUG:
			print("Communities: ")
			print(C1)
			print(C2)
		# Sorted communities
		C3 = util.sort_communities(C1, C2)

		# Insert new partition and its modularity 
		prev_part = partition.get(step)
		part = list(prev_part)
		part.remove(C1)
		part.remove(C2)
		part.append(C3)
		partition[step+1] = part
		compute_modularity(step+1, G)
		if DEBUG:
			print("Partition " + str(step+1))
			print(part)

		# Update comm dict by removing C1 and C2
		# and adding C3
		update_comm(C1, C2, C3)

		# Update new community for each node and
		# find all adjacent vertices in C3
		av = set()
		for v in C3:
			community[v] = sorted(C3)
			av |= set(G.adj[v])

		# Remove duplicates and vertices already in C3
		av = list(av - set(C3))

		# Find existing communities for each vertice
		adj_communities = []
		for C in av:
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

	#util.print_results(partition, Q)

	bp = partition[max(Q, key=Q.get)]

	# Graph Plot
	# util.graph_plot(G, bp)

	# Calculate the rand_index (Only for random graph generation)
	if mode == 0:
		answer = G.graph['partition']
		ri = util.rand_index(bp, answer, N)
		# print("Rand_index: %s" % ri)
		return(max(Q.values()), ri)

if __name__ == '__main__':
	main()