import util
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import itertools
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

t = 2

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

G = nx.read_gml('karate.gml', label='id')
#G = nx.read_adjlist('facebook_combined.txt', nodetype=int)

N = G.number_of_nodes()
 
# Add self loops
for x in range(N):
	G.add_edge(x+1, x+1)

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
		variance[(
			str([s]) + str([d]))] = \
			compute_variance_linear(N, Dd, [s], [d])

# Start algorithm
for step in range(1,N):
	if DEBUG:
		print("Step " + str(step))

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
	if DEBUG:
		print("Adj Vertices: ")
		print(av)

	# Find existing communities for each vertice
	adj_communities = []
	for C in av:
		adj_communities.append(community[C])

	# Remove duplicates from adj_communities
	adj_communities = \
		list(dict((x[0], x) for x in adj_communities).values())
	if DEBUG:
		print("Adj Communities: ")
		print(adj_communities)

	# Update distance between C3 and its adjacent communities
	for C in adj_communities:
		var = 0
		if check_compute_variance(C1, C2, C):
			var = compute_variance_constant(C1, C2, C)
		else:
			var = compute_variance_linear(N, Dd, C3, C)
		update_variance(C1, C2, C3, C, var)

	if DEBUG:
		print("Variance Keys: ")
		print(variance.keys())

if DEBUG:
	print("===== Partitions =====")
	print(partition)
	print("===== Modularities =====")
	print(Q)

print("Best Partition: ")
bp = partition[max(Q, key=Q.get)]
print(bp)
print("Number of Communities: ")
print(len(bp))

# Draw Graph
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


