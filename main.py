import util
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import itertools
from numpy.linalg import inv, norm

# Stores the partitions per iteration
partition = {}

# Stores the variances between two communities
variance = {}

# Stores the probability from a community to all its adjacent nodes
# Example. P^t_C1.
comm = {}

t = 2

# Remove old communities and insert them as a new one
def update_comm(C1, C2):
	new_P_t_C = ( \
		(len(C1) * comm[str(C1)]) + \
		(len(C2) * comm[str(C2)]) ) / \
		(len(C1) + len(C2))

	del comm[str(C1)]
	del comm[str(C2)]

	comm[str(C1 + C2)] = new_P_t_C

# Remove old variances and insert new ones
def update_variance(C1, C2, C3, C, var):
	variance.pop(str(C1) + str(C2), None)
	variance.pop(str(C2) + str(C1), None)
	variance.pop(str(C1) + str(C), None)
	variance.pop(str(C) + str(C1), None)
	variance.pop(str(C2) + str(C), None)
	variance.pop(str(C) + str(C2), None)
	
	variance[str(C3) + str(C)] = var

# Choose communities based on lowest variance
def choose_communities():
	return \
		util.string_list_to_lists(min(variance, key=variance.get)) 

def check_compute_variance(C1, C2, C3):
	return (variance.get(str(C1) + str(C2)) is not None and 
		variance.get(str(C1) + str(C3)) is not None and
		variance.get(str(C2) + str(C3)) is not None)

# (Theorem 4) Compute variance between two communities
# Assumed check_compute_variance is done before calling
def compute_variance_constant(C1, C2, C3):
	return ( ((len(C1) + len(C3)) * variance[str(C1) + str(C3)]) + \
		((len(C2) + len(C3)) * variance[str(C2) + str(C3)]) + \
		(len(C3) * variance[str(C1) + str(C2)]) ) / \
		(len(C1) + len(C2) + len(C3))

# (Theorem 3) Compute variance between two communities 
def compute_variance_linear(N, Dd, C1, C2):
	return (((len(C1) * len(C2)) / (len(C1) + len(C2))) * 
		norm((Dd @ comm[str(C1)]) - (Dd @ comm[str(C2)]))) / N


# # Getter & Setter for comm
# def insert_comm(C, val):
# 	comm[C] = val

# def get_comm(C):
# 	return comm.get(C)

# def remove_comm(C):
# 	del comm[C]

# # Getter & Setter for var
# def insert_var(C, val):
# 	var[C] = val

# def get_var(C):
# 	return var.get(C)

# def remove_var(C):
# 	del var[C]

np.set_printoptions(threshold=np.nan)

G = nx.read_gml('karate.gml', label='id')

# plt.subplot(121)
# nx.draw(G, with_labels=True, font_weight='bold')
# plt.show()

N = G.number_of_nodes()

for x in range(N):
	G.add_edge(x+1, x+1)

print(G.edges)

# Adjacency Matrix with self-loops
A = nx.to_numpy_matrix(G, dtype=int)
print(A)

# Diagonal Matrix
D = nx.laplacian_matrix(G) + A
print(D)

# D^(-1/2). For distance calculation
Dtemp = np.diagonal(D)
print(Dtemp)
Dd = np.diag(np.power(Dtemp, (-0.5)))
print(Dd)

# Transition Matrix P
P = inv(D) @ A

# Transition Matrix P^t
P_t = util.transition_matrix_after_t(P, t)

# Initialize Partition 1
part = []
for n in G.nodes:
	part.append([n])
partition[1] = part

# Populate comm dictionary
for C in part:
	comm[str(C)] = util.community_to_adj(P_t, C)

# Populate initial variance
for (s, d) in G.edges:
	if s != d:
		variance[(
			str([s]) + str([d]))] = \
			compute_variance_linear(N, Dd, [s], [d])

print("--------------")
print(comm)
print("--------------")
print(variance)


# Start algorithm
for step in range(1,N):
	print("Step " + str(step))

	# Choose two communities based on variance
	(C1,C2) = choose_communities()
	print("Communities: ")
	print(C1)
	print(C2)
	C3 = C1 + C2

	# Update partition
	prev_part = partition.get(step)
	part = list(prev_part)
	part.remove(C1)
	part.remove(C2)
	part.append(C3)
	partition[step+1] = part
	print("Partition " + str(step+1))
	print(part)

	# Update comm dict by removing C1 and C2
	# and adding C3
	update_comm(C1, C2)

	# Find all adjacent vertices in C3
	av = []
	for v in C3:
		av += list(G.adj[v])
	# Remove duplicates and vertices already in C3
	av = list(set(av) - set(C3))
	print("Adj Vertices: ")
	print(av)

	# Find existing communities for each vertice
	adj_communities = []
	for community in part:
		for v in av:
			if v in community:
				adj_communities.append(community)
				break

	# Remove duplicates from adj_communities
	adj_communities = \
		list(dict((x[0], x) for x in adj_communities).values())

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

	print(variance.keys())

print("FINISHED")
print(partition)



#SUBGRAPHS - self loop +2 degree. +1 edge.
#G2 = nx.subgraph(G, [1,2,3])
#print(G2.number_of_edges())
#plt.subplot(122)
#nx.draw(G2, with_labels=True, font_weight='bold')
#plt.show()



