import util
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from numpy.linalg import inv, norm

# Compute variance between two communities (Theorem 3)
def compute_variance(N, Dd, C1, C2):
	return (((len(C1) * len(C2)) / (len(C1) + len(C2))) * 
		norm((Dd @ comm[str(C1)]) - (Dd @ comm[str(C2)]))) / N


# Stores the partitions per iteration
partition = {}

# Stores the variances between two communities
variance = {}

# Stores the probability from a community to all its adjacent nodes
# Example. P^t_C1.
comm = {}

t = 2

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

N = G.number_of_nodes()

for x in range(N):
	print(x+1)
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
Dtemp2 = np.power(Dtemp, (-0.5))
print(Dtemp2)
Dd = np.diag(Dtemp2)
print(Dd)

# Transition Matrix P
P = inv(D) @ A

# Transition Matrix P^t
P_t = util.transition_matrix_after_t(P, t)

# Initialize Partition 1
part = []
for n in G.nodes:
	part.append([n])
partition["1"] = part

# Populate comm dictionary
for C in part:
	comm[str(C)] = util.community_to_adj(P_t, C)

# Populate initial variance
for (s, d) in G.edges:
	if s != d:
		variance[(
			str([s]) + str([d]))] = \
			compute_variance(N, Dd, [s], [d])

print("--------------")
print(comm)
print("--------------")
print(variance)

#
#plt.subplot(121)
#nx.draw(G, with_labels=True, font_weight='bold')
#plt.show()

#SUBGRAPHS - self loop +2 degree. +1 edge.
#G2 = nx.subgraph(G, [1,2,3])
#print(G2.number_of_edges())
#plt.subplot(122)
#nx.draw(G2, with_labels=True, font_weight='bold')
#plt.show()



