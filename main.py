import numpy as np
import theano
import theano.tensor as T
import itertools

class Node(object):
    def __init__(self, root=False, name=""):
        self.root = root
        self.name = name
        if root:
            self.var = theano.shared(np.array([.0]*3), name=name)
        else:
            self.var = theano.shared(np.random.randn(3), name=name)

        self.neighbours = []
        self.visited = False

def createCost(root):
    to_visit = set([root])
    visited = set()
    cost = 0.0
    tups = []
    trips = []

    while to_visit:
        current = to_visit.pop()
        for n in current.neighbours:
            if not n in visited:
                to_visit.add(n)
                cost += (1 - T.sum((current.var - n.var)**2))**2

        for n, nn in itertools.combinations(current.neighbours, 2):
            cost += (cos_ta - (n.var - current.var).dot(nn.var - current.var))**2

        visited.add(current)

    updates = []
    vars = [n.var for n in visited if not n.root]
    for v in vars:
        gv = T.grad(cost, v)
        updates.append((v, v - lr*gv))

    return cost, updates

def nodes_from_mat(mat):
    nodes = [Node(root=True)] + [Node() for i in range(len(mat)-1)]

    for i, row in enumerate(mat):
        for j, e in enumerate(row):
            if e:
                nodes[i].neighbours.append(nodes[j])

    return nodes

cos_ta = -1.0 / 3.0
lr = 0.003

#adj_mat = [[0, 1, 0],
#           [1, 0, 1],
#           [0, 1, 0]]

#pentagon
#adj_mat = [[0, 1, 0, 0, 0],
#           [1, 0, 1, 0, 0],
#           [0, 1, 0, 1, 0],
#           [0, 0, 1, 0, 1],
#           [1, 0, 0, 1, 0]]

#adj_mat = [[0, 1, 0, 0, 0, 1],
#           [1, 0, 1, 0, 0, 0],
#           [0, 1, 0, 1, 0, 0],
#           [0, 0, 1, 0, 1, 0],
#           [0, 0, 0, 1, 0, 1],
#           [1, 0, 0, 0, 1, 0]]
adj_mat = np.loadtxt("dodecahedron-adjacency.csv")
print(adj_mat)
nodes = nodes_from_mat(adj_mat)
print(len(nodes))
for node in nodes:
    print(node.name, [n.name for n in node.neighbours])
cost, updates = createCost(nodes[0])
print(cost)
"""
root = theano.shared(np.array([.0, .0, .0]))
nodes = [theano.shared(np.random.randn(3)) for i in range(2)]

cost = (1 - T.sum((root - nodes[0])**2))**2 + (1 - T.sum((nodes[0] - nodes[1])**2))**2 + (cos_ta - (root-nodes[0]).dot(nodes[0] - nodes[1]))**2

#gp1, gp2 = T.grad(cost, [p1, p2])
gnodes = []
updates = []
for node in nodes:
  gnode = T.grad(cost, node)
  gnodes.append(gnode)
  updates.append((node, node - lr*gnode))
"""
minimize = theano.function(
        inputs=[],
        outputs=[cost],
        updates= updates,
        mode="FAST_COMPILE")

print("Minimizing Objective...")
obj = minimize()[0]
while obj > 1e-1:
    obj = minimize()[0]
    print(obj)

print("Final Positions: ")
#root_v = nodes[0].var.get_value()
print("{")
nodes_v = {chr(ord("A") + i) : node.var.get_value().tolist() for i, node in enumerate(nodes)}
#print("nodes: ", nodes_v)
for key, item in nodes_v.items():
    print(key, ":", item, ",")
print("}")
#print("dist(node0, node1) = ", np.sum((nodes_v[0] - nodes_v[1])**2))
#print("cos(root - node1, node1 - node2) = ", (nodes_v[1] -nodes_v[2]).dot(nodes_v[0] - nodes_v[1]))
