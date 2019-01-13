import numpy as np
import theano
import theano.tensor as T

cos_ta = -1.0 / 3.0
lr = 0.003
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
  
minimize = theano.function(
        inputs=[],

        outputs=[cost],
        updates= updates )
#p1_init = np.randn(3)
#p2_init = np.randn(3)
obj = minimize()[0]
while obj > 1e-5:
    obj = minimize()[0]
    print(obj)

print("Final Positions: ")
root_v = root.get_value()
nodes_v = [node.get_value() for node in nodes]
print("nodes: ", nodes_v)
print("dist(node1, node2) = ", np.sum((nodes_v[0] - nodes_v[1])**2))
print("cos(root - node1, node1 - node2) = ", (nodes_v[0] -nodes_v[1]).dot(root_v - nodes_v[0]))
