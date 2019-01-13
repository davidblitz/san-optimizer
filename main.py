import numpy as np
import theano
import theano.tensor as T

lr = 0.1
p1 = theano.shared(np.array([.0, .0, .0]))
p2 = theano.shared(np.array([.1, .2, .1]))


cost = (1 - T.sum((p1 - p2)**2))**2

#gp1, gp2 = T.grad(cost, [p1, p2])
gp2 = T.grad(cost, p2)
minimize = theano.function(
        inputs=[],
        outputs=[cost],
#        updates= [(p1, p1 - lr*gp1), (p2, p2 - lr*gp2)] )
        updates= [(p2, p2 - lr*gp2)] )

#p1_init = np.randn(3)
#p2_init = np.randn(3)

for i in range(30):
    obj = minimize()
    print(obj)

print("Final Positions: ")
p1_v = p1.get_value()
p2_v = p2.get_value()
print("p1: ", p1_v)
print("p2: ", p2_v)
print("dist(p1, p2) = ", (p1_v - p2_v).dot(p1_v - p2_v))
