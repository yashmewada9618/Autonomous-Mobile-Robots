import autograd.numpy as np
import pymanopt
import pymanopt.manifolds
import pymanopt.optimizers


dim = 3
manifoldr3 = pymanopt.manifolds.Euclidean(dim,1)
SO3 = pymanopt.manifolds.SpecialOrthogonalGroup(dim)
SE2 = pymanopt.manifolds.product.Product([SO3,manifoldr3])
wp = np.array([[1,0,0,0],
                [0,1,0,0],
                [0,0,1,0]])
#Position of Landmarks in image of camera
uk = np.array([[0.866,-0.945,-0.189,-0.289],
                [0.289,0.189,-0.567,-0.866],
                [-0.408,-0.267,0.802,-0.408]])
@pymanopt.function.autograd(SE2)
def cost(rot,tr):
    s = 0
    for i in range(4):
        cp = np.dot(rot,wp[:,[i]]) + tr
        cp = cp/np.linalg.norm(cp)
        cp = cp.T @ uk[:,[i]]
        temp = cp[0,0]
        if (temp >= 1.0):
            s += np.arccos(1.0)**2
        elif (temp <= -1.0):
            s += np.arccos(-1.0)**2
        else:
            s += np.arccos(temp)**2
    return s

problem = pymanopt.Problem(SE2, cost)

optimizer = pymanopt.optimizers.SteepestDescent()
result = optimizer.run(problem)

print("Rotation:", result.point[0])
print("Translation:", result.point[1])