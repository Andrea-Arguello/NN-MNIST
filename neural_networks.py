from functools import reduce
import numpy as np

sigmoid = lambda x: (1.0 / (1.0 + np.exp(-x))) # a_l = sigmoid(z_l)

def sigmoid_2(z):
    a = [sigmoid(x) for x in z]
    return np.asarray(a).reshape(z.shape)

flatten_list_of_arrays = lambda list_of_arrays: reduce(
    lambda acc, v: np.array([*acc.flatten(), *v.flatten()]),
    list_of_arrays
)

def inflate_matrices(flat_thetas, shapes):
    layers = len(shapes) + 1
    sizes = [shape[0] * shape[1] for shape in shapes]
    steps = np.zeros(layers, dtype=int)

    for l in range(layers - 1):
        steps[l + 1] = steps[l] + sizes[l]
    
    return [
        flat_thetas[steps[i]: steps[i+1]].reshape(*shapes[i]) for i in range(layers - 1)
    ]

def feed_forward(thetas, X): #Este feed forward entra con todos de un solo
    a = [np.asarray(X)]
    for i in range(len(thetas)):
        a.append(
            sigmoid(
                np.matmul(
                    np.hstack((
                        np.ones(len(X)).reshape(len(X),1),
                        a[i]
                    )), thetas[i].T
                )
            )
        )
    return a

def cost_function(flat_thetas, shapes, X, Y):
    a = feed_forward(
        inflate_matrices(flat_thetas,shapes),
        X
    )
    return -(Y * np.log(a[-1]) + (1 - Y) * np.log(1 - a[-1])).sum() / len(X)

def back_propagation(flat_thetas, shapes, X, y):
    m, layers = len(X), len(shapes) + 1
    thetas = inflate_matrices(flat_thetas, shapes)
    a = feed_forward(thetas,X)
    deltas = [*range(layers-1), a[-1] - y]
    for l in range(layers-2,0,-1):
        deltas[l] = deltas[l+1] @ np.delete(thetas[l], 0, 1) * a[l]*(1-a[l]) # 2.4 - Theta sin el bias
    return flatten_list_of_arrays([
        np.matmul(deltas[l+1].T, np.hstack((
        np.ones(len(a[l])).reshape(len(a[l]),1),
        a[l]))) / m for l in range(layers-1)
    ])
