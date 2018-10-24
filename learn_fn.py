import numpy as np
from numpy.linalg import norm
from bp_conf import *


def forward(X, model):
    H = []
    V = []

    H.append(X)
    V.append(np.dot(X, model['W'][0].T) + model['B'][0].T)
    H.append(activate(V[0]))
    hl_cnt = len(model['W']) - 1

    for i in range(1, hl_cnt):
        V.append(np.dot(H[i], model['W'][i].T) + model['B'][i].T)
        H.append(activate(V[i]))

    V.append(np.dot(H[-1], model['W'][hl_cnt].T) + model['B'][hl_cnt].T)
    Y = activate(V[-1], last=True)
    return Y, H, V


def backward(H, err, V, model, lr, update=True):
    grads = []
    r_model = list(enumerate(model['W']))
    r_model.reverse()

    for i, wl in r_model:
        if i == (len(model['W']) - 1):
            grad = err * a_deriv(V[i], last=True)
            dd = np.mean(grad * H[i], axis=0)
        else:
            grad = a_deriv(V[i]) * np.dot(grad, model['W'][i+1])
            dd = np.mean(np.einsum('ij,ik->ikj', H[i], grad), axis=0)

        grads.append(-dd)
        D = lr * dd

        if update:
            model['B'][i] = model['B'][i] + lr * np.mean(grad, axis=0).reshape((grad.shape[1], 1))
            model['W'][i] = wl + D

    grads.reverse()
    return grads

def cost_fn(err):
    return np.mean(err ** 2, axis=0)[0]/2

def activate(V, last=False):
    act_fn = ll_act_fn if last else hl_act_fn
    if act_fn == 'tanh':
        return np.tanh(V)
    if act_fn == 'relu':
        return np.maximum(V, 0)
    if act_fn == 'sigmoid':
        return 1 / (1 + np.exp(-V))
    if act_fn == 'linear':
        return V

def a_deriv(V, last=False):
    act_fn = ll_act_fn if last else hl_act_fn
    if act_fn == 'tanh':
        return 1 - np.tanh(V)**2
    if act_fn == 'relu':
        V_ = np.array(V, copy=True)
        V_[V_ > 1] = 1
        V_[V_ < 0] = 0
        V_[V_ == 0] = 0.5
        return V_
    if act_fn == 'sigmoid':
        return 1 / (1 + np.exp(-V)) * (1 - 1 / (1 + np.exp(-V)))
    if act_fn == 'linear':
        return 1

def unroll_params(param_list):
    unrolled = []
    layer_shapes = []

    for W in param_list:
        layer_shapes.append(W.shape)
        unrolled = np.append(unrolled, W.flatten())

    return unrolled, layer_shapes

def revert_unroll(param_vec, shapes):
    recovered = []
    p_vector = np.copy(param_vec)

    for layer_shape in shapes:
        layer_size = layer_shape[0] * layer_shape[1]
        layer_params, p_vector = p_vector[:layer_size], p_vector[layer_size:]
        recovered.append(layer_params.reshape(layer_shape))

    return recovered

def gradient_check(grads, X_set, Y_set, model, epsilon=1e-7):

    param_vec, layer_shapes = unroll_params(model['W'])
    grad_vec, _ = unroll_params(grads)

    # Create vector of zeros to be used with epsilon
    grads_approx = np.zeros_like(param_vec)

    for i in range(param_vec.shape[0]):
        theta_plus = np.copy(param_vec)
        theta_plus[i] = theta_plus[i] + epsilon
        Y, H, V = forward(X_set, {'W': revert_unroll(theta_plus, layer_shapes), 'B': model['B']})
        j_plus = cost_fn(Y_set - Y)

        theta_minus = np.copy(param_vec)
        theta_minus[i] = theta_minus[i] - epsilon
        Y, H, V = forward(X_set, {'W': revert_unroll(theta_minus, layer_shapes), 'B': model['B']})
        j_minus = cost_fn(Y_set - Y)

        grads_approx[i] = (j_plus - j_minus) / (2 * epsilon)

    print(grad_vec, grads_approx)
    # Compute the difference of numerical and analytical gradients
    numerator = norm(grad_vec - grads_approx)
    denominator = norm(grads_approx) + norm(grad_vec)
    difference = numerator / denominator

    if difference > 10e-7:
        print("\033[31mThere is a mistake in back-propagation " + \
              "implementation. The difference is: {}".format(difference))
    else:
        print("\033[32mThere implementation of back-propagation is fine! " + \
              "The difference is: {}".format(difference))

    return difference

def get_batches_iter(f_set, target, batch_size):
    c = 0

    while c * batch_size < f_set.shape[0]:
        f_batch = f_set[batch_size*c: batch_size*(c+1)]
        target_batch = target[batch_size*c: batch_size*(c+1)]

        yield f_batch, target_batch
        c += 1
