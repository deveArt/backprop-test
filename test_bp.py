import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from numpy.linalg import norm

np.random.seed(555)

# Load data and do some preprocessing
# https://archive.ics.uci.edu/ml/machine-learning-databases/tic-tac-toe/tic-tac-toe.data

df = shuffle(pd.read_csv('tic_tac.csv'))
df.columns = [1, 2, 3, 4, 5, 6, 7, 8, 9, 'target']

X = pd.get_dummies(df[[1, 2, 3, 4, 5, 6, 7, 8, 9]])

Y = df[['target']].copy()
Y.values[Y == 'positive'] = 1
Y.values[Y == 'negative'] = 0
Y = Y.astype(np.int8)

train_size = 735
X_train = X[:train_size].values
X_test = X[train_size:].values
Y_train = Y[:train_size].values
Y_test = Y[train_size:].values

# Hyper params/optimisation opts

input_dim = 27 #27
hidden_layers = [30, 20]
epochs = 5
lr = 0.2
hl_count = len(hidden_layers)
ll_act_fn = 'sigmoid'
hl_act_fn = 'tanh'

# Init weights

model = []
B = np.zeros([hl_count+1, 1])

for i, layer_units in enumerate(hidden_layers):
    if i == 0:
        model.append(np.random.randn(layer_units, input_dim) / np.sqrt(input_dim))
    else:
        prev_dim = hidden_layers[i - 1]
        model.append(np.random.randn(layer_units, prev_dim) / np.sqrt(prev_dim))
model.append(np.random.randn(1, hidden_layers[-1]) / np.sqrt(hidden_layers[-1]))

def forward(X, model_w):
    H = []
    V = []

    H.append(X)
    V.append(np.dot(X, model_w[0].T) + B[0])
    H.append(activate(V[0]))
    hl_cnt = len(model_w) - 1

    for i in range(1, hl_cnt):
        V.append(np.dot(H[i], model_w[i].T) + B[i])
        H.append(activate(V[i]))

    V.append(np.dot(H[-1], model_w[hl_cnt].T) + B[hl_cnt])
    Y = activate(V[-1], last=True)
    return Y, H, V


def backward(H, err, V, model_w):
    grads = []
    r_model = list(enumerate(model_w))
    r_model.reverse()

    for i, wl in r_model:
        if i == (len(model_w) - 1):
            grad = err * a_deriv(V[i], last=True)
            dd = np.mean(grad * H[i], axis=0)
        else:
            grad = a_deriv(V[i]) * np.dot(grad, model_w[i+1])
            dd = np.mean(np.einsum('ij,ik->ikj', H[i], grad), axis=0)

        grads.append(dd)
        D = lr * dd
        model_w[i] = wl + D

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
        V_[V > 1] = 1
        V_[V < 0] = 0
        V_[V == 0] = 0.5
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

def gradient_check(grads, X_set, Y_set, model_w, epsilon=1e-2):

    param_vec, layer_shapes = unroll_params(model_w)
    grad_vec, _ = unroll_params(grads)

    # Create vector of zeros to be used with epsilon
    grads_approx = np.zeros_like(param_vec)

    for i in range(param_vec.shape[0]):
        theta_plus = np.copy(param_vec)
        theta_plus[i] = theta_plus[i] + epsilon
        Y, H, V = forward(X_set, revert_unroll(theta_plus, layer_shapes))
        j_plus = cost_fn(Y_set - Y)

        theta_minus = np.copy(param_vec)
        theta_minus[i] = theta_minus[i] - epsilon
        Y, H, V = forward(X_set, revert_unroll(theta_minus, layer_shapes))
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

# Model training

C = []
eps = []

for i in range(1):
    eps.append(i)
    Y, H, V = forward(X_train, model)

    err = Y_train - Y
    C.append(cost_fn(err))

    grads = backward(H, err, V, model)
    gradient_check(grads, X_train, Y_train, model)

# Show results

pdata = pd.DataFrame({'epochs': eps, 'cost': C})
pdata.plot(x='epochs', y='cost')

Y_pred, _, _ = forward(X_test, model)
acc = accuracy_score(Y_test, Y_pred.round())

print('Test accuracy score:', acc)
plt.show()