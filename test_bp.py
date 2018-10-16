import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from numpy.linalg import norm
import sys

np.random.seed(555)

# Load data and do some preprocessing
#

features = pd.read_csv('kc_house_data.csv').drop(columns=['id', 'date', 'price', 'yr_built', 'yr_renovated', 'sqft_living15',
                                                          'sqft_lot15', 'sqft_basement']).values
target = pd.read_csv('kc_house_data.csv')[['price']].values

x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.3, shuffle=True)
y_test_orig = np.copy(y_test)

x_scaler = MinMaxScaler((-1,1))
x_scaler.fit(x_train)
x_train = x_scaler.transform(x_train)
x_test = x_scaler.transform(x_test)

y_scaler = MinMaxScaler()
y_scaler.fit(y_train)
y_train = y_scaler.transform(y_train)
y_test = y_scaler.transform(y_test)

#print( x_test)
#sys.exit()
# Hyper params/optimisation opts

input_dim = 13
hidden_layers = [20, 50, 10]
epochs = 85
batch_size = 512
lr = 0.011
hl_count = len(hidden_layers)
ll_act_fn = 'relu'
hl_act_fn = 'tanh'

# Init weights

model = []
B = []

for i, layer_units in enumerate(hidden_layers):
    if i == 0:
        model.append(np.random.randn(layer_units, input_dim) / np.sqrt(input_dim))
    else:
        prev_dim = hidden_layers[i - 1]
        model.append(np.random.randn(layer_units, prev_dim) / np.sqrt(prev_dim))
    B.append(np.zeros([layer_units, 1]))

B.append(np.zeros([1, 1]))
model.append(np.random.randn(1, hidden_layers[-1]) / np.sqrt(hidden_layers[-1]))

def forward(X, model_w, B):
    H = []
    V = []

    H.append(X)
    V.append(np.dot(X, model_w[0].T) + B[0].T)
    H.append(activate(V[0]))
    hl_cnt = len(model_w) - 1

    for i in range(1, hl_cnt):
        V.append(np.dot(H[i], model_w[i].T) + B[i].T)
        H.append(activate(V[i]))

    V.append(np.dot(H[-1], model_w[hl_cnt].T) + B[hl_cnt].T)
    Y = activate(V[-1], last=True)
    return Y, H, V


def backward(H, err, V, model_w, B, update=True):
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

        grads.append(-dd)
        D = lr * dd

        if update:
            B[i] = B[i] + lr * np.mean(grad, axis=0).reshape((grad.shape[1], 1))
            model_w[i] = wl + D

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

def gradient_check(grads, X_set, Y_set, model_w, epsilon=1e-7):

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

# Model training

C = []
t_C = []

for i in range(epochs):
    batch_cost = []

    for train_batch_features, train_batch_target in get_batches_iter(x_train, y_train, batch_size):
        Y, H, V = forward(train_batch_features, model, B)

        err = train_batch_target - Y
        batch_cost.append(cost_fn(err))

        grads = backward(H, err, V, model, B) # pass update = False to make gradient checking

        #gradient_check(grads, X_train, Y_train, model)

    Y_val, _, _ = forward(x_test, model, B) # validate on each epoch
    t_C.append(cost_fn(y_test - Y_val))
    cost = np.mean(batch_cost)
    C.append(cost)
    print('epoch %s - train mse %s' % (i, cost))


# Show results

pdata = pd.DataFrame({'epochs': list(range(epochs)), 'train_loss': C, 'test_loss': t_C})
pdata.plot(x='epochs', y=['train_loss', 'test_loss'])

Y_pred, _, _ = forward(x_test, model, B)
Y_pred = y_scaler.inverse_transform(Y_pred)
rmse = np.sqrt(mean_squared_error(y_test_orig, Y_pred))
print(y_test_orig[:5], Y_pred[:5])
print('Root mean squared error:', rmse)
plt.show()
