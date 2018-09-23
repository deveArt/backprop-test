import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

np.random.seed(555)

def d(*args):
    print(*args)
    sys.exit()

# Load data and some preprocessing
# https://archive.ics.uci.edu/ml/machine-learning-databases/tic-tac-toe/tic-tac-toe.data

df = shuffle(pd.read_csv('tic_tac.csv'))
df.columns = [1, 2, 3, 4 ,5 ,6 ,7 ,8, 9, 'target']

X = pd.get_dummies(df[[1, 2, 3, 4 ,5 ,6 ,7 ,8, 9]])

Y = df[['target']].copy()
Y.values[Y == 'positive'] = 1
Y.values[Y == 'negative'] = 0
Y = Y.astype(np.int8)

train_size = 750
X_train = X[:train_size].values
X_test = X[train_size:].values
Y_train = Y[:train_size].values
Y_test = Y[train_size:].values

# Hyper params/optimisation opts

input_dim = 27
hidden_layers = [80, 50, 30]
epochs = 350
lr = 0.18
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

def forward(X):
    H = []
    V = []

    H.append(X)
    V.append(np.dot(X, model[0].T) + B[0])
    H.append(activate(V[0]))

    hl_cnt = len(model) - 1

    for i in range(1, hl_cnt):
        V.append(np.dot(H[i], model[i].T) + B[i])
        H.append(activate(V[i]))

    V.append(np.dot(H[-1], model[hl_cnt].T) + B[hl_cnt])
    Y = activate(V[-1], last=True)
    return Y, H, V


def backward(H, err, V):
    grad = None
    r_model = list(enumerate(model))
    r_model.reverse()

    for i, wl in r_model:
        if i == (len(model) - 1):
            grad = err * a_deriv(V[i], last=True)
            dd = grad * H[i]
        else:
            grad = a_deriv(V[i]) * np.dot(grad, model[i+1])
            dd = np.einsum('ij,ik->ikj', H[i], grad)

        D = lr * np.mean(dd, axis=0)
        model[i] = wl + D


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

C = []
eps = []
# Training

for i in range(epochs):
    eps.append(i)
    Y, H, V = forward(X_train)

    err = Y_train - Y
    C.append(cost_fn(err))

    backward(H, err, V)

pdata = pd.DataFrame({'epochs': eps, 'cost': C})
pdata.plot(x='epochs', y='cost')

Y_pred, _, _ = forward(X_test)
acc = accuracy_score(Y_test, Y_pred.round())

print('accuracy:', acc)
plt.show()
