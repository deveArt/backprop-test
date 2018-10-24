import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import sys
from bp_conf import *
from learn_fn import *

np.random.seed(555)

# Load data and do some preprocessing
#

kc_data_org = pd.read_csv('kc_house_data.csv')
kc_data_org['sale_yr'] = pd.to_numeric(kc_data_org.date.str.slice(0, 4))
kc_data_org['sale_month'] = pd.to_numeric(kc_data_org.date.str.slice(4, 6))
kc_data_org['sale_day'] = pd.to_numeric(kc_data_org.date.str.slice(6, 8))

features = kc_data_org[['sale_yr','sale_month','sale_day',
        'bedrooms','bathrooms','sqft_living','sqft_lot','floors',
        'condition','grade','sqft_above','sqft_basement','yr_built',
        'zipcode','lat','long','sqft_living15','sqft_lot15']].values
target = kc_data_org[['price']].values

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

# Init weights

weights = []
biases = []

for i, layer_units in enumerate(hidden_layers):
    if i == 0:
        weights.append(np.random.randn(layer_units, input_dim) / np.sqrt(input_dim))
    else:
        prev_dim = hidden_layers[i - 1]
        weights.append(np.random.randn(layer_units, prev_dim) / np.sqrt(prev_dim))
    biases.append(np.zeros([layer_units, 1]))

biases.append(np.zeros([1, 1]))
weights.append(np.random.randn(1, hidden_layers[-1]) / np.sqrt(hidden_layers[-1]))

model = {
    'W': weights,
    'B': biases
}

# Model training

C = []
t_C = []

for i in range(epochs):
    batch_cost = []

    for train_batch_features, train_batch_target in get_batches_iter(x_train, y_train, batch_size):
        Y, H, V = forward(train_batch_features, model)

        err = train_batch_target - Y
        batch_cost.append(cost_fn(err))

        grads = backward(H, err, V, model, lr) # pass update = False to make gradient checking
        #gradient_check(grads, train_batch_features, train_batch_target, model)

    Y_val, _, _ = forward(x_test, model) # validate on each epoch
    t_C.append(cost_fn(y_test - Y_val))
    cost = np.mean(batch_cost)
    C.append(cost)
    print('epoch %s - train mse %s' % (i, cost))


# Show results

pdata = pd.DataFrame({'epochs': list(range(epochs)), 'train_loss': C, 'test_loss': t_C})
pdata.plot(x='epochs', y=['train_loss', 'test_loss'])

Y_pred, _, _ = forward(x_test, model)
Y_pred = y_scaler.inverse_transform(Y_pred)
rmse = np.sqrt(mean_squared_error(y_test_orig, Y_pred))
print('Root mean squared error:', rmse)
plt.show()
