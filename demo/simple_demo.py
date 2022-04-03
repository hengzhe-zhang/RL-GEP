from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from RL_GEP_sklearn import RLGEPRegressor
from simple_utils import boston_house_data

X, y, input_name = boston_house_data()
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
x_train, x_test = StandardScaler().fit_transform(x_train), StandardScaler().fit_transform(x_test)
reg = RLGEPRegressor(n_gen=500, verbose=True, head_length=15, learning_rate=1e-3,
                     variable_length=True, ga_probability=0.75, use_GPU=False,
                     test_data=(x_test, y_test), linear_hidden_units='[32,32]')
reg.fit(x_train, y_train)
print('Training Loss', mean_squared_error(y_train, reg.predict(x_train)))
print('Testing Loss', mean_squared_error(y_test, reg.predict(x_test)))
