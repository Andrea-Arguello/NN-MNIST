import numpy as np
import pandas as pd
import neural_networks as nn
from scipy.optimize import minimize

#df = pd.read_csv(r'./datasets/csv/fashion-mnist_train.csv')
test = pd.read_csv(r'./datasets/csv/fashion-mnist_test.csv')

#shuffled = df.sample(frac=1)

#train, validate = np.split(shuffled, [50000])
#train.to_csv(path_or_buf='./datasets/csv/fashion-mnist_train_50k.csv',index=False)
#validate.to_csv(path_or_buf='./fashion-mnist_validate.csv',index=False)

train = pd.read_csv(r'./datasets/csv/fashion-mnist_train_50k.csv')
validate = pd.read_csv(r'./datasets/csv/fashion-mnist_validate.csv')

NORMALIZE = 1000.0
train_X = train[train.columns[~train.columns.isin(['label'])]].to_numpy() / NORMALIZE
train_y = train[['label']].to_numpy().reshape(len(train),1)
validate_X = validate[validate.columns[~validate.columns.isin(['label'])]].to_numpy() / NORMALIZE
validate_y = validate[['label']].to_numpy().reshape(len(validate),1)
test_X = test[test.columns[~test.columns.isin(['label'])]].to_numpy() / NORMALIZE
test_y = test[['label']].to_numpy().reshape(len(test),1)

train_Y = (train_y == np.asarray(range(10))).astype(int)
validate_Y = (validate_y == np.asarray(range(10))).astype(int)
test_Y = (test_y == np.asarray(range(10))).astype(int)
# A setear variables
NETWORK_ARCH = np.array([
    784,
    90, # ~(784*10)**0.5
    10
    ])
theta_shapes = np.hstack((
    NETWORK_ARCH[1:].reshape(len(NETWORK_ARCH)-1,1),
    (NETWORK_ARCH[:-1]+1).reshape(len(NETWORK_ARCH)-1,1)
))

#flat_thetas = nn.flatten_list_of_arrays([
#   np.random.rand(*theta_shape) for theta_shape in theta_shapes #Le agregué el 2 porque normal no converge
#])
#np.save('flat_thetas3',flat_thetas)
flat_thetas = np.load('flat_thetas3.npy')
#print(flat_thetas, flat_thetas * 2)
#flat_thetas = flat_thetas * 2

#nn.back_propagation(flat_thetas,theta_shapes,train_X,train_y)

result = minimize(
    fun=nn.cost_function,
    x0=flat_thetas,
    args=(theta_shapes,train_X,train_Y),
    method='L-BFGS-B',
    jac=nn.back_propagation,
    options={'disp':True, 'maxiter':3000}
)

print(result)
#np.save('result',result.x) #we don't want to run this again o se overwritea
