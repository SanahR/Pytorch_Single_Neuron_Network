"""
Goal: Find unknowns in the equation of the network/single neuron that given an X, will output values as close as possible to the Y values.

1. Data
- Create X (features) and y (targets)
2. Model
- Use a single neuron, single input with the equation 1/1+e^-(mx+b)
- Single neuron, multiple inputs: 1/1+e^-(m1x1+m2x2....+b)
- Each input is multiplied by a number, and the letter m is used to show weights.
- Example data: X = np.array([[1,2,3],[4,5,6],[7,8,9]])
- Index 0 is [1,2,3] which are the multiple inputs of x 1: 1, x2: 2, x3: 3
3. Training
- Use gradient descent to find the best weights (m) and bias (b)
4. Testing/Prediction
- Use the trained model with calculated m and b to predict targets of new data
"""

X = [1,2,3,4,8,9,10,11]
y = [0,0,0,0,1,1,1,1]
m = 1.5
b = -9.92
e = 2.718281828459045
error_list = []
learning_rate = 0.01
for a in range(250):
  for item in range(len(X)):
    pred = 1/(1+e**-(m*X[item]+b)) #Computing the output of the single neuron
    m = m-learning_rate* (pred-y[item]) * (pred*(1-pred)) * X[item] #Updating the value of M
    b = b-learning_rate* (pred-y[item]) * (pred*(1-pred)) #Updating the value of B
    print(X[item],y[item], "\n",pred)
    error = abs(y[item]-pred)
  error_list.append(error)
  print()
  print(m,"\n",b)
  print()
print(error_list)
