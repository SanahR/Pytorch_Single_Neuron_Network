import torch
from torch import optim as optim
"""
Goal: Single Neuron Neural Network from PyTorch
"""

#Step 1: Making the Data:
trainingX = torch.tensor([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]],dtype=torch.float32)
trainingy = torch.tensor([2,4,6,8,10,12,14,16,18,20],dtype=torch.float32)
testX = torch.tensor([[5],[7],[8]],dtype=torch.float32)
testY = torch.tensor([10,14,16],dtype=torch.float32)

#Step 2: Creating the Model
algo = torch.nn.Sequential(
    torch.nn.Linear(1,1),
    #torch.nn.Sigmoid()
)
#plugs nn.Linear into nn.Sigmoid, which is 1/1+e^-(mx+b)
#The first number of torch.nn.Linear is the number of inputs, and the second number is the number of neurons in the layer
#If you had nn.Linear(3,1), the equation would be 1/1+e^-1(m1x1+m2x2.....+b)
#Some people use A for the output of nn.Linear and Z for the output of nn.Sigmoid. You can also use S.

#Step 3: Training the Model
#Putting the data and the equation together. Finding the unknowns of the model.
loss_fn = torch.nn.MSELoss()
optimizer = optim.SGD(algo.parameters(),lr=0.005)
n_epochs = 15
for epoch in range(n_epochs):
  for i in range(len(trainingX)):
    ypred = algo(trainingX[i])
    print(algo[0].weight.data)
    print(algo[0].bias.data)
    print(trainingX[i])
    print(ypred)
    loss = loss_fn(trainingy[i],ypred[0])
    print((trainingy[i]-ypred)**2)
    print(loss)
    loss.backward() #finds the error slope
    optimizer.step() #Applies the gradient (takes a step)
    optimizer.zero_grad() #sets the gradient equal to zero. If you don't do zero grad, the gradient keeps adding on.

#Step 4: Prediction
ypred = algo(testX)
print(ypred)
