import numpy
import torch

from torch import nn
from pathlib import Path
import matplotlib.pyplot as plt

# print("hello world")
# x = torch.rand(5,3)
# MARK: set the seed
torch.manual_seed(42)

weight = 0.7
bias = 0.3

# create data -- This step is replaced by importing/creating dataset
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)

# linear regression formula
y = weight * X + bias

# MARK: Linear Regression Class
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)
        self.bias = nn.Parameter(torch.randn(1,dtype=torch.float), requires_grad=True)
        
    # forward function 
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias
        
        
# MARK: split into training and testing data 
train_split = int(0.8 * len(X))

X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

print(len(X_train), len(y_train), len(X_test), len(y_test))


# plot 
def plot_predictions(train_data=X_train, train_labels=y_train, test_data=X_test, test_labels=y_test, predictions=None):
    """
        Plots training data, test data and compares predictions
    """
    # this sets the parameters for a figure for a figure of dimensions 10 * 7
    plt.figure(figsize=(10,7))
    
    #Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
    
    #plot the test data in green
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")
    
    if predictions is not None:
        # Plot the predictions in red (predictions were made on the test data)
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")
    
    # show the legend
    plt.legend(prop={"size":14})  

# plot_predictions()

# create instance of the model
model_0 = LinearRegressionModel()

# list model parameters
print(list(model_0.parameters()))

#list named parameters
model_0.state_dict()

#make inference using model
with torch.inference_mode():
    y_preds = model_0(X_test)
    
# print(f"Number of testing sample: {len(X_test)}")
# print(f"Number of predictions made: {len(y_preds)}")
# print(f"Predicted values:\n{y_preds}")

plot_predictions(predictions=y_preds)

# print(y_test - y_preds)

# MARK: Train model
#loss function:
loss_fn = nn.L1Loss() # MAE loss

#optimizer
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)

#MARK: Optimization loop
epochs = 100 #num of times model will pass over the training data

#Create empty loss lists to track values 
train_loss_values = []
test_loss_values = []
epoch_count = []

for epoch in range(epochs):
    ### Training
    
    #put model in training mode (default state)
    model_0.train()
    
    #1. Forward pass on train data using the forward() method inside
    y_prep = model_0(X_train)
    # print(y_prep)
    
    #2. Calculate the loss (difference from predictions to real)
    loss = loss_fn(y_prep, y_train)
    
    #3. Zero grad of the optimzer
    optimizer.zero_grad()
    
    #4. loss backwards
    loss.backward()
    
    #5. Progress the optimizer
    optimizer.step()
    
    ### Testing 
    # Put the model in evaluation mode
    model_0.eval()
    
    with torch.inference_mode():
        #1. Forward pass on test data
        test_pred = model_0(X_test)
        
        #2. Calculate the loss on test data
        test_loss = loss_fn(test_pred, y_test.type(torch.float))

        #print out what's happening
        if epoch % 10 == 0:
            epoch_count.append(epoch)
            train_loss_values.append(loss.detach().numpy())
            test_loss_values.append(test_loss.detach().numpy())
            #print(f"Epoch: {epoch} | MAE Train loss: {loss} | MAE Test loss: {test_loss}")
          
# MARK: Plot the loss curves 
# -- will appear on same graph so make sure to disable this one or the other -- TODO 
# plt.plot(epoch_count, train_loss_values, label="Train loss")
# plt.plot(epoch_count, test_loss_values, label="Test loss")
# plt.title("Training and test loss curves")
# plt.ylabel("Loss")
# plt.xlabel("Epochs")
# plt.legend()

#Find model's learned parameters
print("The model learned the following values for weights and bias:")
print(model_0.state_dict())
print("\nAnd the original values for weights and bias were:\n")
print(f"weights: {weight}, bias: {bias}")

# MARK: Making prediction with inference with trained model
model_0.eval() # set in evaluation mode

#setup the inference mode
with torch.inference_mode():
    y_preds = model_0(X_test)
# print(y_preds)

plot_predictions(predictions=y_preds) #update the prediction line

plt.show()

#MARK: saving the model
# 1. Create models directory
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2. Create model save path
MODEL_NAME = "01_pytorch_workflow_model_0.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3. Save the model state dict
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_0.state_dict(), # only saving the state_dict() only saves the models learned parameters
           f=MODEL_SAVE_PATH)
