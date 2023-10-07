"""
NLI ML Model  : Computation of Non Linear Interference using Neural Networks
Deep Learning framework : pytorch
version 1.0
@author: aneog
"""

import torch 
import torch.nn as nn
from torch.nn import Linear
import torch.optim as optim
import matplotlib.pyplot as plt
import math
import analytics
NUM_EPOCH=60
class NLIModel(nn.Module):
    def __init__(self,name):
        super(NLIModel,self).__init__()
        self.name    = name
        self.hidden1 = Linear(8,128) #8 inputs
        self.act1    = nn.ELU()
        self.hidden2 = Linear(128,128)
        self.act2    = nn.ELU()
        self.output  = Linear(128,1) # 1 output , nli
    #forward propagation 
    def forward(self,X):
        X=self.hidden1(X)
        X=self.act1(X)
        X=self.hidden2(X)
        X=self.act2(X)
        X=self.output(X)
        return X

def Train(model,train_dl,cv_dl,normalizer):
    loss_function    = nn.MSELoss() #lossfn(input,target)
    optimizer        = optim.RMSprop(model.parameters())
    train_loss = []
    val_loss = []
    paitence =0 # for early stopping
    for t in range(NUM_EPOCH):
        print(f"\n")
        print(f"Epoch {t+1}\n-----------------------------------------------------------------------")
        t_loss  = TrainLoop(train_dl,model,normalizer,loss_function,optimizer)
        print(f"\n----------------------------")
        cv_loss = ValidateLoop(cv_dl,model,normalizer,loss_function)
        train_loss.append(t_loss)
        val_loss.append(cv_loss)
        print(f'\n Average Train Loss : {t_loss :<3f} | Average Validation Loss : {cv_loss :<3f}')
        if(cv_loss <0.05):
            if(paitence>5):
                print("Early stopping at epoch : ",t)
                break
            else:
                paitence+=1
    print("Done!")
    plt.plot(train_loss,'g')
    plt.plot(val_loss,'r')
    plt.show()

def TrainLoop(dataloader,model,normalizer,loss_fn,optimizer) :

    size = len(dataloader.dataset)
    total_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        inputs = torch.from_numpy(normalizer.Normalize(X))
        y_pred = model(inputs.float())
        loss = loss_fn(y_pred,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
        if batch % 10000== 0:
            loss, current = loss.item(), batch * len(X)
            print(f"Training loss: {loss:>7f}  Sample : [{current:>5d}/{size:>5d}]  Batch :{batch}/{math.ceil(size/dataloader.batch_size)}")
    avg_loss = total_loss/math.ceil(size/dataloader.batch_size)
    return avg_loss

def ValidateLoop(dataloader, model,normalizer, loss_fn):

    size = len(dataloader.dataset)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            inputs     = torch.from_numpy(normalizer.Normalize(X))
            y_pred     = model(inputs.float())
            loss       = loss_fn(y_pred,y).item()
            test_loss += loss
            current    =  batch * len(X)
            if batch%200==0:
                print(f"Cross Validation loss: {loss:>7f}  Sample : [{current:>5d}/{size:>5d}]  Batch :{batch}/{math.ceil(size/dataloader.batch_size)}")
    test_loss /= math.ceil(size/dataloader.batch_size)
    return test_loss

def Eval(model,normalizer,X,y):
    inputs = torch.from_numpy(normalizer.Normalize(X))
    output = model(inputs.float())
    y_pred = output.detach().numpy()

    error  = y_pred - y
    num_samples = error.shape[0]
    print("Number of samples in Test set : ",num_samples)
    num_correct_prediction=0
    for item in range(0,num_samples):
        if(abs(error[item])<=0.5):
            num_correct_prediction+=1
        else:
            print("Error > 0.5 : ",error[item])

    print("Model Evaluation : Error")
    print(error)
    analyzer = analytics.NLIAnalyzer(model.name,y_pred,y,X,)
    analyzer.PrintData()
    analyzer.Plot()
    print("\nNum Samples in Test set : ",num_samples)
    print("Num Correct Prediction  : ",num_correct_prediction)
    print('\n#########################################')
    print("# Accuracy ( error <=0.5) : ",(num_correct_prediction/num_samples)*100.0, "%    ")
    print('#########################################')


def Predict(model,normalizer,X):
    model.eval()
    inputs = torch.from_numpy(normalizer.Normalize(X))
    print("Normalized Input : ")
    print(inputs)
    output = model(inputs.float())
    y_pred = output.detach().numpy()
    print("Predicted NLI :")
    print(y_pred)
    return y_pred








    


    

        

