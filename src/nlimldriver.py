"""
NLI ML Driver code  : Computation of Non Linear Interference using Neural Networks
Deep Learning framework : pytorch
version 1.0
@author: aneog
"""
import infra
import nliml
import utils
import sys
import getopt
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader



def Usage():
    print("\n\n\n USAGE: ")
    print("For Training        : nlimldriver.py -m <modelname> -t  <trainingfile>")
    print("For Testing         : nlimldriver.py -m <modelname> -v  <testfile>")
    print("For Predicting      : nlimldriver.py -m <modelname> -p  <predictionfile>")
    print("\n\n\n")
    sys.exit(2)

def Help():
    print("\n\n\nHELP: ")
    print("This is a driver program for NLI Computation via Machine Lerning ")
    print(" Please refer documention available in doc/ folder for details ")
    print("You need to provide a model name and a traing/testing file to start using as shown below")
    print("For Training        : nlimldriver.py -m <modelname> -t  <trainingfile>")
    print("For Testing         : nlimldriver.py -m <modelname> -v  <testfile>")
    print("For Predicting      : nlimldriver.py -m <modelname> -p  <predictionfile>")
    print("For Train+Test      : nlimldriver.py -m <modelname> -t <trainingfile> -p  <predictionfile>")
    print("For Test+Predict    : nlimldriver.py -m <modelname> -v <testfile> -p  <predictionfile>")
    print("For Train+Predict   : nlimldriver.py -m <modelname> -t <trainingfile> -p  <predictionfile>")
    print("\n\n\n")
    sys.exit(2)


def Train(model_name,training_file):
    filemanager      = infra.FileManager(model_name,'train',training_file)
    train, cv        = utils.GetTrianTestDataFrames(filemanager.inputfile)
    train_dataset    = infra.NLIDataSet(train)
    cv_dataset       = infra.NLIDataSet(cv)
    input_normalizer = train_dataset.normalizer
    train_dl  = DataLoader(train_dataset, batch_size=32, shuffle=True)
    cv_dl     = DataLoader(cv_dataset, batch_size=512, shuffle=False)
    print("Training Dataset size : ",len(train_dl.dataset))
    print("CV Dataset size : ",len(cv_dl.dataset))
    nlimodel = nliml.NLIModel(model_name)
    nliml.Train(nlimodel,train_dl,cv_dl,input_normalizer)
    filemanager.SaveModel(nlimodel)
    filemanager.SaveAssetAsPickle(input_normalizer,'input_normalizer.pkl')

def Test(model_name,test_file):
    filemanager    = infra.FileManager(model_name,'test',test_file)
    test           = utils.GetDF(filemanager.inputfile)
    #test_dataset   = infra.NLIDataSet(test)
    nlimodel       = filemanager.GetModel()
    nlimodel.eval()
    input_normalizer = filemanager.GetAssetFromPickle('input_normalizer.pkl')
    X = test.values[:, :-1] # all but the last column
    y = test.values[:, 8:9] # the last column
    nliml.Eval(nlimodel,input_normalizer,X,y)



def Predict(model_name, file):
    filemanager    = infra.FileManager(model_name,'test',file)
    test           = utils.GetDF(filemanager.inputfile)
    nlimodel       = filemanager.GetModel()
    nlimodel.eval()
    input_normalizer = filemanager.GetAssetFromPickle('input_normalizer.pkl')
    X = test.values[:, 0:8] # all but the last column
    print("Input : ")
    print(X)
    predicted_nli = nliml.Predict(nlimodel,input_normalizer,X)







def Start():
    arglist = sys.argv[1:]
    opts  = "hm:t:v:p:"
    longopts  = ['help','modelname','train','validate','predict']
    try:
        options, args = getopt.getopt(arglist,opts,longopts)
    except getopt.GetoptError as err:
        print(err)
        Usage()
        sys.exit(2)

    modelname =''
    modelGiven=False
    optionGiven = False

    for o,a in options:
        if o=='-h':
            Help()
        if o=="-m":
            modelname = a
            if(modelname==''):
                Usage()
            print("model name : ",modelname)
            modelGiven=True
        elif o=="-t":
            if(modelname==''):
                Usage()
            trainingfile = a
            if(trainingfile==''):
                Usage()
            optionGiven=True
            Train(modelname,trainingfile)
        elif o=="-v":
            if(modelname==''):
                Usage()
            testingfile = a
            if(testingfile==''):
                Usage()
            optionGiven=True
            Test(modelname,testingfile)
        elif o=="-p":
            if(modelname==''):
                Usage()
            predictfile =a
            if(predictfile==''):
                Usage()
            optionGiven=True
            Predict(modelname,predictfile)
        else:
            assert False,"Unknown Option"
    if(modelGiven==False or optionGiven==False):
        Usage()
    print("\n\nDone!")
  

if __name__=='__main__': #activated only when run as script
    Start()
