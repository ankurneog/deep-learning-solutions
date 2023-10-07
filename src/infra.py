"""
NLI ML Infrastructure  : Computation of Non Linear Interference using Neural Networks
Deep Learning framework : pytorch
version 1.0
@author: aneog
"""
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import torch as torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch import Tensor
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch.nn import Module
from sklearn.model_selection import train_test_split
import numpy as np
import os
import pickle


class NLINormalizer:
        #initialize normalizer with training data
        def __init__(self,X):
            self.avgpsdScalar = preprocessing.MinMaxScaler()
            self.avgpsdScalar.fit(X[:,0].reshape(-1,1))

            self.psdScalar = preprocessing.MinMaxScaler()
            self.psdScalar.fit(X[:,1].reshape(-1,1))
            
            self.scfcutScalar = preprocessing.MinMaxScaler()
            self.scfcutScalar.fit(X[:,2].reshape(-1,1))
            
            self.scfnScalar  = preprocessing.MinMaxScaler()
            self.scfnScalar.fit(X[:,3].reshape(-1,1))
            
            self.phicutScalar   = preprocessing.MinMaxScaler()
            self.phicutScalar.fit(X[:,4].reshape(-1,1))

            self.phinScalar = preprocessing.MinMaxScaler()
            self.phinScalar.fit(X[:,5].reshape(-1,1))

            self.symratecutScalar =preprocessing.MinMaxScaler()
            self.symratecutScalar.fit(X[:,6].reshape(-1,1))

            self.symratenScalar = preprocessing.MinMaxScaler()
            self.symratenScalar.fit(X[:,7].reshape(-1,1))



        def Normalize(self,data):
           # print("Normalizing with MinMaxScalar()...")
            avg_psd_norm         = self.avgpsdScalar.transform(data[:,0].reshape(-1,1))
            psd_norm             = self.psdScalar.transform(data[:,1].reshape(-1,1))
            scf_cut_norm         = self.scfcutScalar.transform(data[:,2].reshape(-1,1))
            scfn_norm            = self.scfnScalar.transform(data[:,3].reshape(-1,1))
            phicut_norm          = self.phicutScalar.transform(data[:,4].reshape(-1,1))
            phi_n_norm           = self.phinScalar.transform(data[:,5].reshape(-1,1))
            sym_rate_norm        = self.symratecutScalar.transform(data[:,6].reshape(-1,1))
            sym_rate_n_norm      = self.symratenScalar.transform( data[:,7].reshape(-1,1))
            
            normalized_data        = np.concatenate((avg_psd_norm,psd_norm,scf_cut_norm,
                                                scfn_norm,phicut_norm,phi_n_norm,
                                                sym_rate_norm,sym_rate_n_norm),axis=1)
            return normalized_data




class NLIDataSet(Dataset):
    import numpy as np

    def __init__(self, df):

        self.X = df.values[:, :-1] # all but the last column
        self.y = df.values[:, 8:9] # the last column

        # ensure input data is floats
        self.X = self.X.astype('float32')
        self.y = self.y.astype('float32')
        self.y = self.y.reshape((len(self.y), 1))
        self.num_features = self.X.shape[1]
        self.normalizer = NLINormalizer(self.X)

        print("X Shape : ",self.X.shape)
        print("Y Shape : ",self.y.shape)
        print("num_features : ",self.num_features)
        # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

    # get indexes for train and test rows
    def get_splits(self, n_test=0.33):
        # determine sizes
        test_size = round(n_test * len(self.X))
        train_size = len(self.X) - test_size
        # calculate the split
        return random_split(self, [train_size, test_size]) 
        
    


class FileManager:
    def __init__(self,model_name,loc,inputfile):
        self.inputfile    = os.path.join('..','dataset',loc,inputfile)
        self.modelpath    = os.path.join('..','model',model_name)
     
        try:
             os.makedirs(self.modelpath)
        except FileExistsError:
            # directory already exists
            print("Directory Already Exists!!")
            pass

        saved_model_name = model_name + ".pt"
        torchscript_file = model_name + "_torchscript.pt"
        self.torchscript_file = os.path.join(self.modelpath,torchscript_file)
        self.saved_model = os.path.join(self.modelpath,saved_model_name)


    def GetInputFile(self):
        print("Input file : ",self.inputfile)
        return self.inputfile


    def SaveModel(self,model):
        example_for_torchscript = torch.tensor([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8])
        traced_script_module = torch.jit.trace(model, example_for_torchscript)
        torch.save(model,self.saved_model) # save Pytorch Model
        traced_script_module.save(self.torchscript_file) #Save torch script model
        
        print("####################################################################")
        print("Model saved in : ",self.saved_model)
        print("Torch Script   : ",self.torchscript_file)

        #print("Logs saved  in : ", self.log_dir) #check how to save pytorch logs
        
    def SaveAssetAsPickle(self,asset,assetname):
        
        assetpath =  os.path.join(self.modelpath,assetname)
        print("Asset : ",assetname, " \nSaved as : ",assetpath)
        pickle.dump(asset, open(assetpath, 'wb'))
    
    def GetAssetFromPickle(self,picklefile):
        assetpath = os.path.join(self.modelpath,picklefile)
        asset = pickle.load(open(assetpath, 'rb'))
        return asset


    def SaveResults(self,results,modelname,inputfilename):
        filename = 'results_'+modelname+'.csv'
        filepath = os.path.join('results',modelname,datetime.datetime.now().strftime("%Y%m%d-%H"))
        try:
         os.makedirs(filepath)
        except FileExistsError:
          print("Directory Already Exists, overwritting!!")
          pass
        
        cwd = os.getcwd() 
        os.chdir(filepath)
        np.savetxt(filename,results,delimiter=',')
        print("\n\n Results saved in :  \n",filepath)
        print("File Name : \n",filename)
        os.chdir(cwd)


    def GetModel(self):
        model = torch.load(self.saved_model)
        return model