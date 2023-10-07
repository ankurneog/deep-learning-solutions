   
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

class NLIAnalyzer:
    def __init__(self,model_name,predicted,actual,X,customtext=''):
        self.modelname    = model_name
        self.predicted    = predicted
        self.actual       = actual
        self.input        = X
        self.customtext   = customtext
    def GetMaxIndices( self,array2d):
        max_result = np.where(array2d==np.amax(array2d))
        return max_result
    def GetMinIndices(self,array2d):
        min_result = np.where(array2d==np.amin(array2d))
        return min_result

    
    def PlotErrorDensity(self,customtext=''):
        error =self.actual-self.predicted
        plt.hist(error,bins=10)
        title = 'Error Density '+ self.modelname + self.customtext
        plt.title(title)
        plt.xlabel('Error')
        plt.ylabel('Num Samples')
        plt.show()
    def PlotError(self):
        plt.subplot(2,1,1)
        plt.plot(self.predicted,'r.')
        plt.plot(self.actual,'b')
        plt.legend(['predicted','actual'],loc='upper left')
        plt.subplot(2,1,2)
        plt.plot((self.actual-self.predicted),'g')
        plt.xlabel('sample index')
        plt.legend(['error'])
        plt.show()
    
    def Plot(self):
        self.PlotError()
        self.PlotErrorDensity()

    def PrintData(self):
        pred_error        = self.actual - self.predicted
        max_pred_postive  = pred_error.max()
        max_pred_negative = pred_error.min()
        pred_error_indice_pos = self.GetMaxIndices(pred_error)
        pred_error_indice_neg = self.GetMinIndices(pred_error)
        X_test_pos_error = self.input[pred_error_indice_pos[0],:]
        X_test_neg_error = self.input[pred_error_indice_neg[0],:]
    
        pred_error_abs = np.abs(pred_error)
        max_pred_error = pred_error_abs.max()
        min_pred_error = pred_error_abs.min()
        pred_error_indice_max = self.GetMaxIndices(pred_error_abs)
        pred_error_indice_min = self.GetMinIndices(pred_error_abs)
        X_test_max_error = self.input[pred_error_indice_max[0],:]
        X_test_min_error = self.input[pred_error_indice_min[0],:]

        #self.model.summary()
        print("-----------------------------------------------------------------")
        print("Inferences for Model : ",self.modelname, " ",self.customtext)
        print(" Root Mean Squared Error (RMSE)             :%.4f"%mean_squared_error(self.predicted,self.actual,squared=False))
        print(" Standard Deviation                         :%.4f"%np.std(pred_error))
        print(" Mean                                       :%.4f"% np.mean(pred_error))
        print("-----------------------------------------------------------------")
    
        print(" Max ABS Prediction Error                   : ",max_pred_error)
        print(" Sample Index in Test Set                   : ",pred_error_indice_max[0])
        print(" X values with Max Prediction Errors        : ")
        print( X_test_max_error)
        print("-----------------------------------------------------------------")
        print(" Min ABS Prediction Error                    : ", min_pred_error)
        print(" Sample Index in Test Set                    : ", pred_error_indice_min[0])
        print(" X values with Min Prediction Errors         : ")
        print( X_test_min_error)
        print("-----------------------------------------------------------------")
    
    
        print(" Max Postive Error                          : ",max_pred_postive)
        print(" Sample Index in Test Set                   : ",pred_error_indice_pos[0])
        print(" X values with Max Postive Prediction Errors: ")
        print( X_test_pos_error)
        print("-----------------------------------------------------------------")
        print(" Max Negative Error                         : ",max_pred_negative)
        print(" Sample Index in Test Set                   : ",pred_error_indice_neg[0])
        print("X values with Min Negative Prediction Errors: ")
        print( X_test_neg_error)
        print("-----------------------------------------------------------------")
    

      