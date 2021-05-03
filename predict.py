
import numpy as np
from acrnn import acrnn
import pickle
from sklearn.metrics import recall_score as recall
from sklearn.metrics import confusion_matrix as confusion
import os
import torch
import torch.optim as optim
import pdb







def load_data(in_dir):
    f = open(in_dir,'rb')
    train_data,train_label,test_data,test_label,valid_data,valid_label,Valid_label,Test_label,pernums_test,pernums_valid = pickle.load(f)
    #train_data,train_label,test_data,test_label,valid_data,valid_label = pickle.load(f)
    return train_data,train_label,test_data,test_label,valid_data,valid_label,Valid_label,Test_label,pernums_test,pernums_valid
    
def test(model_path):
    train_data, train_label, test_data, test_label, valid_data, valid_label, Valid_label, Test_label, pernums_test, pernums_valid = load_data('/content/drive/MyDrive/IEMOCAP.pkl')
    test_label = test_label.reshape(-1)
    model=torch.load(model_path)
    result=model.predict(test_data)
    
    
if __name__ == '__main__':
    traindata_path = '/content/drive/MyDrive/IEMOCAP.pkl'
    model_path = '/content/drive/MyDrive/best_model.pth'
    result=test(model_path)
    print(result)

    
    
    
