import numpy as np
import argparse
from collections import OrderedDict
import matplotlib.pyplot as plt
import json
from PIL import Image
import torch
from torch import nn, optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.autograd import Variable

import utility_data
import model_data

parser = argparse.ArgumentParser(description = 'train.py file parser')

parser.add_argument('data_dir', action="store", default="./flowers/")#data directory to call from where data is to be trained/tested
parser.add_argument('--save_dir', action="store", default="./checkpoint.pth")#checkpoint to save time by not training repeatedly
parser.add_argument('--arch', action="store", default="vgg16")#structure to train on (since we used vgg16 so far we'll continue with that)
parser.add_argument('--gpu', action="store", default="gpu")#device on which we are training (gpu for faster training)
parser.add_argument('--learning_rate', action="store", type=float,default=0.001)#learning rate value for each epoch
parser.add_argument('--epochs', action="store", default=3, type=int)#number of epochs
parser.add_argument('--dropout', action="store", type=float, default=0.2)#dropout rate
parser.add_argument('--hidden_units', action="store", dest="hidden_units", type=int, default=512)#hidden units value

#assigning values to variables through argument parser
args = parser.parse_args()
location = args.data_dir#directory location for main work (flowers)
dir_path = args.save_dir#save directory (ImageClassifier)
architecture = args.arch#architecture model used (vgg16)
gpu_p = args.gpu#gpu (yes)
lr = args.learning_rate#learning rate for epochs (0.001)
epochs = args.epochs#number of epochs (3 with hidden batches for each)
dropout = args.dropout#dropout rate (0.2 (updated from 0.5 used in jupyter n/b for better accuracy))
hidden_units = args.hidden_units#number of hidden units/layer runs per epoch (512)

#checking for gpu
if torch.cuda.is_available() and gpu_p == 'gpu':
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    print("Warning: GPU mode is not enabled.")

def main():
    #call utility_data and model_data to load the data loaders and model
    trainloader, validloader, testloader, train_data = utility_data.load_data(location)
    model, criterion = model_data.network_build(architecture,dropout,hidden_units,lr,gpu_p)
    optimizer = optim.Adam(model.classifier.parameters(), lr= 0.001)
    
    # training the model
    step_num = 0
    Print_every = 5
    run_loss = 0
    print("!! Start of Training of model !!")
    #working through each epoch
    for epoch in range(epochs):
        for datas, target in trainloader:
            step_num += 1
            
            #checking again for gpu 
            if torch.cuda.is_available() and gpu_p =='gpu':
                datas = datas.to(device)
                target = target.to(device)
                model = model.to(device)
                
            #set gradients of all optimized tensors to zero
            optimizer.zero_grad()

            logs = model.forward(datas) #pass the input through the model to get the output logits
            loss = criterion(logs, target) #calculate the loss between the logits and the target
        
            loss.backward() #perform backpropagation to compute the gradients of the loss w.r.t the model parameters
            optimizer.step() #update the model parameters with the computed gradients

            run_loss += loss.item()

            if step_num % Print_every == 0:
                accu = 0
                apparent_loss = 0

                model.eval()
                with torch.no_grad():
                    #looping through validation data
                    for datas, target in validloader:
                        datas = datas.to(device)
                        target = target.to(device)
                        
                        #passing the validation data through model
                        logs = model.forward(datas)
                        
                        #calculating batch loss and validation loss
                        batch_loss = criterion(logs, target)
                        apparent_loss += batch_loss.item()

                        # Calculating accuracy for each epoch
                        xa = torch.exp(logs)
                        
                        #getting highest probability class for each input
                        top_p, top_class = xa.topk(1, dim=1)
                        
                        #checking if predicted class matches target class
                        equals = top_class == target.view(*top_class.shape)
                        
                        #calculating accuracy of model based on matches
                        accu += torch.mean(equals.type(torch.FloatTensor)).item()
                        
                #printing some values to check if each epoch is running smoothly
                print(f"Epoch number {epoch+1}/{epochs}.. "
                      f"Loss: {run_loss/Print_every:.3f}.. "
                      f"Validation Loss: {apparent_loss/len(validloader):.3f}.. "
                      f"Accuracy: {accu/len(validloader):.3f}")
                #setting loss on each run for next epoch/batch
                run_loss = 0
                model.train()
                
    #class_to_idx is necessary to be monitored in case of an issue(list discrepancy/Keyerror)
    model.class_to_idx =  train_data.class_to_idx
    
    #saving checkpoint !Very important! to save time else will have to train every time model is to be run
    torch.save({'structure' :architecture, 'hidden_units':hidden_units, 'dropout':dropout, 'learning_rate':lr, 'no_of_epochs':epochs, 'state_dict':model.state_dict(), 'class_to_idx':model.class_to_idx}, dir_path)
    
    print("Saving Checkpoint......")
    #successful message on saving checkpoint
    print("!! Checkpoint Saved !!")
    
#calling main function
if __name__ == "__main__":
    main()