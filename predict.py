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

parser = argparse.ArgumentParser(description = 'predict.py file parser')

parser.add_argument('input', default='./flowers/test/1/image_06743.jpg', nargs='?', action="store", type = str)#input image path argument
parser.add_argument('--dir', action="store",dest="data_dir", default="./flowers/")#data directory to get data from (here we get images from 'flowers' directory)
parser.add_argument('--gpu', default="gpu", action="store", dest="gpu")#defining gpu/cpu argument
parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')#defining category names mapping file argument
parser.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)#defining topk predictions argument
parser.add_argument('checkpoint', default='./checkpoint.pth', nargs='?', action="store", type = str)#checkpoint path argument

#assigning values to variables through argument parser
args = parser.parse_args()
path_of_image = args.input#get path to input image
no_of_outputs = args.top_k#get number of outputs to show
device = args.gpu#get device to use (gpu here)
checkpoint_path = args.checkpoint#getting path to checkpoint file
json_filename = args.category_names#get path to category names mapping file

def main():
    
    #loading model checkpoint !important!
    model = model_data.loading_checkpoint(checkpoint_path)
    
    #load category names mapping
    with open(json_filename, 'r') as json_data:
        name = json.load(json_data)
    
    #making predictions for input image using model (calculating probability)
    pbs_calc = model_data.predict(path_of_image, model, no_of_outputs, device)
    
    #convert probability calculated and label acquired to numpy arrays
    prob_calc = np.array(pbs_calc[0][0])
    
    target = [name[str(index + 1)] for index in np.array(pbs_calc[1][0])]
    
    #printing top k predictions and their output probability
    k = 0
    while k < no_of_outputs:
        #printing prediction likelihood class
        print("The image is {} with a likelihood of {} ({}%)".format(target[k], prob_calc[k], (prob_calc[k]*100)))
        k += 1
        
    #message indicating end of printing/prediction
    print("End of Prediction! Hope the results were satisfactory!")

#calling main function  
if __name__== "__main__":
    main()