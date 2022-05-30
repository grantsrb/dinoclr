"""
Argue the model paths or a path to a folder with lots of model folders
as command line arguments. The output will be appended to
knn_results.csv

 $ python3 linear_eval.py path_to_csv.csv path_to_model_folder0 path_to_model1 path_to_model2 ...
 
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import models
import utils
from vision_transformer import DINOHead
import vision_transformer as vits
import math
from tqdm import tqdm
import sys
import os

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

import matplotlib
matplotlib.rc('font', **font)

def get_features(model, train_data, step_size=300, layer_name="backbone"):
    outs = {}
    def hook(modu, inps, oupts):
        outs[layer_name] = oupts.cpu().detach().data
    for name,modu in model.named_modules():
        if layer_name == name:
            handle = modu.register_forward_hook(hook)
            break
    model.eval()
    feats = []
    with torch.no_grad():
        for i in tqdm(range(0, len(train_data), step_size)):
            _ = model(train_data[i:i+step_size].cuda())
            feats.append(outs[layer_name])
    handle.remove()
    return torch.cat(feats, dim=0)


def train_linear_classifier(Xtrain, ytrain, batch_size=2000, lr=1e-2):
    """
    Xtrain: torch float tensor (N, D)
        the features from the model
    ytrain: torch long tensor (N,C)
        the training labels
    """
    proj = nn.Linear(Xtrain.shape[-1], int(torch.max(ytrain)+1))
    optim = torch.optim.Adam(proj.parameters(), lr=lr)
    proj.cuda()
    loss_fxn = nn.CrossEntropyLoss()
    old_loss = math.inf
    tot_loss = 100
    eps = 1e-5
    loop = 0
    best_sd = None
    best_loss = math.inf
    while loop < 5:
        loop += 1 
        tot_loss = 0
        for i in range(0,len(Xtrain),batch_size):
            optim.zero_grad()
            preds = proj(Xtrain[i:i+batch_size].cuda())
            loss = loss_fxn(preds, ytrain[i:i+batch_size].cuda())
            loss.backward()
            optim.step()
            tot_loss += loss.item()
        tot_loss = tot_loss/len(Xtrain)
        print("Loop:", loop,
              "| Best:", round(best_loss, 5),
              "| Loss:", round(tot_loss,5),
              end=100*" "+"\r")
        if tot_loss < best_loss and np.abs(tot_loss-best_loss)>eps:
            loop = 0
            best_loss = tot_loss
            best_sd = proj.state_dict()
    proj.load_state_dict(best_sd)
    return proj, best_loss

def get_accs(preds, labels, top_k=5):
    top_k = top_k + 1
    accs = {i:0 for i in range(1,top_k)}
    top_k_args = np.argsort(-preds, axis=-1)[:,:top_k]
    labels = np.tile(labels[:,None],(1,top_k))
    for k in range(1, top_k):
        eqs = (top_k_args[:,:k] == labels[:,:k]).sum(-1)
        accs[k] = eqs.mean()
    return accs


def predict(proj, Xtest, batch_size=None):
    """
    Given a matrix of distances between test points and training points,
    predict a label for each test point.

    Inputs:
    - proj: torch nn module
        a linear layer
    - Xtest: torch float tensor (N,D)
        the testing inputs which should be features from the model

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].
    """
    if batch_size is None:
        return proj(Xtest.cuda()).argmax(-1).cpu().detach().numpy()
    y_preds = []
    for i in range(0, len(Xtest), batch_size):
        y_preds.append(proj(Xtest[i:i+batch_size].cuda()).cpu().detach())
    return torch.cat(y_preds, dim=0).numpy()

csv_file = "/mnt/fs1/grantsrb/dinoclr_saves/linear_eval.csv"

ext = "checkpoint.pth"
paths = sys.argv[1:]
checkpt_paths = []
for path in paths:
    if ".csv" in path:
        csv_file = path
        print("saving to", csv_file)
        continue
    elif not os.path.exists(path):
        print(path, "does not exist")
        continue
    if not os.path.isdir(path) and ext in path:
        checkpt_paths.append(path)
        continue
    for d,subds,files in os.walk(path):
        for f in files:
            if ext in f: checkpt_paths.append(os.path.join(d,f))
print("Searching over checkpts:")
for checkpt in sorted(checkpt_paths):
    print(checkpt)

transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

def unnormalize(X):
    """
    X: torch tensor or ndarray (..., 3, H, W)
    """
    if isinstance(X, type(np.ones((1,)))):
        means = np.asarray([0.485, 0.456, 0.406])
        stds =  np.asarray([0.229, 0.224, 0.225])
    else:
        means = torch.FloatTensor([0.485, 0.456, 0.406])
        stds = torch.FloatTensor([0.229, 0.224, 0.225])
    return X*stds[:,None,None] + means[:,None,None]

batch_size = 4

dataset = "cifar10"
if dataset == "cifar10":
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    X_train = torch.FloatTensor(trainset.data).permute(0,3,1,2)
    y_train = torch.LongTensor(trainset.targets)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    X_test = torch.FloatTensor(testset.data).permute(0,3,1,2)
    y_test = torch.LongTensor( testset.targets)
elif dataset=="cifar100":
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                            download=True, transform=transform)
    X_train = torch.FloatTensor(trainset.data).permute(0,3,1,2)
    y_train = torch.LongTensor(trainset.targets)
    
    testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                           download=True, transform=transform)
    X_test = torch.FloatTensor(testset.data).permute(0,3,1,2)
    y_test = torch.LongTensor( testset.targets)
if os.path.exists(csv_file):
    main_df = pd.read_csv(csv_file, sep="!")
else:
    main_df = pd.DataFrame()
for checkpt_path in tqdm(checkpt_paths):
    if "folder" in main_df.columns and checkpt_path in set(main_df["folder"]):
        print("Already recorded", checkpt_path, "skipping")
        continue
    print("\nBeginning evaluation for", checkpt_path)
    
    df = dict()
    try:
        checkpt = torch.load(checkpt_path, map_location='cpu')
        args = checkpt["args"]
        if args.arch in vits.__dict__.keys():                                  
            hyps = {                                                           
                "patch_size": args.patch_size,                                 
                "img_size": [32]
            }                                                                  
            teacher = vits.__dict__[args.arch]( **hyps )                       
            embed_dim = teacher.embed_dim
        elif args.arch in models.__dict__.keys():                              
            try:
                hyps = checkpt["hyps"]
            except:
                hyps = {
                    "n_cnns": 8, "inpt_shape": (3,32,32), "chans": [8,16,24,48,96],
                    "ksizes": 2, "strides": 1, "paddings": 0, "lnorm": True, "out_dim": 65536,
                }                                                                  
            teacher = models.__dict__[args.arch](**hyps)                       
            embed_dim = hyps["agg_dim"] if "out_dim" not in hyps else hyps["out_dim"] 
        model = utils.MultiCropWrapper(                                      
            teacher,                                                           
            DINOHead(embed_dim, args.out_dim, False),            
        )
        param_count = 0
        for p in teacher.parameters():
            param_count += math.prod(p.shape)
        wrapper_count = 0
        for p in model.parameters():
            wrapper_count += math.prod(p.shape)
        print("Param Count", param_count)
        print("Wrapper Count", wrapper_count)
        try:
            model.load_state_dict(checkpt["teacher"])
        except:
            new_sd = dict()
            for k in checkpt["teacher"]:
                if "module" in k:
                    k2 = ".".join(k.split(".")[1:])
                    new_sd[k2] = checkpt["teacher"][k]
                else:
                    new_sd[k] = checkpt["teacher"][k]
            model.load_state_dict(new_sd)
        layer_name = "backbone"

        model.eval()
        torch.cuda.empty_cache()
        
        failure = True
        bsize = 2000
        while failure and bsize > 10:
            try:
                print("Getting train features")
                train_feats = get_features(
                    model.cuda(),
                    X_train,
                    step_size=bsize,
                    layer_name=layer_name
                )
                print("train_feats:", train_feats.shape)
                print("Getting test features")
                test_feats = get_features(
                    model.cuda(),
                    X_test,
                    step_size=bsize,
                    layer_name=layer_name
                )
                failure = False
            except:
                bsize = bsize//2
                print("Error ocurred, reducing bsize to", bsize)
        model.cpu()
        torch.cuda.empty_cache()

        failure = True
        bsize = 256
        while failure and bsize > 10:
            try:
                print("training linear classifier")
                print("trainfeats:", train_feats.shape)
                proj, loss = train_linear_classifier(train_feats, y_train, bsize)
                print("Linear Loss:", loss)
                print()
                failure = False
            except:
                bsize = bsize//2
                print("Error ocurred, reducing bsize to", bsize)

        with torch.no_grad():
            print("Predicting labels")
            failure = True
            bsize = len(X_test)
            while failure and bsize > 10:
                try:
                    preds = predict(proj, test_feats, batch_size=bsize)
                    failure = False
                except:
                    bsize = bsize//2
                    print("Error ocurred, reducing bsize to", bsize)
            accs = get_accs(preds, y_test.numpy(), top_k=5)
            print("Acc:", accs)
        for k in accs:
            df["top_"+str(k)] = [accs[k]]
        df["acc"] = [accs[1]]
        df["loss"] = [loss]
        df = pd.DataFrame(df)
        df["folder"] = checkpt_path
        df["param_count"] = param_count
        df["wrapper_count"] = wrapper_count-param_count
        df["dataset"] = dataset
        for key in hyps.keys():
            if type(hyps[key])==type([]): hyps[key] = str(hyps[key])
            df[key] = hyps[key]
        args = vars(args)
        for key in args.keys():
            if type(args[key])==type([]): args[key] = str(args[key])
            elif type(args[key])==type((1,)): args[key] = str(args[key])
            df[key] = args[key]
        main_df = main_df.append(df, sort=True)
        main_df.to_csv(csv_file, header=True, index=False, sep="!", mode="w")
    except Exception as e:
        print(e.__class__)
        print(e.__traceback__.tb_frame)
        print("Error ocurred for", checkpt_path,"-- failed to record results")