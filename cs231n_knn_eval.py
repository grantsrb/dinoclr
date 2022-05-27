"""
Argue the model paths or a path to a folder with lots of model folders
as command line arguments. The output will be appended to
knn_results.csv

 $ python3 knn_eval.py path_to_model_folder path_to_model1 path_to_model2 ...
 
"""

import torch
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

def get_features(model, train_data, step_size=300):
    model.eval()
    feats = []
    with torch.no_grad():
        for i in tqdm(range(0, len(train_data), step_size)):
            f = model(train_data[i:i+step_size].cuda())
            feats.append(f.detach().cpu().data)
    return torch.cat(feats, dim=0)

def compute_distances_no_loops(Xtrain, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using no explicit loops.

    Input / Output: Same as compute_distances_two_loops
    """
    if type(X) == type(torch.zeros(1)):
        zeros = torch.zeros
        matmul = torch.matmul
    else:
        zeros = np.zeros
        matmul = np.matmul
    num_test = X.shape[0]
    num_train = Xtrain.shape[0]
    dists = zeros((num_test, num_train))

    X_sq = (X**2).sum(1)
    Train_sq = (Xtrain**2).sum(1)
    dists = X_sq[:,None] -2*matmul(X, Xtrain.T) + Train_sq[None]
    dists = dists**0.5

    return dists

def predict_labels(ytrain, dists, k=1):
    """
    Given a matrix of distances between test points and training points,
    predict a label for each test point.

    Inputs:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      gives the distance betwen the ith test point and the jth training point.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].
    """
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in range(num_test):
        closest_y = []
        closest_y = ytrain[np.argsort(dists[i,:])[:k]]
        y_pred[i] = np.argmax(np.bincount(closest_y))

    return y_pred

csv_file = "knn_eval.csv"

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

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
X_train = torch.FloatTensor(trainset.data).permute(0,3,1,2)
y_train = torch.LongTensor(trainset.targets)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
X_test = torch.FloatTensor(testset.data).permute(0,3,1,2)
y_test = torch.LongTensor( testset.targets)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

if os.path.exists(csv_file):
    main_df = pd.read_csv(csv_file, sep="!")
else:
    main_df = pd.DataFrame()
for checkpt_path in tqdm(checkpt_paths):
    if "folder" in main_df.columns and checkpt_path in set(main_df["folder"]):
        print("Already recorded", checkpt_path, "skipping")
        continue
    print("\nBeginning evaluation for", checkpt_path)
    try:
        df = dict()
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
        model.load_state_dict(checkpt["teacher"])
        model = model.backbone
        model.cuda()
        torch.cuda.empty_cache()
        
        failure = True
        bsize = 500
        while failure and bsize > 10:
            try:
                print("Getting train features")
                train_feats = get_features(model, X_train, step_size=bsize)
                print("Getting test features")
                test_feats = get_features(model, X_test, step_size=bsize)
                failure = False
            except:
                bsize = bsize//2
                print("Error ocurred, reducing bsize to", bsize)
        with torch.no_grad():
            print("Computing distances")
            dists = compute_distances_no_loops(train_feats.cpu().data.numpy(), test_feats.cpu().data.numpy())
            
        accs = []
        ks = list(range(4,20))
        print("Predicting Labels")
        for k in tqdm(ks):
            preds = predict_labels(y_train, dists, k=k)
            accs.append((preds == y_test.numpy()).mean())
        df["acc"] = accs
        df["k"] = ks
        df = pd.DataFrame(df)
        df["folder"] = checkpt_path
        df["param_count"] = param_count
        df["wrapper_count"] = wrapper_count-param_count
        for key in hyps.keys():
            if type(hyps[key])==type([]): hyps[key] = str(hyps[key])
            df[key] = hyps[key]
        args = vars(args)
        for key in args.keys():
            if type(args[key])==type([]): args[key] = str(args[key])
            elif type(args[key])==type((1,)): args[key] = str(args[key])
            df[key] = args[key]
        main_df = main_df.append(df, sort=True)
        main_df.to_csv("knn_eval.csv", header=True, index=False, sep="!", mode="w")
    except Exception as e:
        print(e.__class__)
        print(e.__traceback__.tb_frame)
        print("Error ocurred for", checkpt_path,"-- failed to record results")
