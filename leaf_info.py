"""
Argue the model paths or a path to a folder with lots of model folders
as command line arguments. The output will be appended to
knn_results.csv

 $ python3 best_leaf_linear_eval.py path_to_save_csv.csv path_to_bests.csv
 
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

def get_hook(out_dict, key):
    def hook(module, ins, out):
        out_dict[key] = out.cpu().detach()
    return hook

def get_features(model, train_data, step_size=300, layers={"backbone"}):
    outs = {}
    feats = {}
    handles = []
    for name,modu in model.named_modules():
        if name in layers or layers=="all":
            hook = get_hook(outs, name)
            handle = modu.register_forward_hook(hook)
            handles.append(handle)

    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(train_data), step_size)):
            _ = model(train_data[i:i+step_size].cuda())
            for k in outs:
                if k in feats:
                    feats[k].append(outs[k])
                else:
                    feats[k] = [outs[k]]
    for handle in handles:
        handle.remove()
    del handles
    keys = list(feats.keys())
    for k in keys:
        if len(feats[k]) > 0: feats[k] = torch.cat(feats[k],dim=0)
        else: del feats[k]
    return feats


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

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    #transforms.Resize((32,32)),
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


max_leafs = 30
csv_file = "/mnt/fs1/grantsrb/dinoclr_saves/linear_eval.csv"
main_df = None

for path in sys.argv[1:]:
    if os.path.exists(path):
        df = pd.read_csv(path, sep="!")
        if "leaf_idx" in df.columns:
            leaf_df = df
        else:
            csv_file = path
            main_df = df
    else:
        csv_file = path
        main_df = pd.DataFrame()
if main_df is None: main_df = pd.DataFrame()

leaf_df = leaf_df.loc[leaf_df["n_cnns"]>1]
paths = set(leaf_df["folder"])
ext = "checkpoint.pth"
checkpt_paths = []
for path in paths:
    if not os.path.exists(path):
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


for checkpt_path in tqdm(checkpt_paths):
    model_max = 1
    if "folder" in main_df.columns and checkpt_path in set(main_df["folder"]):
        model_max = np.max(main_df.loc[main_df["folder"]==checkpt_path]["n_leafs"])
        if model_max >= max_leafs:
            print("Already recorded", checkpt_path, "skipping")
            continue
    print("\nBeginning evaluation for", checkpt_path)
    
    max_k = 5
    df = {
        **{"acc":[], "n_leafs":[], "loss": []},
        **{"top_"+str(k):[] for k in range(1,max_k+1)}
    }
    #try:
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
    if model_max >= hyps["n_cnns"]:
        print("Already recorded", checkpt_path, "skipping")
        continue
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

    layer = "cnn.net"
    layers = {layer}
    model = model.backbone
    model.eval()
    model.cuda()
    torch.cuda.empty_cache()

    #X_train = X_train[:100]
    #y_train = y_train[:100]
    #X_test = X_test[:100]
    #y_test = y_test[:100]
    
    failure = True
    bsize = 750
    while failure and bsize >= 5:
        try:
            print("Getting train features")
            train_feats = get_features(
                model.cuda(),
                X_train,
                step_size=bsize,
                layers=layers
            )[layer]
            print("Getting test features")
            test_feats = get_features(
                model.cuda(),
                X_test,
                step_size=bsize,
                layers=layers
            )[layer]
            failure = False
        except:
            bsize = bsize//2
            print("Error ocurred, reducing bsize to", bsize)
    model.cpu()
    torch.cuda.empty_cache()

    leaf_info = leaf_df.loc[leaf_df["folder"]==checkpt_path]
    leaf_info = leaf_info.sort_values(by="acc", ascending=False)
    B1 = len(train_feats)
    B2 = len(test_feats)
    for n_leafs in range(int(model_max),int(max_leafs)):
        idxs = list(leaf_info["leaf_idx"][:n_leafs])
        if len(idxs) < n_leafs: break
        tr_feats = train_feats[:,idxs].reshape(B1,-1)
        tst_feats = test_feats[:,idxs].reshape(B2,-1)
        failure = True
        bsize = 256
        while failure and bsize > 10:
            try:
                print("training linear classifier")
                proj, loss = train_linear_classifier(
                    tr_feats,
                    y_train,
                    bsize
                )
                print("Linear Loss:", loss)
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
                    preds = predict(
                        proj,
                        tst_feats,
                        batch_size=bsize
                    )
                    failure = False
                except:
                    bsize = bsize//2
                    print("Error ocurred, reducing bsize to", bsize)
            accs = get_accs(preds, y_test.numpy(), top_k=max_k)
            print("N Leafs:", n_leafs, "| Acc:", accs)
            print()
        for k in accs:
            df["top_"+str(k)].append(accs[k])
        df["acc"].append(accs[1])
        df["loss"].append(loss)
        df["n_leafs"].append(n_leafs)
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
    #except Exception as e:
    #    print(e.__class__)
    #    print(e.__traceback__.tb_frame)
    #    print("Error ocurred for", checkpt_path,"-- failed to record results")