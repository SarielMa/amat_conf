# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 17:20:18 2019

@author: liang
"""
import sys
sys.path.append('../../core')
#%%
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import torch
import pandas as pd
import time
#%%
random_seed=1
#%%
#https://pytorch.org/docs/stable/notes/randomness.html
#https://pytorch.org/docs/stable/cuda.html
import random
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(random_seed)

#%%

#%% ------ use this line, and then this file can be used as a Python module --------------------
def grid_validation(noise, delta, task):
#%%

    loss_name = "model_AMAT"+task+"_N_"+str(noise)+"_D_"+str(delta)+"_Dice_epoch50_result_wba_L2_"+task+".pt"

    acc_list = []
    noise_list = []
    auc = 0
    for n in os.listdir("./result"):
        if loss_name == n:
            print ("selected filename is ",n)
            checkpoint=torch.load("./result/"+n, map_location=torch.device('cpu'))
            # extract the acc_lsit and noise level list to get the auc and plots
            #print ("good")
            
            acc_list = []

            noise_list = []
            for i in checkpoint['result_100pgd']:
                #if i['noise_norm']<0.3:
                #    continue
                acc_list.append(i['avgDice'])
                noise_list.append(i['noise'])
    return acc_list, noise_list
            
#%%
def cal_AUC_robustness(acc_list, noise_level_list, maxNoise):
    #noise_level_list[0] is 0
    #acc_list[0] is acc on clean data
    auc=0
    for n in range(1, len(acc_list)):
        auc+= (acc_list[n]+acc_list[n-1])*(noise_level_list[n]-noise_level_list[n-1])*0.5
        if noise_level_list[n]==maxNoise:
            break
    auc/=noise_level_list[n]
    
    # times acc clean and sqrt 
    auc_robust = np.sqrt(acc_list[0]*auc)
    return auc_robust



def cal_AUC_old(acc_list, noise_level_list, maxNoise):
    #noise_level_list[0] is 0
    #acc_list[0] is acc on clean data
    auc=0
    for n in range(1, len(acc_list)):
        auc+= (acc_list[n]+acc_list[n-1])*(noise_level_list[n]-noise_level_list[n-1])*0.5
        if noise_level_list[n]==maxNoise:
            break
    auc/=noise_level_list[n]
    

    return auc

def cal_AUC_robustness_2_old(acc_list, noise_level_list, maxNoise):
    #noise_level_list[0] is 0
    #acc_list[0] is acc on clean data
    auc=0
    for n in range(1, len(acc_list)):
        auc+= (acc_list[n]+acc_list[n-1])*(noise_level_list[n]-noise_level_list[n-1])*0.5
        if noise_level_list[n]==maxNoise:
            break
    auc/=noise_level_list[n]
    
    # times acc clean and sqrt 
    auc_robust = np.sqrt(acc_list[0]*acc_list[0]*auc)
    return auc_robust

def cal_AUC_robustness_2(acc_list, noise_level_list, maxNoise):
    return np.sqrt(acc_list[0]*acc_list[-1])



def get_heat_map(data, epsilons, deltas, path):
    #heatmap = np.array(data)
    fig, ax = plt.subplots(figsize=(8,7))
    df = pd.DataFrame(data, index = epsilons, columns = deltas)
    df["mean"] = df.mean(axis = 1)
    df.loc["mean"] = df.mean(axis = 0)
    df_v = df.copy()
    df_v["mean"] = float("nan")
    df_v.loc["mean"] = float("nan")
    df_m = df.copy()
    df_m.loc[:-1,:-1] = float("nan")
    df_m.loc["mean","mean"] = float("nan")
    sns.heatmap(ax = ax, data = df_v, annot=True, fmt = ".4f", cmap="Reds",)
    sns.heatmap(ax = ax, data = df_m, annot=True, fmt = ".4f", cmap="Blues",)
    ax.set(ylabel='Noise', xlabel='Delta')
    ax.figure.savefig(path+".pdf", bbox_inches='tight',pad_inches=0.0)
    ax.figure.clf()

if __name__=="__main__":
    #[0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    #[0.005,0.01, 0.015, 0.02, 0.025, 0.03 ]
    # in paper it is 1.5
    import matplotlib.pyplot as plt
    import seaborn as sns
    import csv
    choice = 2
    task = ""
    mat = []
    matacc=[]
    matrobust = []
    mat2 = []
    
    if choice == 0:
        task = "002"
        epsilons = [float("inf")]
        deltas = [1,3,5,7,9]
        
    if choice == 1:
        task = "004"
        epsilons = [float("inf")]
        deltas = [0.1,0.3,0.5,0.7,0.9,1.1,1.3,1.5]

    if choice == 2:
        task = "005"
        epsilons = [float("inf")]
        deltas = [1,3,5,7,9,11,13,15,17,19,21]       
    
    
    
    accLists = []
    noiseList = []
    for i, noise in enumerate(epsilons):
        line = []
        lineacc = []
        linerobust = []
        line2 = []
        for j, delta in enumerate(deltas): # in paper it is 1.5/100
            accList, noiseList = grid_validation(noise, delta, task)
            accLists.append(accList)
            line.append(round(cal_AUC_robustness(accList, noiseList, noiseList[-1]), 4))
            lineacc.append(round(accList[0],4))
            linerobust.append(round(accList[-1], 4))
            line2.append(round(cal_AUC_robustness_2(accList, noiseList, noiseList[-1]),4))
        mat.append(line)
        matacc.append(lineacc)
        matrobust.append(linerobust)
        mat2.append(line2)
    
    
    
    
    datas = [matacc, matrobust, mat, mat2]
    paths = ["clean_AVG_Dice","noisy_AVG_Dice", "AUC_Robust", "GeoMean"]
    base = "result/"
    
    """
    for i in range(len(datas)):       
        get_heat_map(datas[i], epsilons, deltas, base+task+paths[i])
    """
    #colors = ["b","r","g","c","m","y","k"]
    
    #d = {'5000cca229d10d09': {374851: 1}, '5000cca229cf3f8f': {372496:3},'5000cca229d106f9': {372496: 3, 372455: 2}, '5000cca229d0b3e4': {380904: 2, 380905: 1, 380906: 1, 386569: 1}, '5000cca229d098f8': {379296: 2, 379297: 2, 379299: 2, 379303: 1, 379306: 1, 379469: 1, 379471: 1, 379459: 1, 379476: 1, 379456: 4, 379609: 4}, '5000cca229d03957': {380160: 3, 380736: 3, 380162: 1, 380174: 1, 381072: 2, 379608: 2, 380568: 3, 380569: 1, 380570: 1, 379296: 3, 379300: 1, 380328: 3, 379306: 1, 380331: 1, 379824: 2, 379825: 1, 379827: 1, 380344: 1, 379836: 1, 379456: 3, 380737: 1, 380739: 1, 379462: 1, 379476: 1, 379992: 3, 379609: 1, 379994: 1, 379611: 1, 379621: 1, 380006: 1, 380904: 3, 380905: 1, 380907: 1, 380535: 3, 380536: 1, 380538: 1}, '5000cca229cf6d0b': {372768: 10, 372550: 15, 372616: 14, 372617: 20, 372653: 3, 372505: 2}, '5000cca229cec4f1': {372510: 132}}
    
    #jet = plt.cm.jet
    #colors = jet(np.linspace(0, 1, len(d)))
    
    fig, ax = plt.subplots()
    colors = list(plt.cm.tab10(np.arange(10))) + ["crimson", "indigo"]
    ax.set_prop_cycle('color', colors)
    
    
    for i, delta in enumerate(deltas):
        ax.plot(noiseList, accLists[i], label = "delta = "+str(delta)+", auc_robust ="+str(mat[0][i]) )
    ax.legend()
    fig.savefig(base+task+"aucRobust.pdf", bbox_inches = "tight")



    with open(base+task+"result.csv","w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["delta"]+noiseList+paths)
        for i, delta in enumerate(deltas):
            writer.writerow(["delta = "+str(delta)]+accLists[i]+[matacc[0][i],matrobust[0][i], mat[0][i], mat2[0][i] ])