# Base
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import matplotlib.pyplot as plt 
import numpy as np
import random
import json
import seaborn as sns
from tqdm import tqdm

# Paths - Adjusted for Windows
DIR_DATA = "D:/NUS/BMI5101/DNN/Data_Processed/" 
DIR_NET_SAVING = "D:/NUS/BMI5101/DNN/Models_Testing/" 
DIR_DATA_TEST = "D:/NUS/BMI5101/DNN/Data_Processed/" 
SUBJECT_USED_FOR_TESTING = "S17"

# Create saving dir if not exists
if not os.path.exists(DIR_NET_SAVING):
    os.makedirs(DIR_NET_SAVING)

# Seed
manualSeed = 1
torch.manual_seed(manualSeed)
random.seed(manualSeed)
np.random.seed(manualSeed)
g = torch.Generator()
g.manual_seed(manualSeed)

# Global Performance Settings
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

# Functions
def suppr(dic):
  """ Delete extremums from a dictionary """
  bornemax=np.quantile(dic["features"],0.99,axis=0)
  bornemin=np.quantile(dic["features"],0.01,axis=0)
  indicesmauvais=np.where(np.sum(np.add(bornemin>np.array(dic["features"]),np.array(dic["features"])>bornemax),axis=1)>0)[0]
  k=0
  for i in indicesmauvais:
    del dic["features"][i-k]
    del dic["label"][i-k]
    k+=1
  return dic

def extract_ds_from_dict(data):
  """ Extract dataset and filter outliners """
  Letat=[]
  for i in range(0,4):
    dictio={}
    features=[data["features"][j] for j in np.where(np.array(data["label"])==i+1)[0]] 
    label=[data["label"][j] for j in np.where(np.array(data["label"])==i+1)[0]]
    dictio["features"]=features
    dictio["label"]=label
    Letat.append(dictio.copy())
  neutr=Letat[0]; stress=Letat[1]; amu=Letat[2]; med=Letat[3]
  neutr=suppr(neutr); stress=suppr(stress); amu=suppr(amu); med=suppr(med)
  features=[]; label=[]; dict_id={}
  for m in range(0,4):
    dictio=Letat[m]
    features+=[x for x in dictio["features"]] 
    label+=[x for x in dictio["label"]]
  dict_id["features"]=features
  dict_id["label"]=label
  return dict_id.copy()

def conf_mat(net,datal,trsh):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  x=datal[0].float().to(device, non_blocking=True)
  y=net(x).view(-1)
  pred=(y>trsh).int()
  label=datal[1].float().to(device, non_blocking=True).view(-1).int()
  num=datal[2].float().to(device, non_blocking=True).int()
  comp=torch.eq(label,pred).int()
  mat_label=np.zeros((2,4))
  mat_nolbl=np.zeros((2,2))
  for i in range(0,4):
    tens=torch.where(num==i+1,1,0)
    numtot=torch.sum(tens).item()
    num_G=torch.sum(torch.where(torch.mul(tens,comp)==1,1,0)).item()
    if i ==1:
      mat_nolbl[0,0]+=num_G; mat_nolbl[1,0]+=numtot-num_G
      mat_label[0,i]=num_G; mat_label[1,i]=numtot-num_G
    else:
      mat_nolbl[1,1]+=num_G; mat_nolbl[0,1]+=numtot-num_G
      mat_label[1,i]=num_G; mat_label[0,i]=numtot-num_G
  return mat_label,mat_nolbl

def fusion_dic(list_dic):
  features=[]; label=[]; dic_f={}
  for dic in list_dic:
    features+=dic["features"]; label+=dic["label"]
  dic_f["features"]=features; dic_f["label"]=label
  return dic_f

def proportion(dic, indice, prop):
  tot=len(indice)
  features=[dic["features"][j] for j in indice[::int(np.ceil(tot/prop))]]
  label=[dic["label"][j] for j in indice[::int(np.ceil(tot/prop))]]
  return features,label

def eq_dic(dic):
  indice_neutr=np.where(np.array(dic["label"])==1)[0]
  indice_stress=np.where(np.array(dic["label"])==2)[0]
  indice_amu=np.where(np.array(dic["label"])==3)[0]
  indice_med=np.where(np.array(dic["label"])==4)[0]
  prop=min([3*len(indice_neutr),len(indice_stress),3*len(indice_amu),3*len(indice_med)])
  p_s=prop; p_n=int(0.333*prop); p_a=int(0.333*prop); p_m=int(0.333*prop)
  features=[]; label=[]; dic_f={}
  tf,tl=proportion(dic,indice_neutr,p_n); features+=tf; label+=tl
  tf,tl=proportion(dic,indice_stress,p_s); features+=tf; label+=tl
  tf,tl=proportion(dic,indice_amu,p_a); features+=tf; label+=tl
  tf,tl=proportion(dic,indice_med,p_m); features+=tf; label+=tl
  dic_f["features"]=features; dic_f["label"]=label
  return dic_f

class ds_wesad(Dataset):
    def __init__(self, dic):
        self.samples = []
        for i in range(len(dic["label"])):
            num=dic["label"][i]
            stress=num==2
            x=np.array(dic["features"][i])
            self.samples.append((x,int(stress),num))
    def __len__(self): return len(self.samples)
    def __getitem__(self, id): return self.samples[id]

def init_weight(m):
    if isinstance(m,nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    if isinstance(m, nn.BatchNorm1d):
        m.weight.data.fill_(1)
        m.bias.data.zero_() 

class ClassifierECG(nn.Module):
    def __init__(self, ngpu):
        super(ClassifierECG, self).__init__()
        self.nnECG = nn.Sequential(
            nn.Linear(12,128,bias=True), nn.BatchNorm1d(128), nn.Dropout(0.5), nn.LeakyReLU(0.2),
            nn.Linear(128,64,bias=True), nn.BatchNorm1d(64), nn.Dropout(0.5), nn.LeakyReLU(0.2),
            nn.Linear(64,16,bias=True), nn.BatchNorm1d(16), nn.Dropout(0.5), nn.LeakyReLU(0.2),
            nn.Linear(16,4,bias=True), nn.BatchNorm1d(4), nn.Dropout(0.5), nn.LeakyReLU(0.2),
            nn.Linear(4,1,bias=True), nn.Sigmoid()
        )
        self.nnECG.apply(init_weight)
    def forward(self, input): return self.nnECG(input)

def training(net,dataloader_t,dataloader_v,num_epochs,j,k, lr):
  Loss = []; Lossv= []
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999))
  for epoch in range(num_epochs):
      L_t=[]; L_v=[]
      for i, dataj in enumerate(dataloader_t, 0):
          net.zero_grad()
          x=dataj[0].float().to(device, non_blocking=True)
          yhat=dataj[1].float().to(device, non_blocking=True).view(-1,1)
          y=net(x)
          err_t=nn.BCELoss()(y.float(),yhat.float())
          err_t.backward(); optimizer.step(); L_t.append(err_t.item())
      for i, dataj in enumerate(dataloader_v, 0):
        net.eval()     
        x=dataj[0].float().to(device, non_blocking=True)
        yhat=dataj[1].float().to(device, non_blocking=True).view(-1,1)
        y=net(x)
        err_v=nn.BCELoss()(y.float(),yhat.float()); L_v.append(err_v.item())
      err=np.mean(L_t); errv=np.mean(L_v)
      Loss.append(err); Lossv.append(errv)
      print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {err:.4f} | Val Loss: {errv:.4f}")
      torch.save(net.state_dict(), os.path.join(DIR_NET_SAVING, f"net_{j}_{k}_epoch_{epoch}.pth"))
  return [Lossv,np.argmin(Lossv)]

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed); random.seed(worker_seed)

if __name__ == '__main__':
    name_list= ['WESADECG_S2.json', 'WESADECG_S3.json', 'WESADECG_S4.json', 'WESADECG_S5.json', 'WESADECG_S6.json', 'WESADECG_S7.json',
     'WESADECG_S8.json', 'WESADECG_S9.json', 'WESADECG_S10.json', 'WESADECG_S11.json', 'WESADECG_S13.json', 'WESADECG_S14.json',
     'WESADECG_S15.json', 'WESADECG_S16.json']

    L_data=[]
    for name in name_list:
        with open(os.path.join(DIR_DATA, name), 'r') as f:
            data = json.load(f)
            L_data.append(eq_dic(data))
    
    dic_merge = fusion_dic(L_data)
    ds_training = ds_wesad(extract_ds_from_dict(dic_merge))
    
    name_test = f'WESADECG_{SUBJECT_USED_FOR_TESTING}.json'
    with open(os.path.join(DIR_DATA_TEST, name_test), 'r') as f:
        ds_test = ds_wesad(json.load(f))

    num_workers = 0
    batch_size = 512 # Maximizing GPU utilization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    
    net = ClassifierECG(1).to(device)
    lr = 0.001 
    
    dataloader_t = torch.utils.data.DataLoader(ds_training, batch_size=batch_size, shuffle=True, num_workers=num_workers, worker_init_fn=seed_worker, generator=g, drop_last=True, pin_memory=True)
    dataloader_test = torch.utils.data.DataLoader(ds_test, batch_size=batch_size, shuffle=True, num_workers=num_workers, worker_init_fn=seed_worker, generator=g, drop_last=True, pin_memory=True)
    
    print("Starting Training with optimized GPU settings...")
    Lossv, argmin = training(net, dataloader_t, dataloader_test, 30, 0, 0, lr)
    
    # Visualizing Results
    plt.plot(Lossv); plt.title("Validation Loss"); plt.xlabel("Epoch")
    plt.savefig(os.path.join(DIR_NET_SAVING, "learning_curve.png"), dpi=300)
    plt.close()  

    # Test with the best model
    best_model_path = os.path.join(DIR_NET_SAVING, f"net_0_0_epoch_{argmin}.pth")
    net.load_state_dict(torch.load(best_model_path))
    net.eval()
    
    confusionlabel=np.zeros((2,4)); confusion=np.zeros((2,2)); length_ds=0
    with torch.no_grad():
        for datal in dataloader_test:
              cl, c = conf_mat(net, datal, 0.5)
              confusion += c; confusionlabel += cl
              length_ds += datal[0].size(0)

    # Metrics
    TP=confusion[0,0]; TN=confusion[1,1]; FN=confusion[1,0]; FP=confusion[0,1]
    acc=(TP+TN)/(TP+FP+FN+TN); precision=TP/(TP+FP) if TP+FP>0 else 0
    recall=TP/(TP+FN) if TP+FN>0 else 0; f1=(2*recall*precision)/(recall+precision) if recall+precision>0 else 0
    
    print(f"\nFinal Metrics on {SUBJECT_USED_FOR_TESTING}:")
    print(f"Accuracy: {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

    # Plot Confusion Matrix
    sns.set(rc={"figure.figsize":(18, 6)})
    fig, axs = plt.subplots(ncols=2)
    sns.heatmap(100*confusion/length_ds, annot=True, fmt='.2f', ax=axs[0], xticklabels=['Stress','No Stress'], yticklabels=['Stress','No Stress'])
    axs[0].set_title('Confusion Matrix (Stress/No-Stress %)')
    sns.heatmap(100*confusionlabel/length_ds, annot=True, fmt='.2f', ax=axs[1], xticklabels=['Neutr','Stress','Amu','Med'], yticklabels=['Stress','No Stress'])
    axs[1].set_title('Detailed States Confusion %')
    plt.savefig(os.path.join(DIR_NET_SAVING, "final_test_results.png"), dpi=300)
    plt.close()
    print(f"Results saved in: {DIR_NET_SAVING}")