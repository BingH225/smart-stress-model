import os
import numpy as np
import pandas as pd
import scipy
from IPython.display import clear_output
import scipy.stats as stats
import heartpy
import json
"""Please extract the files in WESAD.zip in a file that will contains S2,S3,...Sk folders of the subject

If you want to use a subject for testing (I have used the S17 for testing) I suggest you to move your S17 file to a Testing folder.
That is to say for eg : /content/data/WESAD contains S2,S3,...Sk folders but does not contain S17 folder
                        /content/data/Testing contains S17 folder
"""

DIR_WESAD="D:/NUS/BMI5101/WESAD/" # Please complete the path to your file that contains S2,S3,...Sk folders. eg : DIR_WESAD=/content/data/WESAD/
DIR_SAVING_DATA= "D:/NUS/BMI5101/DNN/Data_Processed/" #Please complete the path where you want to save your data once treated
# Fonctions extraction ECG
def peak_pos(x : np.array,threshhold: float):
  """ Detect the ECG peaks on the signal x, peaks detected must go above the float value threshold
  
  The signal is smoothed before the peaks detection using the mean on 5 values with a convolution 
  Returns the index of the peaks 
  """
  assert len(x.shape)==1
  x=(x-np.mean(x))/np.std(x)
  smoothed=[]
  conv=np.array([0.2,0.2,0.2,0.2,0.2])
  smoothed=np.convolve(x,conv, mode='same')
  baseline=float(np.mean(smoothed))
  peakindex=-1
  peakindices=[]
  peakvalue=-100

  for index in range(0,len(smoothed)):
    value=smoothed[index]

    if value>baseline:

      if peakvalue ==-100 or value>peakvalue :
        peakindex=index
        peakvalue=value

    if value<baseline and peakindex!=-1 and peakvalue>threshhold:
      peakindices.append(peakindex)
      peakindex=-1
      peakvalue=-100

  if peakindex!=-1 and peakvalue>threshhold:
    peakindices.append(peakindex)

  return np.array(peakindices)


def TINN(x:np.array):
  """ Compute all the triangular interpolation to calculate the TINN scores. It also computes HRV index from an array x which contains 
      all the interbeats times for a given ECG signal.

      The axis is divided in 2 parts respectively on the right and left of the abscissa of the maximum value of the gaussian distribution
      The TINN score calculation is defined in the WESAD Dataset paper, to calculate it we needthe closest triangular interpolation 
      of the gaussian distribution of the interbeats times. The triangular interpolation is defined by 2 lines that meet at the maximum value
      of the gaussian distribution and cross the x-axis in N on the first half of the x-axis and M on the second half of the x-axis. 
      Thus inside ]N;M[ the interpolation function != 0
      Outside of ]N;M[ the interpolation function equals 0.
  """

  kernel = stats.gaussian_kde(x) #Create an approximated kernel for gaussian distribution from the x array (interbeats times)
  absi=np.linspace(np.min(x),np.max(x),len(x)) # Compute the x-axis of the interbeats distribution (from minimum interbeat time to maximum interbeat time)
  val=kernel.evaluate(absi) # Fit the gaussian distribution to the created x-axis
  ecart=absi[1]-absi[0] # Space between 2 values on the axis
  maxind=np.argmax(val) # Select the index for which the gaussian distribution (val array) is maximum 
  max_pos=absi[maxind]  # Interbeat time (abscissa) for which the gaussian distribution is maximum
  maxvalue=np.amax(val) # Max of the gaussian distribution
  N_abs=absi[0:maxind+1] # First half of the x-axis
  M_abs=absi[maxind:] # Second half of the x-axis
  HRVindex=len(x)/maxvalue
  err_N=[]
  err_M=[]

  for i in range(0,len(N_abs)-1):
    N=N_abs[i]
    slope=(maxvalue)/(max_pos-N)
    D=val[0:maxind+1]
    q=np.clip(slope*ecart*np.arange(-i,-i+maxind+1),0,None) #Triangular interpolation on the First half of the x-axis
    diff=D-q 
    err=np.multiply(diff,diff)
    err1=np.delete(err,-1)
    err2=np.delete(err, 0)
    errint=(err1+err2)/2
    errtot=np.linalg.norm(errint) # Error area between the triangular interpolation and the gaussian distribution on the first half of the x-axis
    err_N.append((errtot,N,N_abs,q))
  
  for i in range(1,len(M_abs)):
    M=M_abs[i]
    slope=(maxvalue)/(max_pos-M)
    D=val[maxind:]
    q=np.clip(slope*ecart*np.arange(-i,len(D)-i),0,None) #Triangular interpolation on the second half of the x-axis
    diff=D-q
    err=np.multiply(diff,diff)
    err1=np.delete(err,-1)
    err2=np.delete(err, 0)
    errint=(err1+err2)/2
    errtot=np.linalg.norm(errint) # Error area between the triangular interpolation and the gaussian distribution on the second half of the x-axis
    err_M.append((errtot,M,M_abs,q))

  return (err_N,err_M,absi,val,HRVindex)

def best_TINN(x:np.array):
  """Select the best N and M that give the best triangular interpolation function approximation of the gaussian distrbution and return
    N; M; the TINN score = M-N ; and the HRV index
  
  """
  err_N,err_M,_,_,HRVindex=TINN(x)
  N=np.argmin(np.array(err_N,dtype=object)[:,0])
  M=np.argmin(np.array(err_M,dtype=object)[:,0])
  absN=err_N[N][1]
  absM=err_M[M][1]
  return float(absN),float(absM),float(absM-absN),HRVindex

def num_compare_NN50(x,i):
  """Count the number of HRV intervals differing more than 50 ms for a given HRV interval x[i]
  
  """
  ref=x[i]
  k=0
  diff=np.absolute(x-ref)
  k+=np.sum(np.where(diff>0.05,1,0))
  return k 

def compare_NN50(x):
  """ Returns the number and percentage of HRV intervals differing more than 50ms for all intervals
  
  """
  k=0
  for i in range(0,len(x)):
    k+=num_compare_NN50(x,i)
  if k==0:
    k=1
  return k,(k/(len(x)*len(x)))

def get_freq_features_ecg(x):
  """ Returns frequential features of the Heart Rate Variability signal (interbeats times) by computing FFT, to compute the Fouriers 
  Frequencies the mean of the Heart Rate variability is used as sampling period  
  """
  mean=np.mean(x)
  yf=np.array(scipy.fft.fft(x-mean))
  xf=scipy.fft.fftfreq(len(x),mean)[0:len(x)//2]
  psd=(2/len(yf))*np.abs(yf)[0:len(x)//2]
  fmean=np.mean(xf)
  fstd=np.std(xf)
  sumpsd=np.sum(psd)
  return fmean,fstd,sumpsd

def get_data_ecg(x):
  """ Collect the features of a given ECG signal x, using HeartPy package to compute the peak list (not the previous developed peak 
  detection function).  
  """
  working,mes=heartpy.process(x,700)
  peak=working["peaklist"]
  periods=np.array([(peak[i+1]-peak[i])/700 for i in range(0,len(peak)-1)])
  frequency=1/periods
  meanfreq = np.mean(frequency)
  stdfreq = np.std(frequency)
  HRV=np.array([(peak[i]-peak[i-1])/700 for i in range(1,len(peak))])
  _,_,T,HRVindex=best_TINN(HRV)
  num50,p50=compare_NN50(HRV)
  meanHRV=np.mean(HRV)
  stdHRV=np.std(HRV)
  rmsHRV=np.sqrt(np.mean(HRV**2))
  fmean,fstd,sumpsd=get_freq_features_ecg(HRV)
  return np.array([meanfreq,stdfreq,T,HRVindex,num50,p50,meanHRV,stdHRV,rmsHRV,fmean,fstd,sumpsd])

def slice_per_label(labelseq,flabel,time_window,step):
  """Return a list of index i of the ECG signal that in [i;i+pts_windows] the label (= emotionnal state of the user) is constant.
  The window is defined by the user by time_window (s) and flabel (freq of sampling for the label, eg 700Hz) (Hz).
  """
  pts_window=time_window*flabel
  taken_indices=[]
  conv=np.array([1 for i in range(0,pts_window)])
  for i in range(0,len(labelseq)-pts_window,flabel*step):   #Sliding 5s window, step 1s
    extr=labelseq[i:i+pts_window]
    res=np.sum(np.multiply(extr,conv))
    l=labelseq[i]
    if l in [1,2,3,4]:
      condition=l*pts_window==res
      if condition==True:
        taken_indices.append((i,l))
  return taken_indices

def get_ecg_f_from_file(path):
  """Extract all the ECG features for each window of the ECG signal of the given file. Once it is extracted the features are
  normalized with the features computed on an ECG signal when the subject is in a neutral state. If the extraction fails for a 
  given window the extracted features are discarded.
  At the end the number of discarded extract is printed
  """
  id=[]
  labels=[]
  features=[]
  taken_indices=[]
  fecg=700
  i=0
  discard=0
  discardbis=0
  pts_window=20*700
  df = pd.read_pickle(path)
  label=df['label']
  print("openned")
  indice_neutral=np.where(label==1)[0]        #Find where is the neutral phase for baseline
  taken_indices=slice_per_label(label,700,20,1)   #get all indices of the start of a slice of the sequence with cst label
  print("indiced")
  ECG_neutral = np.array(df["signal"]["chest"]["ECG"][indice_neutral[0]:indice_neutral[-1]*700][:,0]) #Baseline
  features_neutral=get_data_ecg(ECG_neutral)        #Baseline extraction
  for x in taken_indices:
    i+=1
    if i%100==0:
      clear_output(wait=True)
      print(path)
      print(i/len(taken_indices))
    indice = x[0]
    try:
      ECG=np.array(df['signal']['chest']['ECG'][(indice*fecg)//700:(pts_window+indice)*fecg//700][:,0])
      result=np.divide(get_data_ecg(ECG),features_neutral)
      if not (np.isinf(result).any() or np.isnan(result).any()):
        features.append(result.tolist())
        labels.append(int(x[1]))
      else : 
        discard+=1
    except KeyboardInterrupt:
      print(1/0)
    except : 
      discardbis+=1
  print("reject because infinit : " +str(discard))
  print("reject because error in algorithm : " +str(discardbis))
  return features,id,labels,discard,discardbis,ECG_neutral
# Extract ECG
""" The extracted features and label (id is not used anymore) are stored in a dictionnary for each subject and each dictionnary is written
in a json file named with the number of the subject
"""

dir_wesad=" "
l= os.listdir(DIR_WESAD) 
del l[l.index('wesad_readme.pdf')]
try :
  del l[l.index('wesad_readme.pdf')]
except :
  pass
i=0
for name in l:
  i+=1
  data_w={}
  path =str(DIR_WESAD+name+"/"+name+".pkl")
  print(name)
  print(i/len(l))
  print(len(l))
  X,Y,Z,discard,discardbis,neutr=get_ecg_f_from_file(path)
  data_w["id"]=Y
  data_w["label"]=Z
  data_w["features"]=X
  with open(DIR_SAVING_DATA+"WESADECG_"+name+".json", 'w') as f:
    json.dump(data_w, f)