import nibabel as nib
import pickle
import os 
import glob
import zipfile
import numpy as np
from sklearn.preprocessing import StandardScaler


rangeOfNorm = 2
factorMul = 3
factorAdd = 2


def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        ret_di = u.load()
        #print(p)
        #ret_di = pickle.load(f)
    return ret_di

def saveasnii(brain_mask,nii_save_path,nii_data):
    img = nib.load(brain_mask)
    print(img.shape)
    nii_img = nib.Nifti1Image(nii_data, img.affine, img.header)
    nib.save(nii_img, nii_save_path)

def normlize(roi,models,weights):
    min = weights[os.path.split(models[0])[-1].split(".")[0]][roi]
    max = weights[os.path.split(models[0])[-1].split(".")[0]][roi]
    for model in models:
        modelName = os.path.split(model)[-1].split(".")[0]
        x = weights[modelName][roi]
        if(x>max):
            max=x
        if(x<min):
            min=x
    for model in models:
        modelName = os.path.split(model)[-1].split(".")[0]
        x = weights[modelName][roi]
        weights[modelName][roi] = rangeOfNorm*((x-min)/(max-min))

    return weights

def Standardization(roi,models,weights):
    x=[]
    for model in models:
        modelName = os.path.split(model)[-1].split(".")[0]
        x.append(weights[modelName][roi])
    x = np.asarray(x)
    x = scale(x)
    i=0
    for model in models:
        modelName = os.path.split(model)[-1].split(".")[0]
        weights[modelName][roi] = rangeOfNorm*x[i]
        i+=1
    return weights



def scale(x):
    mean = np.mean(x)
    std = np.std(x)
    for i in range(0,len(x)):
        x[i]=(x[i]-mean)/std
    return x

def fun(x):
    
    return (factorMul / (1+np.exp(-x))) + factorAdd

def calcFun(models,ROI,weights):
    
    for roi in ROI:
        sum = 0
        for model in models:
            modelName = os.path.split(model)[-1].split(".")[0]
            weights[modelName][roi] = fun(weights[modelName][roi])
            sum+=weights[modelName][roi]
        for model in models:
            modelName = os.path.split(model)[-1].split(".")[0]
            weights[modelName][roi]/=sum
   

    return weights
def allWeights():
    modelsPath = './models/'
    models = glob.glob(modelsPath + '/*.txt')
    weights = {}
    ROI = ['LOC', 'FFA', 'STS', 'EBA', 'PPA', 'V1', 'V2', 'V3', 'V4']
    for model in models:
        f = open(model, "r")
        ScorePerROI = f.read().split('\n')
        modelName = os.path.split(model)[-1].split(".")[0]
        weights[modelName]={}
        for i in range(0,9):
           s = ScorePerROI[i]
           x = s.split(":")
           weights[modelName][x[0]]=float(x[1])
    for roi in ROI:
      weights=Standardization(roi,models,weights)
    weights = calcFun(models,ROI,weights)
    return weights

def main():
  ROI = ['LOC', 'FFA', 'STS', 'EBA', 'PPA', 'V1', 'V2', 'V3', 'V4']
  SUB = ['sub01', 'sub02', 'sub03', 'sub04', 'sub05', 'sub06', 'sub07', 'sub08', 'sub09', 'sub10']
  weights = allWeights()
  #print(weights)
  modelsPath = './models/'
  models = glob.glob(modelsPath + '/*.pkl')
  results = load_dict(models[0])
#   print(load_dict(models[0])['V1']['sub01'])
  modelName = os.path.split(models[0])[-1].split(".")[0]
  for roi in ROI:
      for sub in SUB:
          results[roi][sub]*=weights[modelName][roi]
  models.pop(0)
  for modelDir in models:
      model = load_dict(modelDir)
      modelName = os.path.split(modelDir)[-1].split(".")[0]
      for roi in ROI:
          for sub in SUB:
              results[roi][sub]+=(model[roi][sub]*weights[modelName][roi])
#   print(load_dict(models[0])['V1']['sub01'])
#   print(results['V1']['sub01'])

  track = 'mini_track'
  save_dict(results,track+".pkl")
  zipped_results = zipfile.ZipFile(track+".zip", 'w')
  zipped_results.write(track+".pkl")
  zipped_results.close()

if __name__ == "__main__":
    main()
