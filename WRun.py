import nibabel as nib
import pickle
import os 
import glob
import zipfile

def calcW(sumAll,model):
  f = open(model, "r")
  ans ={}
  ScorePerROI = f.read().split('\n')
  for i in range(0,9):
      s = ScorePerROI[i]
      x = s.split(":")
      ans[x[0]]=float(x[1])/sumAll[x[0]]
  return ans

# print(ans)
def allWeights():
  modelsPath = './models/'
  models = glob.glob(modelsPath + '/*.txt')
  sumAll = {'V1': 0, 'V2': 0, 'V3': 0, 'V4': 0, 'LOC': 0, 
          'EBA': 0, 'FFA': 0, 'STS': 0, 'PPA': 0}
  for model in models:
      f = open(model, "r")
      ScorePerROI = f.read().split('\n')
      for i in range(0,9):
         s = ScorePerROI[i]
         x = s.split(":")
         sumAll[x[0]]+=float(x[1])
  weights = {}
  for model in models:
    modelName = os.path.split(model)[-1].split(".")[0]
    weights[modelName]=calcW(sumAll,model)
  return weights

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

def main():
  ROI = ['LOC', 'FFA', 'STS', 'EBA', 'PPA', 'V1', 'V2', 'V3', 'V4']
  SUB = ['sub01', 'sub02', 'sub03', 'sub04', 'sub05', 'sub06', 'sub07', 'sub08', 'sub09', 'sub10']
  weights = allWeights()
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
