import nibabel as nib
import pickle
import os 
import glob
import zipfile

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
  modelsPath = './models/'
  models = glob.glob(modelsPath + '/*.pkl')
  results = load_dict(models[0])
  models.pop(0)
  for modelDir in models:
      model = load_dict(modelDir)
      for roi in ROI:
          for sub in SUB:
              results[roi][sub]+=model[roi][sub]
  for roi in ROI:
      for sub in SUB:
          results[roi][sub]/=(len(models)+1)
  track = 'mini_track'
  save_dict(results,track+".pkl")
  zipped_results = zipfile.ZipFile(track+".zip", 'w')
  zipped_results.write(track+".pkl")
  zipped_results.close()

if __name__ == "__main__":
    main()