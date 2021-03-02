import pandas as pd 
from sklearn.model_selection import StratifiedKFold 

# split to folds by category/class_name
nfolds = 4

train_path = '/media/basic/ssd256/VinBigData_CXR/vinbigdata/train.csv'
trnsamples = pd.read_csv(train_path)
trnsamples = trnsamples.drop_duplicates().reset_index(drop=True)
x = trnsamples['image_id']
y = trnsamples['class_name'] 
trnsamples['fold'] = -1
skf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=42)

for fold, (trnid, tstid) in enumerate(skf.split(y, x)):
	trnsamples.loc[tstid, 'fold'] = fold 

# savepath = '/media/basic/ssd256/VinBigData_CXR/vinbigdata/train_folds_bypatid.csv'
# trnsamples.to_csv(savepath, index=False)

for cat in trnsamples['image_id'].unique():
	print(cat)
	cat_samples = trnsamples[trnsamples['image_id'] == cat].reset_index(drop=True)
	for fold in range(nfolds):
		print(fold, len(cat_samples[cat_samples.fold == fold]))