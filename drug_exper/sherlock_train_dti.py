from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import pandas as pd
import numpy as np
from tqdm import tqdm

from DeepPurpose import utils # , dataset, CompoundPred
from DeepPurpose import DTI as models
from DeepPurpose.utils import *
from DeepPurpose.dataset import *
from rdkit import Chem
from rdkit import DataStructs
#from rdkit.ML.Cluster import Butina
#from rdkit.Chem import Draw
#from rdkit.Chem import rdFingerprintGenerator
#from rdkit.Chem.Draw import SimilarityMaps
import warnings
import numpy as np
import sys 
import pandas as pd 
warnings.filterwarnings("ignore")



seed = int(sys.argv[1])
np.random.seed(seed)

drug_encoding = 'CNN'
target_encoding = 'Transformer'


df = pd.read_csv('cleaned_drug_data.csv')

affinity_list = df['affinity'].tolist()
ligand_smiles_list = df['ligand'].tolist()
target_aac_list = df['target'].tolist()

n_total = len(affinity_list)
reind = np.random.permutation(n_total)

X_drugs_train = [ligand_smiles_list[reind[0:int(n_total*0.75+1)][i]] for i in range(int(n_total*0.75+1))]
X_targets_train = [target_aac_list[reind[0:int(n_total*0.75+1)][i]] for i in range(int(n_total*0.75+1))]
y_train = [affinity_list[reind[0:int(n_total*0.75+1)][i]] for i in range(int(n_total*0.75+1))]


X_drugs_other = [ligand_smiles_list[reind[int(1+n_total*0.75):n_total][i]] for i in range(n_total-int(n_total*0.75+1))]
X_targets_other = [target_aac_list[reind[int(1+n_total*0.75):n_total][i]] for i in range(n_total-int(n_total*0.75+1))]
y_other = [affinity_list[reind[int(1+n_total*0.75):n_total][i]] for i in range(n_total-int(n_total*0.75+1))]

ttrain, tval, ttest = utils.data_process(X_drugs_train, X_targets_train, y_train, 
                                  drug_encoding, target_encoding, 
                                  split_method='random', frac=[0.8,0.2,0.],
                                  random_seed = seed)

ddata, _, __ = utils.data_process(X_drugs_other, X_targets_other, y_other, 
                                  drug_encoding, target_encoding, 
                                    split_method='random', frac=[1., 0., 0.],
                                    random_seed = seed)


calib_test_perm = np.random.permutation(len(ddata))
calib_test_ratio = 0.7
dcalib = ddata.iloc[calib_test_perm[0:int(len(ddata)*calib_test_ratio+1)]].reset_index(drop=True)
dtest = ddata.iloc[calib_test_perm[int(1+len(ddata)*calib_test_ratio):len(ddata)]].reset_index(drop=True)


# get quantile cutoffs
testq2 = np.zeros(dtest.shape[0])
testq5 = np.zeros(dtest.shape[0])
testq7 = np.zeros(dtest.shape[0])
testq8 = np.zeros(dtest.shape[0])
testq9 = np.zeros(dtest.shape[0])

for i in range(dtest.shape[0]):
    tenc = dtest['Target Sequence'].iloc[i]
    tsub = ttrain['Target Sequence'] == tenc 
    if sum(tsub) == 0:
      allb = ttrain 
    else:
      allb = ttrain[tsub]

    q2, q5, q7, q8, q9 = np.quantile(allb['Label'], 0.2), np.quantile(allb['Label'], 0.5), np.quantile(allb['Label'], 0.7), np.quantile(allb['Label'], 0.8), np.quantile(allb['Label'], 0.9)
    testq2[i] = q2
    testq5[i] = q5
    testq7[i] = q7
    testq8[i] = q8
    testq9[i] = q9


calibq2 = np.zeros(dcalib.shape[0])
calibq5 = np.zeros(dcalib.shape[0])
calibq7 = np.zeros(dcalib.shape[0])
calibq8 = np.zeros(dcalib.shape[0])
calibq9 = np.zeros(dcalib.shape[0])


for i in range(dcalib.shape[0]):
    tenc = dcalib['Target Sequence'].iloc[i]
    tsub = ttrain['Target Sequence'] == tenc
    # print(tsub)
    if sum(tsub) == 0:
        allb = ttrain 
    else:
        allb = ttrain[tsub]

    # allb = ttrain[]
    # print(allb['Label'])
    # print(allb)
    q2, q5, q7, q8, q9 = np.quantile(allb['Label'], 0.2), np.quantile(allb['Label'], 0.5), np.quantile(allb['Label'], 0.7), np.quantile(allb['Label'], 0.8), np.quantile(allb['Label'], 0.9)
    calibq2[i] = q2
    calibq5[i] = q5
    calibq7[i] = q7 
    calibq8[i] = q8 
    calibq9[i] = q9


config = utils.generate_config(drug_encoding = drug_encoding, 
                          target_encoding = target_encoding, 
                          cls_hidden_dims = [1024,1024,512], 
                          train_epoch = 10, 
                          LR = 0.001, 
                          batch_size = 128,
                          hidden_dim_drug = 128,
                          mpnn_hidden_size = 128,
                          mpnn_depth = 3, 
                          cnn_target_filters = [32,64,96],
                          cnn_target_kernels = [4,8,12]
                          )


model = models.model_initialize(**config)
model.train(ttrain, tval)

calib_pred = model.predict(dcalib)
test_pred = model.predict(dtest)



# get scores

def get_scores(Y, predictions):
  return np.where(Y > 0, np.inf, -predictions)

calib_pred = np.array(calib_pred)
test_pred = np.array(test_pred)

n = len(calib_pred)
m = len(test_pred)
calib_quantiles = [calibq2, calibq5, calibq7, calibq8, calibq9]
test_quantiles = [testq2, testq5, testq7, testq8, testq9]



testS = get_scores(np.zeros(m), test_pred)
np.save(f'./scores_and_Ys/testS_j{seed}.npy', testS)

for quantile_indexer in range(5):
    adjusted_calibY = dcalib['Label'].to_numpy() - calib_quantiles[quantile_indexer]
    adjusted_testY = dtest['Label'].to_numpy() - test_quantiles[quantile_indexer]


    calibS = get_scores(adjusted_calibY, calib_pred)
    np.save(f'./scores_and_Ys/calibS_j{seed}_q{quantile_indexer}.npy', calibS)
    np.save(f'./scores_and_Ys/testY_j{seed}_q{quantile_indexer}.npy', adjusted_testY)







