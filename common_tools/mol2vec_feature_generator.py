# This file is to generate mol2vec features from SMILES strings

# Import necessary libraries
import pandas as pd
import numpy as np
from rdkit import Chem 
#pip install rdkit-pypi if not present
from gensim.models import word2vec
from mol2vec.features import mol2alt_sentence, mol2sentence,MolSentence, DfVec, sentences2vec 
# pip install git+https://github.com/samoturk/mol2vec if not present
from gensim.models import word2vec

# Takes a dataframe with SMILES strings to return dataframe with mol2vec features
def main(dataset_path,trained_mol2vec_model_path):
    mdf = pd.read_csv(dataset_path).iloc[:,:2] # select the SMILES and target columns from the dataframe
    mdf.columns = ['smiles', 'target'] # rename the columns
    mdf = mdf.astype(object) # change data type to object (required to store molecule objects in the dataframe later)
    mdf['mol'] = mdf['smiles'].apply(lambda x: Chem.MolFromSmiles(x)) # generate molecule objects from the SMILES strings                
    model = word2vec.Word2Vec.load(trained_mol2vec_model_path) # load the trained mol2vec model
    mdf['sentence'] = None # create a new column to store the mol2vec sentences

    # generate mol2vec sentences from the molecule objects
    for i in range(mdf.shape[0]):
        try: # the folloing 'try except' code block is require to skip erroneous molecules that could not be processed by rdkit
            m = mdf['mol'][i]
            mdf.loc[i,'sentence'] = MolSentence(mol2alt_sentence(m, 1))
        except: # do the following if there's an exception (or error) while executing the previous lines
            mdf.loc[i,'sentence'] = None
            print('skipped: {}'.format(mdf['smiles'][i]))
    
    mdf.dropna(inplace=True) # drop rows with missing values
    # generate vector representations of molecules using 'molecular sentences' and pre-trained mol2vec model
    mdf['mol2vec'] = [DfVec(x) for x in sentences2vec(mdf['sentence'], model)]

    X = np.array([x.vec for x in mdf['mol2vec']]) # feature matrix
    y = mdf['target'].values.astype(np.float32) # target values
    
    return X

if __name__=='__main__':
    dataset_path = 'D:\Personal\Self_projects\Thermodynamic property estimation\Thermodynamic-property-estimation-using-Machine-Learning-models\Datasets\Group-10\hfus.csv'
    trained_mol2vec_model_path = 'D:\Personal\Self_projects\Thermodynamic property estimation\Thermodynamic-property-estimation-using-Machine-Learning-models\common_tools\model_300dim.pkl'
    X = main(dataset_path,trained_mol2vec_model_path)
    print(X.shape)

