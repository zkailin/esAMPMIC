import numpy as np
import pandas as pd
import collections
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras as keras
import sklearn as sk
from sklearn import metrics
from scipy import stats
import random
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Dense, MultiHeadAttention, Dropout, Flatten, GlobalAveragePooling1D, LSTM, Conv1D
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

def dictionary_word2vec(filename):
    di_word2vec = {}
    fasta_file = filename
    seq_list = []
    for (index,seq_record) in enumerate(SeqIO.parse(fasta_file, "fasta")):
        seq_list.append(str(seq_record.seq))
    arr = np.array(seq_list)
    # print(AAs.map(lambda x :list (x)))
    w2v_model = Word2Vec(arr, vector_size = 20)
    for idx , key in enumerate(w2v_model.wv.key_to_index):
        di_word2vec[key] = list(w2v_model.wv[key])
    # di_word2vec = list(w2v_model.wv.index_to_key)
    return di_word2vec

def dictionary_substitution_matrix_features(filename):
    AAs = ["A","R","N","D","C","Q","E","G","H","I","L","K","M","F","P","S","T","W","Y","V"]
    di_sub_mat_feat = {}
    for line in open(filename):
        if (line[0] in AAs):
            feats = line.split()[1:21] #keep only the first 20 corresponding the "common 20 AAs"
            feats = list(map(np.float32, feats))
            # print(feats)
            di_sub_mat_feat[line[0]] = feats
    return di_sub_mat_feat

def dictionary_one_hot():
    AAs = ["A","R","N","D","C","Q","E","G","H","I","L","K","M","F","P","S","T","W","Y","V"]
    di_one_hot = {}
    for (i,aa) in enumerate(AAs):
        di_one_hot[aa] = np.zeros((len(AAs)), dtype = np.float32)
        di_one_hot[aa][i] = 1.0
    return di_one_hot

def AAindex():
    filename = "features/AAindex.txt"
    AAs = ["A","R","N","D","C","Q","E","G","H","I","L","K","M","F","P","S","T","W","Y","V"]
    with open(filename) as f:
        records = f.readlines()[1:]
    AAindex = []
    AA_encoding = {}
    for i in records:
        AAindex.append(i.rstrip().split()[1:] if i.rstrip() != '' else None)
    for i in range(20):
        AA_encoding[AAs[i]] = [item[i] for item in AAindex]
    return AA_encoding

zscale = {
'A': [0.24,  -2.32,  0.60, -0.14,  1.30], # A
'C': [0.84,  -1.67,  3.71,  0.18, -2.65], # C
'D': [3.98,   0.93,  1.93, -2.46,  0.75], # D
'E': [3.11,   0.26, -0.11, -0.34, -0.25], # E
'F': [-4.22,  1.94,  1.06,  0.54, -0.62], # F
'G': [2.05,  -4.06,  0.36, -0.82, -0.38], # G
'H': [2.47,   1.95,  0.26,  3.90,  0.09], # H
'I': [-3.89, -1.73, -1.71, -0.84,  0.26], # I
'K': [2.29,   0.89, -2.49,  1.49,  0.31], # K
'L': [-4.28, -1.30, -1.49, -0.72,  0.84], # L
'M': [-2.85, -0.22,  0.47,  1.94, -0.98], # M
'N': [3.05,   1.62,  1.04, -1.15,  1.61], # N
'P': [-1.66,  0.27,  1.84,  0.70,  2.00], # P
'Q': [1.75,   0.50, -1.44, -1.34,  0.66], # Q
'R': [3.52,   2.50, -3.50,  1.99, -0.17], # R
'S': [2.39,  -1.07,  1.15, -1.39,  0.67], # S
'T': [0.75,  -2.18, -1.12, -1.46, -0.40], # T
'V': [-2.59, -2.64, -1.54, -0.85, -0.02], # V
'W': [-4.36,  3.94,  0.59,  3.44, -1.59], # W
'Y': [-2.54,  2.44,  0.43,  0.04, -1.47], # Y
}
def seq_to_embedding(seq_list,output_vector,AA_len):
    output_list = list()
    for seq in seq_list:
        output_array = np.zeros((AA_len,len(output_vector['A'])))
        for (index , aa) in enumerate(seq):
            output_array[index,:] = output_vector[aa]
        output_list.append(output_array)
    return np.array(output_list)

def evaluate(y_pred, y_test):
    MSE =  metrics.mean_squared_error(y_test,y_pred)
    MAE = metrics.mean_absolute_error(y_test,y_pred)
    R2 = metrics.r2_score(y_test,y_pred)
    PCC = stats.pearsonr(y_test,y_pred)
    print('Model Performance')
    print('MSE: {:0.3f}.'.format(MSE))
    print('MAE = {:0.3f}.'.format(MAE))
    print('R2 = {:0.3f}.'.format(R2))
    print('PCC = {:0.3f}.'.format(PCC[0]))
    
    return PCC[0]

def train_test_val_DL(train,test,val):
    train_x = train['SEQUENCE']
    train_y = train['NEW-CONCENTRATION']
    test_x = test['SEQUENCE']
    test_y = test['NEW-CONCENTRATION']
    val_x = val['SEQUENCE']
    val_y = val['NEW-CONCENTRATION']
    return train_x,test_x,val_x,train_y,test_y,val_y

def combine_features(X1,X2):
    combine_list = list()
    for i in range(len(X1)):
        combine_list.append(np.hstack((X1[i],X2[i])))
    return np.array(combine_list)

def save_model_history(model,history,model_name):
    model.save('{}.h5'.format(model_name))
    hist_df = pd.DataFrame(history.history) 
    hist_json_file = '{}_history.json.'.format(model_name) 
    with open(hist_json_file, mode='w') as f:
        hist_df.to_json(f)    

def Genomesequence_concat(Feature_array,Genome_array):
    # add an extra dimension to the array
    arr = np.expand_dims(Genome_array, axis=1)
    # replicate the data across the new dimension to match the desired shape 
    arr = np.tile(arr, (1, 1, 1))
    # define the desired shape after padding
    pad_shape = (Feature_array.shape[0], 1, Feature_array.shape[2])
    # pad the array with zeros along the last dimension
    arr_padded = np.pad(arr, [(0, 0), (0, 0), (0, pad_shape[2] - arr.shape[2])], mode='constant')
    # concat two array to one 
    concatenated_array = np.concatenate((Feature_array, arr_padded), axis=1)
    return concatenated_array

def r_squared(y_true, y_pred):
    ss_res = K.sum(K.square(y_true - y_pred))
    ss_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - ss_res/(ss_tot + K.epsilon())



One_hot_encoding = dictionary_one_hot()
BLOSUM62 = dictionary_substitution_matrix_features("features/BLOSUM62.txt")
AAindex = AAindex()
output_vector = {}
AAs = ["A","R","N","D","C","Q","E","G","H","I","L","K","M","F","P","S","T","W","Y","V"]
for aa in AAs:
    output_vector[aa] = np.concatenate((AAindex[aa], BLOSUM62[aa], One_hot_encoding[aa], zscale[aa]))


#My paper's dataset
#EC - 40
EC_train = pd.read_csv('data/EC_X_train_40.csv')
EC_test = pd.read_csv('data/EC_X_test_40.csv')
EC_val = pd.read_csv('data/EC_X_val_40.csv')

#SA - 40
SA_train = pd.read_csv('data/SA_X_train_40.csv')
SA_test = pd.read_csv('data/SA_X_test_40.csv')
SA_val = pd.read_csv('data/SA_X_val_40.csv')

#PA - 40
PA_train = pd.read_csv('data/PA_X_train_40.csv')
PA_test = pd.read_csv('data/PA_X_test_40.csv')
PA_val = pd.read_csv('data/PA_X_val_40.csv')

Three_concat_train = pd.concat([SA_train, EC_train, PA_train], axis=0)

SA_X_train, SA_X_test, SA_X_val, SA_y_train, SA_y_test, SA_y_val = train_test_val_DL(SA_train,SA_test,SA_val)
EC_X_train, EC_X_test, EC_X_val, EC_y_train, EC_y_test, EC_y_val = train_test_val_DL(EC_train,EC_test,EC_val)
PA_X_train, PA_X_test, PA_X_val, PA_y_train, PA_y_test, PA_y_val = train_test_val_DL(PA_train,PA_test,PA_val)


SA_X_train = seq_to_embedding(SA_X_train,output_vector,40)
SA_X_test = seq_to_embedding(SA_X_test,output_vector,40)
SA_X_val = seq_to_embedding(SA_X_val,output_vector,40)

EC_X_train = seq_to_embedding(EC_X_train,output_vector,40)
EC_X_test = seq_to_embedding(EC_X_test,output_vector,40)
EC_X_val = seq_to_embedding(EC_X_val,output_vector,40)

PA_X_train = seq_to_embedding(PA_X_train,output_vector,40)
PA_X_test = seq_to_embedding(PA_X_test,output_vector,40)
PA_X_val = seq_to_embedding(PA_X_val,output_vector,40)

#My paper's dataset
#EC
T5XL_EC_X_train = np.load('data/T5XL_Embeddings_max_40/EC_X_TRAIN.npy')
T5XL_EC_X_test = np.load('data/T5XL_Embeddings_max_40/EC_X_TEST.npy')
T5XL_EC_X_val = np.load('data/T5XL_Embeddings_max_40/EC_X_VAL.npy')

#SA
T5XL_SA_X_train = np.load('data/T5XL_Embeddings_max_40/SA_X_TRAIN.npy')
T5XL_SA_X_test = np.load('data/T5XL_Embeddings_max_40/SA_X_TEST.npy')
T5XL_SA_X_val = np.load('data/T5XL_Embeddings_max_40/SA_X_VAL.npy')

#PA
T5XL_PA_X_train = np.load('data/T5XL_Embeddings_max_40/PA_X_TRAIN.npy')
T5XL_PA_X_test = np.load('data/T5XL_Embeddings_max_40/PA_X_TEST.npy')
T5XL_PA_X_val = np.load('data/T5XL_Embeddings_max_40/PA_X_VAL.npy')

#My paper's dataset
#EC
T5XL_EC_X_train_GS = Genomesequence_concat(T5XL_EC_X_train,EC_train.iloc[:,250:-12])
T5XL_EC_X_test_GS = Genomesequence_concat(T5XL_EC_X_test,EC_test.iloc[:,250:-12])
T5XL_EC_X_val_GS = Genomesequence_concat(T5XL_EC_X_val,EC_val.iloc[:,250:-12])

#SA
T5XL_SA_X_train_GS = Genomesequence_concat(T5XL_SA_X_train,SA_train.iloc[:,250:-12])
T5XL_SA_X_test_GS = Genomesequence_concat(T5XL_SA_X_test,SA_test.iloc[:,250:-12])
T5XL_SA_X_val_GS = Genomesequence_concat(T5XL_SA_X_val,SA_val.iloc[:,250:-12])

#PA
T5XL_PA_X_train_GS = Genomesequence_concat(T5XL_PA_X_train,PA_train.iloc[:,250:-12])
T5XL_PA_X_test_GS = Genomesequence_concat(T5XL_PA_X_test,PA_test.iloc[:,250:-12])
T5XL_PA_X_val_GS = Genomesequence_concat(T5XL_PA_X_val,PA_val.iloc[:,250:-12])


# With Genome sequence
T5XL_three_train_concat = np.concatenate([T5XL_EC_X_train_GS,T5XL_SA_X_train_GS,T5XL_PA_X_train_GS])
T5XL_three_test_concat = np.concatenate([T5XL_EC_X_test_GS,T5XL_SA_X_test_GS,T5XL_PA_X_test_GS])
T5XL_three_val_concat = np.concatenate([T5XL_EC_X_val_GS,T5XL_SA_X_val_GS,T5XL_PA_X_val_GS])

# Concat
Three_train = pd.concat([EC_train,SA_train,PA_train])
Three_test = pd.concat([EC_test,SA_test,PA_test])
Three_val = pd.concat([EC_val,SA_val,PA_val])

# My paper's models
# load model (only with sequence-based features)
EC_bilstm_40 = load_model('model_max_40/EC_bilstm_40.h5')
SA_bilstm_40 = load_model('model_max_40/SA_bilstm_40.h5')
PA_bilstm_40 = load_model('model_max_40/PA_bilstm_40.h5')

EC_CNN_40 = load_model('model_max_40/EC_CNN_40.h5')
SA_CNN_40 = load_model('model_max_40/SA_CNN_40.h5')
PA_CNN_40 = load_model('model_max_40/PA_CNN_40.h5')

T5_EC_bilstm_40 = load_model('model_max_40/T5_EC_bilstm_40.h5')
T5_SA_bilstm_40 = load_model('model_max_40/T5_SA_bilstm_40.h5')
T5_PA_bilstm_40 = load_model('model_max_40/T5_PA_bilstm_40.h5')

T5_EC_CNN_40 = load_model('model_max_40/T5_EC_CNN_40.h5')
T5_SA_CNN_40 = load_model('model_max_40/T5_SA_CNN_40.h5')
T5_PA_CNN_40 = load_model('model_max_40/T5_PA_CNN_40.h5')

EC_Transformer_40 = load_model('model_max_40/EC_Transformer_40.h5')
SA_Transformer_40 = load_model('model_max_40/SA_Transformer_40.h5')
PA_Transformer_40 = load_model('model_max_40/PA_Transformer_40.h5')

EC_MB_40 = load_model('model_max_40/EC_Bi_CNN_40.h5')
SA_MB_40 = load_model('model_max_40/SA_Bi_CNN_40.h5')
PA_MB_40 = load_model('model_max_40/PA_Bi_CNN_40.h5')

# load model (sequenced-based + genome-based features)
T5_Three_Bi_model = load_model('model_max_40/T5_Three_Bi_40.h5')
T5_Three_CNN_model = load_model('model_max_40/T5_Three_CNN_40.h5')
T5_Three_Tf_model = load_model('model_max_40/T5_Three_Tf_40.h5')
T5_Three_MB_model = load_model('model_max_40/T5_Three_MB_40.h5', custom_objects={'r_squared': r_squared})


# Three Concat
SA_pred = pd.DataFrame(columns ={'CNN','BILSTM','MB'})
EC_pred = pd.DataFrame(columns ={'CNN','BILSTM','MB'})
PA_pred = pd.DataFrame(columns ={'CNN','BILSTM','MB'})

X_test = {'SA_X_test':T5XL_SA_X_test_GS,
          'EC_X_test':T5XL_EC_X_test_GS,
          'PA_X_test':T5XL_PA_X_test_GS
}
y_test = {'SA_y_test':SA_test['NEW-CONCENTRATION'],
          'EC_y_test':EC_test['NEW-CONCENTRATION'],
          'PA_y_test':PA_test['NEW-CONCENTRATION']
}
pred = {'SA_pred':SA_pred,
        'EC_pred':EC_pred,
        'PA_pred':PA_pred}

print('T5_Three_CNN')
for X,y,p in zip(X_test,y_test,pred):
    print(X,y)
    CNN_pred = T5_Three_CNN_model.predict([X_test.get(X)[:,:40,:],X_test.get(X)[:,40,:][:,:84]])
    CNN_PCC = evaluate(CNN_pred.reshape(-1),y_test.get(y))
    pred.get(p)['CNN'] = CNN_pred.reshape(-1).tolist()
print('---------------------------------------')
print('T5_Three_Bi')
for X,y,p in zip(X_test,y_test,pred):
    print(X,y)
    Bi_pred = T5_Three_Bi_model.predict([X_test.get(X)[:,:40,:],X_test.get(X)[:,40,:][:,:84]])
    Bi_PCC = evaluate(Bi_pred.reshape(-1),y_test.get(y))
    pred.get(p)['BILSTM'] = Bi_pred.reshape(-1).tolist()
print('---------------------------------------')
print('T5_Three_Transformer')
for X,y in zip(X_test,y_test):
    print(X,y)
    Tf_pred = T5_Three_Tf_model.predict([X_test.get(X)[:,:40,:],X_test.get(X)[:,40,:][:,:84]])
    Tf_PCC = evaluate(Tf_pred.reshape(-1),y_test.get(y))
print('---------------------------------------')    
print('T5_Three_MB')
for X,y,p in zip(X_test,y_test,pred):
    print(X,y)
    MB_40_pred = T5_Three_MB_model.predict([X_test.get(X)[:,:40,:],X_test.get(X)[:,:40,:],X_test.get(X)[:,40,:][:,:84]])
    MB_40_PCC = evaluate(MB_40_pred.reshape(-1),y_test.get(y))
    pred.get(p)['MB'] = MB_40_pred.reshape(-1).tolist()


SA_pred['MIC_Final'] = SA_pred['CNN']*0.3+SA_pred['BILSTM']*0.4+SA_pred['MB']*0.3
EC_pred['MIC_Final'] = EC_pred['CNN']*0.3+EC_pred['BILSTM']*0.4+EC_pred['MB']*0.3
PA_pred['MIC_Final'] = PA_pred['CNN']*0.3+PA_pred['BILSTM']*0.4+PA_pred['MB']*0.3

