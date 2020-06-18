import pandas as pd
import numpy as np
from sklearn import preprocessing, decomposition, metrics, model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from imblearn.over_sampling import SMOTE
from math import radians, cos, sin, asin, sqrt

def scale_data(data):
    scaler = preprocessing.MinMaxScaler()
    scaled = scaler.fit_transform(data)
    return pd.DataFrame(scaled)

def fill_missing_val(data, cols):
    #Fill numerical values missing values with means
    for col in cols:
        data[col] = data[col].fillna(round(data[col].mean()))
    
    return data

def get_haversine(lon1, lat1, lon2, lat2):
    
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, (lon1, lat1, lon2, lat2))

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r

def calculate_distance(data):
    data["distance"] = data.apply(
        lambda x: get_haversine(x["pick_lon"], x["pick_lat"], x["drop_lon"], x["drop_lat"]), axis=1)
    
    data = data.drop(columns=["pick_lon", "pick_lat", "drop_lon", "drop_lat"], axis=1)
    return data

def get_hour(row, col, method="hour"):
    val = str(row)
    #     if not "20" in val:
    #         print("Empty")
    #print(val)
    hour = int(val[-5:-3].replace(" ", ""))
    if method == "hour":
        return hour
    minutes = int(val[-2:].replace(" ", ""))
    minutes = 1 if minutes >= 30 else 0
    bin_val = minutes + 2 * hour
    return bin_val

def get_binned_DF(data, col):
    data[col] = data[col].apply(lambda x: round(float(x)))
    for i in pd.unique(data[col]):
        data[col + "_timebin_" + str(i)] = (data[col] == i)
    
    data = data.drop(columns=[col], axis=1)
    return data

def balance_using_SMOTE(X, Y, k_neighbors=102):
    smote = SMOTE(k_neighbors=k_neighbors)
    datasetX, datasetY = smote.fit_resample(X, Y)
    return datasetX, datasetY

def preprocess(data, is_train=True, use_synthetic_generation=True):
    #Get columns list and remove tripid col and non numerical cols
    cols = list(data.columns)
    cols.remove("tripid")
    cols.remove("pickup_time")
    cols.remove("drop_time")    
    data = data.drop(columns=["tripid"], axis=1)
    
    if is_train:
        #Adjust target label to True or False
        data.loc[data.label == "correct", "label"] = True
        data.loc[data.label == "incorrect", "label"] = False
        data["label"] = data["label"].astype('bool')
        cols.remove("label")
        
    #Fill missing values
    if is_train:
        dataset = fill_missing_val(data, cols)
    else:
        dataset = data
    
    #calculate the distance
    dataset = calculate_distance(dataset)
    
    #Create hour value for timestamps
    dataset["pickup_time"] = data.apply((lambda x: get_hour(x["pickup_time"], "pickup_time")), axis=1).astype("int")
    dataset["drop_time"] = data.apply((lambda x: get_hour(x["drop_time"], "pickup_time")), axis=1).astype("int")

    
    if is_train and use_synthetic_generation:
        #Use SMOTE to resample
        datasetX, datasetY = balance_using_SMOTE(dataset.drop(columns=["label"], axis=1), dataset["label"])
    elif is_train:
        datasetX, datasetY = dataset.drop(columns=["label"], axis=1), dataset["label"]
    else:
        datasetX = dataset
    
    #Bin timestamps
    datasetX = get_binned_DF(datasetX, "pickup_time")
    datasetX = get_binned_DF(datasetX, "drop_time")

    
    #Scale dataset
    datasetX = scale_data(datasetX)
    
    if is_train:
        return datasetX, datasetY
    
    return datasetX

def evalutate_f1(model, X, Y):
    predictions = model.predict(X)
    return metrics.f1_score(Y, predictions)

def hard_vote(predictions, k):
    predictions['sum'] = 0
    for i in range(1, k + 1):
        predictions['sum'] += predictions["prediction_" + str(k)]
    predictions["final_prediction"] = (predictions["sum"] >= k/2)
    return predictions

def train_k_fold_random_forest(X, Y, k=10, use_synthetic_generation=True):
    models = [] #store models
    scores = []
    
    #Convert to numpy
    X = X.to_numpy()
    Y = Y.to_numpy()
    
    #init kfolds
    kfold = KFold(k, True)
    
    #Do k fold validation
    for train_index, test_index in kfold.split(X):
        #Get training and test splits
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        #Use SMOTE for oversampled balanced training data
        if use_synthetic_generation:
            X_train, Y_train = balance_using_SMOTE(X_train, Y_train, k_neighbors=102)
        #Train a single model
        mdl = train_random_forest(X_train, Y_train)
        
        train_score = evalutate_f1(mdl, X_train, Y_train)
        test_score = evalutate_f1(mdl, X_test, Y_test)
        print(train_score, test_score)
        
        models.append(mdl)
        scores.append((train_score, test_score))
    
    return models, scores