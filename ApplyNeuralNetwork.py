#libs & paths
import pandas as pd
import numpy as np
from keras.models import load_model


model = load_model('bestmodel.h5')
inputData = pd.read_csv("valid_track1/result.txt", sep=',', header=None)

x_valid = inputData[inputData.columns[0:16]]
# y_predict = inputData[inputData.columns[16]]
inputMetaData1 = inputData[inputData.columns[17]]
inputMetaData2 = inputData[inputData.columns[18]]
inputMetaData3 = inputData[inputData.columns[19]]
y_predict = model.predict(x_valid)

results = pd.DataFrame()
results["a"]=np.array(y_predict).flatten().tolist()
results["b"]=inputMetaData1.values.tolist()
results["c"]=inputMetaData2.values.tolist()
results["d"]=inputMetaData3.values.tolist()


results.to_csv('networkOutput.txt', header=None, index=None, sep=',', mode='w')
