from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def testModel(model, trainingFile, testingFile, bank, st):
    dataset_train = pd.read_csv(trainingFile)
    training_set = dataset_train.iloc[:, 1:2].values
    dataset_test = pd.read_csv(testingFile)
    real_stock_price = dataset_test.iloc[:, 1:2].values
    
    sc = MinMaxScaler(feature_range=(0,1))
    sc.fit_transform(training_set)

    #Modify the test data set
    dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
    inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
    inputs = inputs.reshape(-1,1)
    inputs = sc.transform(inputs)
    X_test = []
    for i in range(60, 76):
        X_test.append(inputs[i-60:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predicted_stock_price = model.predict(X_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)
    
    fig2 = plt.figure()
    plt.plot(real_stock_price, color = 'black', label = 'Stock Price '+bank)
    plt.plot(predicted_stock_price, color = 'green', label = 'Predicted Stock Price '+bank)
    plt.title(' Stock Price Prediction '+bank)
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    st.pyplot(fig2)