import sqlite3
import streamlit as st
import uuid
import os
import hashlib
import numpy as np

import company_ddl
import model_generator
import modelSaver
import training_job_ddl
import model_tester


def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    if make_hashes(password) == hashed_text:
        return hashed_text
    return False

# DB Management
conn = sqlite3.connect('userdata.db')
c = conn.cursor()
company_ddl.createTables(conn)
training_job_ddl.createTables(conn)

# create needed folders
path = "trainingData"
isExist = os.path.exists(path)
if not isExist:
    os.makedirs(path)
    
path = "models"
isExist = os.path.exists(path)
if not isExist:
    os.makedirs(path)

path = "testingData"
isExist = os.path.exists(path)
if not isExist:
    os.makedirs(path)

def initMLModel():
    '''
    url = 'https://raw.githubusercontent.com/AchiniKarunasinghe/Stock-Prediction/master/Data-repo.csv'
    dataset_train = pd.read_csv(url)
    training_set = dataset_train.iloc[:, 1:2].values

    dataset_train.head()

    #Data normalization

    sc = MinMaxScaler(feature_range=(0,1))
    training_set_scaled = sc.fit_transform(training_set)

    #Incorporating Timesteps Into Data
    X_train = []
    y_train = []
    for i in range(60, 2035):
        X_train.append(training_set_scaled[i-60:i, 0])
        y_train.append(training_set_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


    model = Sequential()

    model.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))

    model.add(LSTM(units=50,return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=50,return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=50))
    model.add(Dropout(0.2))

    model.add(Dense(units=1))

    model.compile(optimizer='adam',loss='mean_squared_error')

    model.fit(X_train,y_train,epochs=10,batch_size=32)

    #Making Predictions on the Test Set
    url = 'https://raw.githubusercontent.com/AchiniKarunasinghe/Stock-Prediction/ed4b480ddafb790b2b39a4693dc27a94040fd570/testdata.csv'
    dataset_test = pd.read_csv(url)
    real_stock_price = dataset_test.iloc[:, 1:2].values

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

    url2 = 'https://raw.githubusercontent.com/AchiniKarunasinghe/Stock-Prediction/60d622e714e002efec35d126e1a8021f085dfc71/testdata2.csv'
    dataset_test2 = pd.read_csv(url2)
    real_stock_price2 = dataset_test2.iloc[:, 1:2].values

    dataset_total2 = pd.concat((dataset_train['Open'], dataset_test2['Open']), axis = 0)
    inputs = dataset_total2[len(dataset_total2) - len(dataset_test2) - 60:].values
    inputs = inputs.reshape(-1,1)
    inputs = sc.transform(inputs)
    X_test = []
    for i in range(60, 76):
        X_test.append(inputs[i-60:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predicted_stock_price2 = model.predict(X_test)
    predicted_stock_price2 = sc.inverse_transform(predicted_stock_price2)

    st.subheader('Predictions vs Original')

    fig = plt.figure()
    plt.plot(real_stock_price2, color = 'black', label = 'Stock Price DFCC')
    plt.plot(predicted_stock_price2, color = 'red', label = 'Predicted Stock Price DFCC')
    plt.title(' Stock Price Prediction DFCC')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    st.pyplot(fig)
    # plt.show()

    fig2 = plt.figure()
    plt.plot(real_stock_price, color = 'black', label = 'Stock Price HNB')
    plt.plot(predicted_stock_price, color = 'green', label = 'Predicted Stock Price HNB')
    plt.title(' Stock Price Prediction HNB')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    st.pyplot(fig2)
    '''

# DB  Functions


def create_usertable():
    c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')


def add_userdata(username, password):
    c.execute('INSERT INTO userstable(username,password) VALUES (?,?)',
              (username, password))
    conn.commit()


def login_user(username, password):
    c.execute('SELECT * FROM userstable WHERE username =? AND password = ?',
              (username, password))
    data = c.fetchall()
    return data


def view_all_users():
    c.execute('SELECT * FROM userstable')
    data = c.fetchall()
    return data

def showCompanies():
    companiesInDB = company_ddl.getCompanies(conn)
    if companiesInDB.empty:
        st.text("No data at the moment")
    else:
        companiesInDB.columns = ['Index', "Bank Code", "Bank Name"]
        st.table(companiesInDB)
        

def companiesSection():
    #company_ddl.createTables(conn)
    newCode = st.text_input("Bank code")
    newName = st.text_input("Bank name")
    if st.button("Add bank"):
        constCompany = {}
        constCompany['code'] = newCode
        constCompany['name'] = newName
        company_ddl.addCompany(constCompany, conn)

def trainSection():
    st.subheader("Train a new model")
    newTrainingJob = {}
    
    companiesInDB = company_ddl.getCompanies(conn)
    banksStr = [' - '.join(val) for val in companiesInDB.astype(str).values.tolist()]
    
    selectedBank = st.selectbox("Select the bank", options=banksStr)
    newTrainingJob['bank'] = selectedBank.split(' - ')[1]
    
    tag = st.text_input("Enter a tag")
    newTrainingJob['tag'] = tag
    
    uploadedFile = st.file_uploader("Select your training dataset", type='csv', key='trainingSetUploader')
    if uploadedFile is not None:
        savedFilename = os.path.join("trainingData", str(uuid.uuid4()) + '_' + uploadedFile.name)
        with open(savedFilename,"wb") as f: 
            f.write(uploadedFile.getbuffer())
        newTrainingJob['trainingFile'] = savedFilename
        if st.button("Train"):
            with st.spinner("Please wait while training is completed..."):
                generatedModel = model_generator.train(savedFilename)
            modelFile = modelSaver.saveModel(generatedModel)
            newTrainingJob['modelFile'] = modelFile
            training_job_ddl.addTrainingJob(newTrainingJob, conn)
            # st.file_uploader("Select your training dataset", type='csv').clear()
            st.success("model trained successfully!")
    else:
        st.error("File should not be empty")

def predictions():
    #pred_uploadedFile = None
    #pred_loadedModel = None
    allJobs = training_job_ddl.getTrainingJobs(conn)
    
    banks = allJobs.loc[:, "bank"].unique()
    selectedBank = st.selectbox("Select the bank", options=banks)
    if selectedBank is not None:
        jobsOfBank = allJobs[allJobs['bank'] == selectedBank]
        tags = jobsOfBank.loc[:, "tag"].unique()
        selectedTag = st.selectbox("Select entered training tag", tags)
        if selectedTag is not None:
            selectedJob = allJobs[np.logical_and(allJobs['bank'] == selectedBank, allJobs['tag'] == selectedTag)]
            if not selectedJob.empty:
                params = selectedJob.iloc[0]
                st.text("Training file is " + params['training_file'])
                st.text("Saved model file is " + params['model_file'])
                pred_uploadedFile = st.file_uploader("Select your testing dataset", type='csv', key='testingSetUploader')
                if pred_uploadedFile is not None:
                    savedFilename = os.path.join("testingData", str(uuid.uuid4()) + '_' + pred_uploadedFile.name)
                    with open(savedFilename,"wb") as f: 
                        f.write(pred_uploadedFile.getbuffer())
                if st.button("Test"):
                    pred_loadedModel = modelSaver.resurrectModel(params['model_file'])
                    st.success('Model loaded')
                    model_tester.testModel(pred_loadedModel, params['training_file'], savedFilename, params['bank'], st)
        
    # select bank
    # select tag
    # load model
    # upload test file
    # generate predictions and graphs

def main():
    """Simple Login App"""

    st.title("Stock Prediction Viewer")

    menu = ["Home", "Login", "SignUp", "All Banks", "Add Bank", "Train", "Predictions"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Home")

    elif choice == "Login":
        st.subheader("Login Section")

        username = st.sidebar.text_input("User Name")
        password = st.sidebar.text_input(
            "Password", value='', type='password', key='login-pw')
        # print(password)
        if st.sidebar.checkbox("Login"):
            # if password == '12345':
            create_usertable()
            hashed_pswd = make_hashes(password)

            result = login_user(username, check_hashes(password, hashed_pswd))
            if result:
                # password = st.sidebar.text_input("Password", value='', type='password')
                st.success("Logged In as {}".format(username))

                task = st.selectbox("Task", ["View Predictions", "Profiles"])
                if task == "View Predictions":
                    st.subheader("Generated Predictions for the companies")
                    initMLModel()

    elif choice == "SignUp":
        st.subheader("Create New Account")
        new_user = st.text_input("Username")
        new_password = st.text_input(
            "Password", value='', type='password', key='register-pw')

        if st.button("Signup"):
            create_usertable()
            add_userdata(new_user, make_hashes(new_password))
            st.success("You have successfully created a valid Account")
            st.info("Go to Login Menu to login")
            # new_password = st.text_input("Password", value='', type='password')

    elif choice == "All Banks":
        st.subheader("All Banks")
        showCompanies()
    elif choice == 'Add Bank':
        companiesSection()
    elif choice == "Train":
        trainSection()
    elif choice == "Predictions":
        predictions()


if __name__ == '__main__':
    main()
