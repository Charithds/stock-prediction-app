#save model
import tensorflow as tf
import uuid
import os


def saveModel(model):
    # Save neural network structure
    # model_structure = model.to_json()
    
    ## Comments - path should be in the UNIX form. 
    ## otherwise it wont work in the server environment
    ## also we can use h5 instead of json
    ## h5 seems to be the standard way of saving
    savedFilename = os.path.join("models", str(uuid.uuid4()) + '_' + '_fullkeras_model.h5')
    
    ## Comments - path should be in the UNIX form. 
    # or you can save the full model via:
    model.save(savedFilename)
    
    #delete your model in memory
    del model
    return savedFilename

def resurrectModel():
    ## Comments - use the exact same file name
    #Know to load your model use:
    my_new_model = tf.keras.models.load_model("C://Users/Rajini/Desktop/StockWebApp_fullkeras_model.h5")
    
    #compile my_new_model:
    my_new_model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    