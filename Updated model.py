#save model
from pathlib import Path
import tensorflow as tf
from keras.models import load_model

# Save neural network structure
model_structure = model.to_json()

## Comments - path should be in the UNIX form. 
## otherwise it wont work in the server environment
## also we can use h5 instead of json
## h5 seems to be the standard way of saving
f = Path("C://Users/Rajini/Desktop/StockWebApp")
f.write_text(model_structure)
print('done')

## Comments - path should be in the UNIX form. 
# Save neural network's trained weights
model.save_weights("C://Users/Rajini/Desktop/StockWebApp_weights.h5")
print('done')

## Comments - path should be in the UNIX form. 
# or you can save the full model via:
model.save('C://Users/Rajini/Desktop/StockWebApp_fullkeras_model.h5')

#delete your model in memory
del model

## Comments - use the exact same file name
#Know to load your model use:
my_new_model = tf.keras.models.load_model("C://Users/Rajini/Desktop/StockWebApp_fullkeras_model.h5")

#compile my_new_model:
my_new_model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# DB Management
import sqlite3 
conn = sqlite3.connect('modelparam.db')
c = conn.cursor()