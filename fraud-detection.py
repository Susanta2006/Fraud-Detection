############## Modules ######################################################
import sys
from datetime import datetime, timedelta
try:
    import pyfiglet
    import warnings
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, classification_report
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras import Input
    warnings.filterwarnings('ignore')
    import os
    tf.get_logger().setLevel('ERROR') 
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Only filters
    os.environ['TERM'] = 'dumb'  # disables color output
#############################################################################
    pf = pyfiglet.figlet_format("Fraud Detection")
    print(pf,"\n version 1.0")
    print('''
**********************************
* ------------------------------ *
* |Created by Mr. Susanta Banik| *
* ------------------------------ *
**********************************
''')
    
    inp=input("[+]Enter <Start> to Start or <Exit> to Stop: ")
    print()
    if inp=="Start" or inp=="start" or inp=="START":
        pass
    else:
        print("[-]Exited at:",str(datetime.now().strftime("%I:%M %p")),"On",str(datetime.now().strftime("%d %B %Y, %A")))
        sys.exit()
    ##########################################################
    # Load data
    df = pd.read_csv("creditcard.csv")
    print()
    print("[*] Dataset Loaded Successfully!")
    print()
    print("[-]Started at:",str(datetime.now().strftime("%I:%M %p")),"On",str(datetime.now().strftime("%d %B %Y, %A")))
    print()
    # Normalize 'Amount'
    df['Amount'] = StandardScaler().fit_transform(df[['Amount']])

    # Drop 'Time' column
    df = df.drop(['Time'], axis=1)

    #X = df.drop('Class', axis=1)
    #y = df['Class']

    # Use only 30,000 rows for testing if your device can handle then make it under comment
    df_sample = df.sample(n=30000, random_state=42)

    # Repeat preprocessing
    X = df_sample.drop('Class', axis=1)
    y = df_sample['Class']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Build ANN model
    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(16, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    early_stops=EarlyStopping(monitor='val_loss',patience=20, min_delta=0.001, restore_best_weights=True)
    
    # Train the model
    history = model.fit(X_train, y_train, epochs=100, verbose=2, batch_size=64, validation_split=0.2, callbacks=[early_stops])

    # Predict
    y_pred = (model.predict(X_test) > 0.5).astype(int)

    # Confusion Matrix and Report
    cm = confusion_matrix(y_test, y_pred)
    print()
    print("[?] Confusion Matrix:")
    print(cm)
    print()
    print("[>] Classification Report:")
    print(classification_report(y_test, y_pred))

    new_data = np.array(X_test.iloc[0]).reshape(1, -1)
    pred = model.predict(new_data)
    print()
    print("[*] Final Verdict: Fraud Detected!" if pred[0][0] > 0.5 else "[*] Final Verdict: Transaction is Normal.")
    print()
    
    # Evaluate
    loss, acc = model.evaluate(X_test, y_test,verbose=2)
    print()
    print("[?] Test Accuracy:",round((round(acc,4)*100),4),"%")
    print()
    
    print("[-]Exited at:",str(datetime.now().strftime("%I:%M %p")),"On",str(datetime.now().strftime("%d %B %Y, %A")))
except KeyboardInterrupt or Exception:
    print("---------------------------------------------------------------------------------------------------------------------------------------------------------")
    print()
    print("[-]Something Went Wrong!")
    print("[-]Exited at:",str(datetime.now().strftime("%I:%M %p")),"On",str(datetime.now().strftime("%d %B %Y, %A")))
    print()
    print("---------------------------------------------------------------------------------------------------------------------------------------------------------")
    sys.exit()
