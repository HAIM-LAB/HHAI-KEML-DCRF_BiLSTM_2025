import os
import numpy as np
import tensorflow as tf

from sklearn.metrics import classification_report
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, Bidirectional, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
#import loadData as ld
from tensorflow.keras.layers import TimeDistributed, Reshape
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import drawing as dw
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.decomposition import PCA


# Set global parameters
FEATURES_COUNT_G = 190
FEATURES_COUNT = 190 #90 #162*3 #282 #194  # Number of features per audio
NUM_CLASSES = 7       # Emotion classes (0-6)
MODEL_TYPE = "LSTM"   # Options: "LSTM", "GRU", "DNN"
ACTIVATION_FN = "LeakyReLU"  # Options: "ReLU", "LeakyReLU", "Softmax"
LOSS_FN = "categorical_crossentropy"  # Options: "", "" categorical_crossentropy 

# Load data function
def load_data(file_path):
    X, y = [], []
    with open(file_path, "r") as f:
        lines = f.readlines()
        for i in range(0, len(lines), 3):
            if lines[i].strip() == "1":  # New audio entry
                features = np.array([float(x) for x in lines[i + 1].strip().split()])
                emotion = int(lines[i + 2].strip())  # Emotion label
                if len(features) == FEATURES_COUNT:
                    X.append(features)
                    y.append(emotion)
    return np.array(X), np.array(y)
# Custom CRF Layer
class CRFLayer(Layer):
    def __init__(self, num_classes, **kwargs):
        super(CRFLayer, self).__init__(**kwargs)
        self.num_classes = num_classes

    def build(self, input_shape):
        self.transitions = self.add_weight(
            shape=(self.num_classes, self.num_classes),
            initializer="glorot_uniform",
            trainable=True,
            name="transitions"
        )
        super(CRFLayer, self).build(input_shape)

    def call(self, inputs, mask=None):
        return inputs  # CRF decoding handled separately

    def compute_output_shape(self, input_shape):
        return input_shape

def build_model(input_dim, model_type="LSTM", activation="LeakyReLU"):
    model = Sequential()
    drop_out = 0.3
    if model_type == "LSTM":
        # First LSTM Layer
        model.add(Bidirectional(LSTM(512, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.01)), input_shape=(1, input_dim)))
        model.add(BatchNormalization())
        model.add(Dropout(drop_out))
       
        # # # Second LSTM Layer (Stacked)
        model.add(Bidirectional(LSTM(512, return_sequences=True,kernel_regularizer=tf.keras.regularizers.l2(0.01))))
        model.add(BatchNormalization())
        model.add(Dropout(drop_out))

        model.add(Bidirectional(LSTM(512, return_sequences=True,kernel_regularizer=tf.keras.regularizers.l2(0.01))))
        model.add(BatchNormalization())
        model.add(Dropout(drop_out))

        model.add(Dense(512, activation="swish"))  # Swish activation
        model.add(Dropout(drop_out))        

        model.add(Dense(512, activation="swish"))  # Swish activation
        model.add(Dropout(drop_out))        

       
    elif model_type == "GRU":
        model.add(Bidirectional(GRU(128, return_sequences=True), input_shape=(1, input_dim)))
    else:
        model.add(Dense(128, input_shape=(1, input_dim), activation="relu"))
    
    if activation == "LeakyReLU":
        model.add(tf.keras.layers.LeakyReLU())
    elif activation == "ReLU":
        model.add(tf.keras.layers.ReLU())
    elif activation == "Softmax":
        model.add(tf.keras.layers.Softmax())

    # Add CRF layer
    model.add(CRFLayer(NUM_CLASSES))

    # Output Layer: Ensure it has the correct shape (batch_size, 7)
    model.add(TimeDistributed(Dense(NUM_CLASSES, activation="softmax")))  

    return model

    
# Custom callback to track and store the best model during training
class BestModelCallback(Callback):
    def __init__(self):
        super().__init__()
        self.best_weights = None
        self.best_accuracy = 0
        self.val_acc_hit = False

    def on_epoch_end(self, epoch, logs=None):
        # If the current validation accuracy is better, save the model weights
        current_accuracy = logs.get('val_accuracy')
        if current_accuracy > self.best_accuracy:
            self.best_accuracy = current_accuracy
            self.best_weights = self.model.get_weights()  # Save the best model weights
        accuracy = logs.get('accuracy')
        if (epoch) % 50 == 0:  # Print every N epochs
            print(f"Epoch {epoch + 1}: Accuracy = {accuracy:.4f}, Validation Accuracy = {self.best_accuracy:.4f}")
        if self.val_acc_hit:
            print(f"\nStopping: One epoch after val_accuracy hit 1.0 (epoch {epoch+1})")
            #self.model.stop_training = True
        elif current_accuracy == 1.0:
            print(f"\nval_accuracy reached 1.0 at epoch {epoch+1}, will stop after next epoch.")
            self.val_acc_hit = True

    def get_best_model(self):
        # Load the best weights into the model
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)


# Training function
def train_model(model, X_train, y_train, X_test, y_test, epochs=200, dataset_name="None", lr=0.0001):

    y_train = to_categorical(y_train, num_classes=NUM_CLASSES)
    y_test = to_categorical(y_test, num_classes=NUM_CLASSES)
    model.add(Reshape((NUM_CLASSES,)))  # Reshape to remove extra dimension

    optimizer = Adam(learning_rate=lr)

    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    model.compile(optimizer=optimizer, loss=LOSS_FN, metrics=["accuracy"])
    best_model_callback = BestModelCallback()
   
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=256, validation_data=(X_test, y_test),callbacks=[best_model_callback],verbose=1)
    
    # After training, set the best model weights (in memory)
    best_model_callback.get_best_model()
    # train_losses = history.history['loss']
    # test_losses = history.history['val_loss']
    # train_accs = history.history['accuracy']
    # test_accs = history.history['val_accuracy']
    # dw.plot_trainHistory(train_losses, test_losses, train_accs, test_accs, dataset_name=dataset_name)

    return model, history
# Evaluation function
def evaluate_model(model, X_test, y_test):
    y_test = to_categorical(y_test, num_classes=NUM_CLASSES)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Final Test Accuracy: {test_acc * 100:.3f}%, and Test Loss: {test_loss:.5f}")
    return test_acc, test_loss

# Calculate weighted average accuracy function
def calculate_weighted_accuracy(y_true, y_pred, dataset_name):
    y_true = to_categorical(y_true, num_classes=7)
    #print("y_true.shape='{}', dim='{}'".format(y_true.shape,y_true.ndim))
    #print("y_pred.shape='{}', dim='{}'".format(y_pred.shape,y_pred.ndim))

    # Convert predictions to one-hot format
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_true, axis=1)

    # Calculate accuracy for each class
    unique_classes = np.unique(y_true_classes)
    class_accuracies = []
    for class_id in unique_classes:
        class_mask = (y_true_classes == class_id)
        class_accuracy = accuracy_score(y_true_classes[class_mask], y_pred_classes[class_mask])
        class_accuracies.append(class_accuracy)
    
    # Calculate the weighted average accuracy
    class_weights = [np.sum(y_true_classes == c) / len(y_true_classes) for c in unique_classes]
    weighted_avg_acc = np.dot(class_accuracies, class_weights)
    print(f"Weighted Average Accuracy for {dataset_name}: {weighted_avg_acc * 100:.2f}%")
    return weighted_avg_acc, class_accuracies
    
def calculate_weighted_accuracy_details(y_true, y_pred, dataset_name, num_classes=7):
    y_true = to_categorical(y_true, num_classes=num_classes)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_true, axis=1)

    # Initialize accuracy for all classes (even if they don't appear in this fold)
    class_accuracies = np.zeros(num_classes)
    class_counts = np.zeros(num_classes)
    
    # Calculate accuracy for each present class
    for class_id in range(num_classes):
        class_mask = (y_true_classes == class_id)
        if np.sum(class_mask) > 0:  # Only calculate if class exists in this fold
            class_accuracies[class_id] = accuracy_score(
                y_true_classes[class_mask], 
                y_pred_classes[class_mask]
            )
            class_counts[class_id] = np.sum(class_mask)
    
    # Calculate weighted average accuracy
    if np.sum(class_counts) > 0:
        weighted_avg_acc = np.sum(class_accuracies * class_counts) / np.sum(class_counts)
    else:
        weighted_avg_acc = 0.0
    
    print(f"Weighted Average Accuracy for {dataset_name}: {weighted_avg_acc * 100:.2f}%")
    return weighted_avg_acc, class_accuracies, class_counts

# Main function to train DeepCNF
def train_DeepCRF(dataset, dataset_name):
    global NUM_CLASSES
    global FEATURES_COUNT
    if dataset_name in ["EmoDB", "CremaD"]:
        NUM_CLASSES = 6
    else:
        NUM_CLASSES = 7    

    vs_code = 1
    if vs_code == 1:
        print(f"reading dataset {dataset_name}")        
        train_file = "2025_ACL_SpeechEMD_proj_code\\datasetFeature\\" + dataset + ".feat"
    else:
        train_file = "D:\\Python\\SpeechEM\\Features_deepCNF\\" + dataset + ".feat"
    X, Y = load_data(train_file)
    # splitting data
    if dataset_name in ["EmoDB"]:
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42, shuffle=True,  stratify=Y)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=0, shuffle=True,  stratify=Y)

    #X_val = X_test
    #y_val = y_test
    #X_temp, X_test, y_temp, y_test = train_test_split(X, Y, test_size=0.20, random_state=0, shuffle=True,  stratify=Y)
    #X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.20, random_state=0, shuffle=True,  stratify=y_temp)

    # Load data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_feat = scaler.transform(X)

    # Step 2: Apply PCA
    pca = PCA(n_components=0.99)  # Keeps 95% variance n_components=100
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    X_feat = pca.transform(X_feat)

    # Step 3: Update FEATURES_COUNT
    FEATURES_COUNT = X_train.shape[1]
    # Reshape data for LSTM/GRU
    X_train = X_train.reshape(X_train.shape[0], 1, FEATURES_COUNT)
    X_test = X_test.reshape(X_test.shape[0], 1, FEATURES_COUNT)
    X_feat = X_feat.reshape(X_feat.shape[0], 1, FEATURES_COUNT)

    # Build and train model
    model = build_model(FEATURES_COUNT, model_type=MODEL_TYPE, activation=ACTIVATION_FN)
    model, history = train_model(model, X_train, y_train, X_test, y_test, epochs=500, dataset_name=dataset_name)
    # Evaluate the model
    test_acc, test_loss = evaluate_model(model, X_test, y_test)

    print(f"Test accuracy with best validation model is {test_acc:.4f}")
    # Calculate weighted average accuracy
    y_pred = model.predict(X_test)
    
    ## Added by Shahana
    save_history(history, dataset_name)

    # added by Z
    save_classification_metrics(model, X_test, y_test, dataset_name, output_path=f"2025_ACL_SpeechEMD_proj_code/report/{dataset_name}_report.txt")

    emotion_labels = ['Neutral', 'Happy', 'Sad', 'Angry', 'Fear', 'Disgust', 'Surprise']
    if dataset_name in ["EmoDB", "CremaD"]:
        emotion_labels = ['Neutral', 'Happy', 'Sad', 'Angry', 'Fear', 'Disgust']    
    
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)

    if y_test.ndim > 1:
        y_test = np.argmax(y_test, axis=1)

    save_confusion_matrix_withEmo(y_test, y_pred, class_labels=emotion_labels, output_path=f"2025_ACL_SpeechEMD_proj_code/report/{dataset_name}_")
    # reset FEATURES_COUNT to load files
    FEATURES_COUNT = FEATURES_COUNT_G
    
     
def save_history(history, dataset_name):

    import pandas as pd
    import os
    
    history_df = pd.DataFrame(history.history)

    # Make directory if doesn't exist
    output_dir = f"2025_ACL_SpeechEMD_proj_code/history/"

    # Save training log to CSV
    history_df.to_csv(f"2025_ACL_SpeechEMD_proj_code/history/{dataset_name}_metrics_training_history.csv", index=False)

    # Plot accuracy and loss
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"2025_ACL_SpeechEMD_proj_code/history/{dataset_name}_training_curves.png")

    # Find the best validation accuracy and corresponding epoch
    best_val_acc = max(history.history['val_accuracy'])
    best_epoch = history.history['val_accuracy'].index(best_val_acc) + 1  # +1 for 1-based index

    print(f"Best Validation Accuracy: {best_val_acc:.4f} at Epoch {best_epoch}")

    # Save it to a text file
    with open(f"2025_ACL_SpeechEMD_proj_code/history/{dataset_name}_best_accuracy.txt", "w") as f:
        f.write(f"Best Validation Accuracy: {best_val_acc:.4f} at Epoch {best_epoch}\n")
    
    #best training accuracy as well
    best_train_acc = max(history.history['accuracy'])
    best_train_epoch = history.history['accuracy'].index(best_train_acc) + 1

    with open(f"2025_ACL_SpeechEMD_proj_code/history/{dataset_name}_best_accuracy.txt", "a") as f:
        f.write(f"Best Training Accuracy: {best_train_acc:.4f} at Epoch {best_train_epoch}\n")

from sklearn.metrics import classification_report
import numpy as np

def save_classification_metrics(model, X_test, y_test, dataset_name, output_path="report.txt"):
    # Ensure y_test is a proper NumPy array
    y_test = np.array(y_test)
    if y_test.ndim == 2 and y_test.shape[1] > 1:  # One-hot encoded
        y_test = np.argmax(y_test, axis=-1)
    y_test_flat = y_test  # No flattening needed

    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=-1)

    report = classification_report(y_test_flat, y_pred, digits=4)
    with open(output_path, "w") as f:
        f.write("=== Evaluation: Precision, Recall, F1-Score ===\n")
        f.write(report)
    

def save_confusion_matrix_withEmo(y_test_final, y_pred_final, class_labels, output_path="./"):
    print(output_path)
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt
    import pandas as pd

    # Compute confusion matrix
    cm = confusion_matrix(y_test_final, y_pred_final, labels=range(len(class_labels)))

    # Display with emotion labels
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)

    plt.figure(figsize=(8, 6))
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.savefig(output_path + "confusion_matrix.png", bbox_inches='tight')
    #plt.show()

    # Save to CSV
    cm_df = pd.DataFrame(cm, index=class_labels, columns=class_labels)
    cm_df.to_csv(output_path + "confusion_matrix.csv")

    # Classification report
    report_dict = classification_report(y_test_final, y_pred_final, target_names=class_labels, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(output_path + "classification_report.csv")

    # Plot F1 scores
    f1_scores = report_df.iloc[:-3]['f1-score']
    f1_scores.plot(kind='bar', color='teal', figsize=(8, 4), title="Per-Class F1 Scores")
    plt.ylabel("F1 Score")
    plt.xlabel("Emotion")
    plt.tight_layout()
    plt.savefig(output_path + "f1_scores.png")
    #plt.show()

    # Save LaTeX report
    latex_report = report_df.to_latex(float_format="%.2f")
    with open(output_path + "classification_report.tex", "w") as f:
        f.write(latex_report)


def save_confusion_matrix(y_test_final, y_pred_final, output_path="/"):

    print(output_path)
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt
    import pandas as pd

    # Compute confusion matrix
    cm = confusion_matrix(y_test_final, y_pred_final)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    # Plot and save confusion matrix
    plt.figure(figsize=(8, 6))
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.savefig(output_path+"confusion_matrix.png", bbox_inches='tight')
    #plt.show()

    # Save to CSV
    cm_df = pd.DataFrame(cm)
    cm_df.to_csv(output_path+"confusion_matrix.csv", index=False)

    from sklearn.metrics import classification_report
    import pandas as pd

    report_dict = classification_report(y_test_final, y_pred_final, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(output_path+"classification_report.csv")

    f1_scores = report_df.iloc[:-3]['f1-score']  # Exclude avg rows
    f1_scores.plot(kind='bar', color='teal', figsize=(8, 4), title="Per-Class F1 Scores")
    plt.ylabel("F1 Score")
    plt.xlabel("Class Label")
    plt.tight_layout()
    plt.savefig(output_path+"f1_scores.png")
    #plt.show()

    latex_report = report_df.to_latex(float_format="%.2f")
    with open(output_path+"classification_report.tex", "w") as f:
        f.write(latex_report)
