import matplotlib.pyplot as plt
import seaborn as sns
#from main import dataset
#from main import dataset_name

#dataset = "Ravdess: "
#path_loss = "D:\\Python\\SpeechEM\\Features_deepCNF\\Images\\Ravdess_Training_loss_vs_Validation_loss.png"
#path_acc = "D:\\Python\\SpeechEM\\Features_deepCNF\\Images\\Ravdess_Training_accuracy_vs_Validation_accuracy.png"
#path_loss = "D:\\Python\\SpeechEM\\Features_deepCNF\\Images\\Tess_Training_loss_vs_Validation_loss.png"
#path_acc = "D:\\Python\\SpeechEM\\Features_deepCNF\\Images\\Tess_Training_accuracy_vs_Validation_accuracy.png"
#path_loss = "D:\\Python\\SpeechEM\\Features_deepCNF\\Images\\Savee_Training_loss_vs_Validation_loss.png"
#path_acc = "D:\\Python\\SpeechEM\\Features_deepCNF\\Images\\Savee_Training_accuracy_vs_Validation_accuracy.png"
#path_loss = "D:\\Python\\SpeechEM\\Features_deepCNF\\Images\\Emodb_Training_loss_vs_Validation_loss.png"
#path_acc = "D:\\Python\\SpeechEM\\Features_deepCNF\\Images\\Emodb_Training_accuracy_vs_Validation_accuracy.png"
#path_loss = "D:\\Python\\SpeechEM\\Features_deepCNF\\Images\\CremaD_Training_loss_vs_Validation_loss.png"
#path_acc = "D:\\Python\\SpeechEM\\Features_deepCNF\\Images\\CremaD_Training_accuracy_vs_Validation_accuracy.png"
#path_loss = "D:\\Python\\SpeechEM\\Features_deepCNF\\Images\\Combined_Training_loss_vs_Validation_loss.png"
#path_acc = "D:\\Python\\SpeechEM\\Features_deepCNF\\Images\\Combined_Training_accuracy_vs_Validation_accuracy.png"
#path_loss = "D:\\Python\\SpeechEM\\Features_deepCNF\\Images\\r_t_s_e_Training_loss_vs_Validation_loss.png"
#path_acc = "D:\\Python\\SpeechEM\\Features_deepCNF\\Images\\r_t_s_e_Training_accuracy_vs_Validation_accuracy.png"
#path_loss = "D:\\Python\\SpeechEM\\Features_deepCNF\\Images\\r_t_s_Training_loss_vs_Validation_loss.png"
#path_acc = "D:\\Python\\SpeechEM\\Features_deepCNF\\Images\\r_t_s_Training_accuracy_vs_Validation_accuracy.png"
# path_loss = "D:\\Python\\SpeechEM\\Features_deepCNF\\Images\\" + dataset_name + "_Training_loss_vs_Validation_loss.png"
# path_acc = "D:\\Python\\SpeechEM\\Features_deepCNF\\Images\\" + dataset_name + "_Training_accuracy_vs_Validation_accuracy.png"

def plot_trainHistory(train_loss, val_loss, train_acc, val_acc, dataset_name="None"):
    path_loss = "..\\Images\\" + dataset_name + "_Training_loss_vs_Validation_loss.png"
    path_acc = "..\\Images\\" + dataset_name + "_Training_accuracy_vs_Validation_accuracy.png"
    # PLOT MODEL HISTORY OF ACCURACY AND LOSS OVER EPOCHS
    # Plot the results
    plt.close('all')
    plt.clf()  # Clears the current figure
    plt.rcParams['figure.dpi'] = 300 
    plt.figure(1,figsize=(6,5))
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.xlabel('Number of epochs')
    plt.ylabel('Loss')
    plt.title(dataset_name +': Training loss vs Validation loss')
    plt.grid(True)
    plt.legend(['Training','Validation'], loc=1)
    sns.set_style("darkgrid") 
    #plt.style.use("darkgrid")
    plt.savefig(path_loss, dpi=300)

    plt.clf()  # Clears the current figure
    plt.close('all')
    plt.rcParams['figure.dpi'] = 300 
    plt.figure(2,figsize=(6,5))
    plt.plot(train_acc)
    plt.plot(val_acc)
    plt.xlabel('Number of epochs')
    plt.ylabel('Accuracy')
    plt.title(dataset_name +': Training accuracy vs Validation accuracy')
    plt.grid(True)
    plt.legend(['Training','Validation'],loc=4)
    #plt.style.use("darkgrid")
    sns.set_style("darkgrid")      
    plt.savefig(path_acc, dpi=300)
    # PRINT LOSS AND ACCURACY PERCENTAGE ON TEST SET
    #print("Loss of the model is - " , model.evaluate(X_test,y_test)[0])
    #print("Accuracy of the model is - " , model.evaluate(X_test,y_test)[1]*100 , "%") 
    #plt.show()
    plt.close('all')
    plt.clf()  # Clears the current figure

