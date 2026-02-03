import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU-only mode
os.environ["OMP_NUM_THREADS"] = "51"  # Adjust based on CPU cores
os.environ["TF_NUM_INTEROP_THREADS"] = "51"
os.environ["TF_NUM_INTRAOP_THREADS"] = "51"
import numpy as np
import tensorflow as tf

from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, Bidirectional, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
from tensorflow.keras.layers import TimeDistributed, Reshape
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import Callback, EarlyStopping
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import train_DeepCRF as dcrf
# Set global parameters
FEATURES_COUNT_G = 190
FEATURES_COUNT = 190
NUM_CLASSES = 7
MODEL_TYPE = "LSTM"
ACTIVATION_FN = "LeakyReLU"
LOSS_FN = "categorical_crossentropy"
N_FOLDS = 5

# (Keep your existing CRFLayer, load_data, build_model, and BestModelCallback implementations here)
# def plot_aggregated_training_curves(all_histories, dataset_name):
#     """Plot and save aggregated training vs validation accuracy curves across all folds"""
#     # Extract all training and validation accuracies
#     all_train_acc = [np.array(h['accuracy']) for h in all_histories]
#     all_val_acc = [np.array(h['val_accuracy']) for h in all_histories]
    
#     # Get the maximum epoch length
#     max_epochs = max(len(acc) for acc in all_train_acc)
    
#     # Pad shorter sequences with their last value
#     padded_train = [np.pad(acc, (0, max_epochs - len(acc)), mode='edge') for acc in all_train_acc]
#     padded_val = [np.pad(acc, (0, max_epochs - len(acc)), mode='edge') for acc in all_val_acc]
    
#     # Convert to numpy arrays
#     train_acc_matrix = np.stack(padded_train)
#     val_acc_matrix = np.stack(padded_val)
    
#     # Calculate mean and standard deviation
#     mean_train = np.mean(train_acc_matrix, axis=0)
#     std_train = np.std(train_acc_matrix, axis=0)
#     mean_val = np.mean(val_acc_matrix, axis=0)
#     std_val = np.std(val_acc_matrix, axis=0)
    
#     # Create plot
#     plt.figure(figsize=(12, 6))
#     epochs = range(1, max_epochs + 1)
    
#     # Plot training accuracy
#     plt.plot(epochs, mean_train, label='Training Accuracy', color='blue')
#     plt.fill_between(epochs, 
#                     mean_train - std_train, 
#                     mean_train + std_train, 
#                     color='blue', alpha=0.1)
    
#     # Plot validation accuracy
#     plt.plot(epochs, mean_val, label='Validation Accuracy', color='orange')
#     plt.fill_between(epochs, 
#                     mean_val - std_val, 
#                     mean_val + std_val, 
#                     color='orange', alpha=0.1)
    
#     plt.title(f'Aggregated Training Curves ({dataset_name}, {len(all_histories)} folds)')
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy')
#     plt.legend()
#     plt.grid(True)
    
#     # Create directory if not exists
#     os.makedirs(f"2025_ACL_SpeechEMD_proj_code/results/{dataset_name}/aggregated", exist_ok=True)
    
#     # Save the plot
#     plot_path = f"2025_ACL_SpeechEMD_proj_code/results/{dataset_name}/aggregated/aggregated_training_curve.png"
#     plt.savefig(plot_path, bbox_inches='tight', dpi=300)
#     plt.close()
    
#     print(f"[INFO] Aggregated training curve saved to: {plot_path}")
def plot_aggregated_training_curves(all_histories, dataset_name):
    """Plot and save aggregated training/validation accuracy and loss curves across all folds"""
    # Extract all metrics
    all_train_acc = [np.array(h['accuracy']) for h in all_histories]
    all_val_acc = [np.array(h['val_accuracy']) for h in all_histories]
    all_train_loss = [np.array(h['loss']) for h in all_histories]
    all_val_loss = [np.array(h['val_loss']) for h in all_histories]
    
    # Get the maximum epoch length
    max_epochs = max(len(acc) for acc in all_train_acc)
    
    # Pad shorter sequences with their last value
    def pad_sequences(sequences):
        return [np.pad(seq, (0, max_epochs - len(seq)), mode='edge') for seq in sequences]
    
    padded_train_acc = pad_sequences(all_train_acc)
    padded_val_acc = pad_sequences(all_val_acc)
    padded_train_loss = pad_sequences(all_train_loss)
    padded_val_loss = pad_sequences(all_val_loss)
    
    # Convert to numpy arrays
    train_acc_matrix = np.stack(padded_train_acc)
    val_acc_matrix = np.stack(padded_val_acc)
    train_loss_matrix = np.stack(padded_train_loss)
    val_loss_matrix = np.stack(padded_val_loss)
    
    # Calculate mean and standard deviation
    def calculate_stats(matrix):
        return np.mean(matrix, axis=0), np.std(matrix, axis=0)
    
    mean_train_acc, std_train_acc = calculate_stats(train_acc_matrix)
    mean_val_acc, std_val_acc = calculate_stats(val_acc_matrix)
    mean_train_loss, std_train_loss = calculate_stats(train_loss_matrix)
    mean_val_loss, std_val_loss = calculate_stats(val_loss_matrix)
    
    # Create figure with two subplots
    plt.figure(figsize=(18, 6))
    epochs = range(1, max_epochs + 1)
    
    # Left subplot - Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, mean_train_acc, label='Training Accuracy', color='blue')
    plt.fill_between(epochs, 
                    mean_train_acc - std_train_acc, 
                    mean_train_acc + std_train_acc, 
                    color='blue', alpha=0.1)
    plt.plot(epochs, mean_val_acc, label='Validation Accuracy', color='orange')
    plt.fill_between(epochs, 
                    mean_val_acc - std_val_acc, 
                    mean_val_acc + std_val_acc, 
                    color='orange', alpha=0.1)
    plt.title(f'Accuracy Curves ({dataset_name}, {len(all_histories)} folds)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1)  # Set y-axis limits for accuracy
    
    # Right subplot - Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, mean_train_loss, label='Training Loss', color='blue')
    plt.fill_between(epochs, 
                    mean_train_loss - std_train_loss, 
                    mean_train_loss + std_train_loss, 
                    color='blue', alpha=0.1)
    plt.plot(epochs, mean_val_loss, label='Validation Loss', color='orange')
    plt.fill_between(epochs, 
                    mean_val_loss - std_val_loss, 
                    mean_val_loss + std_val_loss, 
                    color='orange', alpha=0.1)
    plt.title(f'Loss Curves ({dataset_name}, {len(all_histories)} folds)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Create directory if not exists
    os.makedirs(f"2025_ACL_SpeechEMD_proj_code/results/{dataset_name}/aggregated", exist_ok=True)
    
    # Save the plot
    plot_path = f"2025_ACL_SpeechEMD_proj_code/results/{dataset_name}/aggregated/aggregated_training_curves.png"
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"[INFO] Aggregated training curves saved to: {plot_path}")

def plot_training_curves(history, fold_num, dataset_name):
    """Plot and save training vs validation accuracy curves"""
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'Fold {fold_num} - Training vs Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Create directory if not exists
    os.makedirs(f"2025_ACL_SpeechEMD_proj_code/results/{dataset_name}/training_curves", exist_ok=True)
    plt.savefig(f"2025_ACL_SpeechEMD_proj_code/results/{dataset_name}/training_curves/fold_{fold_num}_training_curve.png")
    plt.close()

def save_confusion_matrix_withEmo(y_test_final, y_pred_final, class_labels, fold_num, dataset_name):
    """Save confusion matrix with emotion labels"""
    # Compute confusion matrix
    cm = confusion_matrix(y_test_final, y_pred_final, labels=range(len(class_labels)))
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_labels, 
                yticklabels=class_labels)
    plt.title(f'Fold {fold_num} - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    # Create directory if not exists
    os.makedirs(f"2025_ACL_SpeechEMD_proj_code/results/{dataset_name}/confusion_matrices", exist_ok=True)
    plt.savefig(f"2025_ACL_SpeechEMD_proj_code/results/{dataset_name}/confusion_matrices/fold_{fold_num}_confusion_matrix.png", 
                bbox_inches='tight')
    plt.close()
    
    # Save to CSV
    cm_df = pd.DataFrame(cm, index=class_labels, columns=class_labels)
    cm_df.to_csv(f"2025_ACL_SpeechEMD_proj_code/results/{dataset_name}/confusion_matrices/fold_{fold_num}_confusion_matrix.csv")

   
def save_classification_metrics(model, X_test, y_test, dataset_name, output_path="report.txt"):
    # Ensure y_test is a proper NumPy array
    y_test = np.array(y_test)
    if y_test.ndim == 2 and y_test.shape[1] > 1:  # One-hot encoded
        y_test = np.argmax(y_test, axis=-1)
    y_test_flat = y_test  # No flattening needed

    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=-1)

    print(f"[DEBUG] y_pred shape: {y_pred.shape}")
    print(f"[DEBUG] y_test shape: {y_test.shape}")
    print(f"[DEBUG] Flattened y_pred shape: {y_pred.shape}")
    print(f"[DEBUG] Flattened y_test shape: {y_test_flat.shape}")

    report = classification_report(y_test_flat, y_pred, digits=4, output_dict=True)
    
    # Save the detailed report
    with open(output_path, "w") as f:
        f.write("=== Evaluation: Precision, Recall, F1-Score ===\n")
        f.write(classification_report(y_test_flat, y_pred, digits=4))
    print(f"[INFO] Classification report saved to: {output_path}")
    
    return report  # Return the report dictionary for aggregation

def save_aggregated_classification_report(all_reports, dataset_name):
    """Aggregate classification reports from all folds and save summary"""
    # Initialize aggregated metrics
    class_names = [k for k in all_reports[0].keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
    aggregated = {class_name: {'precision': [], 'recall': [], 'f1-score': [], 'support': []} 
                 for class_name in class_names}
    aggregated['accuracy'] = []
    aggregated['macro_avg'] = {'precision': [], 'recall': [], 'f1-score': [], 'support': []}
    aggregated['weighted_avg'] = {'precision': [], 'recall': [], 'f1-score': [], 'support': []}
    
    # Collect metrics from all folds
    for report in all_reports:
        for class_name in class_names:
            for metric in ['precision', 'recall', 'f1-score', 'support']:
                aggregated[class_name][metric].append(report[class_name][metric])
        
        aggregated['accuracy'].append(report['accuracy'])
        for metric in ['precision', 'recall', 'f1-score', 'support']:
            aggregated['macro_avg'][metric].append(report['macro avg'][metric])
            aggregated['weighted_avg'][metric].append(report['weighted avg'][metric])
    
    # Calculate mean and std for each metric
    summary = {}
    for class_name in class_names:
        summary[class_name] = {
            'precision': f"{np.mean(aggregated[class_name]['precision']):.4f} ± {np.std(aggregated[class_name]['precision']):.4f}",
            'recall': f"{np.mean(aggregated[class_name]['recall']):.4f} ± {np.std(aggregated[class_name]['recall']):.4f}",
            'f1-score': f"{np.mean(aggregated[class_name]['f1-score']):.4f} ± {np.std(aggregated[class_name]['f1-score']):.4f}",
            'support': int(np.mean(aggregated[class_name]['support']))
        }
    
    # Add overall metrics
    summary['accuracy'] = f"{np.mean(aggregated['accuracy']):.4f} ± {np.std(aggregated['accuracy']):.4f}"
    summary['macro_avg'] = {
        'precision': f"{np.mean(aggregated['macro_avg']['precision']):.4f} ± {np.std(aggregated['macro_avg']['precision']):.4f}",
        'recall': f"{np.mean(aggregated['macro_avg']['recall']):.4f} ± {np.std(aggregated['macro_avg']['recall']):.4f}",
        'f1-score': f"{np.mean(aggregated['macro_avg']['f1-score']):.4f} ± {np.std(aggregated['macro_avg']['f1-score']):.4f}",
        'support': int(np.mean(aggregated['macro_avg']['support']))
    }
    summary['weighted_avg'] = {
        'precision': f"{np.mean(aggregated['weighted_avg']['precision']):.4f} ± {np.std(aggregated['weighted_avg']['precision']):.4f}",
        'recall': f"{np.mean(aggregated['weighted_avg']['recall']):.4f} ± {np.std(aggregated['weighted_avg']['recall']):.4f}",
        'f1-score': f"{np.mean(aggregated['weighted_avg']['f1-score']):.4f} ± {np.std(aggregated['weighted_avg']['f1-score']):.4f}",
        'support': int(np.mean(aggregated['weighted_avg']['support']))
    }
    
    # Convert to pandas DataFrame for nice formatting
    report_df = pd.DataFrame(summary).T
    
    # Create output directory
    os.makedirs(f"2025_ACL_SpeechEMD_proj_code/results/{dataset_name}/aggregated", exist_ok=True)
    
    # Save to CSV
    csv_path = f"2025_ACL_SpeechEMD_proj_code/results/{dataset_name}/aggregated/aggregated_classification_report.csv"
    report_df.to_csv(csv_path)
    
    # Save human-readable version
    txt_path = f"2025_ACL_SpeechEMD_proj_code/results/{dataset_name}/aggregated/aggregated_classification_report.txt"
    with open(txt_path, "w") as f:
        f.write("=== Aggregated Classification Report Across All Folds ===\n")
        f.write(f"Mean ± Standard Deviation for {len(all_reports)} folds\n\n")
        f.write(report_df.to_string())
    
    print(f"[INFO] Aggregated classification report saved to: {csv_path} and {txt_path}")
    return report_df
    
def run_n_fold_cross_validation(dataset, dataset_name):
    global NUM_CLASSES
    global FEATURES_COUNT
    if dataset_name in ["EmoDB", "CremaD"]:
        NUM_CLASSES = 6
    else:
        NUM_CLASSES = 7    

    # Load data
    vs_code = 1
    if vs_code == 1:
        print(f"reading dataset {dataset_name}")
        train_file = "2025_ACL_SpeechEMD_proj_code/datasetFeature/" + dataset + ".feat"
    else:
        train_file = "D:\\Python\\SpeechEM\\Features_deepCNF\\" + dataset + ".feat"
    
    #train_file = f"D:\\Python\\SpeechEM\\Features_deepCNF\\{dataset}.feat"
    X, y = dcrf.load_data(train_file)
    
    # Define emotion labels based on dataset
    if dataset_name in ["EmoDB", "CremaD"]:
        emotion_labels = ['Neutral', 'Happy', 'Sad', 'Angry', 'Fear', 'Disgust']
    else:
        emotion_labels = ['Neutral', 'Happy', 'Sad', 'Angry', 'Fear', 'Disgust', 'Surprise']
    
    # Initialize k-fold cross-validation
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    
    # Store results
    fold_results = {
        'test_acc': [],
        'test_loss': [],
        'weighted_acc': [],
        'class_accuracies': [],
        'class_counts': [],
        'histories': [],
        'confusion_matrices': [],
        'classification_reports': []
    }
    
    best_model = None
    best_val_acc = 0
    all_reports = []  # To store classification reports from all folds
    all_histories = []  # To store training histories from all folds
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"\n{'='*40}")
        print(f"Fold {fold + 1}/{N_FOLDS}")
        print(f"{'='*40}")
        
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # # Further split training data into train/val
        # X_train, X_val, y_train, y_val = train_test_split(
        #     X_train, y_train, test_size=0.20, random_state=42, shuffle=True,  stratify=y_train
        # )
        
        # Preprocessing
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        #X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        
        # Apply PCA
        pca = PCA(n_components=0.99)
        X_train = pca.fit_transform(X_train)
        #X_val = pca.transform(X_val)
        X_test = pca.transform(X_test)
        
        # Update features count
        #global FEATURES_COUNT
        FEATURES_COUNT = X_train.shape[1]
        
        # Reshape for LSTM/GRU
        X_train = X_train.reshape(X_train.shape[0], 1, FEATURES_COUNT)
        #X_val = X_val.reshape(X_val.shape[0], 1, FEATURES_COUNT)
        X_test = X_test.reshape(X_test.shape[0], 1, FEATURES_COUNT)
        
        # Build and train model
        model = dcrf.build_model(FEATURES_COUNT, model_type=MODEL_TYPE, activation=ACTIVATION_FN)
        #model, history = dcrf.train_model(model, X_train, y_train, X_val, y_val, epochs=500)
        model, history = dcrf.train_model(model, X_train, y_train, X_test, y_test, epochs=500)
        
        # After model training:
        all_histories.append(history.history)
        # Plot training curves
        #plot_training_curves(history, fold+1, dataset_name)
        
        # Evaluate on test set
        test_acc, test_loss = dcrf.evaluate_model(model, X_test, y_test)
        print(f"Fold {fold + 1} Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")
        
        # Calculate weighted accuracy
        y_pred = model.predict(X_test)
        weighted_acc, class_accs, class_counts = dcrf.calculate_weighted_accuracy_details(y_test, y_pred, dataset_name, NUM_CLASSES)

        #weighted_acc, class_accs = dcrf.calculate_weighted_accuracy(y_test, y_pred, dataset_name)
        print(f"Fold {fold + 1} Weighted Accuracy: {weighted_acc:.4f}")
        
        # Generate and save confusion matrix
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_classes = np.argmax(to_categorical(y_test, NUM_CLASSES), axis=1) if y_test.ndim == 1 else np.argmax(y_test, axis=1)
        save_confusion_matrix_withEmo(y_test_classes, y_pred_classes, emotion_labels, fold+1, dataset_name)
        
        # After model training and evaluation:
        report = save_classification_metrics(model, X_test, y_test, f"2025_ACL_SpeechEMD_proj_code/results/{dataset_name}/classification_reports/fold_{fold+1}_report.txt")
        all_reports.append(report)
                
        # Store results
        fold_results['test_acc'].append(test_acc)
        fold_results['test_loss'].append(test_loss)
        fold_results['weighted_acc'].append(weighted_acc)
        fold_results['class_accuracies'].append(class_accs)
        fold_results['class_counts'].append(class_counts)
        fold_results['histories'].append(history.history)
        # Ensure we generate square confusion matrix
        cm = confusion_matrix(y_test_classes, y_pred_classes, labels=range(len(emotion_labels)))
        #fold_results['confusion_matrices'].append(confusion_matrix(y_test_classes, y_pred_classes))
        fold_results['confusion_matrices'].append(cm)
        fold_results['classification_reports'].append(report)
        
        # Track best model
        max_val_acc = max(history.history['val_accuracy'])
        if max_val_acc > best_val_acc:
            best_val_acc = max_val_acc
            best_model = model
    
    # Calculate and print final statistics
    print("\nFinal Cross-Validation Results:")
    print(f"Average Test Accuracy: {np.mean(fold_results['test_acc']):.4f} ± {np.std(fold_results['test_acc']):.4f}")
    print(f"Average Weighted Accuracy: {np.mean(fold_results['weighted_acc']):.4f} ± {np.std(fold_results['weighted_acc']):.4f}")
    print(f"Average Test Loss: {np.mean(fold_results['test_loss']):.4f} ± {np.std(fold_results['test_loss']):.4f}")
    
    print(f"Highest Test Accuracy: {np.max(fold_results['test_acc']):.4f}")
    print(f"Highest Weighted Accuracy: {np.max(fold_results['weighted_acc']):.4f}")

    # Save aggregated confusion matrix
    save_aggregated_confusion_matrix(fold_results['confusion_matrices'], emotion_labels, dataset_name)
    
    # Save all results
    save_results(fold_results, emotion_labels, dataset_name)

    # After all folds are complete:
    save_aggregated_classification_report(all_reports, dataset_name)    
    
    # After all folds are complete:
    plot_aggregated_training_curves(all_histories, dataset_name)

    # reset FEATURES_COUNT to load files
    #global FEATURES_COUNT
    FEATURES_COUNT = FEATURES_COUNT_G
    
    return best_model, fold_results

def save_aggregated_confusion_matrix(confusion_matrices, class_labels, dataset_name):
    """Save an averaged confusion matrix across all folds"""
    # First ensure all confusion matrices have the same shape
    target_shape = (len(class_labels), len(class_labels))
    
    # Pad or trim confusion matrices to the target shape
    processed_cms = []
    for cm in confusion_matrices:
        if cm.shape != target_shape:
            # Create a new matrix of target shape
            new_cm = np.zeros(target_shape)
            # Copy available values
            min_rows = min(cm.shape[0], target_shape[0])
            min_cols = min(cm.shape[1], target_shape[1])
            new_cm[:min_rows, :min_cols] = cm[:min_rows, :min_cols]
            processed_cms.append(new_cm)
        else:
            processed_cms.append(cm)
    
    # Now average the properly shaped matrices
    avg_cm = np.mean(processed_cms, axis=0)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(avg_cm, annot=True, fmt='.1f', cmap='Blues', 
                xticklabels=class_labels, 
                yticklabels=class_labels)
    plt.title('Average Confusion Matrix Across All Folds')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    os.makedirs(f"2025_ACL_SpeechEMD_proj_code/results/{dataset_name}/aggregated", exist_ok=True)
    plt.savefig(f"2025_ACL_SpeechEMD_proj_code/results/{dataset_name}/aggregated/average_confusion_matrix.png", 
                bbox_inches='tight')
    plt.close()
    
    # Save to CSV
    avg_cm_df = pd.DataFrame(avg_cm, index=class_labels, columns=class_labels)
    avg_cm_df.to_csv(f"2025_ACL_SpeechEMD_proj_code/results/{dataset_name}/aggregated/average_confusion_matrix.csv")
# For aggregation across folds:
def aggregate_class_accuracies(fold_results, num_classes):
    # Initialize arrays
    all_accs = np.zeros((len(fold_results['class_accuracies']), num_classes))
    all_counts = np.zeros((len(fold_results['class_counts']), num_classes))
    
    # Fill arrays
    for i, (accs, counts) in enumerate(zip(fold_results['class_accuracies'], fold_results['class_counts'])):
        all_accs[i] = accs
        all_counts[i] = counts
    
    # Calculate weighted mean across folds
    total_counts = np.sum(all_counts, axis=0)
    avg_class_accs = np.sum(all_accs * all_counts, axis=0) / np.where(total_counts > 0, total_counts, 1)
    
    return avg_class_accs, total_counts

def save_results(results, class_labels, dataset_name):
    """Save all results to files"""
    # Create output directory
    os.makedirs(f"2025_ACL_SpeechEMD_proj_code/results/{dataset_name}", exist_ok=True)
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        'fold': range(1, N_FOLDS+1),
        'test_accuracy': results['test_acc'],
        'test_loss': results['test_loss'],
        'weighted_accuracy': results['weighted_acc']
    })
    metrics_df.to_csv(f"2025_ACL_SpeechEMD_proj_code/results/{dataset_name}/metrics.csv", index=False)
    
    # Save class accuracies
    class_acc_df = pd.DataFrame(results['class_accuracies'], 
                              columns=[f'class_{i}' for i in range(NUM_CLASSES)])
    class_acc_df['fold'] = range(1, N_FOLDS+1)
    class_acc_df.to_csv(f"2025_ACL_SpeechEMD_proj_code/results/{dataset_name}/class_accuracies.csv", index=False)
    
    # Save summary statistics
    with open(f"2025_ACL_SpeechEMD_proj_code/results/{dataset_name}/summary.txt", "w") as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Number of folds: {N_FOLDS}\n")
        f.write(f"Average Test Accuracy: {np.mean(results['test_acc']):.4f} ± {np.std(results['test_acc']):.4f}\n")
        f.write(f"Average Weighted Accuracy: {np.mean(results['weighted_acc']):.4f} ± {np.std(results['weighted_acc']):.4f}\n")
        f.write(f"Average Test Loss: {np.mean(results['test_loss']):.4f} ± {np.std(results['test_loss']):.4f}\n\n")
        f.write(f"Highest Test Accuracy: {np.max(results['test_acc']):.4f}\n")
        f.write(f"Highest Weighted Accuracy: {np.max(results['weighted_acc']):.4f}\n")
        
        f.write("Average Class-wise Accuracies:\n")
        avg_class_accs, total_counts = aggregate_class_accuracies(results, NUM_CLASSES)
        for i, acc in enumerate(avg_class_accs):
            f.write(f"Class {i}: {acc:.4f}\n")

# Main function
def train_DeepCRF(dataset, dataset_name):
    best_model, results = run_n_fold_cross_validation(dataset, dataset_name)
    
    # Save the best model
    # if best_model:
    #     os.makedirs("2025_ACL_SpeechEMD_proj_code/models", exist_ok=True)
    #     best_model.save(f"2025_ACL_SpeechEMD_proj_code/models/{dataset_name}_best_model.h5")
    
    return best_model, results
