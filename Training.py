import os
import numpy as np
import pandas as pd
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, Flatten, MaxPooling1D, Input
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import permutations


def process_file(file_path):
    """
    Process individual spectral data files and extract intensity values
    
    This function reads tab-separated spectral data files and extracts the intensity
    values from the second column, which contains the SERS spectral measurements.
    
    Args:
        file_path (str): Path to the spectral data file (.txt format)
    
    Returns:
        numpy.ndarray: Array of spectral intensity values from the second column
        None: If file processing fails
    """
    try:
        # Read tab-separated file without header, expecting two columns: wavenumber and intensity
        spectrum = pd.read_csv(file_path, sep='\t', header=None)
        # Return the second column (index 1) containing spectral intensity values
        return spectrum[1].values
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None


def create_unique_pairs(files, class_label):
    """
    Generate unique concatenated pairs from spectral files for data augmentation
    
    This function creates all possible permutations of file pairs within a cancer type
    and concatenates their spectral data to create augmented training samples. This
    approach increases the diversity of training data by combining different patient
    samples from the same cancer type.
    
    Args:
        files (list): List of file paths for a specific cancer type
        class_label (int): Integer label representing the cancer type
    
    Returns:
        tuple: (data_list, labels_list)
            - data_list: List of concatenated spectral arrays
            - labels_list: List of corresponding class labels
    """
    data = []
    labels = []
    seen_combinations = set()  # Track processed combinations to avoid duplicates
    
    # Generate all possible permutations of 2 files from the current cancer type
    for (file1, file2) in permutations(files, 2):
        # Skip if this combination has already been processed
        if (file1, file2) not in seen_combinations:
            seen_combinations.add((file1, file2))
            
            # Process both files to extract spectral data
            data1 = process_file(file1)
            data2 = process_file(file2)
            
            # Only proceed if both files were successfully processed
            if data1 is not None and data2 is not None:
                # Create two concatenated combinations for enhanced data augmentation
                data.append(np.concatenate((data1, data2)))  # AB combination
                data.append(np.concatenate((data2, data1)))  # BA combination
                # Add corresponding class labels for both combinations
                labels.extend([class_label, class_label])
    
    return data, labels


def load_data(base_path, target_samples_per_class, max_samples_per_class=None, training=True):
    """
    Load and prepare spectral data with strict train/test splitting
    
    This function loads spectral data from a directory structure where each subdirectory
    represents a different cancer type. It performs a strict 80/20 split at the file level
    to ensure no data leakage between training and testing sets.
    
    Args:
        base_path (str): Root directory containing cancer type subdirectories
        target_samples_per_class (int): Target number of sample pairs per cancer type
        max_samples_per_class (int, optional): Maximum number of files to process per cancer type
        training (bool): If True, load training data; if False, load validation data
    
    Returns:
        tuple: (data, labels, class_map)
            - data: numpy array of spectral data
            - labels: numpy array of class labels
            - class_map: dictionary mapping class names to indices
    """
    data = []
    labels = []
    class_map = {}  # Maps cancer type names to integer indices
    class_idx = 0
    
    # Iterate through each subdirectory (cancer type)
    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        
        # Process only directories (each represents a cancer type)
        if os.path.isdir(folder_path):
            # Map cancer type name to class index
            class_map[folder] = class_idx
            
            # Collect all .txt files containing spectral data
            all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.txt')]
            
            # Limit number of files per class if specified (for memory management)
            if max_samples_per_class is not None and len(all_files) > max_samples_per_class:
                # Randomly sample maximum number of files to prevent bias
                selected_files = random.sample(all_files, max_samples_per_class)
            else:
                selected_files = all_files

            # Perform strict train/test split at file level (80/20 split)
            split_point = int(len(selected_files) * 0.8)
            if training:
                # Use first 80% of files for training
                selected_files = selected_files[:split_point]
            else:
                # Use remaining 20% of files for validation/testing
                selected_files = selected_files[split_point:]

            # Generate sample pairs based on training/validation mode
            if training:
                # Create training data pairs from training files
                train_data, train_labels = create_unique_pairs(selected_files, class_idx)
                # Limit to target number of samples per class
                data.extend(train_data[:target_samples_per_class])
                labels.extend(train_labels[:target_samples_per_class])
            else:
                # Create validation data pairs from validation files
                val_data, val_labels = create_unique_pairs(selected_files, class_idx)
                # Limit to target number of samples per class
                data.extend(val_data[:target_samples_per_class])
                labels.extend(val_labels[:target_samples_per_class])

            class_idx += 1
            print(f"Processed {folder}: {len(data)} data points.")

    # Convert lists to numpy arrays for efficient computation
    data = np.array(data)
    labels = np.array(labels)
    print(f"Total data points prepared: {len(data)}, Unique labels: {np.unique(labels)}")
    
    return data, labels, class_map


def build_complex_model(input_shape, num_classes):
    """
    Build a Convolutional Neural Network for spectral data classification
    
    This function creates a 1D CNN architecture optimized for SERS spectral analysis.
    The model uses convolutional layers to extract spectral features, followed by
    dense layers for classification.
    
    Args:
        input_shape (int): Length of the input spectral data
        num_classes (int): Number of cancer types to classify
    
    Returns:
        tensorflow.keras.Model: Compiled CNN model ready for training
    """
    # Build sequential CNN model for spectral classification
    model = Sequential([
        # Input layer: expects 1D spectral data with single channel
        Input(shape=(input_shape, 1)),
        
        # First convolutional block: extract low-level spectral features
        Conv1D(64, 3, activation='relu', padding='same'),  # 64 filters, kernel size 3
        MaxPooling1D(2),  # Downsample by factor of 2
        
        # Second convolutional block: extract higher-level spectral patterns
        Conv1D(128, 3, activation='relu', padding='same'),  # 128 filters, kernel size 3
        MaxPooling1D(2),  # Further downsampling
        
        # Flatten convolutional output for dense layers
        Flatten(),
        
        # Dense layers for final classification
        Dense(256, activation='relu'),  # Fully connected layer with 256 neurons
        Dropout(0.3),  # Regularization to prevent overfitting
        
        # Output layer: softmax activation for multi-class classification
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile model with appropriate optimizer, loss function, and metrics
    model.compile(
        optimizer=Adam(0.001),  # Adam optimizer with learning rate 0.001
        loss='categorical_crossentropy',  # Multi-class classification loss
        metrics=['accuracy']  # Track accuracy during training
    )
    
    return model


def main():
    """
    Main training function that orchestrates the complete training pipeline
    
    This function executes the entire training process:
    1. Configure GPU settings for optimal performance
    2. Load and preprocess training and validation data
    3. Build and train the CNN model
    4. Evaluate model performance with various metrics
    5. Generate visualization plots and save results
    6. Save the trained model for future inference
    """
    
    # ==================== CONFIGURATION SECTION ====================
    print("=== TRAINING CONFIGURATION ===")
    
    # Define data paths and training parameters
    base_path = 'C:\\Users\\Administrator\\Desktop\\Train_data'  # Training data directory
    target_samples_per_class = 100000  # Target number of sample pairs per cancer type
    max_samples_per_class = 100  # Maximum number of files to process per cancer type
    
    print(f"Training data path: {base_path}")
    print(f"Target samples per class: {target_samples_per_class}")
    print(f"Max files per class: {max_samples_per_class}")

    # ==================== GPU CONFIGURATION ====================
    print("\n=== GPU CONFIGURATION ===")
    
    # Configure GPU usage for accelerated training
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth to prevent GPU memory allocation issues
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ {len(gpus)} GPU(s) are available and configured for training.")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    else:
        print("⚠ No GPU found, using CPU for training (this will be slower).")

    # ==================== DATA LOADING SECTION ====================
    print("\n=== DATA LOADING ===")
    
    # Load training data (80% of files from each cancer type)
    print("Loading training data...")
    data_train, labels_train, class_map = load_data(
        base_path, 
        target_samples_per_class, 
        max_samples_per_class,
        training=True
    )
    
    # Load validation data (20% of files from each cancer type)
    print("Loading validation data...")
    data_val, labels_val, _ = load_data(
        base_path, 
        target_samples_per_class, 
        max_samples_per_class, 
        training=False
    )

    # Validate that data was successfully loaded
    if data_train.size == 0 or data_val.size == 0:
        print("✗ ERROR: No valid data to process. Check data directory and file formats.")
        return

    print(f"✓ Training data loaded: {data_train.shape[0]} samples")
    print(f"✓ Validation data loaded: {data_val.shape[0]} samples")
    print(f"✓ Number of cancer types: {len(class_map)}")
    print(f"✓ Cancer types: {list(class_map.keys())}")

    # ==================== DATA PREPROCESSING ====================
    print("\n=== DATA PREPROCESSING ===")
    
    # Add channel dimension for 1D convolution (samples, features, channels)
    data_train = np.expand_dims(data_train, -1)
    data_val = np.expand_dims(data_val, -1)
    print(f"✓ Training data shape after preprocessing: {data_train.shape}")
    print(f"✓ Validation data shape after preprocessing: {data_val.shape}")

    # Convert integer labels to categorical (one-hot encoded) format
    labels_train = tf.keras.utils.to_categorical(labels_train, num_classes=len(class_map))
    labels_val = tf.keras.utils.to_categorical(labels_val, num_classes=len(class_map))
    print(f"✓ Labels converted to categorical format: {labels_train.shape}")

    # ==================== MODEL BUILDING AND TRAINING ====================
    print("\n=== MODEL BUILDING AND TRAINING ===")
    
    # Build the CNN model architecture
    print("Building CNN model...")
    model = build_complex_model(data_train.shape[1], len(class_map))
    
    # Display model architecture summary
    print("Model Architecture:")
    model.summary()
    
    # Train the model with validation monitoring
    print(f"\nStarting training for 6 epochs...")
    history = model.fit(
        data_train, labels_train,  # Training data and labels
        epochs=6,  # Number of training epochs
        batch_size=128,  # Batch size for training
        validation_data=(data_val, labels_val),  # Validation data for monitoring
        verbose=2  # Detailed progress output
    )
    
    print("✓ Model training completed successfully!")

    # ==================== MODEL EVALUATION ====================
    print("\n=== MODEL EVALUATION ===")
    
    # Generate predictions on validation data
    print("Generating predictions on validation data...")
    y_pred = model.predict(data_val)
    y_pred_classes = np.argmax(y_pred, axis=1)  # Convert probabilities to class predictions
    y_true = np.argmax(labels_val, axis=1)  # Convert one-hot labels back to integers
    
    print(f"✓ Predictions generated for {len(y_pred)} validation samples")

    # ==================== CLASSIFICATION REPORT ====================
    print("\n=== CLASSIFICATION REPORT ===")
    
    # Generate detailed classification metrics
    cm = confusion_matrix(y_true, y_pred_classes)
    clr = classification_report(y_true, y_pred_classes, target_names=list(class_map.keys()))
    print("Detailed Classification Report:")
    print("=" * 60)
    print(clr)

    # Save classification report to CSV for further analysis
    report_df = pd.DataFrame.from_dict(
        classification_report(y_true, y_pred_classes, target_names=list(class_map.keys()), output_dict=True)
    )
    report_path = os.path.join(base_path, 'classification_report.csv')
    report_df.to_csv(report_path)
    print(f"✓ Classification report saved to: {report_path}")

    # ==================== CONFUSION MATRIX VISUALIZATION ====================
    print("\n=== CONFUSION MATRIX VISUALIZATION ===")
    
    # Create and save confusion matrix heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True,  # Show numeric values in cells
        fmt="d",  # Integer format
        xticklabels=list(class_map.keys()),  # Cancer type names on x-axis
        yticklabels=list(class_map.keys())   # Cancer type names on y-axis
    )
    
    # Configure plot appearance
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.title('Confusion Matrix - Cancer Type Classification', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save confusion matrix plot
    cm_path = os.path.join(base_path, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to: {cm_path}")
    plt.show()

    # ==================== ROC CURVE ANALYSIS ====================
    print("\n=== ROC CURVE ANALYSIS ===")
    
    # Compute ROC curve and AUC for each cancer type (multi-class)
    print("Computing ROC curves for each cancer type...")
    fpr = dict()  # False Positive Rate for each class
    tpr = dict()  # True Positive Rate for each class
    roc_auc = dict()  # Area Under Curve for each class
    
    # Calculate ROC metrics for each cancer type
    for i in range(len(class_map)):
        fpr[i], tpr[i], _ = roc_curve(labels_val[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Create ROC curve visualization
    plt.figure(figsize=(10, 8))
    
    # Plot ROC curve for each cancer type
    for i, class_name in enumerate(class_map.keys()):
        plt.plot(
            fpr[i], tpr[i], 
            linewidth=2,
            label=f'{class_name} (AUC = {roc_auc[i]:.2f})'
        )

    # Add diagonal reference line (random classifier)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
    
    # Configure plot appearance
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title('ROC Curves for Multi-Class Cancer Detection', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save ROC curves plot
    roc_path = os.path.join(base_path, 'roc_curves.png')
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    print(f"✓ ROC curves saved to: {roc_path}")
    plt.show()

    # ==================== MODEL SAVING ====================
    print("\n=== MODEL SAVING ===")
    
    # Save the trained model using TensorFlow's SavedModel format
    model_save_path = os.path.join(base_path, 'saved_model')
    model.save(model_save_path, save_format='tf')
    print(f"✓ Model saved successfully at: {model_save_path}")
    
    # Display model size information
    model_size = sum(os.path.getsize(os.path.join(dirpath, filename)) 
                     for dirpath, dirnames, filenames in os.walk(model_save_path) 
                     for filename in filenames) / (1024 * 1024)  # Convert to MB
    print(f"✓ Model size: {model_size:.2f} MB")
    
    # ==================== TRAINING SUMMARY ====================
    print("\n" + "="*60)
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"📊 Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"📈 Training epochs completed: 6")
    print(f"🎯 Cancer types classified: {len(class_map)}")
    print(f"💾 Model saved for inference: {model_save_path}")
    print("="*60)


if __name__ == '__main__':
    # Execute the main training pipeline
    main()
