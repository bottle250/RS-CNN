import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import permutations
import random
from sklearn.manifold import TSNE
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from numpy import interp
import plotly.graph_objects as go
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Set matplotlib global font to Times New Roman for consistent scientific publication formatting
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'  # Mathematical font setting compatible with Times New Roman

def process_file(file_path):
    """
    Process individual spectral data files
    
    Args:
        file_path: Path to the spectral data file (.txt format)
    
    Returns:
        numpy array: Spectral intensity values from the second column of the file
    """
    try:
        # Read tab-separated file without header, expecting spectral data format
        spectrum = pd.read_csv(file_path, sep='\t', header=None)
        # Return the second column (index 1) which contains the spectral intensity values
        return spectrum[1].values
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None


def create_unique_pairs(files, class_label):
    """
    Create unique pairs from spectral files for data augmentation
    
    This function generates concatenated spectral pairs (AB and BA) from all possible
    combinations of files within a class to increase training data diversity
    
    Args:
        files: List of file paths for a specific cancer type
        class_label: Integer label for the cancer type
    
    Returns:
        tuple: (data_list, labels_list) where data contains concatenated spectral pairs
    """
    data = []
    labels = []
    seen_combinations = set()
    
    # Generate all possible permutations of 2 files from the file list
    for (file1, file2) in permutations(files, 2):
        # Avoid duplicate combinations to ensure unique pairs
        if (file1, file2) not in seen_combinations:
            seen_combinations.add((file1, file2))
            
            # Process both files to get spectral data
            data1 = process_file(file1)
            data2 = process_file(file2)
            
            # Only proceed if both files were successfully processed
            if data1 is not None and data2 is not None:
                # Create concatenated pairs: AB and BA for data augmentation
                data.append(np.concatenate((data1, data2)))  # AB combination
                data.append(np.concatenate((data2, data1)))  # BA combination
                # Add corresponding class labels for both combinations
                labels.extend([class_label, class_label])
    
    return data, labels


def load_all_data(base_path, max_samples_per_class=None, samples_per_class=None):
    """
    Load all spectral data from directory structure without train/test split
    
    Expected directory structure:
    base_path/
    ├── Cancer_Type_1/
    │   ├── sample1.txt
    │   ├── sample2.txt
    │   └── ...
    ├── Cancer_Type_2/
    │   ├── sample1.txt
    │   └── ...
    └── ...
    
    Args:
        base_path: Root directory containing cancer type subdirectories
        max_samples_per_class: Maximum number of files to use per cancer type
        samples_per_class: Maximum number of sample pairs to generate per cancer type
    
    Returns:
        tuple: (data, labels, class_map, idx_to_class)
    """
    data = []
    labels = []
    class_map = {}  # Maps class names to integer indices
    class_idx = 0
    
    # Iterate through each subdirectory (cancer type)
    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        
        # Process only directories (each represents a cancer type)
        if os.path.isdir(folder_path):
            # Map folder name to class index for classification
            class_map[folder] = class_idx
            
            # Collect all .txt files containing spectral data
            all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.txt')]
            
            # Limit number of files per class if specified (for memory management)
            if max_samples_per_class is not None and len(all_files) > max_samples_per_class:
                selected_files = random.sample(all_files, max_samples_per_class)
            else:
                selected_files = all_files
            
            # Generate sample pairs for current cancer type
            class_data, class_labels = create_unique_pairs(selected_files, class_idx)
            
            # Limit number of sample pairs if specified (for computational efficiency)
            if samples_per_class is not None and len(class_data) > samples_per_class:
                # Take first N samples
                class_data = class_data[:samples_per_class]
                class_labels = class_labels[:samples_per_class]
            
            # Add processed data to global dataset
            data.extend(class_data)
            labels.extend(class_labels)
            
            class_idx += 1
            print(f"Processed {folder}: {len(class_data)} data points")
    
    # Convert lists to numpy arrays for efficient computation
    data = np.array(data)
    labels = np.array(labels)
    print(f"Total data points: {len(data)}, Number of classes: {len(class_map)}, Classes: {list(class_map.keys())}")
    
    # Create reverse mapping for result visualization
    idx_to_class = {v: k for k, v in class_map.items()}
    
    return data, labels, class_map, idx_to_class


def plot_2d_lda_scatter(data, labels, class_map,
                        font_size=12, label_font_size=14, label_position='top left',
                        scatter_colors=None, figure_width=800, figure_height=600, resolution=300,
                        font_family='Times New Roman', output_dir=None):
    """
    Create 2D LDA scatter plot for data visualization
    
    This function performs Linear Discriminant Analysis (LDA) dimensionality reduction
    and creates an interactive scatter plot to visualize class separation
    
    Args:
        data: Input spectral data array
        labels: Class labels for each data point
        class_map: Dictionary mapping class names to indices
        font_size: Font size for plot elements
        label_font_size: Font size for axis labels
        label_position: Position of legend ('top left', 'top right', etc.)
        scatter_colors: Custom color palette for different classes
        figure_width: Plot width in pixels
        figure_height: Plot height in pixels
        resolution: Image resolution for saving
        font_family: Font family for text elements
        output_dir: Directory to save the plot
    
    Returns:
        plotly figure object or None if matplotlib fallback is used
    """
    try:
        print("Starting LDA scatter plot generation...")
        print(f"Data shape: {data.shape}, Labels shape: {labels.shape}, Number of classes: {len(class_map)}")
        
        # Validate input data
        if data.size == 0 or labels.size == 0:
            print("Error: Input data or labels are empty")
            return None
            
        # Set sample size per class for balanced visualization
        sample_size_per_class = 2000
        sampled_data = []
        sampled_labels = []

        # Perform stratified sampling to ensure balanced representation
        for class_idx in range(len(class_map)):
            class_data = data[labels == class_idx]
            print(f"Class {list(class_map.keys())[class_idx]} has {len(class_data)} samples")
            
            if len(class_data) == 0:
                print(f"Warning: Class {list(class_map.keys())[class_idx]} has no samples")
                continue
                
            # Random sampling if class has more samples than required
            if class_data.shape[0] > sample_size_per_class:
                idx = np.random.choice(class_data.shape[0], sample_size_per_class, replace=False)
                sampled_data.extend(class_data[idx])
                sampled_labels.extend([class_idx] * sample_size_per_class)
            else:
                # Use all available samples if less than required
                sampled_data.extend(class_data)
                sampled_labels.extend([class_idx] * class_data.shape[0])

        # Convert to numpy arrays for processing
        sampled_data = np.array(sampled_data)
        sampled_labels = np.array(sampled_labels)
        
        print(f"After sampling - Data shape: {sampled_data.shape}, Labels shape: {sampled_labels.shape}")

        # Perform dimensionality reduction using LDA or PCA as fallback
        try:
            print("Attempting LDA dimensionality reduction...")
            lda = LDA(n_components=2)  # Reduce to 2D for visualization
            X_r2 = lda.fit_transform(sampled_data, sampled_labels)
            print(f"LDA reduction successful - Output shape: {X_r2.shape}")
        except Exception as e:
            print(f"LDA reduction failed: {e}")
            print("Trying PCA as fallback...")
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            X_r2 = pca.fit_transform(sampled_data)
            print(f"PCA reduction successful - Output shape: {X_r2.shape}")

        class_labels = list(class_map.keys())  # Extract class names

        # Define color palette for different classes
        if scatter_colors is None:
            scatter_colors = [
                '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'
            ]  # Standard color palette for up to 8 classes
            
        # Create interactive plot using Plotly
        try:
            print("Creating Plotly scatter plot...")
            fig = go.Figure()
            
            # Add scatter trace for each class
            for color, i, label in zip(scatter_colors, range(len(class_labels)), class_labels):
                mask = sampled_labels == i
                if np.sum(mask) > 0:  # Ensure class has data points
                    fig.add_trace(go.Scatter(
                        x=X_r2[mask, 0],
                        y=X_r2[mask, 1],
                        mode='markers',
                        marker=dict(size=7, color=color, opacity=0.7),
                        name=label
                    ))

            # Configure plot layout with scientific formatting
            fig.update_layout(
                xaxis_title='Component 1',
                yaxis_title='Component 2',
                title='2D Scatter Plot of LDA Reduced Data',
                margin=dict(l=0, r=0, b=0, t=40),
                template='plotly_white',
                width=figure_width,
                height=figure_height,
                font=dict(size=font_size, family=font_family)
            )

            # Set axis label font sizes
            fig.update_xaxes(title_font=dict(size=label_font_size, family=font_family))
            fig.update_yaxes(title_font=dict(size=label_font_size, family=font_family))
            fig.update_traces(textfont=dict(size=label_font_size, family=font_family))

            # Position legend according to user preference
            if label_position == 'top left':
                fig.update_layout(legend=dict(x=0, y=1))
            elif label_position == 'top right':
                fig.update_layout(legend=dict(x=1, y=1))
            elif label_position == 'bottom left':
                fig.update_layout(legend=dict(x=0, y=0))
            elif label_position == 'bottom right':
                fig.update_layout(legend=dict(x=1, y=0))

            # Display the interactive plot
            try:
                fig.show()
            except Exception as e:
                print(f"Cannot display Plotly figure: {e}")
                
            # Save the plot if output directory is specified
            if output_dir:
                try:
                    output_path = os.path.join(output_dir, 'lda_visualization.png')
                    print(f"Saving scatter plot to: {output_path}")
                    fig.write_image(output_path, scale=2)
                    print("Scatter plot saved successfully")
                except Exception as e:
                    print(f"Failed to save Plotly figure: {e}")
                    print("Trying matplotlib as fallback for saving...")
                    
                    # Matplotlib fallback for saving
                    plt.figure(figsize=(10, 8))
                    for i, label in enumerate(class_labels):
                        mask = sampled_labels == i
                        if np.sum(mask) > 0:
                            plt.scatter(X_r2[mask, 0], X_r2[mask, 1], 
                                      color=scatter_colors[i % len(scatter_colors)], 
                                      label=label, alpha=0.7, s=30)
                    
                    plt.xlabel('Component 1', fontsize=label_font_size, fontfamily=font_family)
                    plt.ylabel('Component 2', fontsize=label_font_size, fontfamily=font_family)
                    plt.title('2D Scatter Plot of LDA Reduced Data', fontsize=label_font_size+2, fontfamily=font_family)
                    plt.legend(fontsize=font_size)
                    plt.grid(True, linestyle='--', alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(output_path, dpi=300)
                    plt.close()
                    print("Successfully saved scatter plot using matplotlib")
            
            return fig
            
        except Exception as e:
            print(f"Plotly plotting error: {e}")
            print("Using matplotlib as fallback for plotting...")
            
            # Matplotlib fallback plotting method
            plt.figure(figsize=(10, 8))
            for i, label in enumerate(class_labels):
                mask = sampled_labels == i
                if np.sum(mask) > 0:
                    plt.scatter(X_r2[mask, 0], X_r2[mask, 1], 
                              color=scatter_colors[i % len(scatter_colors)], 
                              label=label, alpha=0.7, s=30)
            
            plt.xlabel('Component 1', fontsize=label_font_size, fontfamily=font_family)
            plt.ylabel('Component 2', fontsize=label_font_size, fontfamily=font_family)
            plt.title('2D Scatter Plot of LDA Reduced Data', fontsize=label_font_size+2, fontfamily=font_family)
            plt.legend(fontsize=font_size)
            plt.grid(True, linestyle='--', alpha=0.3)
            
            if output_dir:
                output_path = os.path.join(output_dir, 'lda_visualization.png')
                plt.savefig(output_path, dpi=300)
                print(f"Saved scatter plot using matplotlib to: {output_path}")
            
            plt.show()
            return None
            
    except Exception as e:
        print(f"Error occurred during scatter plot generation: {e}")
        return None


def main():
    """
    Main function to execute the complete cancer prediction pipeline
    
    This function orchestrates the entire process:
    1. Load test data from directory structure
    2. Load pre-trained CNN model
    3. Perform predictions on test data
    4. Generate evaluation metrics and visualizations
    """
    
    # ==================== CONFIGURATION SECTION ====================
    print("=== CONFIGURATION ===")
    
    # Define paths for test data and pre-trained model
    base_path = r"D:\Year4\Paper1\New_data"  # Test data directory path
    model_path = r"E:\Single test-based diagnosis of multiple cancer types using Exosome-SERS-AI for early stage cancers\SURF\TTTRRRRYYYY\MSC-CNN\450\99.42%\saved_model"  # Pre-trained model path
    
    # Set data processing limits for memory and computation management
    max_samples_per_class = 1000  # Maximum number of files per cancer type
    samples_per_class = 100000  # Maximum number of sample pairs per cancer type
    
    # Configure output directory for saving results
    output_dir = r"E:\Single test-based diagnosis of multiple cancer types using Exosome-SERS-AI for early stage cancers\SURF\TTTRRRRYYYY\Confusion matrix\New-data"
    os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist
    print(f"Results will be saved to: {output_dir}")
    
    # ==================== GPU CONFIGURATION ====================
    print("\n=== GPU CONFIGURATION ===")
    
    # Configure GPU settings for TensorFlow
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth to avoid GPU memory allocation issues
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"{len(gpus)} GPU(s) available and configured")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    else:
        print("No GPU found, using CPU for computation")
    
    # ==================== DATA LOADING SECTION ====================
    print("\n=== DATA LOADING ===")
    
    # Load all spectral data from the directory structure
    print("Loading spectral data from directory structure...")
    data, true_labels, class_map, idx_to_class = load_all_data(
        base_path, 
        max_samples_per_class=max_samples_per_class,
        samples_per_class=samples_per_class
    )
    
    # Validate that data was successfully loaded
    if data.size == 0:
        print("ERROR: No valid data found for processing")
        return
    
    print(f"Data loading completed: {data.shape[0]} samples, {data.shape[1]} features")
    
    # ==================== DATA PREPROCESSING ====================
    print("\n=== DATA PREPROCESSING ===")
    
    # Add channel dimension for CNN input (spectral_length, 1)
    data = np.expand_dims(data, -1)
    print(f"Data shape after adding channel dimension: {data.shape}")
    
    # ==================== MODEL LOADING ====================
    print("\n=== MODEL LOADING ===")
    
    # Load the pre-trained CNN model
    print(f"Loading pre-trained model from: {model_path}")
    try:
        model = tf.keras.models.load_model(model_path)
        print("✓ Model loaded successfully")
        print(f"Model input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    # ==================== PREDICTION SECTION ====================
    print("\n=== PREDICTION ===")
    
    # Perform batch prediction on all test data
    print("Performing predictions on test data...")
    predictions = model.predict(data)
    pred_labels = np.argmax(predictions, axis=1)  # Convert probabilities to class predictions
    print(f"✓ Predictions completed for {len(predictions)} samples")
    
    # ==================== EVALUATION METRICS ====================
    print("\n=== EVALUATION METRICS ===")
    
    # Generate class names for reporting
    class_names = [idx_to_class[i] for i in range(len(class_map))]
    
    # Compute confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    print("✓ Confusion matrix computed")
    
    # Generate and display detailed classification report
    print("\nClassification Report:")
    print("=" * 50)
    report = classification_report(true_labels, pred_labels, target_names=class_names)
    print(report)
    
    # Save classification report to CSV for further analysis
    report_dict = classification_report(true_labels, pred_labels, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame.from_dict(report_dict)
    report_path = os.path.join(output_dir, 'classification_report.csv')
    report_df.to_csv(report_path)
    print(f"✓ Classification report saved to: {report_path}")
    
    # ==================== CONFUSION MATRIX VISUALIZATION ====================
    print("\n=== CONFUSION MATRIX VISUALIZATION ===")
    
    # Create and save confusion matrix heatmap
    plt.figure(figsize=(10, 8))
    font_size = 12
    
    # Generate heatmap with annotations
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names)
    
    # Configure plot appearance with scientific formatting
    plt.ylabel('True Labels', fontfamily='Times New Roman', fontsize=font_size+2)
    plt.xlabel('Predicted Labels', fontfamily='Times New Roman', fontsize=font_size+2)
    plt.title('Confusion Matrix', fontfamily='Times New Roman', fontsize=font_size+4)
    
    # Set tick label fonts
    plt.xticks(fontfamily='Times New Roman', fontsize=font_size)
    plt.yticks(fontfamily='Times New Roman', fontsize=font_size)
    
    plt.tight_layout()
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300)
    print(f"✓ Confusion matrix saved to: {cm_path}")
    plt.show()
    
    # ==================== ROC CURVE ANALYSIS ====================
    print("\n=== ROC CURVE ANALYSIS ===")
    
    try:
        print("Computing ROC curves and AUC scores...")
        n_classes = len(class_map)
        
        # Define color palette for ROC curves
        colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        # Binarize labels for multi-class ROC analysis
        y_bin = label_binarize(true_labels, classes=list(range(n_classes)))
        
        # Use prediction probabilities as scores
        y_score = predictions
        
        # Compute ROC curve and AUC for each class
        fpr = dict()  # False Positive Rate
        tpr = dict()  # True Positive Rate
        roc_auc = dict()  # Area Under Curve

        # Create ROC plot with consistent formatting
        plt.figure(figsize=(8, 6))
        
        # Calculate and plot ROC curve for each cancer type
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            plt.plot(fpr[i], tpr[i], lw=1.5, color=colors[i % len(colors)], 
                     label=f'{class_names[i]} (AUC = {roc_auc[i]:0.2f})')

        # Configure ROC plot appearance
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontfamily='Times New Roman', fontsize=12)
        plt.ylabel('True Positive Rate', fontfamily='Times New Roman', fontsize=12)
        plt.title('Receiver Operating Characteristic', fontfamily='Times New Roman', fontsize=14)
        plt.legend(loc="lower right", prop={'family': 'Times New Roman', 'size': 10})
        plt.grid(False)  # Remove grid for cleaner appearance
        
        # Set tick label fonts
        plt.xticks(fontfamily='Times New Roman', fontsize=10)
        plt.yticks(fontfamily='Times New Roman', fontsize=10)
        plt.tight_layout()
        
        # Save ROC curves plot
        roc_path = os.path.join(output_dir, "ROC_curves.png")
        plt.savefig(roc_path, dpi=300)
        print(f"✓ ROC curves saved to: {roc_path}")
        plt.close()  # Close figure to free memory
        
    except Exception as e:
        print(f"✗ Error in ROC curve generation: {e}")

    # ==================== LDA VISUALIZATION ====================
    print("\n=== LDA VISUALIZATION ===")
    
    try:
        print("Generating LDA scatter plot visualization...")
        
        # Flatten data for dimensionality reduction
        vis_data = data.reshape(len(data), -1)  
        
        # Perform stratified sampling for visualization (max 1000 points per class)
        sampled_data = []
        sampled_labels = []
        max_vis_samples = 1000
        
        for i in range(len(class_map)):
            idx = np.where(true_labels == i)[0]
            if len(idx) > max_vis_samples:
                idx = np.random.choice(idx, max_vis_samples, replace=False)
            sampled_data.append(vis_data[idx])
            sampled_labels.append(true_labels[idx])
        
        # Combine sampled data from all classes
        sampled_data = np.vstack(sampled_data)
        sampled_labels = np.concatenate(sampled_labels)
        
        print(f"Data shape before dimensionality reduction: {sampled_data.shape}")
        
        # Attempt LDA dimensionality reduction, with PCA as fallback
        try:
            print("Attempting LDA dimensionality reduction...")
            lda = LDA(n_components=2)
            X_reduced = lda.fit_transform(sampled_data, sampled_labels)
            print("✓ LDA reduction successful")
        except Exception as e:
            print(f"LDA reduction failed: {e}")
            print("Using PCA as fallback...")
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            X_reduced = pca.fit_transform(sampled_data)
            print("✓ PCA reduction successful")
        
        print(f"Data shape after dimensionality reduction: {X_reduced.shape}")
        
        # Create scatter plot with consistent styling
        plt.figure(figsize=(8, 6))
        
        # Plot each cancer type with different colors
        for i, name in enumerate(class_names):
            idx = sampled_labels == i
            if np.sum(idx) > 0:  # Ensure class has data points
                plt.scatter(
                    X_reduced[idx, 0], 
                    X_reduced[idx, 1],
                    color=colors[i % len(colors)],
                    label=name,
                    s=40,  # Point size
                    alpha=0.7  # Transparency
                )
        
        # Configure scatter plot appearance
        plt.xlabel('Component 1', fontfamily='Times New Roman', fontsize=12)
        plt.ylabel('Component 2', fontfamily='Times New Roman', fontsize=12)
        plt.title('2D LDA Visualization', fontfamily='Times New Roman', fontsize=14)
        plt.legend(loc='best', prop={'family': 'Times New Roman', 'size': 10})
        plt.grid(False)  # Remove grid for cleaner appearance
        plt.tight_layout()
        
        # Save LDA visualization
        scatter_path = os.path.join(output_dir, "lda_visualization.png")
        plt.savefig(scatter_path, dpi=300)
        print(f"✓ LDA scatter plot saved to: {scatter_path}")
        plt.close()
        
        # Verify file was saved successfully
        if os.path.exists(scatter_path):
            file_size = os.path.getsize(scatter_path) / 1024
            print(f"✓ Verification: Scatter plot successfully saved ({file_size:.1f} KB)")
        else:
            print("✗ Warning: Scatter plot file not found, save operation failed")
        
    except Exception as e:
        print(f"✗ Error in LDA visualization: {e}")
        import traceback
        traceback.print_exc()  # Print detailed error traceback

    # ==================== CLASS-WISE ACCURACY ANALYSIS ====================
    print("\n=== CLASS-WISE ACCURACY ANALYSIS ===")
    
    # Calculate accuracy for each cancer type
    class_accuracy = {}
    print("Individual class accuracies:")
    print("-" * 40)
    
    for i in range(len(class_map)):
        class_mask = (true_labels == i)
        if np.sum(class_mask) > 0:  # Prevent division by zero
            accuracy = np.sum(pred_labels[class_mask] == i) / np.sum(class_mask)
            class_accuracy[idx_to_class[i]] = accuracy
            print(f"{idx_to_class[i]}: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Calculate overall accuracy
    overall_accuracy = np.sum(pred_labels == true_labels) / len(true_labels)
    print(f"\nOverall Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
    
    print("\n" + "="*60)
    print("PREDICTION PIPELINE COMPLETED SUCCESSFULLY")
    print("="*60)


if __name__ == '__main__':
    # Execute the main prediction pipeline
    main() 