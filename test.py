from ultralytics import YOLO
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas

# Base paths
base_path = "/mmfs1/scratch/utsha.saha/skncncr/"
test_data_dir = "/mmfs1/scratch/utsha.saha/skncncr/data/test/"

labels_dir = {
}

for class_names in os.listdir(test_data_dir):
    class_path = os.path.join(test_data_dir, class_names)
    
    for files in os.listdir(class_path):
        labels_dir[os.path.join(class_path, files)] = class_names



from IPython.display import clear_output

models = ["n", "s", "m", "l", "x"]
# models = ["n"]

for x in models:
    model_path = os.path.join(base_path, f"result_with_aug/yolov8{x}/weights/best.pt")
    model = YOLO(model_path)
    class_names = model.names
    
    true_labels = list(labels_dir.values())
    pred_labels = []
    
    for image_path in labels_dir:
        results = model(image_path, save=True, verbose=False)
        predicted_class = results[0].probs.top1
        pred_labels.append(class_names[predicted_class])
        clear_output(wait=True)  # Clear output for cleaner display

    # Generate classification report and save as CSV
    report = classification_report(true_labels, pred_labels, digits=3, output_dict=True)
    df = pandas.DataFrame(report).transpose()
    df.to_csv(f"yolov8{x}_report.csv")

    # Confusion Matrix
    conf_matrix = confusion_matrix(true_labels, pred_labels)
    
    # Plotting confusion matrix
    plt.figure()  # Create new figure
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='viridis', 
                xticklabels=class_names.values(), yticklabels=class_names.values(), cbar=False)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'YOLOv8{x}')
    plt.savefig(f"yolov8{x}_confusion_matrix.png")
    
    # Clear and close the figure to prevent overlap
    plt.clf()  # Clear the current figure
    plt.close()  # Close the figure explicitly

