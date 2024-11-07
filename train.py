#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.model_selection import KFold
import pandas as pd
from PIL import Image
import os
import csv
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np

# Custom Dataset
class RetinalDataset(Dataset):
    def __init__(self, data, root_dir, target_column, transform=None):
        self.data = data
        self.root_dir = root_dir
        self.target_column = target_column
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data.iloc[idx]['ImagePath'])
        image = Image.open(img_name).convert('RGB')
        label = self.data.iloc[idx][self.target_column]
        sample_id = self.data.iloc[idx]['SampleID']
        label = float(label)

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32), sample_id

# Transformations
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Horizontal flip
    transforms.RandomVerticalFlip(),  # Vertical flip
    transforms.RandomRotation(30),  # Random rotation
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize with ImageNet's mean and std
])

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize with ImageNet's mean and std
])


# In[2]:


from torchvision.models import MobileNet_V3_Large_Weights

class MobileNetV3LargeRegressor(nn.Module):
    def __init__(self):
        super(MobileNetV3LargeRegressor, self).__init__()
        self.model = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
        self.model.classifier = nn.Identity()  # Remove the fully connected layers
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pooling
        self.fc1 = nn.Linear(960, 512)  # Fully connected layer with 512 units
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 1)  # Final layer for regression

    def forward(self, x):
        x = self.model.features(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Initialize the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[29]:


def get_average_metrics(results):
    preds = results.groupby('SampleID')['Prediction'].mean().values.round(2)
    true = results.groupby('SampleID')['TrueValue'].mean().values.round(2)
    mae = mean_absolute_error(preds, true)
    r2 = r2_score(preds, true)
    return mae, r2


# In[17]:


from sklearn.model_selection import train_test_split
import time
from tempfile import TemporaryDirectory


# In[51]:


def cross_validate_biomarker(target_column, results_dir, csv_file, root_dir, num_epochs=150, batch_size=32, n_splits=5):
    dataset = pd.read_csv(csv_file)
    dataset = dataset[dataset[target_column].notna()]
    train, test = train_test_split(dataset, test_size=0.2, random_state=42)
    train = RetinalDataset(data=train, root_dir=root_dir, target_column=target_column, transform=train_transforms)
    test = RetinalDataset(data=test, root_dir=root_dir, target_column=target_column, transform=test_transforms)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)

    # Training
    # Initialize the model
    model = MobileNetV3LargeRegressor().to(device)
    
    # Loss and Optimizer
    criterion = nn.L1Loss()  # MAE Loss
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    # Variables to track the best model
    best_val_loss = float('inf')
    best_model_wts = None

    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels, _ in train_loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

        # Validation step
        model.eval()
        predictions = []
        true_values = []
        sample_ids = []
        with torch.no_grad():
            for images, labels, sample_id in test_loader:
                images = images.to(device)
                labels = labels.to(device).unsqueeze(1)
                outputs = model(images)
                predictions.extend(outputs.cpu().numpy())
                true_values.extend(labels.cpu().numpy())
                sample_ids.extend(sample_id)

        # Calculate validation loss over the entire validation set
        predictions = np.array(predictions).flatten()
        true_values = np.array(true_values).flatten()
        sample_ids = np.array(sample_ids)

        val_loss = mean_absolute_error(true_values, predictions)
        print(f'Validation Loss: {val_loss:.4f}')
        
        # Save the model if it has the best performance so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = model.state_dict().copy()

    # Load the best model weights
    if best_model_wts:
        model.load_state_dict(best_model_wts)
    
    # Calculate metrics
    # Validation step
    model.eval()
    test_predictions = []
    test_true_values = []
    test_sample_ids = []
    with torch.no_grad():
        for images, labels, sample_id in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            test_predictions.extend(outputs.cpu().numpy())
            test_true_values.extend(labels.cpu().numpy())
            test_sample_ids.extend(sample_id)

    # Calculate validation loss over the entire validation set
    r2 = r2_score(test_true_values, test_predictions)
    mae = mean_absolute_error(test_true_values, test_predictions)
    metrics = {'R2': r2, 'MAE': mae}
    biomarker_fold_results_dir = os.path.join(results_dir, target_column)
    os.makedirs(biomarker_fold_results_dir, exist_ok=True)
    model_save_path = os.path.join(biomarker_fold_results_dir, 'mobilenetv3large_regressor_best.pth')
    torch.save(model.state_dict(), model_save_path)

    # Save test results
    results = pd.DataFrame({
        'Biomarker': target_column,
        'SampleID': sample_ids,
        'Prediction': predictions,
        'TrueValue': true_values
    })

    results.to_csv(f'{biomarker_fold_results_dir}/test_reults.csv', index=False)

    avg_mae, avg_r2 = get_average_metrics(results)

    avg_metrics_save_path = os.path.join(biomarker_fold_results_dir, 'avg_test_metrics.csv')
    with open(avg_metrics_save_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['R2', avg_r2])
        writer.writerow(['MAE', avg_mae])
    print(f'Metrics for saved to {avg_metrics_save_path}')
    
    metrics_save_path = os.path.join(biomarker_fold_results_dir, 'test_metrics.csv')
    with open(metrics_save_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Metric', 'Value'])
        for key, value in metrics.items():
            writer.writerow([key, value])
    print(f'Metrics for saved to {metrics_save_path}')


# In[53]:


biomarker_names = ['Age', 'BMI', 'HBA 1C %', 'BP_OUT_CALC_AVG_SYSTOLIC_BP', 'BP_OUT_CALC_AVG_DIASTOLIC_BP',
       'Hemoglobin', 'Hematocrit', 'Red Blood Cell', 'Glucose', 'Creatinine',
       'Cholesterol Total', 'HDL-Cholesterol', 'LDL-Cholesterol Calc',
       'Triglyceride', 'Insulin', 'Testosterone Total', 'Estradiol',
       'SexHormone Binding Globulin']

root_dir = 'mean_subtracted/mean_subtracted'
main_folder = 'stratified_age'
import os

files = os.listdir(main_folder)

for csv_file in files:

    sp = csv_file.split('.')[0]
    results_dir = f'results/experiment #18 - {sp} Stratified'
    for biomarker in biomarker_names:
        print(f'Training and evaluating model for biomarker: {biomarker} in {sp} dataset')
        cross_validate_biomarker(biomarker, results_dir, f'{main_folder}/{csv_file}', root_dir, num_epochs=150)


# In[ ]:
