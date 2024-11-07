#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
from torchmetrics import BootStrapper
from torchmetrics.regression import MeanAbsoluteError, R2Score

def compute_ci(metric, preds, true):
    torch.manual_seed(123)

    quantiles = torch.tensor([0.05, 0.95])
    base_metric = metric()
    bootstrap = BootStrapper(
        base_metric, num_bootstraps=2000, sampling_strategy="multinomial", quantile=quantiles
    )
    bootstrap.update(torch.from_numpy(preds), torch.from_numpy(true))
    output = bootstrap.compute()
    return output


# In[8]:


import os
import pandas as pd

# In[9]

parent_dir = 'results/experiment #14'
dirs = os.listdir(parent_dir)

for folder in dirs:
    results = pd.read_csv(f'{parent_dir}/{folder}/test_reults.csv')
    true_values = results.groupby('SampleID').mean('TrueValue').reset_index()['TrueValue'].values
    preds = results.groupby('SampleID').mean('Prediction').reset_index()['Prediction'].values
    print(f'For {folder}: \n')
    ci_mae = compute_ci(metric=MeanAbsoluteError, preds=preds, true=true_values)
    ci_r2 = compute_ci(metric=R2Score, preds=preds, true=true_values)
    print(f'MAE: {ci_mae}')
    print(f'R2: {ci_r2}')
    print('------------------------------------')

# In[10]

parent_dir = 'results/experiment #16 - control Stratified'
dirs = os.listdir(parent_dir)

for folder in dirs:
    results = pd.read_csv(f'{parent_dir}/{folder}/test_reults.csv')
    true_values = results.groupby('SampleID').mean('TrueValue').reset_index()['TrueValue'].values
    preds = results.groupby('SampleID').mean('Prediction').reset_index()['Prediction'].values
    print(f'For {folder}: \n')
    ci_mae = compute_ci(metric=MeanAbsoluteError, preds=preds, true=true_values)
    ci_r2 = compute_ci(metric=R2Score, preds=preds, true=true_values)
    print(f'MAE: {ci_mae}')
    print(f'R2: {ci_r2}')
    print('------------------------------------')


# In[15]:


parent_dir = 'results/experiment #16 - diabetic Stratified'
dirs = os.listdir(parent_dir)

for folder in dirs:
    results = pd.read_csv(f'{parent_dir}/{folder}/test_reults.csv')
    true_values = results.groupby('SampleID').mean('TrueValue').reset_index()['TrueValue'].values
    preds = results.groupby('SampleID').mean('Prediction').reset_index()['Prediction'].values
    print(f'For {folder}: \n')
    ci_mae = compute_ci(metric=MeanAbsoluteError, preds=preds, true=true_values)
    ci_r2 = compute_ci(metric=R2Score, preds=preds, true=true_values)
    print(f'MAE: {ci_mae}')
    print(f'R2: {ci_r2}')
    print('------------------------------------')


# In[25]:


parent_dir = 'results/experiment #16 - 30_years_and_younger Stratified'
dirs = os.listdir(parent_dir)

for folder in dirs:
    results = pd.read_csv(f'{parent_dir}/{folder}/test_reults.csv')
    true_values = results.groupby('SampleID').mean('TrueValue').reset_index()['TrueValue'].values
    preds = results.groupby('SampleID').mean('Prediction').reset_index()['Prediction'].values
    print(f'For {folder}: \n')
    ci_mae = compute_ci(metric=MeanAbsoluteError, preds=preds, true=true_values)
    ci_r2 = compute_ci(metric=R2Score, preds=preds, true=true_values)
    print(f'MAE: {ci_mae}')
    print(f'R2: {ci_r2}')
    print('------------------------------------')


# In[5]:


import os
import pandas as pd
parent_dir = 'results/experiment #16 - 31-40_years Stratified'
dirs = os.listdir(parent_dir)

for folder in dirs:
    results = pd.read_csv(f'{parent_dir}/{folder}/test_reults.csv')
    true_values = results.groupby('SampleID').mean('TrueValue').reset_index()['TrueValue'].values
    preds = results.groupby('SampleID').mean('Prediction').reset_index()['Prediction'].values
    print(f'For {folder}: \n')
    ci_mae = compute_ci(metric=MeanAbsoluteError, preds=preds, true=true_values)
    ci_r2 = compute_ci(metric=R2Score, preds=preds, true=true_values)
    print(f'MAE: {ci_mae}')
    print(f'R2: {ci_r2}')
    print('------------------------------------')


# In[22]:


#### import os
import pandas as pd

parent_dir = 'results/experiment #16 - control Stratified'
dirs = os.listdir(parent_dir)
dirs.sort()
result_list = []
for folder in dirs:
    results = pd.read_csv(f'{parent_dir}/{folder}/test_reults.csv')
    true_values = results.groupby('SampleID').mean('TrueValue').reset_index()['TrueValue'].values
    preds = results.groupby('SampleID').mean('Prediction').reset_index()['Prediction'].values
    print(f'For {folder}: \n')
    ci_mae = compute_ci(metric=MeanAbsoluteError, preds=preds, true=true_values)
    ci_r2 = compute_ci(metric=R2Score, preds=preds, true=true_values)
    print(f'MAE: {ci_mae}')
    print(f'R2: {ci_r2}')
    result_list.append({'Biomarker': folder, 'MAE': ci_mae['mean'].item(), 
                        'MAE CI 1': ci_mae['quantile'].numpy()[0],
                        'MAE CI 2': ci_mae['quantile'].numpy()[1], 
                        'R2': ci_r2['mean'].item(), 
                        'R2 CI 1': ci_r2['quantile'].numpy()[0], 
                        'R2 CI 2': ci_r2['quantile'].numpy()[1]})
    print('------------------------------------')


# In[24]:


pd.DataFrame(result_list).round(2).to_csv('results_biomarkers_control_ds.csv', index=False)


# In[ ]:




