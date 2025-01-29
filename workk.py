import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("smoking_drinking_dataset_Ver01.csv")
pd.set_option('display.max_columns', None) # in order to see all the columns

# Detection of duplicated rows
duplicates = df.duplicated() 
print("Number of duplicated rows:", len(df[duplicates]))

# Dropping duplicated data
df = df.drop_duplicates()

df_numeric = df.drop(["sex", "DRK_YN"], axis=1)
df_numeric.corr()
"""
### Visulation
sns.scatterplot(data = df, x = 'weight', y = 'height', hue='DRK_YN')
plt.xlabel('Weight')
plt.ylabel('Height')
plt.title('Scatter')
plt.show()

plt.figure(figsize=(8, 6))
sns.countplot(x='SMK_stat_type_cd', data=df, hue='DRK_YN', palette='pastel')
plt.title('The Effect of Smoking on Alcohol Consumption')
plt.xlabel('1 (never smoked)  2 (used to smoke but quit)  3 (still smoke)')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.countplot(x='sight_left', data=df, hue='DRK_YN', palette='pastel')
plt.title('Left Sight - Alcohol Consumption')
plt.xlabel('Categories (vision impairment increases as the value increases)')
plt.xticks(rotation=90)
plt.ylabel('Count')

plt.subplot(1, 2, 2)
sns.countplot(x='sight_right', data=df, hue='DRK_YN', palette='pastel')
plt.title('Right Sight - Alcohol Consumption')
plt.xlabel('Categories (vision impairment increases as the value increases)')
plt.xticks(rotation=90)
plt.ylabel('Count')

plt.suptitle('Vision Effect on Alcohol Consumption', fontsize=16)
plt.show()

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.countplot(x='hear_left', data=df, hue='DRK_YN', palette='pastel')
plt.title('Left Hearing - Alcohol Consumption')
plt.xlabel('Categories (1:normal, 2:abnormal)')
plt.xticks(rotation=90)
plt.ylabel('Count')

plt.subplot(1, 2, 2)
sns.countplot(x='hear_right', data=df, hue='DRK_YN', palette='pastel')
plt.title('Right Hearing - Alcohol Consumption')
plt.xlabel('Categories (1:normal, 2:abnormal)')
plt.xticks(rotation=90)
plt.ylabel('Count')

plt.suptitle('Hearing Effect on Alcohol Consumption', fontsize=16)
plt.show()

fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(14, 18))
fig.suptitle('Boxplot of Selected Columns', y=1.02)

Outliers_features_list = ['waistline', 'BLDS', 'tot_chole', 'HDL_chole', 'LDL_chole',
            'triglyceride', 'serum_creatinine', 'SGOT_AST', 'SGOT_ALT', 'gamma_GTP']

for i, col in enumerate(Outliers_features_list):
    sns.boxplot(x=df[col], ax=axs[i//2, i % 2], palette='pastel')
    axs[i//2, i % 2].set_title(col)

plt.tight_layout()
plt.show()

# To include drinking status, let's give 1 to drinkers and 0 to non-drinkers.
df_ = df.copy()
df_['DRK_YN'] = df_['DRK_YN'].map({'Y': 1, 'N': 0}) 
df_.head()

correlation_matrix = df_.corr()

plt.figure(figsize=(14, 8))
sns.heatmap(round(correlation_matrix,1), annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Korelasyon Matrisi Heatmap')
plt.show()
"""

def remove_outliers(df, feature, iqr_factor):
    data = df[feature]
    
    # Calculate the interquartile range (IQR)
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    
    # Determine the lower and upper bounds
    lower_bound = Q1 - iqr_factor * IQR
    upper_bound = Q3 + iqr_factor * IQR
    
    # Filter rows with values outside the bounds
    filtered_df = df[(data >= lower_bound) & (data <= upper_bound)]
    
    return filtered_df


filtered_df = df.copy()
for i in Outliers_features_list:
    filtered_df = remove_outliers(filtered_df, i, 2)
b=filtered_df.describe()







