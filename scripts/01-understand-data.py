# ============================================================
# This is a script to understand the data and perform some basic analysis.
# ============================================================

# Import necessary libraries for data analysis and visualization
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import os

# Display the main data information, including the number of rows, columns, and data types
df = pd.read_csv('data/raw/players_20.csv')

print('=============================================================')
print('Data Information:')
print(f'Data shape: {df.shape}')
print(f'Data types:')
print(f'{df.dtypes}')
print(f'Number of missing values:')
print(f'{df.isnull().sum()}')
print(f'Number of duplicate rows: {df.duplicated().sum()}')

# Display different positions and their counts in the dataset
print('=============================================================')
print('Player Positions and Counts:')
position_counts = df['team_position'].value_counts()
print(position_counts)

# Create the 'figures' directory if it doesn't exist
if not os.path.exists('figures/understand_data'):
    os.makedirs('figures/understand_data')

# Graphically display the crutial points and save the figures in the 'figures' directory
plt.figure(figsize=(10, 6))
# Plot the distribution of player ages
sns.histplot(data=df, x='age', bins=30, kde=True)
plt.title('Distribution of Player Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig('figures/understand_data/age_distribution.png')
plt.show()

# Plot the distribution of player positions
plt.figure(figsize=(16, 6))
sns.countplot(data=df, x='team_position', order=df['team_position'].value_counts().index)
plt.title('Distribution of Player Positions')
plt.xlabel('Position')
plt.ylabel('Count')
plt.savefig('figures/understand_data/position_distribution.png')
plt.show()