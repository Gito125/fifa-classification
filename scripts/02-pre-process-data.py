# ============================================================
# This is a script to pre-process and split the data for the project.
# It includes functions to clean, normalize, and transform the data
# ============================================================

import os

import pandas as pd
from sklearn.model_selection import train_test_split


RAW_DATA_PATH = 'data/raw/players_20.csv'
PROCESSED_DATA_DIR = 'data/processed'

GOALKEEPER_COLUMNS = [
    'gk_diving',
    'gk_handling',
    'gk_kicking',
    'gk_reflexes',
    'gk_speed',
    'gk_positioning',
]
NON_GOALKEEPER_COLUMNS = [
    'pace',
    'shooting',
    'passing',
    'dribbling',
    'defending',
    'physic',
]

df = pd.read_csv(RAW_DATA_PATH)

# Clean the data by handling missing values, duplicates, and outliers.
print('=============================================================')
print('Data Cleaning:')
print(f'Number of missing values before cleaning: {df.isnull().sum().sum()}')
print(f'Number of duplicate rows before cleaning: {df.duplicated().sum()}')
print(f'Number of rows before cleaning: {df.shape[0]}')

# Drop duplicate rows
df.drop_duplicates(inplace=True)
print(f'Number of rows after removing duplicates: {df.shape[0]}')

# Handle position-specific missing values. These attributes are not relevant
# to players in the other position group, so zero is an appropriate value.
df[GOALKEEPER_COLUMNS] = df[GOALKEEPER_COLUMNS].fillna(0)
df[NON_GOALKEEPER_COLUMNS] = df[NON_GOALKEEPER_COLUMNS].fillna(0)

print(f'Number of missing values after filling position-specific values: {df.isnull().sum().sum()}')

# Remove outliers based on domain knowledge and statistical methods
df = df[(df['age'] >= 16) & (df['age'] <= 40)]
print(f'Number of rows after removing outliers: {df.shape[0]}')

# Map single player positions to team_position. Players with multiple positions
# are excluded because their primary team position is ambiguous.
df['team_position'] = df['player_positions'].apply(
    lambda positions: positions if isinstance(positions, str) and ',' not in positions else None
)

print('Sample of players with missing team_position values:')
print(
    df.loc[
        df['team_position'].isna(),
        ['short_name', 'player_positions', 'team_position']
    ].head(20)
)
df = df.dropna(subset=['team_position'])

# Remove columns that are not needed for analysis or modeling.
print('Number of columns before removing unnecessary columns:', df.shape[1])
columns_to_drop = [
    'sofifa_id',
    'player_url',
    'short_name',
    'player_traits',
    'long_name',
    'club_name',
    'nationality',
    'league_name',
    'dob',
    'preferred_foot',
    'international_reputation',
    'body_type',
    'work_rate',
    'height_cm',
    'weight_kg',
    'real_face',
    'player_positions',
    'team_jersey_number',
    'loaned_from',
    'joined',
    'contract_valid_until',
    'nation_jersey_number',
    'age',
    'league_rank',
    'release_clause_eur',
    'player_tags',
    'nation_position',
    'defending_marking',
    'ls',
    'st',
    'rs',
    'lw',
    'lf',
    'cf',
    'rf',
    'rw',
    'lam',
    'cam',
    'ram',
    'lm',
    'rcm',
    'lcm',
    'cm',
    'rm',
    'lwb',
    'ldm',
    'cdm',
    'rdm',
    'rwb',
    'lb',
    'lcb',
    'cb',
    'rcb',
    'rb',
]
df.drop(columns=columns_to_drop, inplace=True)
print(f'Number of rows after removing unnecessary columns: {df.shape[0]}')
print('Number of columns after removing unnecessary columns:', df.shape[1])

# Show remaining missing values after data cleaning.
print('Missing values after data cleaning:', df.isnull().sum().sum())
print('Missing values by column after data cleaning:', df.isnull().sum())

# Map player positions to broader categories.
position_mapping = {
    'GK': 'GK',
    'LCB': 'CB',
    'RCB': 'CB',
    'CB': 'CB',
    'LB': 'LB',
    'RB': 'RB',
    'LWB': 'LB',
    'RWB': 'RB',
    'LCM': 'CM',
    'RCM': 'CM',
    'CM': 'CM',
    'LDM': 'CDM',
    'RDM': 'CDM',
    'CDM': 'CDM',
    'CAM': 'CAM',
    'LAM': 'CAM',
    'RAM': 'CAM',
    'LM': 'CM',
    'RM': 'CM',
    'LW': 'LW',
    'RW': 'RW',
    'LS': 'ST',
    'RS': 'ST',
    'ST': 'ST',
    'LF': 'LF',
    'RF': 'RF',
    'CF': 'ST',
}

df['team_position'] = df['team_position'].map(position_mapping)
df.dropna(subset=['team_position'], inplace=True)

# Split the data into training and testing sets then save them as separate files for model training and evaluation
print('==============================================================')
print('Splitting the data into training and testing sets...')
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

print(f'Number of rows in training set: {train_df.shape[0]}')
print(f'Number of rows in testing set: {test_df.shape[0]}')

print(f'Number of columns in training set: {train_df.shape[1]}')
print(f'Number of columns in testing set: {test_df.shape[1]}')

# Save the pre-processed training and testing data to CSV files
print('Saving pre-processed training and testing data to CSV files...')
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
train_df.to_csv(os.path.join(PROCESSED_DATA_DIR, 'train_data.csv'), index=False)
test_df.to_csv(os.path.join(PROCESSED_DATA_DIR, 'test_data.csv'), index=False)
