# ==============================================================================
# Script to preprocess and split data according to version configuration.
# Supports random split (v1/v1.1) and by-file temporal split (e.g. 15-20 train / 21 test).
# ==============================================================================

import argparse
import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split

from common import get_paths, load_config

# Attribute groups for zero-imputation
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

# Position mapping to 10 broad categories
POSITION_MAPPING = {
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
TARGET_CLASSES = ['CAM', 'CB', 'CDM', 'CM', 'GK', 'LB', 'LW', 'RB', 'RW', 'ST']

# Top traits with high positional discrimination
SIGNIFICANT_TRAITS = [
    'Comes For Crosses',
    'GK Long Throw',
    'Cautious With Crosses',
    'Rushes Out Of Goal',
    'Saves with Feet',
    'Technical Dribbler (AI)',
    'Long Shot Taker (AI)',
    'Speed Dribbler (AI)',
    'Flair',
    'Long Passer (AI)',
    'Power Header',
    'Playmaker (AI)',
    'Dives Into Tackles (AI)',
    'Finesse Shot',
    'Outside Foot Shot',
    'Early Crosser',
    'Long Throw-in',
    'Chip Shot (AI)',
]

# Custom body type mappings for star players with custom models
CUSTOM_BODY_TYPES = {
    'Messi': 'Lean',
    'Neymar': 'Lean',
    'Mohamed Salah': 'Lean',
    'Courtois': 'Normal',
    'C. Ronaldo': 'Normal',
    'Shaqiri': 'Stocky',
    'Akinfenwa': 'Stocky',
}


def clean_dataframe(df, config):
    """Clean raw dataset and optionally engineer version-specific features."""
    cfg_prep = config.get('preprocessing', {})
    feature_flags = cfg_prep.get('features', {})
    age_min, age_max = cfg_prep.get('age_range', [16, 40])

    # 1. Remove duplicate rows
    df = df.drop_duplicates().copy()

    # 2. Impute role-specific missing values with 0
    df[GOALKEEPER_COLUMNS] = df[GOALKEEPER_COLUMNS].fillna(0)
    df[NON_GOALKEEPER_COLUMNS] = df[NON_GOALKEEPER_COLUMNS].fillna(0)

    # 3. Filter age outliers
    df = df[(df['age'] >= age_min) & (df['age'] <= age_max)].copy()

    # 4. Filter single-position players to remove ambiguity
    df['team_position'] = df['player_positions'].apply(
        lambda pos: pos if isinstance(pos, str) and ',' not in pos else None
    )
    df = df.dropna(subset=['team_position']).copy()

    # 5. Map positions into 10 target classes
    df['team_position'] = df['team_position'].map(POSITION_MAPPING)
    df = df.dropna(subset=['team_position']).copy()
    df = df[df['team_position'].isin(TARGET_CLASSES)].copy()

    # 6. Feature engineering (if enabled in config)
    if feature_flags.get('include_preferred_foot', False):
        df['preferred_foot_right'] = (df['preferred_foot'] == 'Right').astype(int)

    if feature_flags.get('include_work_rate', False):
        wr_map = {'Low': 0, 'Medium': 1, 'High': 2}
        wr_split = df['work_rate'].str.split('/', expand=True)
        df['work_rate_attack'] = wr_split[0].str.strip().map(wr_map).fillna(1).astype(int)
        df['work_rate_defense'] = wr_split[1].str.strip().map(wr_map).fillna(1).astype(int)

    if feature_flags.get('include_body_type', False):
        cleaned_body = df['body_type'].replace(CUSTOM_BODY_TYPES)
        df['body_type_lean'] = (cleaned_body == 'Lean').astype(int)
        df['body_type_stocky'] = (cleaned_body == 'Stocky').astype(int)

    if feature_flags.get('include_player_traits', False):
        for trait in SIGNIFICANT_TRAITS:
            col_name = 'trait_' + re.sub(r'[^a-z0-9]+', '_', trait.lower()).strip('_')
            escaped = re.escape(trait)
            df[col_name] = df['player_traits'].str.contains(escaped, na=False, regex=True).astype(int)

    # 7. Define columns to drop
    always_drop = [
        'sofifa_id', 'player_url', 'short_name', 'long_name', 'dob', 'nationality',
        'club_name', 'league_name', 'league_rank', 'team_jersey_number', 'loaned_from',
        'joined', 'contract_valid_until', 'release_clause_eur', 'player_tags',
        'nation_position', 'nation_jersey_number', 'player_traits', 'defending_marking',
        'real_face', 'player_positions', 'age',
        # In-game positional ratings (target leakage)
        'ls', 'st', 'rs', 'lw', 'lf', 'cf', 'rf', 'rw', 'lam', 'cam', 'ram',
        'lm', 'rcm', 'lcm', 'cm', 'rm', 'lwb', 'ldm', 'cdm', 'rdm', 'rwb',
        'lb', 'lcb', 'cb', 'rcb', 'rb'
    ]

    optional_drops = []
    if not feature_flags.get('include_height_weight', False):
        optional_drops.extend(['height_cm', 'weight_kg'])
    if not feature_flags.get('include_international_reputation', False):
        optional_drops.append('international_reputation')

    # Drop original raw columns that were transformed
    transformed_drops = ['body_type', 'work_rate', 'preferred_foot']

    all_drops = set(always_drop + optional_drops + transformed_drops)
    existing_drops = [c for c in all_drops if c in df.columns]
    df = df.drop(columns=existing_drops)

    return df


def load_file_list(files):
    """Load and concatenate multiple CSV files."""
    dfs = []
    for file_path in files:
        print(f'Reading: {file_path}')
        sub_df = pd.read_csv(file_path)
        dfs.append(sub_df)
    return pd.concat(dfs, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description='Preprocess FIFA data based on version config.')
    parser.add_argument('--config', type=str, default='configs/v1.json', help='Path to configuration JSON file.')
    args = parser.parse_args()

    config = load_config(args.config)
    paths = get_paths(config)
    data_sources = config.get('data_sources', {})
    split_strategy = data_sources.get('split_strategy', 'random_split')

    print(f"=== Preprocessing for {config.get('model_version', 'Unknown Version')} ===")
    print(f'Split strategy: {split_strategy}')

    if split_strategy == 'by_files':
        # Temporal / file-based split (e.g. 15-20 train, 21 test)
        raw_train_df = load_file_list(data_sources['train_files'])
        raw_test_df = load_file_list(data_sources['test_files'])

        train_df = clean_dataframe(raw_train_df, config)
        test_df = clean_dataframe(raw_test_df, config)

    else:
        # Standard random split
        raw_files = data_sources.get('raw_files', ['data/raw/players_20.csv'])
        raw_df = load_file_list(raw_files)
        clean_df = clean_dataframe(raw_df, config)

        test_size = data_sources.get('test_size', 0.2)
        random_state = data_sources.get('random_state', 42)

        print(f'Splitting data (test_size={test_size}, random_state={random_state})...')
        train_df, test_df = train_test_split(clean_df, test_size=test_size, random_state=random_state)

    train_out = os.path.join(paths['processed_dir'], 'train_data.csv')
    test_out = os.path.join(paths['processed_dir'], 'test_data.csv')

    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)

    print('Preprocessing complete!')
    print(f'Training samples: {len(train_df)} ({train_df.shape[1]} columns) -> {train_out}')
    print(f'Testing samples:  {len(test_df)} ({test_df.shape[1]} columns) -> {test_out}')


if __name__ == '__main__':
    main()
