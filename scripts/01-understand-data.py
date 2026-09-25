# ==============================================================================
# Script: 01-understand-data.py
# Purpose: Comprehensive Exploratory Data Analysis (EDA) for FIFA Player Position
#          Classification. Supports version-driven analysis (v1, v1.1, v1.2),
#          diagnosing data hygiene, structural missingness, target contamination,
#          positional attribute signatures, and multi-season stability.
# ==============================================================================

import argparse
import os
import sys
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch, Rectangle

# Ensure scripts directory is on sys.path for common imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from common import get_paths, load_config, resolve_config_path

# ------------------------------------------------------------------------------
# Constants and Position Mappings
# ------------------------------------------------------------------------------

ROLE_CATEGORIES = {
    'Goalkeepers': ['GK'],
    'Defenders': ['CB', 'LB', 'RB'],
    'Midfielders': ['CDM', 'CM', 'CAM'],
    'Attackers': ['ST', 'LW', 'RW'],
}

ROLE_COLORS = {
    'Goalkeepers': '#1f77b4',  # Blue
    'Defenders': '#2ca02c',    # Green
    'Midfielders': '#ff7f0e',  # Amber
    'Attackers': '#d62728',    # Crimson
}

POSITION_TO_ROLE = {}
for role, positions in ROLE_CATEGORIES.items():
    for pos in positions:
        POSITION_TO_ROLE[pos] = role

POSITION_MAPPING = {
    'GK': 'GK',
    'LCB': 'CB', 'RCB': 'CB', 'CB': 'CB',
    'LB': 'LB', 'RB': 'RB', 'LWB': 'LB', 'RWB': 'RB',
    'LCM': 'CM', 'RCM': 'CM', 'CM': 'CM',
    'LDM': 'CDM', 'RDM': 'CDM', 'CDM': 'CDM',
    'CAM': 'CAM', 'LAM': 'CAM', 'RAM': 'CAM',
    'LM': 'CM', 'RM': 'CM',
    'LW': 'LW', 'RW': 'RW',
    'LS': 'ST', 'RS': 'ST', 'ST': 'ST',
    'LF': 'LF', 'RF': 'RF', 'CF': 'ST',
}

TARGET_CLASSES = ['CAM', 'CB', 'CDM', 'CM', 'GK', 'LB', 'LW', 'RB', 'RW', 'ST']
OUTFIELD_POSITIONS = ['CB', 'LB', 'RB', 'CDM', 'CM', 'CAM', 'LW', 'RW', 'ST']

CORE_ATTRIBUTES = ['pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic']
GK_COLUMNS = [
    'gk_diving', 'gk_handling', 'gk_kicking',
    'gk_reflexes', 'gk_speed', 'gk_positioning',
]

CUSTOM_BODY_TYPES = {
    'Messi': 'Lean',
    'Neymar': 'Lean',
    'Mohamed Salah': 'Lean',
    'Courtois': 'Normal',
    'C. Ronaldo': 'Normal',
    'Shaqiri': 'Stocky',
    'Akinfenwa': 'Stocky',
}


def setup_plot_style():
    """Configure modern, publication-ready styling for seaborn and matplotlib."""
    sns.set_theme(style='whitegrid', font_scale=1.05)
    plt.rcParams.update({
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'axes.edgecolor': '#cccccc',
        'axes.linewidth': 0.8,
        'grid.color': '#e0e0e0',
        'grid.linestyle': '--',
        'grid.alpha': 0.7,
        'figure.autolayout': False,
    })


def print_section_header(title):
    print('\n' + '=' * 75)
    print(f'  {title.upper()}')
    print('=' * 75)


# ------------------------------------------------------------------------------
# Data Loading and Parsing
# ------------------------------------------------------------------------------

def load_dataset(config, input_path=None):
    """Load dataset based on config specification or explicit input override."""
    if input_path:
        print(f"Loading custom input file: {input_path}")
        df = pd.read_csv(input_path, low_memory=False)
        df['season'] = 'Custom'
        df['split_group'] = 'Custom'
        return df

    data_sources = config.get('data_sources', {})
    split_strategy = data_sources.get('split_strategy', 'random_split')

    if split_strategy == 'by_files':
        train_files = data_sources.get('train_files', [])
        test_files = data_sources.get('test_files', [])
        all_files = train_files + test_files
        print(f"Loading multi-season dataset ({len(all_files)} files: {len(train_files)} train, {len(test_files)} test)...")

        dfs = []
        for file_path in all_files:
            file_name = os.path.basename(file_path)
            season = file_name.replace('players_', 'FIFA ').replace('.csv', '')
            is_test = file_path in test_files
            split_group = f"Test ({season})" if is_test else f"Train (FIFA 15-20)"

            print(f"  Reading: {file_path} [{season}] ({'Test' if is_test else 'Train'})")
            sub_df = pd.read_csv(file_path, low_memory=False)
            sub_df['season'] = season
            sub_df['split_group'] = split_group
            dfs.append(sub_df)

        return pd.concat(dfs, ignore_index=True)

    else:
        raw_files = data_sources.get('raw_files', ['data/raw/players_20.csv'])
        dfs = []
        for file_path in raw_files:
            print(f"  Reading: {file_path}")
            sub_df = pd.read_csv(file_path, low_memory=False)
            sub_df['season'] = 'FIFA 20'
            sub_df['split_group'] = 'Full Dataset'
            dfs.append(sub_df)
        return pd.concat(dfs, ignore_index=True)


def parse_clean_position(df):
    """Filter single-position players and map them to the 10 target classes."""
    df_clean = df.copy()

    # Determine if single-position player
    df_clean['is_single_position'] = df_clean['player_positions'].apply(
        lambda pos: isinstance(pos, str) and ',' not in pos
    )

    # Clean primary position for single-position specialists
    df_clean['clean_position'] = df_clean['player_positions'].apply(
        lambda pos: pos if isinstance(pos, str) and ',' not in pos else None
    )
    df_clean['mapped_position'] = df_clean['clean_position'].map(POSITION_MAPPING)
    df_clean['macro_role'] = df_clean['mapped_position'].map(POSITION_TO_ROLE)
    return df_clean


# ------------------------------------------------------------------------------
# Statistical Analysis & Console Reporting
# ------------------------------------------------------------------------------

def analyze_dataset_hygiene(df, version):
    print_section_header(f'1. Dataset Overview & Data Hygiene ({version})')
    print(f'Total records (rows):     {df.shape[0]:,}')
    print(f'Total attributes (cols):  {df.shape[1]:,}')
    memory_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    print(f'Memory usage:             {memory_mb:.2f} MB')
    duplicates = df.duplicated(subset=['sofifa_id', 'season'] if 'sofifa_id' in df.columns else None).sum()
    print(f'Duplicate records:        {duplicates} ({(duplicates / len(df)) * 100:.2f}%)')

    num_cols = df.select_dtypes(include=['number']).columns
    cat_cols = df.select_dtypes(exclude=['number']).columns
    print(f'Numerical features:       {len(num_cols)}')
    print(f'Categorical features:     {len(cat_cols)}')


def analyze_missingness_patterns(df):
    print_section_header('2. Missing Value Analysis (Structural vs Random)')
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / len(df)) * 100
    missing_df = pd.DataFrame({'Missing_Count': missing_counts, 'Missing_Pct': missing_pct})
    filtered_missing = missing_df.loc[missing_df['Missing_Count'] > 0]
    sorted_missing = filtered_missing.sort_values(by='Missing_Count', ascending=False)

    print(f'Columns with missing values: {len(sorted_missing)} out of {df.shape[1]}')
    print('\nTop 15 columns with missing values:')
    for row in sorted_missing.head(15).itertuples():
        missing_cnt = int(getattr(row, 'Missing_Count'))
        missing_pct_val = float(getattr(row, 'Missing_Pct'))
        print(f"  - {row.Index:<26}: {missing_cnt:>8,} ({missing_pct_val:>5.1f}%)")

    # Quantify Goalkeeper vs Outfield structural missingness
    gk_count = df['player_positions'].str.startswith('GK', na=False).sum()
    pace_missing = df['pace'].isnull().sum()
    gk_div_missing = df['gk_diving'].isnull().sum()

    print('\n[Key Finding] Structural Missingness Partition:')
    print(f'  * Goalkeepers in dataset:          {gk_count:>8,} players')
    print(f'  * Missing outfield core stats:     {pace_missing:>8,} ({pace_missing / len(df) * 100:.1f}%) -> GKs completely lack outfield stats')
    print(f'  * Missing goalkeeper core stats:   {gk_div_missing:>8,} ({gk_div_missing / len(df) * 100:.1f}%) -> Outfielders completely lack GK stats')
    print('  * Preprocessing implication: DropNA on all columns would erase 100% of rows! Role-specific zero-imputation is essential.')


def analyze_target_labels(df_clean):
    print_section_header('3. Target Labels & Tactical Position Distribution')

    # Team position contamination
    team_pos_counts = df_clean['team_position'].value_counts()
    sub_count = team_pos_counts.get('SUB', 0)
    res_count = team_pos_counts.get('RES', 0)
    sub_res_count = sub_count + res_count
    print("Raw 'team_position' contamination:")
    print(f"  * Total 'SUB' (Substitute):  {sub_count:>8,} ({sub_count / len(df_clean) * 100:.1f}%)")
    print(f"  * Total 'RES' (Reserve):     {res_count:>8,} ({res_count / len(df_clean) * 100:.1f}%)")
    print(f"  * Total Matchday bench roles:{sub_res_count:>8,} ({sub_res_count / len(df_clean) * 100:.1f}%)")
    print("  -> Insight: 'team_position' reflects match squad status, NOT player tactical position!")

    # Single vs multi-position
    single_count = df_clean['is_single_position'].sum()
    multi_count = len(df_clean) - single_count
    print(f"\nPlayer Positions specialization:")
    print(f"  * Single-position specialists: {single_count:>8,} ({single_count / len(df_clean) * 100:.1f}%)")
    print(f"  * Multi-position utility:     {multi_count:>8,} ({multi_count / len(df_clean) * 100:.1f}%)")

    # Mapped target classes distribution
    single_df = df_clean[df_clean['is_single_position'] & df_clean['mapped_position'].isin(TARGET_CLASSES)].copy()
    counts = single_df['mapped_position'].value_counts()
    max_count = counts.max()
    print('\nTarget Class Distribution (Clean Single-Position Players):')
    print(f"{'Position':<8} {'Role':<14} {'Count':>8} {'Percent':>8} {'Imbalance Ratio':>17}")
    print('-' * 60)
    for pos in TARGET_CLASSES:
        cnt = counts.get(pos, 0)
        pct = (cnt / len(single_df)) * 100
        ratio = f'1 : {max_count / cnt:.1f}' if cnt > 0 else 'N/A'
        role = POSITION_TO_ROLE.get(pos, 'Unknown')
        print(f'{pos:<8} {role:<14} {cnt:>8,} {pct:>7.1f}% {ratio:>17}')
    print('-' * 60)
    print(f"Total clean single-position samples: {len(single_df):,}")


def analyze_attribute_signatures(df_clean):
    print_section_header('4. Positional Attribute Signatures (Mean Values)')
    single_df = df_clean[df_clean['is_single_position'] & df_clean['mapped_position'].isin(OUTFIELD_POSITIONS)].copy()

    means = single_df.groupby('mapped_position')[CORE_ATTRIBUTES].mean()
    means = means.reindex(OUTFIELD_POSITIONS)

    header = f"{'Position':<8}" + ''.join([f'{attr.capitalize():>10}' for attr in CORE_ATTRIBUTES])
    print(header)
    print('-' * len(header))
    for pos, row in means.iterrows():
        row_str = f'{pos:<8}' + ''.join([f'{val:>10.1f}' for val in row])
        print(row_str)

    print('\nKey Tactical Insights:')
    print('  - Strikers (ST) dominate in Shooting and Physicality, lowest in Defending.')
    print('  - Center Backs (CB) dominate in Defending and Physicality, lowest in Shooting.')
    print('  - Central Midfielders (CM) lead in Passing and Dribbling with balanced defending.')
    print('  - Wingers (LW/RW) have the highest Pace (>75.0) and strong Dribbling.')


def analyze_demographics_and_physics(df_clean):
    print_section_header('5. Physical Dimensions & Demographic Profiles')
    single_df = df_clean[df_clean['is_single_position'] & df_clean['mapped_position'].isin(TARGET_CLASSES)].copy()

    agg_df = single_df.groupby('mapped_position').agg({
        'height_cm': 'mean',
        'weight_kg': 'mean',
        'age': 'mean',
        'preferred_foot': lambda s: (s == 'Left').mean() * 100,
    }).rename(columns={'preferred_foot': 'pct_left_foot'}).reindex(TARGET_CLASSES)

    print(f"{'Position':<8} {'Mean Height (cm)':>18} {'Mean Weight (kg)':>18} {'Mean Age':>10} {'% Left-Footed':>15}")
    print('-' * 75)
    for pos, row in agg_df.iterrows():
        print(f"{pos:<8} {row['height_cm']:>18.1f} {row['weight_kg']:>18.1f} {row['age']:>10.1f} {row['pct_left_foot']:>14.1f}%")


def print_modeling_recommendations():
    print_section_header('6. Key Takeaways & Preprocessing Pipeline Recommendations')
    recommendations = [
        ("Target Variable", "Never train on 'team_position' (SUB/RES contamination). Use mapped 'player_positions'."),
        ("Ambiguity Filter", "Filter for single-position players in baseline models to ensure unconfounded labels."),
        ("Zero-Imputation", "Impute GK attributes with 0 for outfielders, and outfield attributes with 0 for GKs."),
        ("Target Leakage", "Drop in-game positional rating columns (ls, st, rcm, lcb, etc.) which leak ground-truth."),
        ("Physical Features", "Include height_cm and weight_kg: they provide strong orthogonal separation between CB/GK and Wingers."),
        ("Demographics", "Include preferred_foot: it discriminates LB from RB almost perfectly."),
        ("Class Imbalance", "Majority classes (CB, GK, ST) vastly outnumber minority classes (LW, RW). Optimize for Macro-F1."),
    ]
    for cat, rec in recommendations:
        print(f"  * {cat:<18}: {rec}")
    print('=' * 75 + '\n')


# ------------------------------------------------------------------------------
# Core Visualizations (All Versions)
# ------------------------------------------------------------------------------

def plot_target_class_distribution(df_clean, output_dir, version, show=False):
    """Plot distribution of 10 target classes with counts, percentages, and role colors."""
    single_df = df_clean[df_clean['is_single_position'] & df_clean['mapped_position'].isin(TARGET_CLASSES)]
    counts = single_df['mapped_position'].value_counts()

    order = counts.index.tolist()
    palette = [ROLE_COLORS[POSITION_TO_ROLE[p]] for p in order]

    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    bars = sns.barplot(
        x=counts.index, y=counts.values, hue=counts.index,
        palette=palette, legend=False, ax=ax, edgecolor='#333333', linewidth=0.8
    )

    total = len(single_df)
    for p in ax.patches:
        if isinstance(p, Rectangle):
            height = p.get_height()
            pct = (height / total) * 100
            ax.annotate(f'{int(height):,}\n({pct:.1f}%)',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=9.5, fontweight='bold',
                        xytext=(0, 4), textcoords='offset points')

    ax.set_title(f'Target Class Distribution - Clean Single-Position Players ({version}, N = {total:,})',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Player Position', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('Number of Players', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylim(0, counts.max() * 1.15)

    legend_elements = [Patch(facecolor=color, edgecolor='#333333', label=role)
                       for role, color in ROLE_COLORS.items()]
    ax.legend(handles=legend_elements, title='Tactical Macro-Role', loc='upper right', frameon=True)

    plt.tight_layout()
    path = os.path.join(output_dir, '01_target_class_distribution.png')
    plt.savefig(path)
    legacy_path = os.path.join(output_dir, 'position_distribution.png')
    plt.savefig(legacy_path)
    if show:
        plt.show()
    plt.close()
    print(f'  Saved: {path}')


def plot_squad_role_vs_tactical(df_clean, output_dir, show=False):
    """Subplot 1: squad role contamination in team_position. Subplot 2: single vs multi-position."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    top_team_pos = df_clean['team_position'].value_counts().head(10)
    colors1 = ['#e74c3c' if pos in ['SUB', 'RES'] else '#3498db' for pos in top_team_pos.index]
    sns.barplot(
        x=top_team_pos.index, y=top_team_pos.values, hue=top_team_pos.index,
        palette=colors1, legend=False, ax=ax1, edgecolor='#333333'
    )
    ax1.set_title("Raw 'team_position': Dominated by Matchday Squad Roles (SUB & RES)", fontsize=12, fontweight='bold')
    ax1.set_xlabel('Recorded Team Position', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Player Count', fontsize=11, fontweight='bold')
    total1 = len(df_clean)
    for p in ax1.patches:
        if isinstance(p, Rectangle):
            h = p.get_height()
            ax1.annotate(f'{int(h):,}\n({h / total1 * 100:.1f}%)',
                         (p.get_x() + p.get_width() / 2., h),
                         ha='center', va='bottom', fontsize=8.5, xytext=(0, 3), textcoords='offset points')
    ax1.set_ylim(0, top_team_pos.max() * 1.15)

    single_count = df_clean['is_single_position'].sum()
    multi_count = len(df_clean) - single_count
    wedges, texts, autotexts = ax2.pie(
        [single_count, multi_count],
        labels=['Single-Position Specialists\n(Clean Ground Truth)', 'Multi-Position Utility\n(Secondary/Ambiguous)'],
        autopct='%1.1f%%',
        startangle=140,
        colors=['#2ecc71', '#95a5a6'],
        explode=(0.06, 0),
        textprops={'fontsize': 11, 'fontweight': 'bold'},
        wedgeprops={'edgecolor': '#333333', 'linewidth': 1.2}
    )
    for at in autotexts:
        at.set_color('white')
        at.set_fontsize(12)
    ax2.set_title("'player_positions': Specialization vs Multi-Position Ambiguity", fontsize=12, fontweight='bold')

    plt.suptitle("Why 'player_positions' is Required for Position Classification", fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    path = os.path.join(output_dir, '02_squad_role_vs_tactical_position.png')
    plt.savefig(path)
    if show:
        plt.show()
    plt.close()
    print(f'  Saved: {path}')


def plot_core_attributes_by_position(df_clean, output_dir, show=False):
    """2x3 grid of boxplots for 6 core attributes across outfield positions."""
    single_df = df_clean[df_clean['is_single_position'] & df_clean['mapped_position'].isin(OUTFIELD_POSITIONS)].copy()

    fig, axes = plt.subplots(2, 3, figsize=(16, 10), dpi=300)
    axes = axes.flatten()

    palette = [ROLE_COLORS[POSITION_TO_ROLE[p]] for p in OUTFIELD_POSITIONS]

    for idx, attr in enumerate(CORE_ATTRIBUTES):
        ax = axes[idx]
        sns.boxplot(
            data=single_df, x='mapped_position', y=attr, order=OUTFIELD_POSITIONS,
            hue='mapped_position', palette=palette, legend=False, ax=ax,
            fliersize=2, linewidth=1.1
        )
        ax.set_title(f'{attr.capitalize()} by Tactical Position', fontsize=12, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('Attribute Rating (0-100)', fontsize=10)
        ax.set_ylim(10, 100)

    fig.suptitle('Core FIFA Attribute Profiles Across Outfield Positions', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    path = os.path.join(output_dir, '03_core_attributes_by_position.png')
    plt.savefig(path)
    if show:
        plt.show()
    plt.close()
    print(f'  Saved: {path}')


def plot_physical_profiles(df_clean, output_dir, show=False):
    """Height and weight distributions across all 10 positions."""
    single_df = df_clean[df_clean['is_single_position'] & df_clean['mapped_position'].isin(TARGET_CLASSES)].copy()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

    order = ['GK', 'CB', 'CDM', 'ST', 'LB', 'RB', 'CM', 'CAM', 'LW', 'RW']
    palette = [ROLE_COLORS[POSITION_TO_ROLE[p]] for p in order]

    # Height boxplot
    sns.boxplot(
        data=single_df, x='mapped_position', y='height_cm', order=order,
        hue='mapped_position', palette=palette, legend=False, ax=ax1,
        fliersize=2, linewidth=1.1
    )
    mean_h = single_df['height_cm'].mean()
    ax1.set_title('Player Height (cm) by Position', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Position (Ordered by Height Hierarchy)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Height in cm', fontsize=11, fontweight='bold')
    ax1.axhline(mean_h, color='red', linestyle='--', alpha=0.7, label=f"Overall Mean ({mean_h:.1f} cm)")
    ax1.legend(loc='lower left')

    # Weight boxplot
    sns.boxplot(
        data=single_df, x='mapped_position', y='weight_kg', order=order,
        hue='mapped_position', palette=palette, legend=False, ax=ax2,
        fliersize=2, linewidth=1.1
    )
    mean_w = single_df['weight_kg'].mean()
    ax2.set_title('Player Weight (kg) by Position', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Position (Ordered by Height Hierarchy)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Weight in kg', fontsize=11, fontweight='bold')
    ax2.axhline(mean_w, color='red', linestyle='--', alpha=0.7, label=f"Overall Mean ({mean_w:.1f} kg)")
    ax2.legend(loc='lower left')

    fig.suptitle('Physical Dimension Breakdown: GKs & CBs vs Agile Wingers & Fullbacks', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    path = os.path.join(output_dir, '04_physical_profile_by_position.png')
    plt.savefig(path)
    if show:
        plt.show()
    plt.close()
    print(f'  Saved: {path}')


def plot_correlation_heatmap(df_clean, output_dir, show=False):
    """Annotated correlation heatmap among core skills and physical dimensions."""
    corr_cols = ['overall', 'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic', 'height_cm', 'weight_kg', 'age']
    sub = df_clean[corr_cols].dropna()
    corr = sub.corr()

    labels = ['Overall', 'Pace', 'Shooting', 'Passing', 'Dribbling', 'Defending', 'Physic', 'Height', 'Weight', 'Age']

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', vmin=-0.8, vmax=0.8,
                xticklabels=labels, yticklabels=labels, ax=ax, cbar_kws={'label': 'Pearson Correlation', 'shrink': 0.8},
                linewidths=0.5, linecolor='white')
    ax.set_title('Correlation Matrix of Core FIFA Attributes & Physical Dimensions', fontsize=13, fontweight='bold', pad=15)

    plt.tight_layout()
    path = os.path.join(output_dir, '05_attribute_correlation_heatmap.png')
    plt.savefig(path)
    if show:
        plt.show()
    plt.close()
    print(f'  Saved: {path}')


def plot_age_distribution_and_curves(df_clean, output_dir, show=False):
    """Overall age distribution + age curves by tactical role."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    sns.histplot(data=df_clean, x='age', bins=26, kde=True, color='#2b5c8f', ax=ax1, edgecolor='black', alpha=0.6)
    mean_age = df_clean['age'].mean()
    median_age = df_clean['age'].median()
    ax1.axvline(mean_age, color='#e74c3c', linestyle='-', linewidth=2, label=f'Mean Age ({mean_age:.1f})')
    ax1.axvline(median_age, color='#f39c12', linestyle='--', linewidth=2, label=f'Median Age ({median_age:.1f})')
    ax1.set_title('Overall Player Age Distribution', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Age', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax1.legend()

    single_df = df_clean[df_clean['is_single_position'] & df_clean['macro_role'].notnull()].copy()
    roles_order = ['Goalkeepers', 'Defenders', 'Midfielders', 'Attackers']
    palette = [ROLE_COLORS[r] for r in roles_order]
    sns.boxplot(
        data=single_df, x='macro_role', y='age', order=roles_order,
        hue='macro_role', palette=palette, legend=False, ax=ax2, linewidth=1.2
    )
    ax2.set_title('Career Longevity & Age Spread Across Macro-Roles', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Tactical Macro-Role', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Age', fontsize=11, fontweight='bold')

    plt.suptitle('Player Demographics: Age Distribution & Tactical Longevity', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    path = os.path.join(output_dir, '06_age_distribution_and_career_curves.png')
    plt.savefig(path)
    legacy_path = os.path.join(output_dir, 'age_distribution.png')
    plt.savefig(legacy_path)
    if show:
        plt.show()
    plt.close()
    print(f'  Saved: {path}')


def plot_preferred_foot(df_clean, output_dir, show=False):
    """100% stacked bar chart showing % Left vs % Right foot by position."""
    single_df = df_clean[df_clean['is_single_position'] & df_clean['mapped_position'].isin(TARGET_CLASSES)].copy()

    foot_counts = pd.crosstab(single_df['mapped_position'], single_df['preferred_foot'], normalize='index') * 100
    if 'Left' not in foot_counts.columns:
        foot_counts['Left'] = 0.0
    if 'Right' not in foot_counts.columns:
        foot_counts['Right'] = 0.0

    order = ['LB', 'LW', 'CAM', 'RW', 'CM', 'CB', 'ST', 'GK', 'CDM', 'RB']
    foot_counts = foot_counts.reindex(order)

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    bars_left = ax.barh(order, foot_counts['Left'], color='#e74c3c', label='Left Foot', edgecolor='#333333')
    bars_right = ax.barh(order, foot_counts['Right'], left=foot_counts['Left'], color='#3498db', label='Right Foot', edgecolor='#333333')

    for idx, pos in enumerate(order):
        left_pct = float(foot_counts.loc[pos, 'Left'])
        right_pct = float(foot_counts.loc[pos, 'Right'])
        if left_pct > 12:
            ax.text(left_pct / 2, idx, f'{left_pct:.1f}%', va='center', ha='center', color='white', fontweight='bold', fontsize=9.5)
        if right_pct > 12:
            ax.text(left_pct + right_pct / 2, idx, f'{right_pct:.1f}%', va='center', ha='center', color='white', fontweight='bold', fontsize=9.5)

    ax.set_title('Tactical Foot Bias: Preferred Foot Ratio by Position', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Percentage (%)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Position', fontsize=11, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.legend(loc='lower right', frameon=True)

    plt.tight_layout()
    path = os.path.join(output_dir, '07_preferred_foot_by_position.png')
    plt.savefig(path)
    if show:
        plt.show()
    plt.close()
    print(f'  Saved: {path}')


def plot_missingness_structure(df_clean, output_dir, show=False):
    """Bar chart illustrating Goalkeeper vs Outfield structural missingness partition."""
    is_gk = df_clean['player_positions'].str.startswith('GK', na=False)

    outfield_stats = ['pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic']
    gk_stats = ['gk_diving', 'gk_handling', 'gk_kicking', 'gk_reflexes', 'gk_speed', 'gk_positioning']

    gk_missing_outfield = df_clean[is_gk][outfield_stats].isnull().mean().mean() * 100
    outfield_missing_outfield = df_clean[~is_gk][outfield_stats].isnull().mean().mean() * 100

    gk_missing_gkstats = df_clean[is_gk][gk_stats].isnull().mean().mean() * 100
    outfield_missing_gkstats = df_clean[~is_gk][gk_stats].isnull().mean().mean() * 100

    gk_total = is_gk.sum()
    outfield_total = (~is_gk).sum()

    labels = ['Outfield Core Stats\n(Pace, Shoot, Pass, etc.)', 'Goalkeeping Stats\n(Diving, Handling, Reflexes, etc.)']
    gk_values = [gk_missing_outfield, gk_missing_gkstats]
    outfield_values = [outfield_missing_outfield, outfield_missing_gkstats]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    rects1 = ax.bar(x - width/2, gk_values, width, label=f'Goalkeepers (N = {gk_total:,})', color='#1f77b4', edgecolor='#333333')
    rects2 = ax.bar(x + width/2, outfield_values, width, label=f'Outfield Players (N = {outfield_total:,})', color='#e67e22', edgecolor='#333333')

    for rect in rects1 + rects2:
        h = rect.get_height()
        ax.annotate(f'{h:.1f}%',
                    (rect.get_x() + rect.get_width() / 2., h),
                    ha='center', va='bottom', fontsize=11, fontweight='bold', xytext=(0, 4), textcoords='offset points')

    ax.set_title('Structural Missingness Partition: Goalkeepers vs Outfielders', fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('Missing Value Rate (%)', fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 115)
    ax.legend(loc='center right', frameon=True)

    plt.tight_layout()
    path = os.path.join(output_dir, '08_systematic_missingness_structure.png')
    plt.savefig(path)
    if show:
        plt.show()
    plt.close()
    print(f'  Saved: {path}')


def plot_positional_radars(df_clean, output_dir, show=False):
    """3-panel radar chart comparing key tactical archetypes."""
    single_df = df_clean[df_clean['is_single_position'] & df_clean['mapped_position'].isin(OUTFIELD_POSITIONS)].copy()
    means = single_df.groupby('mapped_position')[CORE_ATTRIBUTES].mean()

    categories = [c.capitalize() for c in CORE_ATTRIBUTES]
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    pairs = [
        ('CB', 'ST', 'Center Back (CB) vs Striker (ST)', '#2ca02c', '#d62728'),
        ('CM', 'LW', 'Central Midfield (CM) vs Winger (LW)', '#ff7f0e', '#9b59b6'),
        ('CDM', 'CAM', 'Defensive Mid (CDM) vs Attacking Mid (CAM)', '#16a085', '#e67e22'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(19, 7), subplot_kw=dict(polar=True), dpi=300)

    for idx, (p1, p2, title, c1, c2) in enumerate(pairs):
        ax = axes[idx]
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=10, fontweight='bold')
        ax.tick_params(pad=14)
        ax.set_ylim(20, 85)

        v1 = means.loc[p1].tolist()
        v1 += v1[:1]
        ax.plot(angles, v1, linewidth=2.2, color=c1, label=p1)
        ax.fill(angles, v1, color=c1, alpha=0.2)

        v2 = means.loc[p2].tolist()
        v2 += v2[:1]
        ax.plot(angles, v2, linewidth=2.2, color=c2, label=p2)
        ax.fill(angles, v2, color=c2, alpha=0.2)

        ax.set_title(title, fontsize=12, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.15))

    fig.suptitle('Tactical Archetype Comparisons: Polar Skill Profiles', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=(0.0, 0.05, 1.0, 0.93))
    path = os.path.join(output_dir, '09_position_radar_profiles.png')
    plt.savefig(path)
    if show:
        plt.show()
    plt.close()
    print(f'  Saved: {path}')


# ------------------------------------------------------------------------------
# Version-Specific Visualizations (v1 Detailed Skills)
# ------------------------------------------------------------------------------

def plot_v1_detailed_skills(df_clean, output_dir, show=False):
    """Detailed in-game technical skills breakdown across positions for baseline v1."""
    single_df = df_clean[df_clean['is_single_position'] & df_clean['mapped_position'].isin(OUTFIELD_POSITIONS)].copy()

    skills = ['attacking_finishing', 'attacking_short_passing', 'defending_standing_tackle', 'skill_ball_control']
    skill_names = ['Finishing', 'Short Passing', 'Standing Tackle', 'Ball Control']

    fig, axes = plt.subplots(2, 2, figsize=(16, 9), dpi=300)
    axes = axes.flatten()

    palette = [ROLE_COLORS[POSITION_TO_ROLE[p]] for p in OUTFIELD_POSITIONS]

    for idx, (skill_col, display_name) in enumerate(zip(skills, skill_names)):
        if skill_col in single_df.columns:
            ax = axes[idx]
            sns.boxplot(
                data=single_df, x='mapped_position', y=skill_col, order=OUTFIELD_POSITIONS,
                hue='mapped_position', palette=palette, legend=False, ax=ax, fliersize=2, linewidth=1.1
            )
            ax.set_title(f'Technical Skill: {display_name} by Position', fontsize=12, fontweight='bold')
            ax.set_xlabel('')
            ax.set_ylabel('In-Game Rating (0-100)', fontsize=10)
            ax.set_ylim(10, 100)

    fig.suptitle('Baseline FIFA 20 In-Game Technical Skill Breakdown across Positions', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    path = os.path.join(output_dir, '10_detailed_skills_by_position.png')
    plt.savefig(path)
    if show:
        plt.show()
    plt.close()
    print(f'  Saved: {path}')


# ------------------------------------------------------------------------------
# Version-Specific Visualizations (v1.1 Engineered Features)
# ------------------------------------------------------------------------------

def plot_v1_1_work_rate(df_clean, output_dir, show=False):
    """Attacking and defensive work rate distributions by position for v1.1."""
    single_df = df_clean[df_clean['is_single_position'] & df_clean['mapped_position'].isin(TARGET_CLASSES)].copy()
    wr_split = single_df['work_rate'].str.split('/', expand=True)
    single_df['wr_att'] = wr_split[0].str.strip()
    single_df['wr_def'] = wr_split[1].str.strip()

    rates = single_df.groupby('mapped_position').agg({
        'wr_att': lambda s: (s == 'High').mean() * 100,
        'wr_def': lambda s: (s == 'High').mean() * 100,
    }).reindex(TARGET_CLASSES)

    x = np.arange(len(TARGET_CLASSES))
    width = 0.38

    fig, ax = plt.subplots(figsize=(13, 6), dpi=300)
    b1 = ax.bar(x - width/2, rates['wr_att'], width, label='High Attacking Work Rate (%)', color='#e74c3c', edgecolor='#333333')
    b2 = ax.bar(x + width/2, rates['wr_def'], width, label='High Defensive Work Rate (%)', color='#2980b9', edgecolor='#333333')

    for container in (b1, b2):
        for bar in container:
            h = bar.get_height()
            if h > 2:
                ax.annotate(f'{h:.1f}%',
                            (bar.get_x() + bar.get_width() / 2., h),
                            ha='center', va='bottom', fontsize=8.5, fontweight='bold', xytext=(0, 2), textcoords='offset points')

    ax.set_title('Engineered Feature: High Attacking vs High Defensive Work Rate by Position (v1.1)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(TARGET_CLASSES, fontsize=11, fontweight='bold')
    ax.set_ylabel('Percentage with High Work Rate (%)', fontsize=11, fontweight='bold')
    ax.set_ylim(0, rates.values.max() * 1.18)
    ax.legend(loc='upper right', frameon=True)

    plt.tight_layout()
    path = os.path.join(output_dir, '10_work_rate_by_position.png')
    plt.savefig(path)
    if show:
        plt.show()
    plt.close()
    print(f'  Saved: {path}')


def plot_v1_1_body_type(df_clean, output_dir, show=False):
    """Body type breakdown (Lean, Normal, Stocky) across positions for v1.1."""
    single_df = df_clean[df_clean['is_single_position'] & df_clean['mapped_position'].isin(TARGET_CLASSES)].copy()
    cleaned_body = single_df['body_type'].replace(CUSTOM_BODY_TYPES)
    single_df['clean_body'] = cleaned_body

    bt_counts = pd.crosstab(single_df['mapped_position'], single_df['clean_body'], normalize='index') * 100
    for col in ['Lean', 'Normal', 'Stocky']:
        if col not in bt_counts.columns:
            bt_counts[col] = 0.0
    bt_counts = bt_counts[['Lean', 'Normal', 'Stocky']].reindex(TARGET_CLASSES)

    fig, ax = plt.subplots(figsize=(13, 6), dpi=300)
    bars_lean = ax.bar(TARGET_CLASSES, bt_counts['Lean'], label='Lean', color='#2ecc71', edgecolor='#333333')
    bars_norm = ax.bar(TARGET_CLASSES, bt_counts['Normal'], bottom=bt_counts['Lean'], label='Normal', color='#3498db', edgecolor='#333333')
    bars_stocky = ax.bar(TARGET_CLASSES, bt_counts['Stocky'], bottom=bt_counts['Lean'] + bt_counts['Normal'], label='Stocky', color='#e67e22', edgecolor='#333333')

    for idx, pos in enumerate(TARGET_CLASSES):
        l_val = float(bt_counts.loc[pos, 'Lean'])
        n_val = float(bt_counts.loc[pos, 'Normal'])
        s_val = float(bt_counts.loc[pos, 'Stocky'])
        if l_val > 10:
            ax.text(idx, l_val / 2, f'{l_val:.0f}%', ha='center', va='center', color='white', fontweight='bold', fontsize=8.5)
        if n_val > 15:
            ax.text(idx, l_val + n_val / 2, f'{n_val:.0f}%', ha='center', va='center', color='white', fontweight='bold', fontsize=8.5)
        if s_val > 8:
            ax.text(idx, l_val + n_val + s_val / 2, f'{s_val:.0f}%', ha='center', va='center', color='white', fontweight='bold', fontsize=8.5)

    ax.set_title('Engineered Feature: Player Body Type Proportions Across Positions (v1.1)', fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('Proportion (%)', fontsize=11, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.legend(loc='lower right', frameon=True)

    plt.tight_layout()
    path = os.path.join(output_dir, '11_body_type_distribution.png')
    plt.savefig(path)
    if show:
        plt.show()
    plt.close()
    print(f'  Saved: {path}')


def plot_v1_1_traits(df_clean, output_dir, show=False):
    """Presence of top discriminative player traits across positions for v1.1."""
    single_df = df_clean[df_clean['is_single_position'] & df_clean['mapped_position'].isin(TARGET_CLASSES)].copy()

    traits = [
        'Comes For Crosses',
        'Speed Dribbler (AI)',
        'Power Header',
        'Long Passer (AI)',
    ]

    trait_df = pd.DataFrame(index=TARGET_CLASSES)
    for trait in traits:
        escaped = trait.replace('(', r'\(').replace(')', r'\)')
        trait_df[trait] = single_df.groupby('mapped_position')['player_traits'].apply(
            lambda s: s.str.contains(escaped, na=False, regex=True).mean() * 100
        ).reindex(TARGET_CLASSES)

    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
    sns.heatmap(trait_df.T, annot=True, fmt='.1f', cmap='YlGnBu', cbar_kws={'label': 'Occurrence Rate (%)'}, ax=ax, linewidths=0.8)
    ax.set_title('Engineered Feature: Discriminative Player Traits Prevalence by Position (v1.1)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Player Position', fontsize=11, fontweight='bold')
    ax.set_ylabel('Player Trait', fontsize=11, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(output_dir, '12_significant_traits_by_position.png')
    plt.savefig(path)
    if show:
        plt.show()
    plt.close()
    print(f'  Saved: {path}')


# ------------------------------------------------------------------------------
# Version-Specific Visualizations (v1.2 Multi-Season Temporal Validation)
# ------------------------------------------------------------------------------

def plot_v1_2_seasonal_volume(df_clean, output_dir, show=False):
    """Track raw player volume and single-position clean samples from FIFA 15 to FIFA 21."""
    seasons_order = ['FIFA 15', 'FIFA 16', 'FIFA 17', 'FIFA 18', 'FIFA 19', 'FIFA 20', 'FIFA 21']
    existing_seasons = [s for s in seasons_order if s in df_clean['season'].unique()]

    raw_counts = df_clean.groupby('season').size().reindex(existing_seasons)
    single_counts = df_clean[df_clean['is_single_position'] & df_clean['mapped_position'].isin(TARGET_CLASSES)].groupby('season').size().reindex(existing_seasons)

    x = np.arange(len(existing_seasons))
    width = 0.38

    fig, ax = plt.subplots(figsize=(13, 6), dpi=300)
    b1 = ax.bar(x - width/2, raw_counts.values, width, label='Raw Total Players', color='#34495e', edgecolor='#333333')
    b2 = ax.bar(x + width/2, single_counts.values, width, label='Clean Single-Position Players', color='#27ae60', edgecolor='#333333')

    for container in (b1, b2):
        for bar in container:
            h = bar.get_height()
            ax.annotate(f'{int(h):,}',
                        (bar.get_x() + bar.get_width() / 2., h),
                        ha='center', va='bottom', fontsize=9, fontweight='bold', xytext=(0, 3), textcoords='offset points')

    ax.set_title('Multi-Season Dataset Volume: FIFA 15 through FIFA 21 (v1.2)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(existing_seasons, fontsize=11, fontweight='bold')
    ax.set_ylabel('Player Count', fontsize=11, fontweight='bold')
    ax.set_ylim(0, raw_counts.max() * 1.15)
    ax.legend(loc='lower right', frameon=True)

    plt.tight_layout()
    path = os.path.join(output_dir, '10_seasonal_volume_and_growth.png')
    plt.savefig(path)
    if show:
        plt.show()
    plt.close()
    print(f'  Saved: {path}')


def plot_v1_2_temporal_stability(df_clean, output_dir, show=False):
    """Compare class percentage distribution between Train (FIFA 15-20) and Test (FIFA 21)."""
    single_df = df_clean[df_clean['is_single_position'] & df_clean['mapped_position'].isin(TARGET_CLASSES)].copy()

    single_df['temporal_split'] = single_df['season'].apply(
        lambda s: 'Test (FIFA 21)' if '21' in str(s) else 'Train (FIFA 15-20)'
    )

    ct = pd.crosstab(single_df['mapped_position'], single_df['temporal_split'], normalize='columns') * 100
    ct = ct.reindex(TARGET_CLASSES)

    x = np.arange(len(TARGET_CLASSES))
    width = 0.38

    fig, ax = plt.subplots(figsize=(14, 6), dpi=300)
    b1 = ax.bar(x - width/2, ct['Train (FIFA 15-20)'], width, label='Train Set (FIFA 15-20, N ~ 57k)', color='#2980b9', edgecolor='#333333')
    b2 = ax.bar(x + width/2, ct['Test (FIFA 21)'], width, label='Test Set (FIFA 21, N ~ 9.5k)', color='#e67e22', edgecolor='#333333')

    for container in (b1, b2):
        for bar in container:
            h = bar.get_height()
            ax.annotate(f'{h:.1f}%',
                        (bar.get_x() + bar.get_width() / 2., h),
                        ha='center', va='bottom', fontsize=8.5, fontweight='bold', xytext=(0, 2), textcoords='offset points')

    ax.set_title('Temporal Class Distribution Stability: Train (FIFA 15-20) vs Unseen Test (FIFA 21)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(TARGET_CLASSES, fontsize=11, fontweight='bold')
    ax.set_ylabel('Class Proportion (%)', fontsize=11, fontweight='bold')
    ax.set_ylim(0, ct.values.max() * 1.15)
    ax.legend(loc='upper right', frameon=True)

    plt.tight_layout()
    path = os.path.join(output_dir, '11_temporal_train_test_class_stability.png')
    plt.savefig(path)
    if show:
        plt.show()
    plt.close()
    print(f'  Saved: {path}')


def plot_v1_2_attribute_evolution(df_clean, output_dir, show=False):
    """Track mean attribute scores and ratings across FIFA 15 to FIFA 21."""
    seasons_order = ['FIFA 15', 'FIFA 16', 'FIFA 17', 'FIFA 18', 'FIFA 19', 'FIFA 20', 'FIFA 21']
    existing_seasons = [s for s in seasons_order if s in df_clean['season'].unique()]

    single_df = df_clean[df_clean['is_single_position'] & df_clean['mapped_position'].isin(OUTFIELD_POSITIONS)].copy()

    season_means = single_df.groupby('season')[['pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic']].mean()
    season_means = season_means.reindex(existing_seasons)

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    markers = ['o', 's', '^', 'D', 'v', 'p']
    colors = ['#e74c3c', '#e67e22', '#2ecc71', '#3498db', '#9b59b6', '#34495e']

    for col, marker, color in zip(season_means.columns, markers, colors):
        ax.plot(existing_seasons, season_means[col], marker=marker, linewidth=2.2, label=col.capitalize(), color=color)

    ax.set_title('Cross-Season Core Attribute Calibration: FIFA 15 through FIFA 21 (v1.2)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('FIFA Edition', fontsize=11, fontweight='bold')
    ax.set_ylabel('Mean Rating (0-100)', fontsize=11, fontweight='bold')
    ax.set_ylim(40, 75)
    ax.legend(loc='lower left', ncol=3, frameon=True)

    plt.tight_layout()
    path = os.path.join(output_dir, '12_cross_season_attribute_evolution.png')
    plt.savefig(path)
    if show:
        plt.show()
    plt.close()
    print(f'  Saved: {path}')


# ------------------------------------------------------------------------------
# Main Entry Point
# ------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Comprehensive Exploratory Data Analysis for FIFA Position Classification.')
    parser.add_argument('--config', type=str, default=None, help='Path to configuration JSON. Defaults to active_version in config.json.')
    parser.add_argument('--input', type=str, default=None, help='Optional explicit CSV input path.')
    parser.add_argument('--output-dir', type=str, default=None, help='Optional custom directory to save output figures.')
    parser.add_argument('--show', action='store_true', help='Display figures interactively with plt.show().')
    args = parser.parse_args()

    setup_plot_style()

    # Load configuration
    config = load_config(args.config)
    version = config.get('model_version', 'v1')
    figures_dir = config.get('paths', {}).get('figures_dir', f'figures/{version}')

    # Output directory resolution: ensure understand_data is located inside figures/<version>/
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(figures_dir, 'understand_data')

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n===========================================================================")
    print(f"  FIFA DATA UNDERSTANDING PIPELINE: VERSION {version.upper()}")
    print(f"  Target Figures Directory: {output_dir}")
    print(f"===========================================================================\n")

    # Load data
    df = load_dataset(config, input_path=args.input)
    df_clean = parse_clean_position(df)

    # 1. Terminal Statistical Reports
    analyze_dataset_hygiene(df, version)
    analyze_missingness_patterns(df)
    analyze_target_labels(df_clean)
    analyze_attribute_signatures(df_clean)
    analyze_demographics_and_physics(df_clean)
    print_modeling_recommendations()

    # 2. Visualizations
    print_section_header(f'Generating Publication-Quality Visualizations for {version}')

    # Core figures (for all versions)
    plot_target_class_distribution(df_clean, output_dir, version, show=args.show)
    plot_squad_role_vs_tactical(df_clean, output_dir, show=args.show)
    plot_core_attributes_by_position(df_clean, output_dir, show=args.show)
    plot_physical_profiles(df_clean, output_dir, show=args.show)
    plot_correlation_heatmap(df_clean, output_dir, show=args.show)
    plot_age_distribution_and_curves(df_clean, output_dir, show=args.show)
    plot_preferred_foot(df_clean, output_dir, show=args.show)
    plot_missingness_structure(df_clean, output_dir, show=args.show)
    plot_positional_radars(df_clean, output_dir, show=args.show)

    # Version-specific figures
    if version == 'v1':
        print('  Generating v1-specific in-game technical skills analysis...')
        plot_v1_detailed_skills(df_clean, output_dir, show=args.show)

    elif version == 'v1.1':
        print('  Generating v1.1-specific engineered feature analysis (Work Rates, Body Types, Traits)...')
        plot_v1_1_work_rate(df_clean, output_dir, show=args.show)
        plot_v1_1_body_type(df_clean, output_dir, show=args.show)
        plot_v1_1_traits(df_clean, output_dir, show=args.show)

    elif version == 'v1.2':
        print('  Generating v1.2-specific multi-season temporal validation analysis (FIFA 15-21)...')
        plot_v1_2_seasonal_volume(df_clean, output_dir, show=args.show)
        plot_v1_2_temporal_stability(df_clean, output_dir, show=args.show)
        plot_v1_2_attribute_evolution(df_clean, output_dir, show=args.show)

    print(f"\nAll data understanding analysis and visualizations completed successfully for {version}!")
    print(f"Figures saved in: {output_dir}/\n")


if __name__ == '__main__':
    main()