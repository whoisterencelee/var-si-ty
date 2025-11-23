import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load the cleaned data
filename = 'data/events_en_US_cleaned.csv'
ev_df = pd.read_csv(filename, encoding='utf-8')

print(f"Loaded {len(ev_df)} events from {filename}\n")

# ============================================================================
# VISUALIZATION 1: Overall Faculty Distribution
# ============================================================================

print("="*80)
print("PREPARING VISUALIZATION 1: Faculty Distribution")
print("="*80)

# Count events per faculty (handling multi-label)
faculty_counts = {}
for faculties_str in ev_df['faculties']:
    if pd.notna(faculties_str) and faculties_str.strip():
        # Split comma-separated faculties
        faculties_list = [f.strip() for f in str(faculties_str).split(',')]
        for fac in faculties_list:
            if fac:
                faculty_counts[fac] = faculty_counts.get(fac, 0) + 1

# Count events with no faculty assigned
no_faculty_count = len(ev_df[ev_df['faculties'].isna() | (ev_df['faculties'] == '')])
if no_faculty_count > 0:
    faculty_counts['Other/Unclassified'] = no_faculty_count

# Create DataFrame for plotting
faculty_df = pd.DataFrame(list(faculty_counts.items()), columns=['Faculty', 'Count'])
faculty_df = faculty_df.sort_values('Count', ascending=False)

print(f"\nFaculty distribution:")
for _, row in faculty_df.iterrows():
    print(f"  {row['Faculty']}: {row['Count']} events ({row['Count']/len(ev_df)*100:.1f}%)")

# Create the bar chart
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Bar chart
colors = plt.cm.Set3(np.linspace(0, 1, len(faculty_df)))
bars = ax1.bar(faculty_df['Faculty'], faculty_df['Count'], color=colors, edgecolor='black', linewidth=1.2)
ax1.set_xlabel('Faculty', fontsize=12, fontweight='bold')
ax1.set_ylabel('Number of Events', fontsize=12, fontweight='bold')
ax1.set_title('Distribution of Events by Faculty\n(Multi-label events counted for each faculty)',
              fontsize=14, fontweight='bold', pad=20)
ax1.tick_params(axis='x', rotation=45)
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}',
            ha='center', va='bottom', fontweight='bold')

# Pie chart
colors_pie = plt.cm.Pastel1(np.linspace(0, 1, len(faculty_df)))
wedges, texts, autotexts = ax2.pie(faculty_df['Count'], labels=faculty_df['Faculty'],
                                     autopct='%1.1f%%', startangle=90, colors=colors_pie,
                                     textprops={'fontsize': 10, 'fontweight': 'bold'})
ax2.set_title('Percentage Distribution of Events by Faculty',
              fontsize=14, fontweight='bold', pad=20)

# Improve percentage text visibility
for autotext in autotexts:
    autotext.set_color('black')
    autotext.set_fontsize(9)

plt.tight_layout()
plt.savefig('visuals/faculty_distribution.png', dpi=300, bbox_inches='tight')
print(f"\n✅ Saved: visuals/faculty_distribution.png")

# ============================================================================
# VISUALIZATION 2: Monthly Event Distribution
# ============================================================================

print("\n" + "="*80)
print("PREPARING VISUALIZATION 2: Monthly Distribution")
print("="*80)

# Find date columns
date_cols = [col for col in ev_df.columns if any(x in col.lower() for x in ['date', 'start', 'begin', 'when'])]
print(f"\nFound potential date columns: {date_cols}")

if not date_cols:
    print("⚠ No date column found. Using index as placeholder.")
    date_col = None
else:
    date_col = date_cols[0]
    print(f"Using date column: '{date_col}'")

# Parse dates and extract months
if date_col:
    # Try to parse the date column
    ev_df['parsed_date'] = pd.to_datetime(ev_df[date_col], errors='coerce')
    ev_df['month'] = ev_df['parsed_date'].dt.month
    ev_df['month_name'] = ev_df['parsed_date'].dt.strftime('%B')
    ev_df['year'] = ev_df['parsed_date'].dt.year

    # Remove rows with invalid dates
    valid_dates = ev_df['parsed_date'].notna()
    print(f"\nValid dates: {valid_dates.sum()} / {len(ev_df)} ({valid_dates.sum()/len(ev_df)*100:.1f}%)")

    ev_df_with_dates = ev_df[valid_dates].copy()

    # Count events by month
    month_counts = ev_df_with_dates.groupby('month').size().reset_index(name='count')
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # Create complete month range
    full_months = pd.DataFrame({'month': range(1, 13)})
    month_counts = full_months.merge(month_counts, on='month', how='left').fillna(0)
    month_counts['month_name'] = month_names
    month_counts['count'] = month_counts['count'].astype(int)

    print(f"\nMonthly distribution:")
    for _, row in month_counts.iterrows():
        print(f"  {row['month_name']}: {row['count']} events")

    # Create monthly visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Bar chart by month
    colors_month = plt.cm.viridis(np.linspace(0, 1, 12))
    bars = ax1.bar(month_counts['month_name'], month_counts['count'],
                   color=colors_month, edgecolor='black', linewidth=1.2)
    ax1.set_xlabel('Month', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Events', fontsize=12, fontweight='bold')
    ax1.set_title('Distribution of Events Throughout the Year',
                  fontsize=14, fontweight='bold', pad=20)
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontweight='bold')

    # Line chart with area fill
    ax2.plot(month_counts['month_name'], month_counts['count'],
             marker='o', linewidth=2.5, markersize=8, color='#2E86AB')
    ax2.fill_between(range(len(month_counts)), month_counts['count'],
                     alpha=0.3, color='#2E86AB')
    ax2.set_xlabel('Month', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Events', fontsize=12, fontweight='bold')
    ax2.set_title('Event Trend Throughout the Year',
                  fontsize=14, fontweight='bold', pad=20)
    ax2.grid(True, alpha=0.3)

    # Add value labels on points
    for i, (x, y) in enumerate(zip(month_counts['month_name'], month_counts['count'])):
        if y > 0:
            ax2.text(i, y, f'{int(y)}', ha='center', va='bottom',
                    fontweight='bold', fontsize=9)

    plt.tight_layout()
    plt.savefig('visuals/monthly_distribution.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved: visuals/monthly_distribution.png")

    # ============================================================================
    # VISUALIZATION 3: Faculty Distribution by Month (Heatmap)
    # ============================================================================

    print("\n" + "="*80)
    print("PREPARING VISUALIZATION 3: Faculty x Month Heatmap")
    print("="*80)

    # Create faculty-month matrix
    faculty_month_data = []

    for _, row in ev_df_with_dates.iterrows():
        month = row['month']
        faculties_str = row['faculties']
        if pd.notna(faculties_str) and faculties_str.strip():
            faculties_list = [f.strip() for f in str(faculties_str).split(',')]
            for fac in faculties_list:
                faculty_month_data.append({'Faculty': fac, 'Month': month})
        else:
            faculty_month_data.append({'Faculty': 'Other', 'Month': month})

    fm_df = pd.DataFrame(faculty_month_data)

    # Create pivot table
    heatmap_data = fm_df.groupby(['Faculty', 'Month']).size().reset_index(name='Count')
    heatmap_pivot = heatmap_data.pivot(index='Faculty', columns='Month', values='Count').fillna(0)

    # Reorder columns to be 1-12
    all_months = list(range(1, 13))
    for month in all_months:
        if month not in heatmap_pivot.columns:
            heatmap_pivot[month] = 0
    heatmap_pivot = heatmap_pivot[all_months]
    heatmap_pivot.columns = month_names

    # Sort by total events
    heatmap_pivot['Total'] = heatmap_pivot.sum(axis=1)
    heatmap_pivot = heatmap_pivot.sort_values('Total', ascending=False)
    heatmap_pivot = heatmap_pivot.drop('Total', axis=1)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(heatmap_pivot, annot=True, fmt='.0f', cmap='YlOrRd',
                linewidths=0.5, linecolor='gray', cbar_kws={'label': 'Number of Events'},
                ax=ax)
    ax.set_title('Faculty Event Distribution by Month',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Month', fontsize=12, fontweight='bold')
    ax.set_ylabel('Faculty', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('visuals/faculty_month_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved: visuals/faculty_month_heatmap.png")

else:
    print("\n⚠ Cannot create monthly visualizations without date information.")

print("\n" + "="*80)
print("VISUALIZATION GENERATION COMPLETE")
print("="*80)
print("\nGenerated files:")
print("  1. visuals/faculty_distribution.png - Overall faculty distribution (bar + pie)")
print("  2. visuals/monthly_distribution.png - Monthly event distribution (bar + line)")
print("  3. visuals/faculty_month_heatmap.png - Faculty x Month heatmap")