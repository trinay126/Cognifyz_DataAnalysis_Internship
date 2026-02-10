"""
Main Level-2 Task-1: Restaurant Ratings Analysis
Analyzes aggregate ratings distribution and voting patterns
Generates visualizations and summary report

Folder Structure:
Y:\Main\
├── Data\
│   └── Dataset_.csv
└── Level-2\
    └── Task-1\
        ├── task1.py (this file)
        └── output\
            ├── restaurant_ratings_analysis.png
            └── restaurant_ratings_summary.txt
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def main():
    print("=" * 70)
    print("⭐ RESTAURANT RATINGS ANALYSIS - LEVEL 2 TASK 1")
    print("=" * 70)
    print(f"\n🔍 Current working directory: {os.getcwd()}")
    
    # ========== STEP 1: LOAD DATA ==========
    print("\n📂 STEP 1: Loading Dataset")
    print("-" * 70)
    
    # Smart path detection - works from any subfolder
    current_dir = os.getcwd()
    
    # Try multiple possible paths and filenames
    possible_paths = []
    filenames = ['Dataset.csv', 'Dataset_.csv', 'dataset.csv', 'dataset_.csv']
    
    # Method 1: Navigate up from current directory
    if "Level-2" in current_dir or "Level 2" in current_dir or "Task" in current_dir:
        main_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
        for fname in filenames:
            possible_paths.append(os.path.join(main_dir, "Data", fname))
    
    # Method 2: Hardcoded Y:\Main
    for fname in filenames:
        possible_paths.append(os.path.join("Y:", "Main", "Data", fname))
    
    # Method 3: Check parent directories
    for i in range(1, 4):
        parent = os.path.abspath(os.path.join(current_dir, *[".." for _ in range(i)]))
        for fname in filenames:
            possible_paths.append(os.path.join(parent, "Data", fname))
    
    # Method 4: Check current directory
    for fname in filenames:
        possible_paths.append(os.path.join(current_dir, fname))
    
    # Find the first path that exists
    dataset_path = None
    print("🔍 Searching for dataset file...")
    for path in possible_paths:
        if os.path.exists(path):
            dataset_path = path
            print(f"✅ Found: {path}")
            break
    
    if dataset_path is None:
        print(f"\n❌ ERROR: Dataset file not found!")
        print(f"\n💡 Please place Dataset.csv or Dataset_.csv in Y:\\Main\\Data\\")
        return
    
    # Load dataset
    try:
        df = pd.read_csv(dataset_path, encoding='utf-8-sig')
        print(f"✅ Dataset loaded successfully")
        print(f"📊 Total records: {len(df):,}")
        print(f"📊 Total columns: {len(df.columns)}")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # ========== STEP 2: DATA VALIDATION ==========
    print("\n🔍 STEP 2: Validating Data")
    print("-" * 70)
    
    # Check required columns
    required_cols = ['Aggregate rating', 'Votes']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"❌ ERROR: Missing columns: {', '.join(missing_cols)}")
        print(f"Available columns: {', '.join(df.columns)}")
        return
    
    print(f"✅ Required columns found: Aggregate rating, Votes")
    
    # Clean data
    initial_count = len(df)
    df_clean = df.dropna(subset=['Aggregate rating', 'Votes']).copy()
    removed = initial_count - len(df_clean)
    
    print(f"📊 Valid records: {len(df_clean):,}")
    print(f"🗑️  Removed (null values): {removed:,}")
    
    # ========== STEP 3: ANALYZE RATINGS DISTRIBUTION ==========
    print("\n📊 STEP 3: Analyzing Ratings Distribution")
    print("-" * 70)
    
    total_restaurants = len(df_clean)
    
    # Overall statistics
    avg_rating = df_clean['Aggregate rating'].mean()
    median_rating = df_clean['Aggregate rating'].median()
    std_rating = df_clean['Aggregate rating'].std()
    min_rating = df_clean['Aggregate rating'].min()
    max_rating = df_clean['Aggregate rating'].max()
    
    print(f"📊 OVERALL RATING STATISTICS:")
    print(f"  • Total Restaurants:  {total_restaurants:,}")
    print(f"  • Average Rating:     {avg_rating:.2f}/5.0")
    print(f"  • Median Rating:      {median_rating:.2f}/5.0")
    print(f"  • Std Deviation:      {std_rating:.2f}")
    print(f"  • Min Rating:         {min_rating:.1f}")
    print(f"  • Max Rating:         {max_rating:.1f}")
    
    # Create rating ranges
    bins = [0, 1, 2, 3, 4, 5]
    labels = ['0-1 (Poor)', '1-2 (Below Avg)', '2-3 (Average)', '3-4 (Good)', '4-5 (Excellent)']
    df_clean['Rating_Range'] = pd.cut(df_clean['Aggregate rating'], bins=bins, labels=labels, include_lowest=True)
    
    # Count distribution by range
    rating_range_counts = df_clean['Rating_Range'].value_counts().sort_index()
    rating_range_pct = (rating_range_counts / total_restaurants * 100).round(2)
    
    print(f"\n⭐ RATING RANGE DISTRIBUTION:")
    print("-" * 70)
    print(f"{'Range':<20} {'Count':<12} {'Percentage':<12}")
    print("-" * 70)
    for range_name in labels:
        count = rating_range_counts.get(range_name, 0)
        pct = rating_range_pct.get(range_name, 0)
        print(f"{range_name:<20} {count:<12,} {pct:<11.2f}%")
    
    # Most common range
    most_common_range = rating_range_counts.idxmax()
    most_common_count = rating_range_counts.max()
    most_common_pct = rating_range_pct[most_common_range]
    
    print(f"\n🏆 MOST COMMON RATING RANGE:")
    print(f"  • Range: {most_common_range}")
    print(f"  • Count: {most_common_count:,} restaurants")
    print(f"  • Percentage: {most_common_pct:.2f}%")
    
    # ========== STEP 4: ANALYZE VOTES ==========
    print("\n📊 STEP 4: Analyzing Votes")
    print("-" * 70)
    
    avg_votes = df_clean['Votes'].mean()
    median_votes = df_clean['Votes'].median()
    std_votes = df_clean['Votes'].std()
    min_votes = df_clean['Votes'].min()
    max_votes = df_clean['Votes'].max()
    total_votes = df_clean['Votes'].sum()
    
    print(f"📊 VOTING STATISTICS:")
    print(f"  • Average Votes per Restaurant: {avg_votes:.2f}")
    print(f"  • Median Votes:                 {median_votes:.0f}")
    print(f"  • Std Deviation:                {std_votes:.2f}")
    print(f"  • Min Votes:                    {min_votes:,}")
    print(f"  • Max Votes:                    {max_votes:,}")
    print(f"  • Total Votes (all):            {total_votes:,}")
    
    # Vote ranges
    vote_bins = [0, 10, 50, 100, 500, 15000]
    vote_labels = ['0-10', '11-50', '51-100', '101-500', '500+']
    df_clean['Vote_Range'] = pd.cut(df_clean['Votes'], bins=vote_bins, labels=vote_labels, include_lowest=True)
    vote_range_counts = df_clean['Vote_Range'].value_counts().sort_index()
    
    print(f"\n📊 VOTE DISTRIBUTION:")
    for vote_range in vote_labels:
        count = vote_range_counts.get(vote_range, 0)
        pct = (count / total_restaurants * 100)
        print(f"  • {vote_range:<10} votes: {count:5,} restaurants ({pct:5.2f}%)")
    
    # Correlation between rating and votes
    correlation = df_clean['Aggregate rating'].corr(df_clean['Votes'])
    print(f"\n📈 CORRELATION:")
    print(f"  • Rating vs Votes: {correlation:.3f}")
    if abs(correlation) > 0.5:
        print(f"  • Strong {'positive' if correlation > 0 else 'negative'} correlation")
    elif abs(correlation) > 0.3:
        print(f"  • Moderate {'positive' if correlation > 0 else 'negative'} correlation")
    else:
        print(f"  • Weak correlation")
    
    # ========== STEP 5: CREATE VISUALIZATIONS ==========
    print("\n📊 STEP 5: Creating Visualizations")
    print("-" * 70)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("viridis")
    
    # Create figure with 6 subplots
    fig = plt.figure(figsize=(18, 12))
    
    # ===== Chart 1: Histogram - Rating Distribution =====
    ax1 = plt.subplot(2, 3, 1)
    
    ax1.hist(df_clean['Aggregate rating'], bins=50, color='#3498db', alpha=0.7, edgecolor='black')
    ax1.axvline(avg_rating, color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_rating:.2f}')
    ax1.axvline(median_rating, color='green', linestyle='--', linewidth=2, label=f'Median: {median_rating:.2f}')
    ax1.set_xlabel('Aggregate Rating', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title('📊 Rating Distribution', fontsize=14, fontweight='bold', pad=20)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ===== Chart 2: Bar Chart - Rating Ranges =====
    ax2 = plt.subplot(2, 3, 2)
    
    colors2 = ['#e74c3c', '#e67e22', '#f39c12', '#2ecc71', '#27ae60']
    x_pos = np.arange(len(labels))
    counts = [rating_range_counts.get(label, 0) for label in labels]
    
    bars2 = ax2.bar(x_pos, counts, color=colors2, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([l.split(' ')[0] for l in labels], rotation=45, ha='right')
    ax2.set_ylabel('Number of Restaurants', fontsize=12, fontweight='bold')
    ax2.set_title('⭐ Restaurants by Rating Range', fontsize=14, fontweight='bold', pad=20)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, count in zip(bars2, counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                 f'{count:,}',
                 ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # ===== Chart 3: Pie Chart - Rating Ranges =====
    ax3 = plt.subplot(2, 3, 3)
    
    explode = [0.05 if label == most_common_range else 0 for label in labels]
    wedges, texts, autotexts = ax3.pie(counts, labels=[l.split(' ')[0] for l in labels],
                                        autopct='%1.1f%%', colors=colors2,
                                        explode=explode, startangle=90, shadow=True,
                                        textprops={'fontsize': 10, 'fontweight': 'bold'})
    ax3.set_title('🥧 Rating Distribution (%)', fontsize=14, fontweight='bold', pad=20)
    
    # ===== Chart 4: Histogram - Votes Distribution =====
    ax4 = plt.subplot(2, 3, 4)
    
    ax4.hist(df_clean['Votes'], bins=50, color='#9b59b6', alpha=0.7, edgecolor='black')
    ax4.axvline(avg_votes, color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_votes:.0f}')
    ax4.axvline(median_votes, color='green', linestyle='--', linewidth=2, label=f'Median: {median_votes:.0f}')
    ax4.set_xlabel('Number of Votes', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax4.set_title('📊 Votes Distribution', fontsize=14, fontweight='bold', pad=20)
    ax4.set_xlim(0, 1000)  # Focus on main distribution
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # ===== Chart 5: Scatter Plot - Rating vs Votes =====
    ax5 = plt.subplot(2, 3, 5)
    
    scatter = ax5.scatter(df_clean['Votes'], df_clean['Aggregate rating'],
                         alpha=0.5, s=20, c=df_clean['Aggregate rating'],
                         cmap='RdYlGn', edgecolors='none')
    ax5.set_xlabel('Number of Votes', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Aggregate Rating', fontsize=12, fontweight='bold')
    ax5.set_title(f'📈 Rating vs Votes (r={correlation:.3f})', fontsize=14, fontweight='bold', pad=20)
    ax5.set_xlim(0, 1000)  # Focus on main cluster
    ax5.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax5)
    cbar.set_label('Rating', fontsize=10, fontweight='bold')
    
    # ===== Chart 6: Box Plot - Ratings by Range =====
    ax6 = plt.subplot(2, 3, 6)
    
    # Prepare data for box plot
    box_data = [df_clean[df_clean['Rating_Range'] == label]['Aggregate rating'].values 
                for label in labels if len(df_clean[df_clean['Rating_Range'] == label]) > 0]
    box_labels = [l.split(' ')[0] for l in labels if l in rating_range_counts.index]
    
    bp = ax6.boxplot(box_data, labels=box_labels, patch_artist=True,
                     notch=True, showmeans=True)
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], colors2[:len(box_data)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax6.set_ylabel('Rating', fontsize=12, fontweight='bold')
    ax6.set_title('📦 Rating Distribution by Range', fontsize=14, fontweight='bold', pad=20)
    ax6.grid(axis='y', alpha=0.3)
    ax6.tick_params(axis='x', rotation=45)
    
    # Add main title
    fig.suptitle('⭐ Restaurant Ratings & Votes Analysis Dashboard', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    # ========== STEP 6: SAVE OUTPUT ==========
    print("\n💾 STEP 6: Saving Results")
    print("-" * 70)
    
    # Create output directory
    output_dir = os.path.join(os.getcwd(), 'output')
    os.makedirs(output_dir, exist_ok=True)
    print(f"✅ Output directory created: {output_dir}")
    
    # Save chart
    output_path = os.path.join(output_dir, 'restaurant_ratings_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Charts saved: {output_path}")
    
    # ========== STEP 7: CREATE SUMMARY REPORT ==========
    print("\n📋 STEP 7: Creating Summary Report")
    print("-" * 70)
    
    # Create detailed summary text file
    summary_path = os.path.join(output_dir, 'restaurant_ratings_summary.txt')
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("RESTAURANT RATINGS ANALYSIS - SUMMARY REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Dataset: {os.path.basename(dataset_path)}\n")
        f.write(f"Total Restaurants Analyzed: {total_restaurants:,}\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("RATING STATISTICS\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Average Rating:          {avg_rating:.2f}/5.0\n")
        f.write(f"Median Rating:           {median_rating:.2f}/5.0\n")
        f.write(f"Standard Deviation:      {std_rating:.2f}\n")
        f.write(f"Minimum Rating:          {min_rating:.1f}\n")
        f.write(f"Maximum Rating:          {max_rating:.1f}\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("RATING RANGE DISTRIBUTION\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"{'Range':<22} {'Count':<14} {'Percentage':<12}\n")
        f.write("-" * 70 + "\n")
        for range_name in labels:
            count = rating_range_counts.get(range_name, 0)
            pct = rating_range_pct.get(range_name, 0)
            f.write(f"{range_name:<22} {count:<14,} {pct:<11.2f}%\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("MOST COMMON RATING RANGE\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Range:             {most_common_range}\n")
        f.write(f"Restaurant Count:  {most_common_count:,}\n")
        f.write(f"Percentage:        {most_common_pct:.2f}%\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("VOTING STATISTICS\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Average Votes per Restaurant:  {avg_votes:.2f}\n")
        f.write(f"Median Votes:                  {median_votes:.0f}\n")
        f.write(f"Standard Deviation:            {std_votes:.2f}\n")
        f.write(f"Minimum Votes:                 {min_votes:,}\n")
        f.write(f"Maximum Votes:                 {max_votes:,}\n")
        f.write(f"Total Votes (All):             {total_votes:,}\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("VOTE DISTRIBUTION\n")
        f.write("=" * 70 + "\n\n")
        
        for vote_range in vote_labels:
            count = vote_range_counts.get(vote_range, 0)
            pct = (count / total_restaurants * 100)
            f.write(f"{vote_range:<15} votes: {count:6,} restaurants ({pct:6.2f}%)\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("CORRELATION ANALYSIS\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Correlation (Rating vs Votes):  {correlation:.3f}\n")
        if abs(correlation) > 0.5:
            strength = "Strong"
        elif abs(correlation) > 0.3:
            strength = "Moderate"
        else:
            strength = "Weak"
        direction = "positive" if correlation > 0 else "negative"
        f.write(f"Interpretation:                 {strength} {direction} correlation\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("KEY INSIGHTS\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"1. RATING DISTRIBUTION:\n")
        f.write(f"   • The most common rating range is {most_common_range}\n")
        f.write(f"   • {most_common_pct:.1f}% of restaurants fall in this range\n")
        f.write(f"   • Average rating ({avg_rating:.2f}) is {'above' if avg_rating > 2.5 else 'below'} mid-point (2.5)\n\n")
        
        f.write(f"2. VOTING PATTERNS:\n")
        f.write(f"   • Average restaurant receives {avg_votes:.0f} votes\n")
        f.write(f"   • Median votes ({median_votes:.0f}) is {'much ' if avg_votes/median_votes > 2 else ''}lower than mean\n")
        f.write(f"   • This indicates some restaurants receive very high votes\n\n")
        
        f.write(f"3. RATING-VOTES RELATIONSHIP:\n")
        if correlation > 0.3:
            f.write(f"   • Positive correlation suggests higher-rated restaurants\n")
            f.write(f"     tend to receive more votes\n")
            f.write(f"   • Popular restaurants (more votes) generally have better ratings\n")
        elif correlation < -0.3:
            f.write(f"   • Negative correlation suggests lower-rated restaurants\n")
            f.write(f"     tend to receive more votes\n")
        else:
            f.write(f"   • Weak correlation means votes and ratings are largely independent\n")
            f.write(f"   • Number of votes doesn't strongly predict rating quality\n")
        
        f.write("\n" + "=" * 70 + "\n")
    
    print(f"✅ Summary saved: {summary_path}")
    
    # Verify file was created
    if os.path.exists(summary_path):
        file_size = os.path.getsize(summary_path)
        print(f"✅ VERIFIED: Summary file exists ({file_size} bytes)")
    else:
        print(f"❌ WARNING: Summary file was not created!")
    
    # Display summary table in console
    print("\n📊 SUMMARY TABLE:")
    print("=" * 70)
    summary_df = pd.DataFrame({
        'Rating Range': [l.split(' ')[0] for l in labels],
        'Count': [rating_range_counts.get(label, 0) for label in labels],
        'Percentage': [f"{rating_range_pct.get(label, 0):.2f}%" for label in labels]
    })
    print(summary_df.to_string(index=False))
    print("=" * 70)
    
    print("\n" + "=" * 70)
    print("✅ LEVEL 2 TASK 1 COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\n📁 Output files location: {os.path.abspath(output_dir)}")
    print("   • restaurant_ratings_analysis.png")
    print("   • restaurant_ratings_summary.txt")
    
    # Show plot at the very end after all files are saved
    print("\n📊 Displaying chart...")
    plt.show()
    plt.close()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user")
    except Exception as e:
        print(f"\n\n❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()