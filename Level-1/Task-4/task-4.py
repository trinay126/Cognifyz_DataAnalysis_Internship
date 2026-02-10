"""
Main Level-1 Task-4: Online Delivery Analysis
Analyzes online delivery availability and its impact on ratings
Generates visualizations and summary report

Folder Structure:
Y:\Main\
├── Data\
│   └── Dataset_.csv
└── Level-1\
    └── Task-4\
        ├── task4.py (this file)
        └── output\
            ├── online_delivery_analysis.png
            └── online_delivery_summary.txt
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def main():
    print("=" * 70)
    print("🚚 RESTAURANT ONLINE DELIVERY ANALYSIS - TASK 4")
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
    if "Level-1" in current_dir or "Level 1" in current_dir or "Task" in current_dir:
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
    required_cols = ['Has Online delivery', 'Aggregate rating']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"❌ ERROR: Missing columns: {', '.join(missing_cols)}")
        print(f"Available columns: {', '.join(df.columns)}")
        return
    
    print(f"✅ Required columns found: Has Online delivery, Aggregate rating")
    
    # Clean data
    initial_count = len(df)
    df_clean = df.dropna(subset=['Has Online delivery', 'Aggregate rating']).copy()
    removed = initial_count - len(df_clean)
    
    print(f"📊 Valid records: {len(df_clean):,}")
    print(f"🗑️  Removed (null values): {removed:,}")
    
    # ========== STEP 3: ANALYZE ONLINE DELIVERY ==========
    print("\n📊 STEP 3: Analyzing Online Delivery")
    print("-" * 70)
    
    # Count delivery availability
    delivery_counts = df_clean['Has Online delivery'].value_counts()
    total_restaurants = len(df_clean)
    
    # Calculate percentages
    delivery_percentages = (delivery_counts / total_restaurants * 100).round(2)
    
    # Get counts
    with_delivery = delivery_counts.get('Yes', 0)
    without_delivery = delivery_counts.get('No', 0)
    pct_with_delivery = delivery_percentages.get('Yes', 0)
    pct_without_delivery = delivery_percentages.get('No', 0)
    
    print(f"📊 Total Restaurants: {total_restaurants:,}")
    print(f"\n🚚 ONLINE DELIVERY AVAILABILITY:")
    print("-" * 70)
    print(f"✅ With Online Delivery:    {with_delivery:,} restaurants ({pct_with_delivery:.2f}%)")
    print(f"❌ Without Online Delivery: {without_delivery:,} restaurants ({pct_without_delivery:.2f}%)")
    
    # Calculate average ratings
    avg_ratings = df_clean.groupby('Has Online delivery')['Aggregate rating'].agg(['mean', 'median', 'std', 'count'])
    
    avg_rating_with = avg_ratings.loc['Yes', 'mean']
    median_rating_with = avg_ratings.loc['Yes', 'median']
    std_rating_with = avg_ratings.loc['Yes', 'std']
    
    avg_rating_without = avg_ratings.loc['No', 'mean']
    median_rating_without = avg_ratings.loc['No', 'median']
    std_rating_without = avg_ratings.loc['No', 'std']
    
    rating_difference = avg_rating_with - avg_rating_without
    
    print(f"\n⭐ AVERAGE RATINGS COMPARISON:")
    print("-" * 70)
    print(f"With Online Delivery:")
    print(f"  • Average Rating:  {avg_rating_with:.2f}/5.0")
    print(f"  • Median Rating:   {median_rating_with:.2f}/5.0")
    print(f"  • Std Deviation:   {std_rating_with:.2f}")
    
    print(f"\nWithout Online Delivery:")
    print(f"  • Average Rating:  {avg_rating_without:.2f}/5.0")
    print(f"  • Median Rating:   {median_rating_without:.2f}/5.0")
    print(f"  • Std Deviation:   {std_rating_without:.2f}")
    
    print(f"\n📈 RATING DIFFERENCE:")
    print(f"  • Restaurants with delivery rate {abs(rating_difference):.2f} points {'HIGHER' if rating_difference > 0 else 'LOWER'}")
    
    # ========== STEP 4: CREATE VISUALIZATIONS ==========
    print("\n📊 STEP 4: Creating Visualizations")
    print("-" * 70)
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("Set2")
    
    # Create figure with 4 subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Color scheme
    colors_delivery = ['#2ecc71', '#e74c3c']  # Green for Yes, Red for No
    
    # ===== Chart 1: Pie Chart - Delivery Availability =====
    ax1 = plt.subplot(2, 2, 1)
    
    labels = ['With Online Delivery', 'Without Online Delivery']
    sizes = [with_delivery, without_delivery]
    explode = (0.05, 0)
    
    wedges, texts, autotexts = ax1.pie(sizes, 
                                        labels=labels,
                                        autopct='%1.1f%%',
                                        colors=colors_delivery,
                                        explode=explode,
                                        startangle=90,
                                        shadow=True,
                                        textprops={'fontsize': 11, 'fontweight': 'bold'})
    
    ax1.set_title('🚚 Online Delivery Availability', fontsize=16, fontweight='bold', pad=20)
    
    # ===== Chart 2: Bar Chart - Restaurant Counts =====
    ax2 = plt.subplot(2, 2, 2)
    
    categories = ['With Delivery', 'Without Delivery']
    counts = [with_delivery, without_delivery]
    
    bars = ax2.bar(categories, counts, color=colors_delivery, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_title('📊 Restaurant Count Comparison', fontsize=16, fontweight='bold', pad=20)
    ax2.set_ylabel('Number of Restaurants', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for bar, count, pct in zip(bars, counts, [pct_with_delivery, pct_without_delivery]):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                 f'{count:,}\n({pct:.1f}%)',
                 ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # ===== Chart 3: Bar Chart - Average Ratings Comparison =====
    ax3 = plt.subplot(2, 2, 3)
    
    avg_ratings_list = [avg_rating_with, avg_rating_without]
    
    bars3 = ax3.bar(categories, avg_ratings_list, color=colors_delivery, alpha=0.8, 
                    edgecolor='black', linewidth=1.5)
    ax3.set_title('⭐ Average Rating Comparison', fontsize=16, fontweight='bold', pad=20)
    ax3.set_ylabel('Average Rating (out of 5.0)', fontsize=12, fontweight='bold')
    ax3.set_ylim(0, 5)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    ax3.axhline(y=3.0, color='gray', linestyle='--', alpha=0.5, label='Mid-point (3.0)')
    ax3.legend()
    
    # Add value labels
    for bar, rating in zip(bars3, avg_ratings_list):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{rating:.2f}',
                 ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # ===== Chart 4: Box Plot - Rating Distribution =====
    ax4 = plt.subplot(2, 2, 4)
    
    # Prepare data for box plot
    data_with = df_clean[df_clean['Has Online delivery'] == 'Yes']['Aggregate rating']
    data_without = df_clean[df_clean['Has Online delivery'] == 'No']['Aggregate rating']
    
    box_data = [data_with, data_without]
    
    bp = ax4.boxplot(box_data, labels=categories, patch_artist=True,
                     notch=True, showmeans=True,
                     boxprops=dict(facecolor='lightblue', alpha=0.7),
                     medianprops=dict(color='red', linewidth=2),
                     meanprops=dict(marker='D', markerfacecolor='green', markersize=8))
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], colors_delivery):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax4.set_title('📦 Rating Distribution (Box Plot)', fontsize=16, fontweight='bold', pad=20)
    ax4.set_ylabel('Rating', fontsize=12, fontweight='bold')
    ax4.set_ylim(0, 5)
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.5, label='Median'),
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='green', markersize=8, label='Mean')
    ]
    ax4.legend(handles=legend_elements, loc='upper right')
    
    # Add main title
    fig.suptitle('🚚 Online Delivery Analysis Dashboard', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    # ========== STEP 5: SAVE OUTPUT ==========
    print("\n💾 STEP 5: Saving Results")
    print("-" * 70)
    
    # Create output directory
    output_dir = os.path.join(os.getcwd(), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save chart
    output_path = os.path.join(output_dir, 'online_delivery_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Charts saved: {output_path}")
    
    # ========== STEP 6: CREATE SUMMARY REPORT ==========
    print("\n📋 STEP 6: Creating Summary Report")
    print("-" * 70)
    
    # Create detailed summary text file
    summary_path = os.path.join(output_dir, 'online_delivery_summary.txt')
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("RESTAURANT ONLINE DELIVERY ANALYSIS - SUMMARY REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Dataset: {os.path.basename(dataset_path)}\n")
        f.write(f"Total Restaurants Analyzed: {total_restaurants:,}\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("ONLINE DELIVERY AVAILABILITY\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"{'Status':<30} {'Count':<15} {'Percentage':<15}\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'With Online Delivery':<30} {with_delivery:<15,} {pct_with_delivery:<14.2f}%\n")
        f.write(f"{'Without Online Delivery':<30} {without_delivery:<15,} {pct_without_delivery:<14.2f}%\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("RATING COMPARISON\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("WITH ONLINE DELIVERY:\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Restaurant Count:      {with_delivery:,}\n")
        f.write(f"  Average Rating:        {avg_rating_with:.2f}/5.0\n")
        f.write(f"  Median Rating:         {median_rating_with:.2f}/5.0\n")
        f.write(f"  Standard Deviation:    {std_rating_with:.2f}\n\n")
        
        f.write("WITHOUT ONLINE DELIVERY:\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Restaurant Count:      {without_delivery:,}\n")
        f.write(f"  Average Rating:        {avg_rating_without:.2f}/5.0\n")
        f.write(f"  Median Rating:         {median_rating_without:.2f}/5.0\n")
        f.write(f"  Standard Deviation:    {std_rating_without:.2f}\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("KEY FINDINGS\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"1. ONLINE DELIVERY PENETRATION:\n")
        f.write(f"   • {pct_with_delivery:.2f}% of restaurants offer online delivery\n")
        f.write(f"   • {pct_without_delivery:.2f}% do not offer online delivery\n\n")
        
        f.write(f"2. RATING IMPACT:\n")
        if rating_difference > 0:
            f.write(f"   • Restaurants WITH online delivery have HIGHER ratings\n")
            f.write(f"   • Average difference: +{rating_difference:.2f} points\n")
            f.write(f"   • Improvement: {(rating_difference/avg_rating_without*100):.1f}%\n")
        else:
            f.write(f"   • Restaurants WITHOUT online delivery have HIGHER ratings\n")
            f.write(f"   • Average difference: {rating_difference:.2f} points\n")
        
        f.write(f"\n3. STATISTICAL INSIGHTS:\n")
        f.write(f"   • Median rating WITH delivery: {median_rating_with:.2f}\n")
        f.write(f"   • Median rating WITHOUT delivery: {median_rating_without:.2f}\n")
        f.write(f"   • More consistent ratings: {'WITH' if std_rating_with < std_rating_without else 'WITHOUT'} delivery\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("BUSINESS IMPLICATIONS\n")
        f.write("=" * 70 + "\n\n")
        
        if rating_difference > 0.5:
            f.write("• Strong positive correlation between online delivery and ratings\n")
            f.write("• Offering online delivery may improve customer satisfaction\n")
            f.write("• Consider expanding delivery services for better ratings\n")
        elif rating_difference > 0:
            f.write("• Slight positive correlation between online delivery and ratings\n")
            f.write("• Online delivery shows modest impact on ratings\n")
        else:
            f.write("• Online delivery does not guarantee higher ratings\n")
            f.write("• Focus on food quality and service over delivery availability\n")
        
        if pct_with_delivery < 30:
            f.write(f"\n• Low delivery penetration ({pct_with_delivery:.1f}%) suggests market opportunity\n")
        elif pct_with_delivery > 50:
            f.write(f"\n• High delivery adoption ({pct_with_delivery:.1f}%) indicates market maturity\n")
        
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
        'Delivery Status': ['With Online Delivery', 'Without Online Delivery'],
        'Count': [with_delivery, without_delivery],
        'Percentage': [f'{pct_with_delivery:.2f}%', f'{pct_without_delivery:.2f}%'],
        'Avg Rating': [f'{avg_rating_with:.2f}', f'{avg_rating_without:.2f}']
    })
    print(summary_df.to_string(index=False))
    print("=" * 70)
    
    print("\n" + "=" * 70)
    print("✅ TASK 4 COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\n📁 Output files location: {os.path.abspath(output_dir)}")
    print("   • online_delivery_analysis.png")
    print("   • online_delivery_summary.txt")
    
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