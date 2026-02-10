"""
Main Level-2 Task-2: Cuisine Combination Analysis
Identifies most common cuisine combinations and their ratings
Generates visualizations and summary report

Folder Structure:
Y:\Main\
├── Data\
│   └── Dataset_.csv
└── Level-2\
    └── Task-2\
        ├── task2.py (this file)
        └── output\
            ├── cuisine_combination_analysis.png
            └── cuisine_combination_summary.txt
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import os

def main():
    print("=" * 70)
    print("🍽️  CUISINE COMBINATION ANALYSIS - LEVEL 2 TASK 2")
    print("=" * 70)
    print(f"\n🔍 Current working directory: {os.getcwd()}")
    
    # ========== STEP 1: LOAD DATA ==========
    print("\n📂 STEP 1: Loading Dataset")
    print("-" * 70)
    
    # Smart path detection
    current_dir = os.getcwd()
    possible_paths = []
    filenames = ['Dataset.csv', 'Dataset_.csv', 'dataset.csv', 'dataset_.csv']
    
    if "Level-2" in current_dir or "Level 2" in current_dir or "Task" in current_dir:
        main_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
        for fname in filenames:
            possible_paths.append(os.path.join(main_dir, "Data", fname))
    
    for fname in filenames:
        possible_paths.append(os.path.join("Y:", "Main", "Data", fname))
    
    for i in range(1, 4):
        parent = os.path.abspath(os.path.join(current_dir, *[".." for _ in range(i)]))
        for fname in filenames:
            possible_paths.append(os.path.join(parent, "Data", fname))
    
    for fname in filenames:
        possible_paths.append(os.path.join(current_dir, fname))
    
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
    
    required_cols = ['Cuisines', 'Aggregate rating']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"❌ ERROR: Missing columns: {', '.join(missing_cols)}")
        print(f"Available columns: {', '.join(df.columns)}")
        return
    
    print(f"✅ Required columns found: Cuisines, Aggregate rating")
    
    initial_count = len(df)
    df_clean = df.dropna(subset=['Cuisines', 'Aggregate rating']).copy()
    df_clean = df_clean[df_clean['Cuisines'].str.strip() != '']
    removed = initial_count - len(df_clean)
    
    print(f"📊 Valid records: {len(df_clean):,}")
    print(f"🗑️  Removed (null/empty): {removed:,}")
    
    # ========== STEP 3: ANALYZE CUISINE COMBINATIONS ==========
    print("\n📊 STEP 3: Analyzing Cuisine Combinations")
    print("-" * 70)
    
    total_restaurants = len(df_clean)
    
    # Separate single vs multi-cuisine restaurants
    df_clean['Cuisine_Count'] = df_clean['Cuisines'].str.count(',') + 1
    df_multi = df_clean[df_clean['Cuisines'].str.contains(',', na=False)].copy()
    df_single = df_clean[~df_clean['Cuisines'].str.contains(',', na=False)].copy()
    
    multi_count = len(df_multi)
    single_count = len(df_single)
    multi_pct = (multi_count / total_restaurants * 100)
    single_pct = (single_count / total_restaurants * 100)
    
    print(f"📊 RESTAURANT TYPES:")
    print(f"  • Multi-cuisine restaurants:  {multi_count:,} ({multi_pct:.2f}%)")
    print(f"  • Single-cuisine restaurants: {single_count:,} ({single_pct:.2f}%)")
    
    # Extract cuisine combinations
    cuisine_combos = []
    combo_ratings = {}
    
    for idx, row in df_multi.iterrows():
        cuisines = row['Cuisines']
        rating = row['Aggregate rating']
        cuisine_list = [c.strip() for c in str(cuisines).split(',')]
        
        # Sort to normalize combinations
        cuisine_list.sort()
        combo = ', '.join(cuisine_list)
        cuisine_combos.append(combo)
        
        if combo not in combo_ratings:
            combo_ratings[combo] = []
        combo_ratings[combo].append(rating)
    
    # Count combinations
    combo_counts = Counter(cuisine_combos)
    total_combos = len(combo_counts)
    
    print(f"\n📊 COMBINATION STATISTICS:")
    print(f"  • Total unique combinations: {total_combos:,}")
    print(f"  • Average cuisines per restaurant: {df_clean['Cuisine_Count'].mean():.2f}")
    
    # Get top 15 combinations
    top15_combos = combo_counts.most_common(15)
    
    print(f"\n🏆 TOP 15 CUISINE COMBINATIONS:")
    print("-" * 70)
    print(f"{'Rank':<6} {'Count':<8} {'Combination':<50}")
    print("-" * 70)
    for i, (combo, count) in enumerate(top15_combos, 1):
        pct = (count / multi_count * 100)
        avg_rating = np.mean(combo_ratings[combo])
        print(f"{i:<6} {count:<8} {combo[:47]:<47} ({avg_rating:.2f}★)")
    
    # ========== STEP 4: RATING ANALYSIS ==========
    print("\n⭐ STEP 4: Analyzing Ratings by Combination Type")
    print("-" * 70)
    
    avg_rating_multi = df_multi['Aggregate rating'].mean()
    avg_rating_single = df_single['Aggregate rating'].mean()
    median_rating_multi = df_multi['Aggregate rating'].median()
    median_rating_single = df_single['Aggregate rating'].median()
    
    rating_diff = avg_rating_multi - avg_rating_single
    
    print(f"⭐ MULTI-CUISINE RESTAURANTS:")
    print(f"  • Average Rating: {avg_rating_multi:.2f}/5.0")
    print(f"  • Median Rating:  {median_rating_multi:.2f}/5.0")
    
    print(f"\n⭐ SINGLE-CUISINE RESTAURANTS:")
    print(f"  • Average Rating: {avg_rating_single:.2f}/5.0")
    print(f"  • Median Rating:  {median_rating_single:.2f}/5.0")
    
    print(f"\n📈 RATING DIFFERENCE:")
    print(f"  • Multi-cuisine restaurants rate {abs(rating_diff):.2f} points {'HIGHER' if rating_diff > 0 else 'LOWER'}")
    
    # Top rated combinations
    top10_combos_data = top15_combos[:10]
    top_rated_combos = sorted(
        [(combo, count, np.mean(combo_ratings[combo])) 
         for combo, count in top10_combos_data],
        key=lambda x: x[2], reverse=True
    )[:10]
    
    print(f"\n🌟 TOP 10 HIGHEST-RATED COMBINATIONS (from top 15):")
    print("-" * 70)
    for i, (combo, count, avg_rating) in enumerate(top_rated_combos, 1):
        print(f"{i:2}. {combo[:50]:<50} | {avg_rating:.2f}★ ({count} restaurants)")
    
    # ========== STEP 5: CREATE VISUALIZATIONS ==========
    print("\n📊 STEP 5: Creating Visualizations")
    print("-" * 70)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    
    fig = plt.figure(figsize=(18, 12))
    
    # ===== Chart 1: Pie Chart - Single vs Multi =====
    ax1 = plt.subplot(2, 3, 1)
    
    labels1 = ['Multi-Cuisine', 'Single-Cuisine']
    sizes1 = [multi_count, single_count]
    colors1 = ['#3498db', '#e74c3c']
    explode1 = (0.05, 0)
    
    wedges, texts, autotexts = ax1.pie(sizes1, labels=labels1, autopct='%1.1f%%',
                                        colors=colors1, explode=explode1,
                                        startangle=90, shadow=True,
                                        textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax1.set_title('🍽️  Restaurant Types', fontsize=14, fontweight='bold', pad=20)
    
    # ===== Chart 2: Top 10 Combinations Bar Chart =====
    ax2 = plt.subplot(2, 3, 2)
    
    top10_names = [combo[:30] + '...' if len(combo) > 30 else combo 
                   for combo, _ in top10_combos_data]
    top10_counts = [count for _, count in top10_combos_data]
    
    y_pos = np.arange(len(top10_names))
    colors2 = plt.cm.viridis(np.linspace(0, 1, len(top10_names)))
    
    bars2 = ax2.barh(y_pos, top10_counts, color=colors2, alpha=0.8, edgecolor='black')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(top10_names, fontsize=9)
    ax2.set_xlabel('Number of Restaurants', fontsize=12, fontweight='bold')
    ax2.set_title('🏆 Top 10 Cuisine Combinations', fontsize=14, fontweight='bold', pad=20)
    ax2.invert_yaxis()
    
    for i, (bar, count) in enumerate(zip(bars2, top10_counts)):
        ax2.text(count + max(top10_counts)*0.01, i, f'{count}',
                va='center', fontweight='bold', fontsize=9)
    
    # ===== Chart 3: Rating Comparison Bar =====
    ax3 = plt.subplot(2, 3, 3)
    
    categories = ['Multi-Cuisine', 'Single-Cuisine']
    ratings = [avg_rating_multi, avg_rating_single]
    colors3 = ['#2ecc71', '#e67e22']
    
    bars3 = ax3.bar(categories, ratings, color=colors3, alpha=0.8, 
                    edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('Average Rating', fontsize=12, fontweight='bold')
    ax3.set_title('⭐ Average Rating Comparison', fontsize=14, fontweight='bold', pad=20)
    ax3.set_ylim(0, 5)
    ax3.grid(axis='y', alpha=0.3)
    ax3.axhline(y=3.0, color='gray', linestyle='--', alpha=0.5)
    
    for bar, rating in zip(bars3, ratings):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{rating:.2f}',
                ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # ===== Chart 4: Top Rated Combinations =====
    ax4 = plt.subplot(2, 3, 4)
    
    top_rated_names = [combo[:30] + '...' if len(combo) > 30 else combo 
                       for combo, _, _ in top_rated_combos[:10]]
    top_rated_ratings = [rating for _, _, rating in top_rated_combos[:10]]
    
    y_pos4 = np.arange(len(top_rated_names))
    colors4 = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top_rated_names)))
    
    bars4 = ax4.barh(y_pos4, top_rated_ratings, color=colors4, alpha=0.8, edgecolor='black')
    ax4.set_yticks(y_pos4)
    ax4.set_yticklabels(top_rated_names, fontsize=9)
    ax4.set_xlabel('Average Rating', fontsize=12, fontweight='bold')
    ax4.set_title('🌟 Highest-Rated Combinations', fontsize=14, fontweight='bold', pad=20)
    ax4.set_xlim(0, 5)
    ax4.invert_yaxis()
    
    for i, (bar, rating) in enumerate(zip(bars4, top_rated_ratings)):
        ax4.text(rating + 0.05, i, f'{rating:.2f}',
                va='center', fontweight='bold', fontsize=9)
    
    # ===== Chart 5: Distribution of Cuisine Count =====
    ax5 = plt.subplot(2, 3, 5)
    
    cuisine_count_dist = df_clean['Cuisine_Count'].value_counts().sort_index()
    x_counts = cuisine_count_dist.index.tolist()
    y_counts = cuisine_count_dist.values.tolist()
    
    bars5 = ax5.bar(x_counts, y_counts, color='#9b59b6', alpha=0.8, edgecolor='black')
    ax5.set_xlabel('Number of Cuisines', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Number of Restaurants', fontsize=12, fontweight='bold')
    ax5.set_title('📊 Cuisines per Restaurant', fontsize=14, fontweight='bold', pad=20)
    ax5.grid(axis='y', alpha=0.3)
    
    for bar, count in zip(bars5, y_counts):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + max(y_counts)*0.01,
                f'{count:,}',
                ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # ===== Chart 6: Box Plot - Rating by Cuisine Count =====
    ax6 = plt.subplot(2, 3, 6)
    
    # Group by cuisine count
    box_data = []
    box_labels = []
    for count in sorted(df_clean['Cuisine_Count'].unique())[:6]:  # Limit to 6 for clarity
        data = df_clean[df_clean['Cuisine_Count'] == count]['Aggregate rating']
        if len(data) > 0:
            box_data.append(data)
            box_labels.append(str(count))
    
    bp = ax6.boxplot(box_data, labels=box_labels, patch_artist=True,
                     notch=True, showmeans=True)
    
    for patch in bp['boxes']:
        patch.set_facecolor('#3498db')
        patch.set_alpha(0.6)
    
    ax6.set_xlabel('Number of Cuisines', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Rating', fontsize=12, fontweight='bold')
    ax6.set_title('📦 Rating Distribution by Cuisine Count', fontsize=14, fontweight='bold', pad=20)
    ax6.grid(axis='y', alpha=0.3)
    
    fig.suptitle('🍽️  Cuisine Combination Analysis Dashboard', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    # ========== STEP 6: SAVE OUTPUT ==========
    print("\n💾 STEP 6: Saving Results")
    print("-" * 70)
    
    output_dir = os.path.join(os.getcwd(), 'output')
    os.makedirs(output_dir, exist_ok=True)
    print(f"✅ Output directory created: {output_dir}")
    
    output_path = os.path.join(output_dir, 'cuisine_combination_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Charts saved: {output_path}")
    
    # ========== STEP 7: CREATE SUMMARY REPORT ==========
    print("\n📋 STEP 7: Creating Summary Report")
    print("-" * 70)
    
    summary_path = os.path.join(output_dir, 'cuisine_combination_summary.txt')
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("CUISINE COMBINATION ANALYSIS - SUMMARY REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Dataset: {os.path.basename(dataset_path)}\n")
        f.write(f"Total Restaurants Analyzed: {total_restaurants:,}\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("RESTAURANT TYPES\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Multi-Cuisine Restaurants:  {multi_count:,} ({multi_pct:.2f}%)\n")
        f.write(f"Single-Cuisine Restaurants: {single_count:,} ({single_pct:.2f}%)\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("COMBINATION STATISTICS\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Total Unique Combinations:       {total_combos:,}\n")
        f.write(f"Average Cuisines per Restaurant: {df_clean['Cuisine_Count'].mean():.2f}\n")
        f.write(f"Max Cuisines in One Restaurant:  {df_clean['Cuisine_Count'].max()}\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("TOP 15 MOST COMMON CUISINE COMBINATIONS\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"{'Rank':<6} {'Count':<8} {'Avg Rating':<12} {'Combination':<45}\n")
        f.write("-" * 70 + "\n")
        for i, (combo, count) in enumerate(top15_combos, 1):
            avg_rating = np.mean(combo_ratings[combo])
            f.write(f"{i:<6} {count:<8} {avg_rating:<12.2f} {combo}\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("RATING ANALYSIS\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("MULTI-CUISINE RESTAURANTS:\n")
        f.write(f"  Average Rating:    {avg_rating_multi:.2f}/5.0\n")
        f.write(f"  Median Rating:     {median_rating_multi:.2f}/5.0\n")
        f.write(f"  Restaurant Count:  {multi_count:,}\n\n")
        
        f.write("SINGLE-CUISINE RESTAURANTS:\n")
        f.write(f"  Average Rating:    {avg_rating_single:.2f}/5.0\n")
        f.write(f"  Median Rating:     {median_rating_single:.2f}/5.0\n")
        f.write(f"  Restaurant Count:  {single_count:,}\n\n")
        
        f.write(f"RATING DIFFERENCE:\n")
        f.write(f"  Multi-cuisine restaurants rate {abs(rating_diff):.2f} points ")
        f.write(f"{'HIGHER' if rating_diff > 0 else 'LOWER'}\n")
        if rating_diff > 0:
            f.write(f"  Percentage improvement: {(rating_diff/avg_rating_single*100):.1f}%\n")
        f.write("\n")
        
        f.write("=" * 70 + "\n")
        f.write("TOP 10 HIGHEST-RATED COMBINATIONS\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"{'Rank':<6} {'Rating':<10} {'Count':<8} {'Combination':<45}\n")
        f.write("-" * 70 + "\n")
        for i, (combo, count, rating) in enumerate(top_rated_combos, 1):
            f.write(f"{i:<6} {rating:<10.2f} {count:<8} {combo}\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("KEY INSIGHTS\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("1. COMBINATION PREVALENCE:\n")
        f.write(f"   • {multi_pct:.1f}% of restaurants serve multiple cuisines\n")
        f.write(f"   • Most popular combination: {top15_combos[0][0]}\n")
        f.write(f"   • This combination appears in {top15_combos[0][1]} restaurants\n\n")
        
        f.write("2. RATING IMPACT:\n")
        if rating_diff > 0.2:
            f.write(f"   • Multi-cuisine restaurants significantly outperform\n")
            f.write(f"   • +{rating_diff:.2f} points higher rating on average\n")
            f.write(f"   • Offering variety may improve customer satisfaction\n")
        elif rating_diff > 0:
            f.write(f"   • Multi-cuisine restaurants have slightly better ratings\n")
            f.write(f"   • +{rating_diff:.2f} points difference\n")
        else:
            f.write(f"   • Single-cuisine restaurants have better ratings\n")
            f.write(f"   • Specialization may improve quality perception\n")
        f.write("\n")
        
        f.write("3. POPULAR PATTERNS:\n")
        north_indian_combos = sum(1 for combo, _ in top15_combos if 'North Indian' in combo)
        chinese_combos = sum(1 for combo, _ in top15_combos if 'Chinese' in combo)
        f.write(f"   • North Indian appears in {north_indian_combos}/15 top combinations\n")
        f.write(f"   • Chinese appears in {chinese_combos}/15 top combinations\n")
        f.write(f"   • These cuisines are frequently paired with others\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("BUSINESS RECOMMENDATIONS\n")
        f.write("=" * 70 + "\n\n")
        
        if rating_diff > 0:
            f.write("• Consider offering multiple cuisines to improve ratings\n")
            f.write(f"• Popular combinations like '{top15_combos[0][0]}' work well\n")
        else:
            f.write("• Specialization in single cuisine may yield better ratings\n")
            f.write("• Focus on quality over variety\n")
        
        f.write(f"• Study high-rated combinations for menu planning\n")
        f.write(f"• Regional preferences strongly influence combinations\n")
        
        f.write("\n" + "=" * 70 + "\n")
    
    print(f"✅ Summary saved: {summary_path}")
    
    if os.path.exists(summary_path):
        file_size = os.path.getsize(summary_path)
        print(f"✅ VERIFIED: Summary file exists ({file_size} bytes)")
    else:
        print(f"❌ WARNING: Summary file was not created!")
    
    print("\n📊 SUMMARY TABLE:")
    print("=" * 70)
    summary_df = pd.DataFrame({
        'Type': ['Multi-Cuisine', 'Single-Cuisine'],
        'Count': [multi_count, single_count],
        'Percentage': [f'{multi_pct:.2f}%', f'{single_pct:.2f}%'],
        'Avg Rating': [f'{avg_rating_multi:.2f}', f'{avg_rating_single:.2f}']
    })
    print(summary_df.to_string(index=False))
    print("=" * 70)
    
    print("\n" + "=" * 70)
    print("✅ LEVEL 2 TASK 2 COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\n📁 Output files location: {os.path.abspath(output_dir)}")
    print("   • cuisine_combination_analysis.png")
    print("   • cuisine_combination_summary.txt")
    
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