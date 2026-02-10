"""
Main Level-2 Task-4: Restaurant Chains Analysis
Identifies restaurant chains and analyzes their ratings and popularity
Generates visualizations and summary report

Folder Structure:
Y:\Main\
├── Data\
│   └── Dataset_.csv
└── Level-2\
    └── Task-4\
        ├── task4.py (this file)
        └── output\
            ├── restaurant_chains_analysis.png
            └── restaurant_chains_summary.txt
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import os

def main():
    print("=" * 70)
    print("🏢 RESTAURANT CHAINS ANALYSIS - LEVEL 2 TASK 4")
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
    
    required_cols = ['Restaurant Name', 'Aggregate rating', 'Votes']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"❌ ERROR: Missing columns: {', '.join(missing_cols)}")
        print(f"Available columns: {', '.join(df.columns)}")
        return
    
    print(f"✅ Required columns found: Restaurant Name, Aggregate rating, Votes")
    
    initial_count = len(df)
    df_clean = df.dropna(subset=['Restaurant Name', 'Aggregate rating', 'Votes']).copy()
    df_clean = df_clean[df_clean['Restaurant Name'].str.strip() != '']
    removed = initial_count - len(df_clean)
    
    print(f"📊 Valid records: {len(df_clean):,}")
    print(f"🗑️  Removed (null/empty): {removed:,}")
    
    # ========== STEP 3: IDENTIFY RESTAURANT CHAINS ==========
    print("\n📊 STEP 3: Identifying Restaurant Chains")
    print("-" * 70)
    
    total_restaurants = len(df_clean)
    
    # Count occurrences of each restaurant name
    name_counts = df_clean['Restaurant Name'].value_counts()
    total_unique_names = len(name_counts)
    
    # Identify chains (appearing more than once)
    chains = name_counts[name_counts > 1]
    single_locations = name_counts[name_counts == 1]
    
    num_chains = len(chains)
    num_singles = len(single_locations)
    total_chain_locations = chains.sum()
    
    print(f"📊 OVERALL STATISTICS:")
    print(f"  • Total unique restaurant names: {total_unique_names:,}")
    print(f"  • Restaurant chains (>1 location): {num_chains:,}")
    print(f"  • Single-location restaurants:     {num_singles:,}")
    print(f"  • Total chain locations:           {total_chain_locations:,}")
    print(f"  • Chain percentage:                {(total_chain_locations/total_restaurants*100):.2f}%")
    
    # Mark chains in dataframe
    df_clean['Is_Chain'] = df_clean['Restaurant Name'].map(lambda x: name_counts[x] > 1)
    df_clean['Location_Count'] = df_clean['Restaurant Name'].map(name_counts)
    
    # Get top chains
    top20_chains = chains.head(20)
    
    print(f"\n🏆 TOP 20 RESTAURANT CHAINS:")
    print("-" * 70)
    print(f"{'Rank':<6} {'Chain Name':<35} {'Locations':<12} {'Avg Rating':<12} {'Avg Votes':<12}")
    print("-" * 70)
    
    chain_stats = {}
    for i, (chain_name, location_count) in enumerate(top20_chains.items(), 1):
        chain_data = df_clean[df_clean['Restaurant Name'] == chain_name]
        avg_rating = chain_data['Aggregate rating'].mean()
        avg_votes = chain_data['Votes'].mean()
        total_votes = chain_data['Votes'].sum()
        
        chain_stats[chain_name] = {
            'locations': location_count,
            'avg_rating': avg_rating,
            'avg_votes': avg_votes,
            'total_votes': total_votes
        }
        
        print(f"{i:<6} {chain_name[:34]:<35} {location_count:<12} {avg_rating:<12.2f} {avg_votes:<12.0f}")
    
    # ========== STEP 4: ANALYZE CHAINS VS SINGLES ==========
    print("\n⭐ STEP 4: Comparing Chains vs Single Locations")
    print("-" * 70)
    
    chains_df = df_clean[df_clean['Is_Chain'] == True]
    singles_df = df_clean[df_clean['Is_Chain'] == False]
    
    chain_avg_rating = chains_df['Aggregate rating'].mean()
    single_avg_rating = singles_df['Aggregate rating'].mean()
    chain_median_rating = chains_df['Aggregate rating'].median()
    single_median_rating = singles_df['Aggregate rating'].median()
    
    chain_avg_votes = chains_df['Votes'].mean()
    single_avg_votes = singles_df['Votes'].mean()
    
    rating_diff = chain_avg_rating - single_avg_rating
    votes_diff = chain_avg_votes - single_avg_votes
    
    print(f"CHAIN RESTAURANTS ({len(chains_df):,} locations):")
    print(f"  • Average Rating:  {chain_avg_rating:.2f}/5.0")
    print(f"  • Median Rating:   {chain_median_rating:.2f}/5.0")
    print(f"  • Average Votes:   {chain_avg_votes:.0f}")
    
    print(f"\nSINGLE LOCATIONS ({len(singles_df):,} restaurants):")
    print(f"  • Average Rating:  {single_avg_rating:.2f}/5.0")
    print(f"  • Median Rating:   {single_median_rating:.2f}/5.0")
    print(f"  • Average Votes:   {single_avg_votes:.0f}")
    
    print(f"\nDIFFERENCES:")
    print(f"  • Rating:  Chains are {abs(rating_diff):.2f} points {'HIGHER' if rating_diff > 0 else 'LOWER'}")
    print(f"  • Votes:   Chains have {abs(votes_diff):.0f} {'MORE' if votes_diff > 0 else 'FEWER'} votes on average")
    
    # Top rated chains
    top_rated_chains = sorted(
        [(name, stats['locations'], stats['avg_rating']) 
         for name, stats in chain_stats.items()],
        key=lambda x: x[2], reverse=True
    )[:10]
    
    print(f"\n🌟 TOP 10 HIGHEST-RATED CHAINS:")
    print("-" * 70)
    for i, (chain, locations, rating) in enumerate(top_rated_chains, 1):
        print(f"{i:2}. {chain[:45]:<45} | {rating:.2f}★ ({locations} locations)")
    
    # Most popular chains (by votes)
    top_popular_chains = sorted(
        [(name, stats['locations'], stats['avg_votes']) 
         for name, stats in chain_stats.items()],
        key=lambda x: x[2], reverse=True
    )[:10]
    
    print(f"\n👥 TOP 10 MOST POPULAR CHAINS (by avg votes):")
    print("-" * 70)
    for i, (chain, locations, votes) in enumerate(top_popular_chains, 1):
        print(f"{i:2}. {chain[:45]:<45} | {votes:.0f} votes ({locations} locations)")
    
    # ========== STEP 5: CREATE VISUALIZATIONS ==========
    print("\n📊 STEP 5: Creating Visualizations")
    print("-" * 70)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    
    fig = plt.figure(figsize=(18, 12))
    
    # ===== Chart 1: Pie Chart - Chains vs Singles =====
    ax1 = plt.subplot(2, 3, 1)
    
    labels1 = ['Chain Locations', 'Single Locations']
    sizes1 = [total_chain_locations, len(singles_df)]
    colors1 = ['#3498db', '#e74c3c']
    explode1 = (0.05, 0)
    
    wedges, texts, autotexts = ax1.pie(sizes1, labels=labels1, autopct='%1.1f%%',
                                        colors=colors1, explode=explode1,
                                        startangle=90, shadow=True,
                                        textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax1.set_title('🏢 Chain vs Single Locations', fontsize=14, fontweight='bold', pad=20)
    
    # ===== Chart 2: Top 15 Chains Bar Chart =====
    ax2 = plt.subplot(2, 3, 2)
    
    top15_names = [name[:25] + '...' if len(name) > 25 else name 
                   for name in top20_chains.head(15).index]
    top15_counts = top20_chains.head(15).values
    
    y_pos = np.arange(len(top15_names))
    colors2 = plt.cm.viridis(np.linspace(0, 1, len(top15_names)))
    
    bars2 = ax2.barh(y_pos, top15_counts, color=colors2, alpha=0.8, edgecolor='black')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(top15_names, fontsize=9)
    ax2.set_xlabel('Number of Locations', fontsize=12, fontweight='bold')
    ax2.set_title('🏆 Top 15 Chains by Locations', fontsize=14, fontweight='bold', pad=20)
    ax2.invert_yaxis()
    
    for i, (bar, count) in enumerate(zip(bars2, top15_counts)):
        ax2.text(count + max(top15_counts)*0.01, i, f'{count}',
                va='center', fontweight='bold', fontsize=9)
    
    # ===== Chart 3: Rating Comparison =====
    ax3 = plt.subplot(2, 3, 3)
    
    categories = ['Chain\nRestaurants', 'Single\nLocations']
    ratings = [chain_avg_rating, single_avg_rating]
    colors3 = ['#2ecc71', '#e67e22']
    
    bars3 = ax3.bar(categories, ratings, color=colors3, alpha=0.8, 
                    edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('Average Rating', fontsize=12, fontweight='bold')
    ax3.set_title('⭐ Rating Comparison', fontsize=14, fontweight='bold', pad=20)
    ax3.set_ylim(0, 5)
    ax3.grid(axis='y', alpha=0.3)
    ax3.axhline(y=3.0, color='gray', linestyle='--', alpha=0.5)
    
    for bar, rating in zip(bars3, ratings):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{rating:.2f}',
                ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # ===== Chart 4: Top Rated Chains =====
    ax4 = plt.subplot(2, 3, 4)
    
    top_rated_names = [chain[:25] + '...' if len(chain) > 25 else chain 
                       for chain, _, _ in top_rated_chains[:10]]
    top_rated_ratings = [rating for _, _, rating in top_rated_chains[:10]]
    
    y_pos4 = np.arange(len(top_rated_names))
    colors4 = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top_rated_names)))
    
    bars4 = ax4.barh(y_pos4, top_rated_ratings, color=colors4, alpha=0.8, edgecolor='black')
    ax4.set_yticks(y_pos4)
    ax4.set_yticklabels(top_rated_names, fontsize=9)
    ax4.set_xlabel('Average Rating', fontsize=12, fontweight='bold')
    ax4.set_title('🌟 Highest-Rated Chains', fontsize=14, fontweight='bold', pad=20)
    ax4.set_xlim(0, 5)
    ax4.invert_yaxis()
    
    for i, (bar, rating) in enumerate(zip(bars4, top_rated_ratings)):
        ax4.text(rating + 0.05, i, f'{rating:.2f}',
                va='center', fontweight='bold', fontsize=9)
    
    # ===== Chart 5: Votes Comparison =====
    ax5 = plt.subplot(2, 3, 5)
    
    votes_data = [chain_avg_votes, single_avg_votes]
    
    bars5 = ax5.bar(categories, votes_data, color=['#9b59b6', '#f39c12'], 
                    alpha=0.8, edgecolor='black', linewidth=1.5)
    ax5.set_ylabel('Average Votes', fontsize=12, fontweight='bold')
    ax5.set_title('👥 Popularity (Votes) Comparison', fontsize=14, fontweight='bold', pad=20)
    ax5.grid(axis='y', alpha=0.3)
    
    for bar, votes in zip(bars5, votes_data):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + max(votes_data)*0.02,
                f'{votes:.0f}',
                ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # ===== Chart 6: Scatter - Locations vs Rating =====
    ax6 = plt.subplot(2, 3, 6)
    
    # Get data for scatter plot
    scatter_chains = list(chain_stats.keys())[:30]  # Top 30 for clarity
    scatter_locations = [chain_stats[c]['locations'] for c in scatter_chains]
    scatter_ratings = [chain_stats[c]['avg_rating'] for c in scatter_chains]
    scatter_votes = [chain_stats[c]['avg_votes'] for c in scatter_chains]
    
    scatter = ax6.scatter(scatter_locations, scatter_ratings,
                         s=[v*2 for v in scatter_votes], alpha=0.6,
                         c=scatter_ratings, cmap='RdYlGn',
                         edgecolors='black', linewidth=0.5)
    ax6.set_xlabel('Number of Locations', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Average Rating', fontsize=12, fontweight='bold')
    ax6.set_title('📈 Locations vs Rating\n(bubble size = votes)', fontsize=14, fontweight='bold', pad=20)
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim(0, 5)
    
    cbar6 = plt.colorbar(scatter, ax=ax6)
    cbar6.set_label('Rating', fontsize=10, fontweight='bold')
    
    fig.suptitle('🏢 Restaurant Chains Analysis Dashboard', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    # ========== STEP 6: SAVE OUTPUT ==========
    print("\n💾 STEP 6: Saving Results")
    print("-" * 70)
    
    output_dir = os.path.join(os.getcwd(), 'output')
    os.makedirs(output_dir, exist_ok=True)
    print(f"✅ Output directory created: {output_dir}")
    
    output_path = os.path.join(output_dir, 'restaurant_chains_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Charts saved: {output_path}")
    
    # ========== STEP 7: CREATE SUMMARY REPORT ==========
    print("\n📋 STEP 7: Creating Summary Report")
    print("-" * 70)
    
    summary_path = os.path.join(output_dir, 'restaurant_chains_summary.txt')
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("RESTAURANT CHAINS ANALYSIS - SUMMARY REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Dataset: {os.path.basename(dataset_path)}\n")
        f.write(f"Total Restaurants Analyzed: {total_restaurants:,}\n")
        f.write(f"Unique Restaurant Names: {total_unique_names:,}\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("CHAIN IDENTIFICATION\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Restaurant Chains (>1 location):  {num_chains:,}\n")
        f.write(f"Single-Location Restaurants:       {num_singles:,}\n")
        f.write(f"Total Chain Locations:             {total_chain_locations:,}\n")
        f.write(f"Chain Percentage:                  {(total_chain_locations/total_restaurants*100):.2f}%\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("TOP 20 RESTAURANT CHAINS\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"{'Rank':<6} {'Chain Name':<40} {'Locations':<12} {'Avg Rating':<12} {'Avg Votes':<12}\n")
        f.write("-" * 70 + "\n")
        for i, (chain_name, stats) in enumerate(list(chain_stats.items())[:20], 1):
            f.write(f"{i:<6} {chain_name:<40} {stats['locations']:<12} {stats['avg_rating']:<12.2f} {stats['avg_votes']:<12.0f}\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("CHAIN VS SINGLE LOCATION COMPARISON\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("CHAIN RESTAURANTS:\n")
        f.write(f"  Total Locations:   {len(chains_df):,}\n")
        f.write(f"  Average Rating:    {chain_avg_rating:.2f}/5.0\n")
        f.write(f"  Median Rating:     {chain_median_rating:.2f}/5.0\n")
        f.write(f"  Average Votes:     {chain_avg_votes:.0f}\n\n")
        
        f.write("SINGLE LOCATIONS:\n")
        f.write(f"  Total Restaurants: {len(singles_df):,}\n")
        f.write(f"  Average Rating:    {single_avg_rating:.2f}/5.0\n")
        f.write(f"  Median Rating:     {single_median_rating:.2f}/5.0\n")
        f.write(f"  Average Votes:     {single_avg_votes:.0f}\n\n")
        
        f.write("DIFFERENCES:\n")
        f.write(f"  Rating Difference: {rating_diff:+.2f} points ")
        f.write(f"(chains rate {'HIGHER' if rating_diff > 0 else 'LOWER'})\n")
        f.write(f"  Votes Difference:  {votes_diff:+.0f} votes ")
        f.write(f"(chains are {'MORE' if votes_diff > 0 else 'LESS'} popular)\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("TOP 10 HIGHEST-RATED CHAINS\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"{'Rank':<6} {'Chain Name':<45} {'Rating':<10} {'Locations':<10}\n")
        f.write("-" * 70 + "\n")
        for i, (chain, locations, rating) in enumerate(top_rated_chains, 1):
            f.write(f"{i:<6} {chain:<45} {rating:<10.2f} {locations:<10}\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("TOP 10 MOST POPULAR CHAINS (by votes)\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"{'Rank':<6} {'Chain Name':<45} {'Avg Votes':<12} {'Locations':<10}\n")
        f.write("-" * 70 + "\n")
        for i, (chain, locations, votes) in enumerate(top_popular_chains, 1):
            f.write(f"{i:<6} {chain:<45} {votes:<12.0f} {locations:<10}\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("KEY INSIGHTS\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("1. CHAIN PRESENCE:\n")
        f.write(f"   • {num_chains} restaurant chains identified in dataset\n")
        f.write(f"   • Chains represent {(total_chain_locations/total_restaurants*100):.1f}% of all locations\n")
        f.write(f"   • Largest chain: {top20_chains.index[0]} ({top20_chains.iloc[0]} locations)\n\n")
        
        f.write("2. PERFORMANCE COMPARISON:\n")
        if rating_diff > 0.1:
            f.write(f"   • Chains significantly outperform single locations\n")
            f.write(f"   • +{rating_diff:.2f} points higher rating on average\n")
            f.write(f"   • Brand consistency may improve customer satisfaction\n")
        elif rating_diff > 0:
            f.write(f"   • Chains have slightly better ratings\n")
            f.write(f"   • +{rating_diff:.2f} points difference\n")
        else:
            f.write(f"   • Single locations outperform chains\n")
            f.write(f"   • Independent restaurants may offer better quality\n")
        f.write("\n")
        
        f.write("3. POPULARITY PATTERNS:\n")
        if votes_diff > 0:
            f.write(f"   • Chains receive {votes_diff:.0f} more votes on average\n")
            f.write(f"   • Brand recognition drives higher engagement\n")
        else:
            f.write(f"   • Single locations more popular in voting\n")
        
        top_chain_type = "international" if any(c in top20_chains.index[0].lower() 
                                                for c in ['mcdon', 'subway', 'pizza', 'domino']) else "local"
        f.write(f"   • Top chain ({top20_chains.index[0]}) is {top_chain_type}\n\n")
        
        f.write("4. RATING VS SCALE:\n")
        f.write(f"   • High-rated chains maintain quality across locations\n")
        f.write(f"   • {top_rated_chains[0][0]} achieves {top_rated_chains[0][2]:.2f} rating\n")
        f.write(f"   • Scale doesn't necessarily compromise quality\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("BUSINESS RECOMMENDATIONS\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("FOR CHAIN OPERATORS:\n")
        f.write(f"  • Study top-rated chains' quality control methods\n")
        f.write(f"  • Maintain consistency across all locations\n")
        f.write(f"  • Brand recognition is a significant advantage\n\n")
        
        f.write("FOR INDEPENDENT RESTAURANTS:\n")
        if rating_diff <= 0:
            f.write(f"  • Current advantage in ratings should be maintained\n")
            f.write(f"  • Emphasize unique offerings and quality\n")
        else:
            f.write(f"  • Consider standardizing successful practices\n")
            f.write(f"  • Build local brand recognition\n")
        
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
        'Category': ['Chains', 'Singles'],
        'Count': [len(chains_df), len(singles_df)],
        'Avg Rating': [f'{chain_avg_rating:.2f}', f'{single_avg_rating:.2f}'],
        'Avg Votes': [f'{chain_avg_votes:.0f}', f'{single_avg_votes:.0f}']
    })
    print(summary_df.to_string(index=False))
    print("=" * 70)
    
    print("\n" + "=" * 70)
    print("✅ LEVEL 2 TASK 4 COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\n📁 Output files location: {os.path.abspath(output_dir)}")
    print("   • restaurant_chains_analysis.png")
    print("   • restaurant_chains_summary.txt")
    
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