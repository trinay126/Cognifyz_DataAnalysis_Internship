"""
Main Level-1 Task-2: City Analysis
Identifies cities with most restaurants and highest ratings
Generates visualizations and summary report

Folder Structure:
Y:\Main\
├── Data\
│   └── Dataset_.csv
└── Level-1\
    └── Task-2\
        ├── task2.py (this file)
        └── output\
            ├── city_analysis_charts.png
            └── city_analysis_summary.txt
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    print("=" * 70)
    print("🏙️  RESTAURANT CITY ANALYSIS - TASK 2")
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
    required_cols = ['City', 'Aggregate rating']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"❌ ERROR: Missing columns: {', '.join(missing_cols)}")
        print(f"Available columns: {', '.join(df.columns)}")
        return
    
    print(f"✅ Required columns found: City, Aggregate rating")
    
    # Clean data
    initial_count = len(df)
    df_clean = df.dropna(subset=['City', 'Aggregate rating']).copy()
    df_clean = df_clean[df_clean['City'].str.strip() != '']
    removed = initial_count - len(df_clean)
    
    print(f"📊 Valid records: {len(df_clean):,}")
    print(f"🗑️  Removed (null/empty): {removed:,}")
    
    # ========== STEP 3: ANALYZE CITIES ==========
    print("\n📊 STEP 3: Analyzing Cities")
    print("-" * 70)
    
    # Count restaurants per city
    city_counts = df_clean['City'].value_counts()
    total_cities = len(city_counts)
    
    print(f"📊 Total unique cities: {total_cities:,}")
    
    # City with most restaurants
    top_city = city_counts.index[0]
    top_city_count = city_counts.iloc[0]
    
    print(f"\n🏆 CITY WITH MOST RESTAURANTS:")
    print(f"   City: {top_city}")
    print(f"   Restaurant Count: {top_city_count:,}")
    
    # Calculate average rating per city
    city_avg_rating = df_clean.groupby('City')['Aggregate rating'].mean().sort_values(ascending=False)
    
    # City with highest average rating
    highest_rated_city = city_avg_rating.index[0]
    highest_avg_rating = city_avg_rating.iloc[0]
    
    print(f"\n⭐ CITY WITH HIGHEST AVERAGE RATING:")
    print(f"   City: {highest_rated_city}")
    print(f"   Average Rating: {highest_avg_rating:.2f}")
    
    # Get top 10 cities by restaurant count
    top10_cities = city_counts.head(10)
    
    # Get top 10 cities by average rating (with at least 10 restaurants)
    city_counts_dict = city_counts.to_dict()
    top10_rated = city_avg_rating[city_avg_rating.index.map(lambda x: city_counts_dict.get(x, 0) >= 10)].head(10)
    
    print(f"\n📋 Top 10 Cities (by restaurant count):")
    for i, (city, count) in enumerate(top10_cities.items(), 1):
        avg_rating = city_avg_rating[city]
        print(f"   {i:2}. {city:25} | Restaurants: {count:4,} | Avg Rating: {avg_rating:.2f}")
    
    # ========== STEP 4: CREATE VISUALIZATIONS ==========
    print("\n📊 STEP 4: Creating Visualizations")
    print("-" * 70)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # Create figure with 3 subplots
    fig = plt.figure(figsize=(18, 12))
    
    # ===== Chart 1: Top 10 Cities by Restaurant Count (Bar Chart) =====
    ax1 = plt.subplot(2, 2, 1)
    colors1 = plt.cm.viridis(range(len(top10_cities)))
    bars1 = ax1.barh(range(len(top10_cities)), top10_cities.values, color=colors1, edgecolor='black')
    ax1.set_yticks(range(len(top10_cities)))
    ax1.set_yticklabels(top10_cities.index)
    ax1.set_xlabel('Number of Restaurants', fontsize=12, fontweight='bold')
    ax1.set_title('🏆 Top 10 Cities by Restaurant Count', fontsize=14, fontweight='bold', pad=20)
    ax1.invert_yaxis()
    
    # Add value labels
    for i, (bar, count) in enumerate(zip(bars1, top10_cities.values)):
        ax1.text(count + max(top10_cities.values)*0.01, i, f'{int(count):,}', 
                va='center', fontweight='bold', fontsize=10)
    
    # ===== Chart 2: Top 10 Cities by Average Rating (Bar Chart) =====
    ax2 = plt.subplot(2, 2, 2)
    colors2 = plt.cm.plasma(range(len(top10_rated)))
    bars2 = ax2.barh(range(len(top10_rated)), top10_rated.values, color=colors2, edgecolor='black')
    ax2.set_yticks(range(len(top10_rated)))
    ax2.set_yticklabels(top10_rated.index)
    ax2.set_xlabel('Average Rating', fontsize=12, fontweight='bold')
    ax2.set_title('⭐ Top 10 Cities by Average Rating\n(min. 10 restaurants)', fontsize=14, fontweight='bold', pad=20)
    ax2.set_xlim(0, 5)
    ax2.invert_yaxis()
    
    # Add value labels
    for i, (bar, rating) in enumerate(zip(bars2, top10_rated.values)):
        ax2.text(rating + 0.05, i, f'{rating:.2f}', 
                va='center', fontweight='bold', fontsize=10)
    
    # ===== Chart 3: Restaurant Count vs Average Rating (Scatter) =====
    ax3 = plt.subplot(2, 2, 3)
    
    # Prepare data for scatter plot
    scatter_data = pd.DataFrame({
        'City': city_counts.index,
        'Restaurant_Count': city_counts.values,
        'Avg_Rating': [city_avg_rating[city] for city in city_counts.index]
    })
    
    # Create scatter plot
    scatter = ax3.scatter(scatter_data['Restaurant_Count'], 
                         scatter_data['Avg_Rating'],
                         s=100, alpha=0.6, c=scatter_data['Avg_Rating'],
                         cmap='RdYlGn', edgecolors='black', linewidth=0.5)
    
    # Highlight top cities
    top_city_data = scatter_data[scatter_data['City'] == top_city]
    ax3.scatter(top_city_data['Restaurant_Count'], top_city_data['Avg_Rating'],
               s=300, marker='*', color='red', edgecolors='black', linewidth=2,
               label=f'Most Restaurants: {top_city}', zorder=5)
    
    highest_rated_data = scatter_data[scatter_data['City'] == highest_rated_city]
    ax3.scatter(highest_rated_data['Restaurant_Count'], highest_rated_data['Avg_Rating'],
               s=300, marker='D', color='gold', edgecolors='black', linewidth=2,
               label=f'Highest Rating: {highest_rated_city}', zorder=5)
    
    ax3.set_xlabel('Number of Restaurants', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Average Rating', fontsize=12, fontweight='bold')
    ax3.set_title('📈 Restaurant Count vs Average Rating by City', fontsize=14, fontweight='bold', pad=20)
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Average Rating', fontsize=10, fontweight='bold')
    
    # ===== Chart 4: Distribution Summary (Text Summary) =====
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    
    # Create summary statistics
    total_restaurants = len(df_clean)
    avg_restaurants_per_city = total_restaurants / total_cities
    overall_avg_rating = df_clean['Aggregate rating'].mean()
    
    summary_text = f"""
    📊 CITY ANALYSIS SUMMARY
    {'='*45}
    
    🏙️  TOTAL CITIES: {total_cities:,}
    🍽️  TOTAL RESTAURANTS: {total_restaurants:,}
    
    {'─'*45}
    
    🏆 MOST RESTAURANTS:
       • City: {top_city}
       • Count: {top_city_count:,} restaurants
       • Avg Rating: {city_avg_rating[top_city]:.2f}/5.0
    
    {'─'*45}
    
    ⭐ HIGHEST RATED CITY:
       • City: {highest_rated_city}
       • Avg Rating: {highest_avg_rating:.2f}/5.0
       • Restaurants: {city_counts[highest_rated_city]:,}
    
    {'─'*45}
    
    📈 OVERALL STATISTICS:
       • Avg Restaurants/City: {avg_restaurants_per_city:.1f}
       • Overall Avg Rating: {overall_avg_rating:.2f}/5.0
    
    {'='*45}
    """
    
    ax4.text(0.1, 0.95, summary_text, transform=ax4.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Add main title
    fig.suptitle('🏙️ Restaurant City Analysis Dashboard', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    # ========== STEP 5: SAVE OUTPUT ==========
    print("\n💾 STEP 5: Saving Results")
    print("-" * 70)
    
    # Create output directory
    output_dir = os.path.join(os.getcwd(), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save chart
    output_path = os.path.join(output_dir, 'city_analysis_charts.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Charts saved: {output_path}")
    
    # ========== STEP 6: CREATE SUMMARY REPORT ==========
    print("\n📋 STEP 6: Creating Summary Report")
    print("-" * 70)
    
    # Create detailed summary text file
    summary_path = os.path.join(output_dir, 'city_analysis_summary.txt')
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("RESTAURANT CITY ANALYSIS - DETAILED SUMMARY REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Dataset: {os.path.basename(dataset_path)}\n")
        f.write(f"Total Records Analyzed: {len(df_clean):,}\n")
        f.write(f"Total Cities: {total_cities:,}\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("KEY FINDINGS\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("1. CITY WITH MOST RESTAURANTS:\n")
        f.write("-" * 70 + "\n")
        f.write(f"   City Name:           {top_city}\n")
        f.write(f"   Restaurant Count:    {top_city_count:,}\n")
        f.write(f"   Average Rating:      {city_avg_rating[top_city]:.2f}/5.0\n")
        f.write(f"   % of Total:          {(top_city_count/total_restaurants*100):.2f}%\n\n")
        
        f.write("2. CITY WITH HIGHEST AVERAGE RATING:\n")
        f.write("-" * 70 + "\n")
        f.write(f"   City Name:           {highest_rated_city}\n")
        f.write(f"   Average Rating:      {highest_avg_rating:.2f}/5.0\n")
        f.write(f"   Restaurant Count:    {city_counts[highest_rated_city]:,}\n")
        f.write(f"   % of Total:          {(city_counts[highest_rated_city]/total_restaurants*100):.2f}%\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("TOP 10 CITIES BY RESTAURANT COUNT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"{'Rank':<6} {'City':<30} {'Count':<12} {'Avg Rating':<12}\n")
        f.write("-" * 70 + "\n")
        for i, (city, count) in enumerate(top10_cities.items(), 1):
            avg_rating = city_avg_rating[city]
            f.write(f"{i:<6} {city:<30} {count:<12,} {avg_rating:<12.2f}\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("TOP 10 CITIES BY AVERAGE RATING (Min. 10 Restaurants)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"{'Rank':<6} {'City':<30} {'Avg Rating':<12} {'Count':<12}\n")
        f.write("-" * 70 + "\n")
        for i, (city, rating) in enumerate(top10_rated.items(), 1):
            count = city_counts[city]
            f.write(f"{i:<6} {city:<30} {rating:<12.2f} {count:<12,}\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("OVERALL STATISTICS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Average Restaurants per City:    {avg_restaurants_per_city:.1f}\n")
        f.write(f"Overall Average Rating:          {overall_avg_rating:.2f}/5.0\n")
        f.write(f"Median Restaurants per City:     {city_counts.median():.0f}\n")
        f.write(f"Max Restaurants in a City:       {city_counts.max():,}\n")
        f.write(f"Min Restaurants in a City:       {city_counts.min():,}\n")
        
        f.write("\n" + "=" * 70 + "\n")
    
    print(f"✅ Summary saved: {summary_path}")
    
    # Verify file was created
    if os.path.exists(summary_path):
        file_size = os.path.getsize(summary_path)
        print(f"✅ VERIFIED: Summary file exists ({file_size} bytes)")
    else:
        print(f"❌ WARNING: Summary file was not created!")
    
    # Display summary table in console
    print("\n📊 SUMMARY TABLE - TOP 10 CITIES BY RESTAURANT COUNT:")
    print("=" * 70)
    summary_df = pd.DataFrame({
        'City': top10_cities.index,
        'Restaurant Count': top10_cities.values,
        'Average Rating': [city_avg_rating[city] for city in top10_cities.index]
    })
    print(summary_df.to_string(index=False))
    print("=" * 70)
    
    print("\n" + "=" * 70)
    print("✅ TASK 2 COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\n📁 Output files location: {os.path.abspath(output_dir)}")
    print("   • city_analysis_charts.png")
    print("   • city_analysis_summary.txt")
    
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