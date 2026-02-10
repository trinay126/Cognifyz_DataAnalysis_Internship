"""
Main Level-1 Task-1: Top Cuisines Analysis
Analyzes restaurant dataset to find top 3 most popular cuisines
Generates bar chart, pie chart, and summary statistics
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    print("=" * 70)
    print("🍽️  RESTAURANT CUISINE ANALYSIS - TASK 1")
    print("=" * 70)
    print(f"\n🔍 Current working directory: {os.getcwd()}")
    
    # ========== STEP 1: LOAD DATA ==========
    print("\n📂 STEP 1: Loading Dataset")
    print("-" * 70)
    
    # Smart path detection - works from any subfolder
    current_dir = os.getcwd()
    
    # Try multiple possible paths and filenames
    possible_paths = []
    
    # Possible filenames (with and without underscore)
    filenames = ['Dataset.csv', 'Dataset_.csv', 'dataset.csv', 'dataset_.csv']
    
    # Method 1: Navigate up from current directory
    if "Level-1" in current_dir or "Level 1" in current_dir or "Task" in current_dir:
        # Go up two levels: Task-1 -> Level-1 -> Main
        main_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
        for fname in filenames:
            possible_paths.append(os.path.join(main_dir, "Data", fname))
    
    # Method 2: Hardcoded Y:\Main
    for fname in filenames:
        possible_paths.append(os.path.join("Y:", "Main", "Data", fname))
    
    # Method 3: Check if Data folder is in parent
    parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
    for fname in filenames:
        possible_paths.append(os.path.join(parent_dir, "Data", fname))
    
    # Method 4: Check if Data folder is two levels up
    grandparent_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
    for fname in filenames:
        possible_paths.append(os.path.join(grandparent_dir, "Data", fname))
    
    # Method 5: Check current directory
    for fname in filenames:
        possible_paths.append(os.path.join(current_dir, fname))
        possible_paths.append(os.path.join(current_dir, "Data", fname))
    
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
        print(f"\n💡 Tried these locations:")
        # Show unique paths only
        shown = set()
        for path in possible_paths:
            if path not in shown:
                print(f"   • {path}")
                shown.add(path)
        print(f"\n💡 Please place Dataset.csv or Dataset_.csv in Y:\\Main\\Data\\")
        print(f"\n📁 Files in current directory:")
        for item in os.listdir(current_dir):
            print(f"   • {item}")
        return
    
    # Load dataset
    try:
        df = pd.read_csv(dataset_path, encoding='utf-8-sig')  # Handle BOM if present
        print(f"✅ Dataset loaded successfully")
        print(f"📊 Total records: {len(df):,}")
        print(f"📊 Total columns: {len(df.columns)}")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # ========== STEP 2: DATA CLEANING ==========
    print("\n🧹 STEP 2: Cleaning Data")
    print("-" * 70)
    
    # Check if Cuisines column exists
    if 'Cuisines' not in df.columns:
        print(f"❌ ERROR: 'Cuisines' column not found")
        print(f"Available columns: {', '.join(df.columns)}")
        return
    
    # Remove null and empty cuisines
    initial_count = len(df)
    df_cuisines = df.dropna(subset=['Cuisines']).copy()
    df_cuisines['Cuisines'] = df_cuisines['Cuisines'].str.strip()
    df_cuisines = df_cuisines[df_cuisines['Cuisines'] != '']
    
    removed_count = initial_count - len(df_cuisines)
    print(f"📊 Restaurants with valid cuisines: {len(df_cuisines):,}")
    print(f"🗑️  Removed (null/empty): {removed_count:,}")
    
    # ========== STEP 3: ANALYZE CUISINES ==========
    print("\n🔍 STEP 3: Analyzing Cuisines")
    print("-" * 70)
    
    # Split and count all cuisines
    all_cuisines = []
    for cuisines in df_cuisines['Cuisines']:
        # Split by comma and clean each cuisine
        cuisine_list = [c.strip() for c in str(cuisines).split(',')]
        all_cuisines.extend(cuisine_list)
    
    print(f"📊 Total cuisine entries: {len(all_cuisines):,}")
    
    # Count cuisine frequencies
    cuisine_counts = pd.Series(all_cuisines).value_counts()
    print(f"📊 Unique cuisines found: {len(cuisine_counts):,}")
    
    # Get top 3
    top3 = cuisine_counts.head(3)
    percentages = (top3 / len(df_cuisines) * 100).round(2)
    
    print("\n🏆 TOP 3 CUISINES:")
    print("-" * 70)
    for i, (cuisine, count) in enumerate(top3.items(), 1):
        pct = percentages[cuisine]
        print(f"{i}. {cuisine:20} | Count: {count:6,} | Percentage: {pct:6.2f}%")
    
    # ========== STEP 4: CREATE VISUALIZATIONS ==========
    print("\n📊 STEP 4: Creating Visualizations")
    print("-" * 70)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Color scheme
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    # ===== Chart 1: Bar Chart =====
    bars = ax1.bar(top3.index, top3.values, color=colors, alpha=0.8, 
                   edgecolor='black', linewidth=1.5)
    ax1.set_title('🏆 Top 3 Most Popular Cuisines', fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlabel('Cuisine Type', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Restaurants', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45, labelsize=11)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bar, count in zip(bars, top3.values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(top3.values)*0.02,
                 f'{int(count):,}',
                 ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # ===== Chart 2: Pie Chart =====
    wedges, texts, autotexts = ax2.pie(
        top3.values, 
        labels=top3.index, 
        autopct='%1.1f%%',
        colors=colors, 
        startangle=90, 
        explode=(0.05, 0, 0),
        shadow=True,
        textprops={'fontsize': 11, 'fontweight': 'bold'}
    )
    ax2.set_title('📈 Market Share Distribution', fontsize=16, fontweight='bold', pad=20)
    
    # Add main title
    fig.suptitle('Restaurant Cuisine Analysis - Top 3 Cuisines', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    # ========== STEP 5: SAVE OUTPUT ==========
    print("\n💾 STEP 5: Saving Results")
    print("-" * 70)
    
    # Create output directory in current folder
    output_dir = os.path.join(os.getcwd(), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save chart
    output_path = os.path.join(output_dir, 'top_cuisines_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Chart saved: {output_path}")
    
    # Show plot
    plt.show()
    
    # Close figure to free memory
    plt.close()
    
    # ========== STEP 6: SUMMARY REPORT ==========
    print("\n📋 STEP 6: Summary Report")
    print("-" * 70)
    
    # Create results dataframe
    result_df = pd.DataFrame({
        'Rank': [1, 2, 3],
        'Cuisine': top3.index,
        'Restaurant Count': top3.values,
        'Percentage (%)': percentages.values
    })
    
    print("\n📊 FINAL RESULTS TABLE:")
    print("=" * 70)
    print(result_df.to_string(index=False))
    print("=" * 70)
    
    # Save summary to text file
    summary_path = os.path.join(output_dir, 'cuisine_analysis_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("RESTAURANT CUISINE ANALYSIS - SUMMARY REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Dataset: {os.path.basename(dataset_path)}\n")
        f.write(f"Total Restaurants Analyzed: {len(df_cuisines):,}\n")
        f.write(f"Total Cuisine Entries: {len(all_cuisines):,}\n")
        f.write(f"Unique Cuisines Found: {len(cuisine_counts):,}\n\n")
        f.write("TOP 3 CUISINES:\n")
        f.write("-" * 70 + "\n")
        f.write(result_df.to_string(index=False))
        f.write("\n" + "=" * 70 + "\n")
    
    print(f"\n✅ Summary saved: {summary_path}")
    
    print("\n" + "=" * 70)
    print("✅ TASK 1 COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\n📁 Output files location: {os.path.abspath(output_dir)}")
    print("   • top_cuisines_analysis.png")
    print("   • cuisine_analysis_summary.txt")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user")
    except Exception as e:
        print(f"\n\n❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()