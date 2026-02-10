"""
Main Level-1 Task-3: Price Range Distribution Analysis
Analyzes the distribution of price ranges among restaurants
Generates histogram, bar chart, pie chart and summary report

Folder Structure:
Y:\Main\
├── Data\
│   └── Dataset_.csv
└── Level-1\
    └── Task-3\
        ├── task3.py (this file)
        └── output\
            ├── price_range_distribution.png
            └── price_range_summary.txt
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def main():
    print("=" * 70)
    print("💰 RESTAURANT PRICE RANGE DISTRIBUTION - TASK 3")
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
    
    # Check if Price range column exists
    if 'Price range' not in df.columns:
        print(f"❌ ERROR: 'Price range' column not found")
        print(f"Available columns: {', '.join(df.columns)}")
        return
    
    print(f"✅ 'Price range' column found")
    
    # Clean data
    initial_count = len(df)
    df_clean = df.dropna(subset=['Price range']).copy()
    removed = initial_count - len(df_clean)
    
    print(f"📊 Valid records: {len(df_clean):,}")
    print(f"🗑️  Removed (null values): {removed:,}")
    
    # ========== STEP 3: ANALYZE PRICE RANGES ==========
    print("\n📊 STEP 3: Analyzing Price Range Distribution")
    print("-" * 70)
    
    # Get price range counts
    price_counts = df_clean['Price range'].value_counts().sort_index()
    total_restaurants = len(df_clean)
    
    # Calculate percentages
    price_percentages = (price_counts / total_restaurants * 100).round(2)
    
    # Define price range labels
    price_labels = {
        1: 'Budget (₹)',
        2: 'Mid-Range (₹₹)',
        3: 'Expensive (₹₹₹)',
        4: 'Very Expensive (₹₹₹₹)'
    }
    
    print(f"📊 Total Restaurants Analyzed: {total_restaurants:,}")
    print(f"📊 Price Range Categories: {len(price_counts)}")
    
    print(f"\n💰 PRICE RANGE DISTRIBUTION:")
    print("-" * 70)
    print(f"{'Range':<10} {'Category':<25} {'Count':<12} {'Percentage':<12}")
    print("-" * 70)
    for price_range in sorted(price_counts.index):
        count = price_counts[price_range]
        percentage = price_percentages[price_range]
        label = price_labels.get(price_range, f'Range {price_range}')
        print(f"{price_range:<10} {label:<25} {count:<12,} {percentage:<11.2f}%")
    
    # Find most common and least common
    most_common_range = price_counts.idxmax()
    most_common_count = price_counts.max()
    least_common_range = price_counts.idxmin()
    least_common_count = price_counts.min()
    
    print(f"\n🏆 MOST COMMON: {price_labels[most_common_range]} - {most_common_count:,} restaurants ({price_percentages[most_common_range]:.2f}%)")
    print(f"📉 LEAST COMMON: {price_labels[least_common_range]} - {least_common_count:,} restaurants ({price_percentages[least_common_range]:.2f}%)")
    
    # ========== STEP 4: CREATE VISUALIZATIONS ==========
    print("\n📊 STEP 4: Creating Visualizations")
    print("-" * 70)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # Create figure with 4 subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Prepare data with labels
    price_ranges_labeled = [price_labels[i] for i in sorted(price_counts.index)]
    counts = [price_counts[i] for i in sorted(price_counts.index)]
    percentages = [price_percentages[i] for i in sorted(price_counts.index)]
    
    # Color scheme
    colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']  # Green, Blue, Orange, Red
    
    # ===== Chart 1: Bar Chart =====
    ax1 = plt.subplot(2, 2, 1)
    bars = ax1.bar(price_ranges_labeled, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_title('💰 Restaurant Count by Price Range', fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlabel('Price Range Category', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Restaurants', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45, labelsize=10)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bar, count, pct in zip(bars, counts, percentages):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                 f'{count:,}\n({pct:.1f}%)',
                 ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # ===== Chart 2: Histogram =====
    ax2 = plt.subplot(2, 2, 2)
    
    # Create histogram data
    price_range_values = df_clean['Price range'].values
    bins = [0.5, 1.5, 2.5, 3.5, 4.5]
    
    n, bins_edges, patches = ax2.hist(price_range_values, bins=bins, 
                                       alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Color each bar individually
    for i, patch in enumerate(patches):
        patch.set_facecolor(colors[i])
    
    ax2.set_title('📊 Price Range Distribution (Histogram)', fontsize=16, fontweight='bold', pad=20)
    ax2.set_xlabel('Price Range', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax2.set_xticks([1, 2, 3, 4])
    ax2.set_xticklabels(['₹', '₹₹', '₹₹₹', '₹₹₹₹'])
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on histogram
    for i, (count, patch) in enumerate(zip(n, patches)):
        height = patch.get_height()
        ax2.text(patch.get_x() + patch.get_width()/2., height + max(n)*0.01,
                 f'{int(count):,}',
                 ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # ===== Chart 3: Pie Chart =====
    ax3 = plt.subplot(2, 2, 3)
    
    # Create pie chart
    explode = [0.05 if i == most_common_range-1 else 0 for i in range(len(counts))]
    wedges, texts, autotexts = ax3.pie(counts, 
                                        labels=price_ranges_labeled,
                                        autopct='%1.1f%%',
                                        colors=colors,
                                        explode=explode,
                                        startangle=90,
                                        shadow=True,
                                        textprops={'fontsize': 11, 'fontweight': 'bold'})
    
    ax3.set_title('🥧 Market Share by Price Range', fontsize=16, fontweight='bold', pad=20)
    
    # ===== Chart 4: Horizontal Bar with Percentages =====
    ax4 = plt.subplot(2, 2, 4)
    
    y_pos = np.arange(len(price_ranges_labeled))
    bars4 = ax4.barh(y_pos, percentages, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(price_ranges_labeled)
    ax4.set_xlabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax4.set_title('📈 Percentage Distribution', fontsize=16, fontweight='bold', pad=20)
    ax4.set_xlim(0, max(percentages) * 1.15)
    ax4.invert_yaxis()
    ax4.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add percentage labels
    for i, (bar, pct, count) in enumerate(zip(bars4, percentages, counts)):
        width = bar.get_width()
        ax4.text(width + 1, i, f'{pct:.1f}% ({count:,})',
                va='center', fontweight='bold', fontsize=10)
    
    # Add main title
    fig.suptitle('💰 Restaurant Price Range Distribution Analysis', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    # ========== STEP 5: SAVE OUTPUT ==========
    print("\n💾 STEP 5: Saving Results")
    print("-" * 70)
    
    # Create output directory
    output_dir = os.path.join(os.getcwd(), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save chart
    output_path = os.path.join(output_dir, 'price_range_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Charts saved: {output_path}")
    
    # ========== STEP 6: CREATE SUMMARY REPORT ==========
    print("\n📋 STEP 6: Creating Summary Report")
    print("-" * 70)
    
    # Create detailed summary text file
    summary_path = os.path.join(output_dir, 'price_range_summary.txt')
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("RESTAURANT PRICE RANGE DISTRIBUTION - SUMMARY REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Dataset: {os.path.basename(dataset_path)}\n")
        f.write(f"Total Restaurants Analyzed: {total_restaurants:,}\n")
        f.write(f"Price Range Categories: {len(price_counts)}\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("PRICE RANGE DISTRIBUTION\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"{'Range':<8} {'Category':<28} {'Count':<14} {'Percentage':<12}\n")
        f.write("-" * 70 + "\n")
        
        for price_range in sorted(price_counts.index):
            count = price_counts[price_range]
            percentage = price_percentages[price_range]
            label = price_labels.get(price_range, f'Range {price_range}')
            f.write(f"{price_range:<8} {label:<28} {count:<14,} {percentage:<11.2f}%\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("KEY INSIGHTS\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"🏆 MOST COMMON PRICE RANGE:\n")
        f.write(f"   Category: {price_labels[most_common_range]}\n")
        f.write(f"   Count: {most_common_count:,} restaurants\n")
        f.write(f"   Percentage: {price_percentages[most_common_range]:.2f}%\n\n")
        
        f.write(f"📉 LEAST COMMON PRICE RANGE:\n")
        f.write(f"   Category: {price_labels[least_common_range]}\n")
        f.write(f"   Count: {least_common_count:,} restaurants\n")
        f.write(f"   Percentage: {price_percentages[least_common_range]:.2f}%\n\n")
        
        # Calculate statistics
        budget_and_mid = price_counts.get(1, 0) + price_counts.get(2, 0)
        budget_mid_pct = (budget_and_mid / total_restaurants * 100)
        expensive = price_counts.get(3, 0) + price_counts.get(4, 0)
        expensive_pct = (expensive / total_restaurants * 100)
        
        f.write("📊 MARKET SEGMENTS:\n")
        f.write(f"   Affordable (₹ + ₹₹):          {budget_and_mid:,} ({budget_mid_pct:.2f}%)\n")
        f.write(f"   Premium (₹₹₹ + ₹₹₹₹):         {expensive:,} ({expensive_pct:.2f}%)\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("OBSERVATIONS\n")
        f.write("=" * 70 + "\n\n")
        
        # Generate insights
        if most_common_range == 1:
            f.write("• Budget-friendly restaurants dominate the market.\n")
        elif most_common_range == 2:
            f.write("• Mid-range restaurants are the market leaders.\n")
        
        if budget_mid_pct > 75:
            f.write(f"• Over {budget_mid_pct:.0f}% of restaurants are affordable (₹ or ₹₹).\n")
        
        if expensive_pct < 25:
            f.write(f"• Premium dining options (₹₹₹/₹₹₹₹) represent only {expensive_pct:.1f}% of the market.\n")
        
        avg_range = df_clean['Price range'].mean()
        f.write(f"• Average price range: {avg_range:.2f}\n")
        
        f.write("\n" + "=" * 70 + "\n")
    
    print(f"✅ Summary saved: {summary_path}")
    
    # Verify file was created
    if os.path.exists(summary_path):
        file_size = os.path.getsize(summary_path)
        print(f"✅ VERIFIED: Summary file exists ({file_size} bytes)")
    else:
        print(f"❌ WARNING: Summary file was not created!")
    
    # Display summary table in console
    print("\n📊 PRICE RANGE SUMMARY TABLE:")
    print("=" * 70)
    summary_df = pd.DataFrame({
        'Price Range': [price_labels[i] for i in sorted(price_counts.index)],
        'Count': [price_counts[i] for i in sorted(price_counts.index)],
        'Percentage': [f"{price_percentages[i]:.2f}%" for i in sorted(price_counts.index)]
    })
    print(summary_df.to_string(index=False))
    print("=" * 70)
    
    print("\n" + "=" * 70)
    print("✅ TASK 3 COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\n📁 Output files location: {os.path.abspath(output_dir)}")
    print("   • price_range_distribution.png")
    print("   • price_range_summary.txt")
    
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