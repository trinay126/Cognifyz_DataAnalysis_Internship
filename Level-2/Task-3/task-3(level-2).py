"""
Main Level-2 Task-3: Geographic Analysis
Analyzes restaurant locations using coordinates and identifies clusters
Generates visualizations and summary report

Folder Structure:
Y:\Main\
├── Data\
│   └── Dataset_.csv
└── Level-2\
    └── Task-3\
        ├── task3.py (this file)
        └── output\
            ├── geographic_analysis.png
            └── geographic_analysis_summary.txt
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
import os

def main():
    print("=" * 70)
    print("🗺️  RESTAURANT GEOGRAPHIC ANALYSIS - LEVEL 2 TASK 3")
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
    
    required_cols = ['Longitude', 'Latitude', 'City']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"❌ ERROR: Missing columns: {', '.join(missing_cols)}")
        print(f"Available columns: {', '.join(df.columns)}")
        return
    
    print(f"✅ Required columns found: Longitude, Latitude, City")
    
    initial_count = len(df)
    df_geo = df.dropna(subset=['Longitude', 'Latitude', 'City']).copy()
    
    # Filter out invalid coordinates (0,0 or extreme outliers)
    df_geo = df_geo[
        (df_geo['Latitude'] != 0) & 
        (df_geo['Longitude'] != 0) &
        (df_geo['Latitude'].between(-90, 90)) &
        (df_geo['Longitude'].between(-180, 180))
    ]
    
    removed = initial_count - len(df_geo)
    
    print(f"📊 Valid records: {len(df_geo):,}")
    print(f"🗑️  Removed (null/invalid): {removed:,}")
    
    # ========== STEP 3: GEOGRAPHIC ANALYSIS ==========
    print("\n📊 STEP 3: Analyzing Geographic Distribution")
    print("-" * 70)
    
    total_restaurants = len(df_geo)
    
    # Overall statistics
    lat_min, lat_max = df_geo['Latitude'].min(), df_geo['Latitude'].max()
    lon_min, lon_max = df_geo['Longitude'].min(), df_geo['Longitude'].max()
    lat_center = df_geo['Latitude'].mean()
    lon_center = df_geo['Longitude'].mean()
    
    print(f"📊 COORDINATE RANGES:")
    print(f"  • Latitude:  {lat_min:.4f}° to {lat_max:.4f}°")
    print(f"  • Longitude: {lon_min:.4f}° to {lon_max:.4f}°")
    print(f"  • Center point: ({lat_center:.4f}°, {lon_center:.4f}°)")
    
    # City analysis
    city_counts = df_geo['City'].value_counts()
    total_cities = len(city_counts)
    
    print(f"\n🏙️  CITY DISTRIBUTION:")
    print(f"  • Total cities: {total_cities:,}")
    print(f"  • Avg restaurants per city: {total_restaurants/total_cities:.1f}")
    
    # Top 10 cities
    top10_cities = city_counts.head(10)
    
    print(f"\n🏆 TOP 10 CITIES BY RESTAURANT COUNT:")
    print("-" * 70)
    print(f"{'Rank':<6} {'City':<20} {'Count':<10} {'Coordinates':<25}")
    print("-" * 70)
    
    city_locations = {}
    for i, (city, count) in enumerate(top10_cities.items(), 1):
        city_data = df_geo[df_geo['City'] == city]
        avg_lat = city_data['Latitude'].mean()
        avg_lon = city_data['Longitude'].mean()
        city_locations[city] = (avg_lat, avg_lon)
        print(f"{i:<6} {city:<20} {count:<10,} ({avg_lat:7.4f}, {avg_lon:8.4f})")
    
    # ========== STEP 4: CLUSTERING ANALYSIS ==========
    print("\n📊 STEP 4: Performing Clustering Analysis")
    print("-" * 70)
    
    # Use KMeans clustering to identify geographic clusters
    n_clusters = 5
    coords = df_geo[['Latitude', 'Longitude']].values
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_geo['Cluster'] = kmeans.fit_predict(coords)
    
    cluster_centers = kmeans.cluster_centers_
    
    print(f"🎯 IDENTIFIED {n_clusters} GEOGRAPHIC CLUSTERS:")
    print("-" * 70)
    
    for i in range(n_clusters):
        cluster_data = df_geo[df_geo['Cluster'] == i]
        cluster_size = len(cluster_data)
        cluster_pct = (cluster_size / total_restaurants * 100)
        avg_rating = cluster_data['Aggregate rating'].mean() if 'Aggregate rating' in df_geo.columns else 0
        center_lat, center_lon = cluster_centers[i]
        
        # Find most common city in cluster
        if len(cluster_data) > 0:
            main_city = cluster_data['City'].value_counts().index[0]
        else:
            main_city = "N/A"
        
        print(f"\nCluster {i+1}:")
        print(f"  • Restaurants: {cluster_size:,} ({cluster_pct:.1f}%)")
        print(f"  • Center: ({center_lat:.4f}°, {center_lon:.4f}°)")
        print(f"  • Main city: {main_city}")
        if 'Aggregate rating' in df_geo.columns:
            print(f"  • Avg rating: {avg_rating:.2f}")
    
    # Density analysis
    print(f"\n📍 DENSITY ANALYSIS:")
    
    # Calculate area coverage (rough approximation)
    lat_span = lat_max - lat_min
    lon_span = lon_max - lon_min
    area = lat_span * lon_span  # Approximate area in square degrees
    density = total_restaurants / area if area > 0 else 0
    
    print(f"  • Geographic span: {lat_span:.2f}° × {lon_span:.2f}°")
    print(f"  • Approximate density: {density:.2f} restaurants per sq degree")
    
    # Top city density
    top_city = top10_cities.index[0]
    top_city_data = df_geo[df_geo['City'] == top_city]
    top_city_lat_span = top_city_data['Latitude'].max() - top_city_data['Latitude'].min()
    top_city_lon_span = top_city_data['Longitude'].max() - top_city_data['Longitude'].min()
    
    print(f"\n  {top_city} (largest cluster):")
    print(f"  • Restaurants: {len(top_city_data):,}")
    print(f"  • Span: {top_city_lat_span:.4f}° × {top_city_lon_span:.4f}°")
    
    # ========== STEP 5: CREATE VISUALIZATIONS ==========
    print("\n📊 STEP 5: Creating Visualizations")
    print("-" * 70)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig = plt.figure(figsize=(18, 12))
    
    # ===== Chart 1: Scatter Plot - All Restaurants =====
    ax1 = plt.subplot(2, 3, 1)
    
    scatter1 = ax1.scatter(df_geo['Longitude'], df_geo['Latitude'],
                          c=df_geo['Aggregate rating'] if 'Aggregate rating' in df_geo.columns else 'blue',
                          cmap='RdYlGn', alpha=0.6, s=10, edgecolors='none')
    ax1.set_xlabel('Longitude', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Latitude', fontsize=12, fontweight='bold')
    ax1.set_title('🗺️  Restaurant Locations (Global View)', fontsize=14, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3)
    
    if 'Aggregate rating' in df_geo.columns:
        cbar1 = plt.colorbar(scatter1, ax=ax1)
        cbar1.set_label('Rating', fontsize=10, fontweight='bold')
    
    # ===== Chart 2: Cluster Map =====
    ax2 = plt.subplot(2, 3, 2)
    
    colors_cluster = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    for i in range(n_clusters):
        cluster_points = df_geo[df_geo['Cluster'] == i]
        ax2.scatter(cluster_points['Longitude'], cluster_points['Latitude'],
                   c=[colors_cluster[i]], label=f'Cluster {i+1}',
                   alpha=0.6, s=15, edgecolors='none')
    
    # Plot cluster centers
    ax2.scatter(cluster_centers[:, 1], cluster_centers[:, 0],
               c='red', marker='X', s=200, edgecolors='black', linewidths=2,
               label='Centers', zorder=5)
    
    ax2.set_xlabel('Longitude', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Latitude', fontsize=12, fontweight='bold')
    ax2.set_title(f'🎯 {n_clusters} Geographic Clusters', fontsize=14, fontweight='bold', pad=20)
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # ===== Chart 3: Top 10 Cities Bar Chart =====
    ax3 = plt.subplot(2, 3, 3)
    
    y_pos = np.arange(len(top10_cities))
    colors3 = plt.cm.viridis(np.linspace(0, 1, len(top10_cities)))
    
    bars3 = ax3.barh(y_pos, top10_cities.values, color=colors3, alpha=0.8, edgecolor='black')
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(top10_cities.index)
    ax3.set_xlabel('Number of Restaurants', fontsize=12, fontweight='bold')
    ax3.set_title('🏙️  Top 10 Cities', fontsize=14, fontweight='bold', pad=20)
    ax3.invert_yaxis()
    
    for i, (bar, count) in enumerate(zip(bars3, top10_cities.values)):
        ax3.text(count + max(top10_cities.values)*0.01, i, f'{count:,}',
                va='center', fontweight='bold', fontsize=9)
    
    # ===== Chart 4: Heatmap-style Density (Top City) =====
    ax4 = plt.subplot(2, 3, 4)
    
    top_city_data = df_geo[df_geo['City'] == top_city]
    
    # Create 2D histogram for density
    h, xedges, yedges = np.histogram2d(
        top_city_data['Longitude'].values,
        top_city_data['Latitude'].values,
        bins=30
    )
    
    im = ax4.imshow(h.T, origin='lower', cmap='YlOrRd', aspect='auto',
                    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    ax4.set_xlabel('Longitude', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Latitude', fontsize=12, fontweight='bold')
    ax4.set_title(f'🔥 Density Map: {top_city}', fontsize=14, fontweight='bold', pad=20)
    
    cbar4 = plt.colorbar(im, ax=ax4)
    cbar4.set_label('Restaurant Count', fontsize=10, fontweight='bold')
    
    # ===== Chart 5: Cluster Size Pie Chart =====
    ax5 = plt.subplot(2, 3, 5)
    
    cluster_sizes = [len(df_geo[df_geo['Cluster'] == i]) for i in range(n_clusters)]
    cluster_labels = [f'Cluster {i+1}' for i in range(n_clusters)]
    
    wedges, texts, autotexts = ax5.pie(cluster_sizes, labels=cluster_labels,
                                        autopct='%1.1f%%', colors=colors_cluster,
                                        startangle=90, shadow=True,
                                        textprops={'fontsize': 10, 'fontweight': 'bold'})
    ax5.set_title('📊 Cluster Distribution', fontsize=14, fontweight='bold', pad=20)
    
    # ===== Chart 6: Rating by Cluster =====
    ax6 = plt.subplot(2, 3, 6)
    
    if 'Aggregate rating' in df_geo.columns:
        cluster_ratings = [df_geo[df_geo['Cluster'] == i]['Aggregate rating'].mean() 
                          for i in range(n_clusters)]
        
        bars6 = ax6.bar(range(1, n_clusters+1), cluster_ratings,
                       color=colors_cluster, alpha=0.8, edgecolor='black')
        ax6.set_xlabel('Cluster', fontsize=12, fontweight='bold')
        ax6.set_ylabel('Average Rating', fontsize=12, fontweight='bold')
        ax6.set_title('⭐ Average Rating by Cluster', fontsize=14, fontweight='bold', pad=20)
        ax6.set_ylim(0, 5)
        ax6.grid(axis='y', alpha=0.3)
        
        for i, (bar, rating) in enumerate(zip(bars6, cluster_ratings)):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{rating:.2f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
    else:
        ax6.text(0.5, 0.5, 'Rating data not available',
                ha='center', va='center', fontsize=12)
        ax6.axis('off')
    
    fig.suptitle('🗺️  Restaurant Geographic Analysis Dashboard', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    # ========== STEP 6: SAVE OUTPUT ==========
    print("\n💾 STEP 6: Saving Results")
    print("-" * 70)
    
    output_dir = os.path.join(os.getcwd(), 'output')
    os.makedirs(output_dir, exist_ok=True)
    print(f"✅ Output directory created: {output_dir}")
    
    output_path = os.path.join(output_dir, 'geographic_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Charts saved: {output_path}")
    
    # ========== STEP 7: CREATE SUMMARY REPORT ==========
    print("\n📋 STEP 7: Creating Summary Report")
    print("-" * 70)
    
    summary_path = os.path.join(output_dir, 'geographic_analysis_summary.txt')
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("RESTAURANT GEOGRAPHIC ANALYSIS - SUMMARY REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Dataset: {os.path.basename(dataset_path)}\n")
        f.write(f"Total Restaurants Analyzed: {total_restaurants:,}\n")
        f.write(f"Total Cities: {total_cities:,}\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("COORDINATE RANGES\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Latitude Range:   {lat_min:.4f}° to {lat_max:.4f}°\n")
        f.write(f"Longitude Range:  {lon_min:.4f}° to {lon_max:.4f}°\n")
        f.write(f"Center Point:     ({lat_center:.4f}°, {lon_center:.4f}°)\n")
        f.write(f"Geographic Span:  {lat_span:.2f}° × {lon_span:.2f}°\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("TOP 10 CITIES BY RESTAURANT COUNT\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"{'Rank':<6} {'City':<20} {'Count':<10} {'Latitude':<12} {'Longitude':<12}\n")
        f.write("-" * 70 + "\n")
        for i, (city, count) in enumerate(top10_cities.items(), 1):
            lat, lon = city_locations[city]
            f.write(f"{i:<6} {city:<20} {count:<10,} {lat:<12.4f} {lon:<12.4f}\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write(f"CLUSTERING ANALYSIS ({n_clusters} CLUSTERS)\n")
        f.write("=" * 70 + "\n\n")
        
        for i in range(n_clusters):
            cluster_data = df_geo[df_geo['Cluster'] == i]
            cluster_size = len(cluster_data)
            cluster_pct = (cluster_size / total_restaurants * 100)
            center_lat, center_lon = cluster_centers[i]
            main_city = cluster_data['City'].value_counts().index[0] if len(cluster_data) > 0 else "N/A"
            avg_rating = cluster_data['Aggregate rating'].mean() if 'Aggregate rating' in df_geo.columns else 0
            
            f.write(f"CLUSTER {i+1}:\n")
            f.write(f"  Restaurant Count:  {cluster_size:,} ({cluster_pct:.1f}%)\n")
            f.write(f"  Center Point:      ({center_lat:.4f}°, {center_lon:.4f}°)\n")
            f.write(f"  Main City:         {main_city}\n")
            if 'Aggregate rating' in df_geo.columns:
                f.write(f"  Average Rating:    {avg_rating:.2f}/5.0\n")
            f.write("\n")
        
        f.write("=" * 70 + "\n")
        f.write("DENSITY ANALYSIS\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Overall Density:  {density:.2f} restaurants per square degree\n")
        f.write(f"Avg per City:     {total_restaurants/total_cities:.1f} restaurants\n\n")
        
        f.write(f"{top_city} (Largest Concentration):\n")
        f.write(f"  Restaurants:  {len(top_city_data):,}\n")
        f.write(f"  Coverage:     {top_city_lat_span:.4f}° × {top_city_lon_span:.4f}°\n")
        f.write(f"  % of Total:   {(len(top_city_data)/total_restaurants*100):.1f}%\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("KEY PATTERNS & INSIGHTS\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("1. GEOGRAPHIC CONCENTRATION:\n")
        top3_total = sum(top10_cities.values[:3])
        top3_pct = (top3_total / total_restaurants * 100)
        f.write(f"   • Top 3 cities contain {top3_pct:.1f}% of all restaurants\n")
        f.write(f"   • Highest concentration: {top_city} ({len(top_city_data):,} restaurants)\n")
        f.write(f"   • This represents {(len(top_city_data)/total_restaurants*100):.1f}% of all locations\n\n")
        
        f.write("2. CLUSTER CHARACTERISTICS:\n")
        largest_cluster = max(range(n_clusters), key=lambda i: len(df_geo[df_geo['Cluster'] == i]))
        largest_cluster_size = len(df_geo[df_geo['Cluster'] == largest_cluster])
        f.write(f"   • Largest cluster: Cluster {largest_cluster+1} ({largest_cluster_size:,} restaurants)\n")
        f.write(f"   • Clusters identified using K-Means algorithm\n")
        f.write(f"   • Geographic spread varies significantly between clusters\n\n")
        
        if 'Aggregate rating' in df_geo.columns:
            f.write("3. RATING PATTERNS BY LOCATION:\n")
            best_cluster = max(range(n_clusters), 
                             key=lambda i: df_geo[df_geo['Cluster'] == i]['Aggregate rating'].mean())
            best_cluster_rating = df_geo[df_geo['Cluster'] == best_cluster]['Aggregate rating'].mean()
            f.write(f"   • Highest-rated cluster: Cluster {best_cluster+1} ({best_cluster_rating:.2f} avg)\n")
            f.write(f"   • Rating variation across geographic areas suggests\n")
            f.write(f"     location-based quality differences\n\n")
        
        f.write("4. URBAN CONCENTRATION:\n")
        f.write(f"   • {total_cities} cities represented in dataset\n")
        f.write(f"   • Strong urban concentration pattern observed\n")
        f.write(f"   • Major metropolitan areas dominate distribution\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("BUSINESS RECOMMENDATIONS\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("• EXPANSION OPPORTUNITIES:\n")
        f.write(f"  - {total_cities - 10} cities have fewer restaurants\n")
        f.write(f"  - Consider expansion in underserved geographic areas\n\n")
        
        f.write("• MARKET SATURATION:\n")
        f.write(f"  - {top_city} shows high concentration\n")
        f.write(f"  - May indicate market saturation in this area\n\n")
        
        f.write("• GEOGRAPHIC STRATEGY:\n")
        f.write(f"  - Focus on identified clusters for targeted marketing\n")
        f.write(f"  - Different clusters may have different demographics\n")
        
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
        'City': top10_cities.index[:5],
        'Count': top10_cities.values[:5],
        'Latitude': [city_locations[city][0] for city in top10_cities.index[:5]],
        'Longitude': [city_locations[city][1] for city in top10_cities.index[:5]]
    })
    print(summary_df.to_string(index=False))
    print("=" * 70)
    
    print("\n" + "=" * 70)
    print("✅ LEVEL 2 TASK 3 COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\n📁 Output files location: {os.path.abspath(output_dir)}")
    print("   • geographic_analysis.png")
    print("   • geographic_analysis_summary.txt")
    
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