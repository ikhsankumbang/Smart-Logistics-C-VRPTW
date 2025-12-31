import streamlit as st
import folium
from streamlit_folium import st_folium
import numpy as np
import random
import math
import requests
import polyline
import pandas as pd
import matplotlib.pyplot as plt
from folium.plugins import AntPath, Fullscreen
from sklearn.cluster import KMeans

# --- KONFIGURASI HALAMAN ---
st.set_page_config(layout="wide", page_title="Advanced VRPTW Logistics AI")

# --- 1. DATASET REALISTIS DENGAN JAM OPERASIONAL ---
LOKASI_INDUSTRI = [
    {"nama": "HUB PUSAT (MM2100)", "lat": -6.307, "lon": 107.103, "tipe": "GUDANG", "jam_buka": 6.0, "jam_tutup": 24.0},
    {"nama": "Toyota Tsusho (KIIC)", "lat": -6.358, "lon": 107.284, "tipe": "DEALER", "jam_buka": 8.0, "jam_tutup": 9.0},
    {"nama": "Astra Otoparts (Surya)", "lat": -6.393, "lon": 107.330, "tipe": "DEALER", "jam_buka": 8.0, "jam_tutup": 17.0},
    {"nama": "Honda Prospect (Karawang)", "lat": -6.367, "lon": 107.288, "tipe": "DEALER", "jam_buka": 7.5, "jam_tutup": 16.5},
    {"nama": "Suzuki Indomobil (Cikarang)", "lat": -6.327, "lon": 107.144, "tipe": "DEALER", "jam_buka": 8.0, "jam_tutup": 16.0},
    {"nama": "Wuling Motors (Deltamas)", "lat": -6.372, "lon": 107.185, "tipe": "DEALER", "jam_buka": 9.0, "jam_tutup": 17.0},
    {"nama": "Hyundai Motor (Deltamas)", "lat": -6.386, "lon": 107.195, "tipe": "DEALER", "jam_buka": 8.0, "jam_tutup": 16.0},
    {"nama": "Mitsubishi Motors (GIIC)", "lat": -6.387, "lon": 107.198, "tipe": "DEALER", "jam_buka": 8.5, "jam_tutup": 17.0},
    {"nama": "Isuzu Astra (Suryacipta)", "lat": -6.389, "lon": 107.325, "tipe": "DEALER", "jam_buka": 8.0, "jam_tutup": 16.0},
    {"nama": "Yamaha Motor (KIIC)", "lat": -6.353, "lon": 107.289, "tipe": "DEALER", "jam_buka": 7.0, "jam_tutup": 15.0},
    {"nama": "Daihatsu (Suryacipta)", "lat": -6.395, "lon": 107.332, "tipe": "DEALER", "jam_buka": 8.0, "jam_tutup": 17.0}
]

# --- 2. ENGINE FISIKA & WAKTU ---

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + \
        math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * \
        math.sin(dlon/2) * math.sin(dlon/2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def get_osrm_geometry(waypoints):
    if len(waypoints) < 2: return waypoints
    locs = ";".join([f"{lon},{lat}" for lat, lon in waypoints])
    url = f"http://router.project-osrm.org/route/v1/driving/{locs}?overview=full&geometries=polyline"
    try:
        r = requests.get(url, timeout=2)
        if r.status_code == 200:
            return polyline.decode(r.json()['routes'][0]['geometry'])
    except:
        pass
    return waypoints

def format_jam(float_jam):
    jam = int(float_jam)
    menit = int((float_jam - jam) * 60)
    return f"{jam:02d}:{menit:02d}"

# --- FUNGSI CLUSTERING (K-MEANS) ---
def get_clusters(dataset, n_clusters):
    # Ambil koordinat dealer (kecuali Gudang index 0)
    # Gunakan index 1 sampai akhir
    dealers = dataset[1:]
    coords = [[d['lat'], d['lon']] for d in dealers]
    
    # Jalankan K-Means
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    kmeans.fit(coords)
    labels = kmeans.labels_
    
    # Kelompokkan dealer berdasarkan label cluster
    # cluster_data = {0: [dealer_obj, ...], 1: [...]}
    clustered_indices = {i: [] for i in range(n_clusters)}
    
    for idx, label in enumerate(labels):
        # idx adalah index di list 'dealers', kita butuh index di 'dataset' (offset +1)
        clustered_indices[label].append(idx + 1)
        
    return clustered_indices

# --- ENGINE PERHITUNGAN BIAYA ---
def hitung_biaya_kompleks(urutan_idx, dataset, max_cap, k_dasar, f_beban, speed_kmh, wage_per_hour, penalty_per_hour):
    rute_truk_list = []
    
    # 1. SPLIT (Logic Kapasitas)
    truk_curr = []
    load_curr = 0
    for idx in urutan_idx:
        berat = dataset[idx]['berat']
        if load_curr + berat <= max_cap:
            truk_curr.append(idx)
            load_curr += berat
        else:
            rute_truk_list.append(truk_curr)
            truk_curr = [idx]
            load_curr = berat
    if truk_curr: rute_truk_list.append(truk_curr)

    # 2. HITUNG FISIKA & WAKTU
    total_bbm = 0
    total_waktu_jam = 0
    total_penalty = 0 
    jumlah_telat = 0
    
    START_TIME = 8.0 
    SERVICE_TIME = 0.5 
    
    for rute in rute_truk_list:
        full_path = [0] + rute + [0]
        muatan = sum([dataset[i]['berat'] for i in rute])
        current_time = START_TIME
        
        for i in range(len(full_path)-1):
            u, v = full_path[i], full_path[i+1]
            
            # Fisika Jarak
            dist = haversine(dataset[u]['lat'], dataset[u]['lon'], 
                             dataset[v]['lat'], dataset[v]['lon']) * 1.3
            
            # BBM
            konsumsi = k_dasar + (muatan * f_beban)
            bbm_segmen = dist * konsumsi
            total_bbm += bbm_segmen
            
            # Waktu
            travel_time = dist / speed_kmh
            current_time += travel_time
            
            # Time Window Check
            if v != 0: 
                jam_buka = dataset[v]['jam_buka']
                jam_tutup = dataset[v]['jam_tutup']
                
                # Nunggu kepagian
                if current_time < jam_buka:
                    current_time = jam_buka 
                
                # Telat
                if current_time > jam_tutup:
                    keterlambatan = current_time - jam_tutup
                    total_penalty += keterlambatan * penalty_per_hour 
                    jumlah_telat += 1
                
                current_time += SERVICE_TIME
                muatan -= dataset[v]['berat']
        
        durasi_kerja = current_time - START_TIME
        total_waktu_jam += durasi_kerja
            
    # Objective Function
    biaya_bbm_rp = total_bbm * 18500 # Harga Solar Industri
    biaya_driver_rp = total_waktu_jam * wage_per_hour
    
    score_fitness = biaya_bbm_rp + biaya_driver_rp + total_penalty
    
    detail_biaya = {
        'biaya_bbm': biaya_bbm_rp,
        'biaya_driver': biaya_driver_rp,
        'biaya_denda': total_penalty
    }
    
    return score_fitness, total_bbm, total_waktu_jam, jumlah_telat, rute_truk_list, detail_biaya

# --- 3. ALGORITMA GENETIKA (OPTIMIZER) ---
def run_optimization_complex(dataset, max_cap, speed, wage, penalty, pop_size, generations):
    # dataset disini adalah subset (Gudang + Dealer Cluster)
    # Maka index 1..len(dataset) adalah index lokal subset
    customer_indices = list(range(1, len(dataset)))
    
    if not customer_indices: # Handle jika cluster kosong
        return {
            'fitness': 0, 'bbm': 0, 'time': 0, 'late': 0, 
            'rute': [], 'detail': {'biaya_bbm':0, 'biaya_driver':0, 'biaya_denda':0}, 'history': []
        }

    populasi = []
    for _ in range(pop_size):
        p = customer_indices.copy()
        random.shuffle(p)
        populasi.append(p)
        
    best_fitness = float('inf')
    best_res = None
    history_fitness = []
    
    progress = st.progress(0)
    
    for gen in range(generations):
        scores = []
        for chrom in populasi:
            fit, bbm, time, late, rute, detail = hitung_biaya_kompleks(chrom, dataset, max_cap, 0.15, 0.0002, speed, wage, penalty)
            scores.append((fit, chrom))
            
        scores.sort()
        
        current_best = scores[0][0]
        history_fitness.append(current_best)
        
        if current_best < best_fitness:
            best_fitness = current_best
            _, bbm_f, time_f, late_f, rute_f, detail_f = hitung_biaya_kompleks(scores[0][1], dataset, max_cap, 0.15, 0.0002, speed, wage, penalty)
            best_res = {
                'fitness': best_fitness, 'bbm': bbm_f, 'time': time_f, 
                'late': late_f, 'rute': rute_f, 'detail': detail_f
            }
            
        # Evolution
        parents = [x[1] for x in scores[:pop_size//2]]
        children = []
        while len(children) < pop_size:
            p = random.choice(parents).copy()
            if random.random() < 0.4:
                if len(p) > 1: # Mutasi butuh minimal 2 node
                    i, j = random.sample(range(len(p)), 2)
                    p[i], p[j] = p[j], p[i]
            children.append(p)
        populasi = children
        
        # if gen % 5 == 0: progress.progress((gen+1)/generations)
        
    progress.empty()
    best_res['history'] = history_fitness
    return best_res

# --- 4. UI DASHBOARD ---
st.title("🚛 Smart Logistics: C-VRPTW Optimization Dashboard")
st.markdown("Sistem optimasi **Cluster-First, Route-Second** dengan Algoritma Genetika.")

# SIDEBAR
st.sidebar.header("⚙️ Skenario & Parameter")

# Parameter K-Means
n_clusters = st.sidebar.slider("Jumlah Cluster (Wilayah)", 1, 4, 2, help="Membagi area pengiriman menjadi beberapa zona.")

kapasitas = st.sidebar.number_input("Kapasitas Truk (kg)", 1000, 5000, 2500)
traffic_level = st.sidebar.select_slider("Kondisi Lalu Lintas", options=["Lancar", "Normal", "Macet", "Macet Parah"])
speed_map = {"Lancar": 50, "Normal": 35, "Macet": 20, "Macet Parah": 10}
avg_speed = speed_map[traffic_level]

st.sidebar.markdown("### 💰 Parameter Biaya")
# Menampilkan parameter biaya agar user tahu dasar perhitungannya
col_wage, col_penalty = st.sidebar.columns(2)
with col_wage:
    wage_driver = st.number_input("Upah/Jam (Rp)", 20000, 100000, 50000, step=5000)
with col_penalty:
    penalty_per_hour = st.number_input("Denda Telat/Jam", 50000, 500000, 100000, step=10000)

st.sidebar.info(
    f"**Asumsi Biaya:**\n"
    f"- Harga Solar: Rp 18.500/Liter\n"
    f"- Kecepatan: {avg_speed} km/jam\n"
)

st.sidebar.markdown("---")
jumlah_order = st.sidebar.slider("Jumlah Dealer Order", 5, 10, 8)

st.sidebar.header("🧬 Parameter AI")
pop_size = st.sidebar.slider("Population Size", 10, 100, 30)
generations = st.sidebar.slider("Generations", 10, 200, 40)

# DATA GENERATION
if 'dataset_clustered' not in st.session_state: st.session_state['dataset_clustered'] = None

if st.sidebar.button("📦 Generate Order Baru"):
    subset = [LOKASI_INDUSTRI[0]] + random.sample(LOKASI_INDUSTRI[1:], jumlah_order)
    for item in subset:
        if item['tipe'] == 'GUDANG': item['berat'] = 0
        else: item['berat'] = random.randint(200, 800)
    st.session_state['dataset_clustered'] = subset
    if 'res_clustered' in st.session_state: del st.session_state['res_clustered']

dataset = st.session_state['dataset_clustered']

if dataset:
    # MANIFEST VIEW
    st.subheader("1. Manifest Order & Jam Operasional")
    display_data = []
    for d in dataset:
        jam_str = f"{format_jam(d['jam_buka'])} - {format_jam(d['jam_tutup'])}" if d['tipe'] != 'GUDANG' else "24 Jam"
        display_data.append({'Lokasi': d['nama'], 'Muatan (kg)': d['berat'], 'Jam Buka': jam_str})
    st.dataframe(pd.DataFrame(display_data), use_container_width=True)
    
    if st.button("🚀 MULAI OPTIMASI (CLUSTERING + GA)", type="primary"):
        
        # A. TAHAP CLUSTERING
        with st.spinner("Membagi wilayah (K-Means Clustering)..."):
            clusters = get_clusters(dataset, n_clusters)
            
        final_agg = {
            'fitness': 0, 'bbm': 0, 'time': 0, 'late': 0, 
            'detail': {'biaya_bbm': 0, 'biaya_driver': 0, 'biaya_denda': 0},
            'all_routes_data': [], # Untuk visualisasi
            'history': []
        }
        
        # B. TAHAP OPTIMASI PER CLUSTER
        progress_bar = st.progress(0)
        
        for i in range(n_clusters):
            target_indices = clusters[i]
            if not target_indices: continue
            
            st.toast(f"Mengoptimalkan Cluster {i+1}...")
            
            # Buat Subset Dataset: Gudang (idx 0) + Dealer Cluster Ini
            subset_dataset = [dataset[0]] + [dataset[x] for x in target_indices]
            
            # Jalankan GA
            res = run_optimization_complex(subset_dataset, kapasitas, avg_speed, wage_driver, penalty_per_hour, pop_size, generations)
            
            # Aggregasi Hasil
            final_agg['fitness'] += res['fitness']
            final_agg['bbm'] += res['bbm']
            final_agg['time'] += res['time']
            final_agg['late'] += res['late']
            final_agg['detail']['biaya_bbm'] += res['detail']['biaya_bbm']
            final_agg['detail']['biaya_driver'] += res['detail']['biaya_driver']
            final_agg['detail']['biaya_denda'] += res['detail']['biaya_denda']
            
            # Mapping Rute Lokal (Subset) ke Rute Global (Dataset Utama) untuk Visualisasi
            for rute_lokal in res['rute']:
                rute_global = []
                muatan_rute = 0
                for idx_lokal in rute_lokal:
                    dealer_obj = subset_dataset[idx_lokal]
                    idx_global = dataset.index(dealer_obj) # Cari index asli
                    rute_global.append(idx_global)
                    muatan_rute += dealer_obj['berat']
                
                final_agg['all_routes_data'].append({
                    'cluster_id': i,
                    'rute_indices': rute_global,
                    'muatan': muatan_rute
                })
            
            progress_bar.progress((i + 1) / n_clusters)
            
        st.session_state['res_clustered'] = final_agg

    # HASIL ANALISIS
    if 'res_clustered' in st.session_state:
        res = st.session_state['res_clustered']
        
        st.divider()
        st.subheader("2. Dashboard Analisis")
        
        # A. METRICS UTAMA
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Biaya Logistik", f"Rp {int(res['fitness']):,}")
        c2.metric("Konsumsi BBM", f"{res['bbm']:.1f} Liter")
        c3.metric("Total Jam Kerja", f"{res['time']:.1f} Jam")
        
        if res['late'] == 0: 
            c4.success("✅ 0 Terlambat")
        else: 
            c4.error(f"⚠️ {res['late']} Dealer Terlambat")

        # B. GRAFIK PIE & BAR
        col_pie, col_bar = st.columns(2)
        
        with col_pie:
            st.markdown("**💰 Cost Breakdown**")
            costs = [res['detail']['biaya_bbm'], res['detail']['biaya_driver'], res['detail']['biaya_denda']]
            labels = ['BBM', 'Gaji Driver', 'Denda Telat']
            colors = ['#ff9999','#66b3ff','#ffcc99']
            
            if sum(costs) > 0:
                fig1, ax1 = plt.subplots(figsize=(4, 4))
                ax1.pie(costs, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
                st.pyplot(fig1)
            else:
                st.info("Belum ada biaya terhitung.")

        with col_bar:
            st.markdown("**🚛 Utilisasi Truk per Cluster**")
            load_data = []
            truck_labels = []
            colors_bar = []
            cluster_colors = ['red', 'blue', 'green', 'purple']
            
            for idx, route_data in enumerate(res['all_routes_data']):
                persen = (route_data['muatan'] / kapasitas) * 100
                load_data.append(persen)
                truck_labels.append(f"C{route_data['cluster_id']+1}-Truk{idx+1}")
                colors_bar.append(cluster_colors[route_data['cluster_id'] % len(cluster_colors)])
            
            if load_data:
                fig2, ax2 = plt.subplots(figsize=(5, 3))
                ax2.barh(truck_labels, load_data, color=colors_bar)
                ax2.set_xlim(0, 100)
                ax2.set_xlabel("Persentase Penuh (%)")
                st.pyplot(fig2)
                st.caption("*Warna Bar sesuai Cluster*")

        # C. PETA & ITINERARY
        st.subheader("3. Peta Rute Terklasterisasi")
        
        m = folium.Map(location=[dataset[0]['lat'], dataset[0]['lon']], zoom_start=11)
        Fullscreen().add_to(m)
        
        cluster_colors = ['red', 'blue', 'green', 'purple', 'orange']
        export_data = []
        
        # Marker Gudang
        folium.Marker(
            [dataset[0]['lat'], dataset[0]['lon']], 
            tooltip="GUDANG PUSAT", 
            icon=folium.Icon(color="black", icon="home", prefix='fa')
        ).add_to(m)

        # Loop setiap rute dari hasil aggregasi
        for idx_rute_global, route_data in enumerate(res['all_routes_data']):
            cluster_id = route_data['cluster_id']
            rute_indices = route_data['rute_indices']
            truk_name = f"Cluster {cluster_id+1} - Truk {idx_rute_global+1}"
            warnanya = cluster_colors[cluster_id % len(cluster_colors)]
            
            # Visualisasi Rute
            full_path_idx = [0] + rute_indices + [0]
            coords_osrm = [[dataset[idx]['lat'], dataset[idx]['lon']] for idx in full_path_idx]
            geometry_jalan = get_osrm_geometry(coords_osrm)
            
            AntPath(
                locations=geometry_jalan, color=warnanya, pulse_color='white', delay=1000,
                weight=6, opacity=0.8, tooltip=truk_name, name=truk_name
            ).add_to(m)
            
            # Hitung Waktu Itinerary untuk Export
            curr_time = 8.0
            for k in range(len(full_path_idx)-1):
                u, v = full_path_idx[k], full_path_idx[k+1]
                dist = haversine(dataset[u]['lat'], dataset[u]['lon'], dataset[v]['lat'], dataset[v]['lon']) * 1.3
                travel = dist / avg_speed
                curr_time += travel
                
                if v != 0:
                    status = "On-Time"
                    if curr_time > dataset[v]['jam_tutup']: status = "LATE"
                    if curr_time < dataset[v]['jam_buka']: curr_time = dataset[v]['jam_buka']
                    
                    folium.CircleMarker(
                        location=[dataset[v]['lat'], dataset[v]['lon']],
                        radius=6, color=warnanya, fill=True, fill_color='white', fill_opacity=1,
                        popup=f"<b>{dataset[v]['nama']}</b><br>{truk_name}<br>Tiba: {format_jam(curr_time)}"
                    ).add_to(m)
                    
                    export_data.append({
                        "Cluster": f"Wilayah {cluster_id+1}",
                        "Truk": truk_name,
                        "Urutan": k+1,
                        "Lokasi": dataset[v]['nama'],
                        "Waktu Tiba": format_jam(curr_time),
                        "Status": status
                    })
                    curr_time += 0.5
        
        folium.LayerControl().add_to(m)
        st_folium(m, width=1200, height=500)
        
        # Download
        st.markdown("---")
        df_export = pd.DataFrame(export_data)
        csv = df_export.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Laporan (CSV)", data=csv, file_name='laporan_cluster_vrp.csv', mime='text/csv')