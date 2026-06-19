import pandas as pd
import os
import time

def solve_hamiltonian_path_nearest_neighbor(file_path, start_node, harga_bensin):
    # Membaca dataset
    df = pd.read_csv(file_path)
    
    # Ekstraksi semua node unik di dalam graf
    nodes = set(df['source']).union(set(df['target']))
    nodes = sorted(list(nodes))
    n = len(nodes)
    
    # Total barang keseluruhan 
    total_barang = n - 1 
    
    if start_node not in nodes:
        raise ValueError(f"Node awal {start_node} tidak ditemukan di dalam graf.")
        
    # Mapping node ID asli ke indeks
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    idx_to_node = {i: node for i, node in enumerate(nodes)}
    
    # Inisialisasi adjacency matrix
    adj_matrix = [[float('inf')] * n for _ in range(n)]
    
    for _, row in df.iterrows():
        u_idx = node_to_idx[int(row['source'])]
        v_idx = node_to_idx[int(row['target'])]
        w = row['weight']
        
        adj_matrix[u_idx][v_idx] = min(adj_matrix[u_idx][v_idx], w)
        adj_matrix[v_idx][u_idx] = min(adj_matrix[v_idx][u_idx], w)
        
    # --- IMPLEMENTASI HEURISTIC NEAREST NEIGHBOR DENGAN BACKTRACKING ---
    visited = [False] * n
    start_idx = node_to_idx[start_node]
    
    best_path_idx = []
    best_min_dist = float('inf')
    
    def dfs_nn(curr_idx, current_dist, path):
        nonlocal best_min_dist, best_path_idx
        
        # Jika semua node sudah dikunjungi, simpan jalurnya
        if len(path) == n:
            if current_dist < best_min_dist:
                best_min_dist = current_dist
                best_path_idx = path.copy()
            return True
            
        # Kumpulkan semua tetangga yang valid beserta jaraknya
        neighbors = []
        for v_idx in range(n):
            if not visited[v_idx] and adj_matrix[curr_idx][v_idx] != float('inf'):
                neighbors.append((adj_matrix[curr_idx][v_idx], v_idx))
                
        # Urutkan tetangga berdasarkan jarak terdekat (Sifat Heuristic Nearest Neighbor)
        neighbors.sort(key=lambda x: x[0])
        
        # Coba kunjungi tetangga dari yang terdekat
        for dist, next_idx in neighbors:
            # Pruning: jika jarak saat ini sudah melebihi jarak terbaik yang pernah ditemukan, abaikan
            if current_dist + dist >= best_min_dist:
                continue
                
            visited[next_idx] = True
            path.append(next_idx)
            
            # Lanjutkan pencarian ke node berikutnya
            dfs_nn(next_idx, current_dist + dist, path)
            
            # Backtrack jika jalur di depan buntu atau tidak optimal
            path.pop()
            visited[next_idx] = False
            
        return len(best_path_idx) == n

    # Jalankan pencarian awal
    visited[start_idx] = True
    dfs_nn(start_idx, 0, [start_idx])
    
    # Jika setelah backtracking tetap tidak ditemukan jalur yang mencakup semua node
    if best_min_dist == float('inf'):
        return float('inf'), [], total_barang, [], 0
        
    # Mengembalikan urutan indeks ke bentuk Node ID asli
    path = [idx_to_node[idx] for idx in best_path_idx]
    min_dist = best_min_dist
    
    # Menghitung Biaya BBM
    rincian_perjalanan = []
    total_biaya_bbm = 0
    current_barang = total_barang
    
    for i in range(len(path) - 1):
        u = path[i]
        v = path[i+1]
        
        if total_barang > 0:
            rasio_bbm_rute = 0.02 + (current_barang / total_barang) * (0.05 - 0.02)
        else:
            rasio_bbm_rute = 0.02
        
        u_idx = node_to_idx[u]
        v_idx = node_to_idx[v]
        jarak_rute = adj_matrix[u_idx][v_idx]
        
        biaya_rute = jarak_rute * rasio_bbm_rute * harga_bensin
        total_biaya_bbm += biaya_rute
        
        rincian_perjalanan.append({
            'dari': u,
            'ke': v,
            'jarak': jarak_rute,
            'barang_dibawa': current_barang, 
            'biaya': biaya_rute
        })
        
        if current_barang > 0:
            current_barang -= 1
    
    return min_dist, path, total_barang, rincian_perjalanan, total_biaya_bbm

def tampilkan_hasil_tsp(jarak, jalur, total_barang, rincian_perjalanan, total_biaya, waktu_eksekusi, harga_bensin, biaya_server):
    """
    Void function murni untuk mencetak hasil metrik pengiriman dan server.
    """
    print("\n" + "="*75)
    print(" Laporan Optimasi Rute, Biaya Kurir & Komputasi Server (Nearest Neighbor - Incomplete Graph)")
    print("="*75)
    
    print(f"Total Barang Start             : {total_barang} barang")
    print(f"Harga Bensin (Input)           : Rp {harga_bensin:,.2f}")
    print(f"Waktu komputasi algoritma      : {waktu_eksekusi:.6f} detik")
    print(f"Biaya Komputasi Server         : Rp {biaya_server:,.6f}\n")
    
    if jarak == float('inf'):
        print("Status                         : GAGAL.")
        print("Keterangan                     : Tidak ada rute yang memungkinkan untuk mengunjungi SEMUA vertex tepat satu kali.")
    else:
        print(f"Jarak total minimum            : {jarak}")
        print(f"Jalur optimal                  : {' -> '.join(map(str, jalur))}")
        
        print("\n--- RINCIAN AKTIVITAS DI SETIAP VERTEX & BIAYA JALAN ---")
        for rincian in rincian_perjalanan:
            print(f" [Tiba di Node {rincian['dari']:>2}] Drop {rincian['dari']:>2} brg | Sisa Bawaan: {rincian['barang_dibawa']:>2} brg")
            print(f"   -> Berkendara ke Node {rincian['ke']:>2} | Jarak: {rincian['jarak']:>2} | Biaya BBM: Rp {rincian['biaya']:,.2f}")
            
        node_terakhir = jalur[-1]
        print(f" [Tiba di Node {node_terakhir:>2}] Drop {node_terakhir:>2} brg | Sisa Bawaan:  0 brg")
        print("   -> (Seluruh Barang Telah Terkirim - Perjalanan Selesai)")
        
        print("-" * 75)
        print(f" TOTAL KESELURUHAN BIAYA BBM   : Rp {total_biaya:,.2f}")
        
        total_operasi = total_biaya + biaya_server
        print(f" TOTAL COST OF OWNERSHIP : Rp {total_operasi:,.2f}")
        print("=" * 75)

if __name__ == "__main__":
    print("--- Program Perangkat Lunak Manajemen Rute Kurir (Heuristic) ---")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    
    # Default path: data/data.csv
    file_path = os.path.join(root_dir, 'data', 'data.csv')
    
    if not os.path.exists(file_path):
        nama_file = input("Masukkan nama file CSV Anda (contoh: data.csv): ")
        file_path = os.path.join(root_dir, 'data', nama_file)
        
    if os.path.exists(file_path):
        try:
            node_awal = int(input("Masukkan titik node untuk memulai perjalanan: "))
            input_bensin = float(input("Masukkan harga bensin per liter (contoh: 10000): "))
            
            start_time = time.perf_counter()
            
            jarak_optimal, jalur_optimal, banyak_barang, rincian_jalan, total_biaya = solve_hamiltonian_path_nearest_neighbor(
                file_path, 
                start_node=node_awal, 
                harga_bensin=input_bensin
            )
            
            end_time = time.perf_counter()
            durasi_eksekusi = end_time - start_time
            
            # Menghitung biaya komputasi server: durasi dikali tarif 50
            biaya_komputasi = durasi_eksekusi * 50
            
            tampilkan_hasil_tsp(
                jarak_optimal, jalur_optimal, banyak_barang,
                rincian_jalan, total_biaya, durasi_eksekusi, input_bensin, biaya_komputasi
            )
            
        except ValueError:
            print("Error: Input node awal / harga bensin harus berupa angka.")
        except Exception as e:
            print(f"Terjadi kesalahan: {e}")
    else:
        print(f"Error: File tidak ditemukan di path: {file_path}")