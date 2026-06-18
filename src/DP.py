import pandas as pd
import os
import time

def solve_hamiltonian_path_dp_undirected(file_path, start_node, harga_bensin):
    # Membaca dataset
    df = pd.read_csv(file_path)
    
    # Ekstraksi semua node unik di dalam graf
    nodes = set(df['source']).union(set(df['target']))
    nodes = sorted(list(nodes))
    n = len(nodes)
    
    # Total barang keseluruhan
    total_barang = sum(nodes)
    
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
        
    # Inisialisasi DP Bitmask 
    dp = [[float('inf')] * n for _ in range(1 << n)]
    parent = [[-1] * n for _ in range(1 << n)]
    
    start_idx = node_to_idx[start_node]
    dp[1 << start_idx][start_idx] = 0
    
    # Transisi State DP
    for mask in range(1 << n):
        for u in range(n):
            if (mask & (1 << u)) and dp[mask][u] != float('inf'):
                for v in range(n):
                    if not (mask & (1 << v)) and adj_matrix[u][v] != float('inf'):
                        new_mask = mask | (1 << v)
                        new_dist = dp[mask][u] + adj_matrix[u][v]
                        
                        if new_dist < dp[new_mask][v]:
                            dp[new_mask][v] = new_dist
                            parent[new_mask][v] = u
                            
    # Mencari titik akhir 
    final_mask = (1 << n) - 1
    min_dist = float('inf')
    last_node_idx = -1
    
    for v in range(n):
        if dp[final_mask][v] < min_dist:
            min_dist = dp[final_mask][v]
            last_node_idx = v
            
    if min_dist == float('inf'):
        return float('inf'), [], total_barang, [], 0
        
    # Backtracking jalur
    path_idx = []
    curr_mask = final_mask
    curr_node = last_node_idx
    
    while curr_node != -1:
        path_idx.append(curr_node)
        prev_node = parent[curr_mask][curr_node]
        curr_mask = curr_mask ^ (1 << curr_node)
        curr_node = prev_node
        
    path_idx.reverse()
    path = [idx_to_node[idx] for idx in path_idx]
    
    # Menghitung Biaya BBM
    rincian_perjalanan = []
    total_biaya_bbm = 0
    current_barang = total_barang
    
    for i in range(len(path) - 1):
        u = path[i]
        v = path[i+1]
        
        current_barang -= u  
        rasio_bbm_rute = 0.05 * current_barang
        
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
    
    return min_dist, path, total_barang, rincian_perjalanan, total_biaya_bbm

def tampilkan_hasil_tsp(jarak, jalur, total_barang, rincian_perjalanan, total_biaya, waktu_eksekusi, harga_bensin, biaya_server):
    """
    Void function murni untuk mencetak hasil metrik pengiriman dan server.
    """
    print("\n" + "="*75)
    print(" Laporan Optimasi Rute, Biaya Kurir & Komputasi Server")
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
    print("--- Program Perangkat Lunak Manajemen Rute Kurir ---")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    
    nama_file = input("Masukkan nama file CSV yang ada di folder 'data' (contoh: data.csv): ")
    file_path = os.path.join(root_dir, 'data', nama_file)
    
    if os.path.exists(file_path):
        try:
            node_awal = int(input("Masukkan titik node untuk memulai perjalanan: "))
            input_bensin = float(input("Masukkan harga bensin per liter (contoh: 10000): "))
            
            start_time = time.perf_counter()
            
            jarak_optimal, jalur_optimal, banyak_barang, rincian_jalan, total_biaya = solve_hamiltonian_path_dp_undirected(
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