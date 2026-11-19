# Hasil terukur v6

Semua angka berasal dari run yang benar-benar dijalankan dengan kode akhir v6 di mesin pengembangan (Linux, 2 inti CPU,
tanpa GPU, MuJoCo 4.0.1). File sumber: `outputs/eval_realistic/*/result.json`, `outputs/eval_xgh/*/result.json`,
`checkpoints/*/eval_log.csv`. Hasil v5 lama: `docs/HASIL_V5.md`.

## Target pengguna dan hasilnya
| Target | Hasil terukur | Tercapai |
|---|---|---|
| Observasi → keranjang rata-rata < 40 s | Misi 1: 26,3 s · Misi 2: 21,9 s · Misi 3 (baris penuh): 23,2 s | ya |
| Panen < 40 s per buah | 100 % buah yang dipanen < 40 s (terlama 36,8 s); seluruh waktu run / buah: 26–30 s | ya |
| Menyelesaikan satu baris ladang | 2 × 30 tanaman, 15,1 m, satu run: 59/60 buah di keranjang, 1 dilewati | ya (1 buah tertinggal) |
| Tingkat keberhasilan panen > 85 % | Misi 1: 96,7 % · Misi 2: 92,5 % buah (90 % episode) · Misi 3: 98,3 % | ya |
| Gripper tidak menjatuhkan buah karena tabrakan | 0 kejadian `fruit_knocked_off` / `plant_hit` di semua evaluasi akhir | ya |
| Kamera gripper memastikan posisi buah pada pose ideal | satu pandangan dekat pada pose ideal + pemeriksaan ulang di T1; galat median 0,6–0,8 cm | ya |
| Baris ladang diam kiri/kanan, jarak tepi dapat diatur | 128 tanaman mesh diam, `edge_gap` awal 0,50 m | ya |

"Waktu" = waktu gerak robot (simulasi) + waktu komputasi nyata persepsi/keputusan/perencanaan di mesin ini (definisi di
`pinesim/timing.py` dan `docs/PERUBAHAN_V6.md` bagian 7).

## Evaluasi realistis akhir (seed uji yang belum pernah dipakai saat pengembangan)
| Misi | Policy | Episode sukses | Buah di keranjang | Observasi → keranjang (rata-rata / median / terlama) | Seluruh run per buah | Kegagalan |
|---|---|---|---|---|---|---|
| Misi 1 (seed 940000–940029) | `mission1_v1.onnx` | **29/30** | 29/30 | 26,3 / 25,6 / 36,8 s | 30,1 s | 1 × `fruit_dropped` (angkut) |
| Misi 1 (seed yang sama) | controller referensi | 29/30 | 29/30 | 25,3 / 24,3 / 36,1 s | 28,9 s | 1 × `fruit_dropped` (seed yang sama) |
| Misi 2 (seed 950000–950019) | `mission2_v1.onnx` | **18/20** | **37/40** | 21,9 / 21,0 / 30,4 s | 26,1 s | 1 × `plan_no_feasible_route`, 1 × `plan_window_no_shoulder` |
| Misi 2 (seed yang sama) | controller referensi | 18/20 | 37/40 | 21,2 / 20,2 / 30,6 s | 25,5 s | sama |
| Misi 3, baris penuh 2 × 30 tanaman (ladang seed 2026) | controller referensi | baris selesai | **59/60** | 23,2 / 22,3 / 34,4 s | 27,4 s | 1 buah dilewati (`plan_no_feasible_route`) |
| Misi 3, 20 tanaman pertama (ladang seed 2026) | `mission3_v1.onnx` | – | 20/20 | 24,4 / 24,6 / 35,8 s | 29,6 s | – |
| Misi 3, ladang lain 2 × 10 tanaman (ladang seed 2027) | controller referensi | – | 19/20 | 23,1 / 23,6 / 27,5 s | 29,4 s | 2 kejadian lewati (`pull_slip`, `plan_window_no_shoulder`) |

Catatan:
- Tidak ada daun tercabut dan tidak ada buah tersenggol jatuh di evaluasi akhir. "Daun tertekuk > 52°" (dihitung,
  tidak menghentikan panen): Misi 1 = 3, Misi 2 = 3, Misi 3 baris penuh = 18.
- Misi 3 baris penuh: kendaraan berhenti 55 kali, menempuh 15,13 m, geser maksimum saat berhenti 0,8 mm, selalu hanya
  3 tanaman interaktif, 3 kali bongkar keranjang oleh operator (kapasitas 8 buah per keranjang), 1.456 s gerak +
  160 s komputasi untuk 59 buah.
- Episode Misi 2 yang gagal berakhir pada buah yang gagal (buah kedua tidak dicoba), karena itu angka per episode
  lebih rendah daripada angka per buah.

## HTML Meshcat (folder evaluasi)
| File | Isi |
|---|---|
| `eval_realistic/mission1_onnx/mission1_seed940000.html`, `…940001.html` | dua episode Misi 1 yang sukses |
| `eval_realistic/mission1_onnx/mission1_seed940024_gagal.html` | satu-satunya episode Misi 1 yang gagal |
| `eval_realistic/mission2_onnx/mission2_seed950000.html`, `…950001.html` | dua episode Misi 2 yang sukses |
| `eval_realistic/mission2_onnx/mission2_seed950004_gagal.html`, `…950014_gagal.html` | dua episode Misi 2 yang gagal |
| `eval_xgh/ref_baris_penuh/html/part_01.html`, `part_02.html` | baris penuh Misi 3 (x 0–8 m dan 8–15 m) dengan baris tetangga diam |

Kotak info di kanan bawah setiap HTML menampilkan jumlah buah dan waktu observasi → keranjang.

## Training (adegan ringan) dan apa yang dipelajari policy
Rincian per evaluasi periodik: `checkpoints/RINGKASAN_TRAINING.md`.

| Misi | Langkah | Waktu | Controller referensi (validasi) | SAC terbaik (validasi) | SAC di akhir training |
|---|---|---|---|---|---|
| Misi 1 | 60.000 | 21 menit | 9/10 | 9/10 (langkah 10.000) | 8/10 |
| Misi 2 | 30.000 | 14 menit | 5/6 episode, 11/12 buah | 5/6, 11/12 (langkah 30.000) | sama |
| Misi 3 | 12.000 | 15 menit | 7/8 buah (2 episode) | 6/8 buah | 6/8 buah |

**Kesimpulan jujur tentang RL:** policy residual SAC tidak melampaui controller referensi di misi mana pun. Di Misi 1
hasilnya turun bila training diteruskan, di Misi 3 lebih buruk daripada referensi pada seed validasi. Pada evaluasi
realistis akhir ONNX dan referensi memberi jumlah buah yang sama; ONNX sedikit lebih lambat (±1 s per buah). Peningkatan
v6 (Misi 1: 8/12 → 29/30; Misi 3: 37/60 pada evaluasi v5 Anda → 59/60; ±75–95 s → ±27 s per buah; angka "sebelum"
diukur dengan kode v5 sebelum diubah) seluruhnya berasal dari
perbaikan persepsi, perencanaan, cengkeraman dan pemulihan. Ini menjawab pertanyaan awal ("mengapa model tidak membaik
walau dilatih lama"): kegagalan yang tersisa terjadi di bagian yang tidak dapat dipengaruhi oleh aksi residual.
Agar policy tidak merugikan, residual hanya bekerja di fase approach/line, dimatikan oleh watchdog bila tidak ada
kemajuan, dan bila shield tetap terblokir lengan mundur ke awal fase lalu menyelesaikan buah itu dengan referensi saja
(terjadi 2 kali pada 20 episode Misi 2; kedua buah tetap terpanen).

## Yang masih kurang (terukur)
- Misi 2: 2 dari 20 episode gagal di tahap observasi buah kedua (tidak ada pose pandang yang bebas tabrakan; shoulder
  buah tidak terlihat). Buah seperti ini di Misi 3 dilewati dan baris dilanjutkan; di Misi 2 episode berakhir.
- Misi 3: 1 dari 60 buah dilewati (tidak ada rute yang layak dari semua pandangan).
- Misi 1: 1 buah jatuh dari cengkeraman saat diangkut (cengkeraman setelah puntir tidak di tengah pad).
- Adegan ringan (training) lebih sulit bagi controller daripada adegan realistis: daun kapsul yang kaku menahan buah
  saat ditarik keluar (`leaf_uprooted`/`fruit_dropped` saat retreat adalah kegagalan validasi yang tersisa).

## Batasan pengukuran (baca sebelum memakai angka di atas)
- Jumlah episode kecil (30 / 20 / 60 buah): selang kepercayaan 95 % untuk 29/30 kira-kira 83–99 %, untuk 37/40 kira-kira
  80–97 %, untuk 59/60 kira-kira 91–100 %.
- Seed 920000–920029 (Misi 1) dan 930000–930019 (Misi 2) serta tanaman 0–24 ladang seed 2026 dipakai untuk mencari
  kesalahan selama pengembangan. Angka akhir Misi 1–2 di atas memakai seed lain (940000…, 950000…); untuk Misi 3
  tanaman 24–60 ladang 2026 dan ladang seed 2027 tidak pernah dipakai untuk penyetelan.
- Training terakhir dijalankan sebelum mekanisme "mundur ke awal fase" ditambahkan; mekanisme itu hanya mengubah apa
  yang terjadi setelah shield memblokir rute, policy yang dievaluasi adalah policy hasil training tersebut.
- Waktu komputasi diukur di mesin 2 inti; penjaga daun memakai tekukan pangkal daun dari simulator dan cengkeram ulang
  memakai posisi kontak pad (lihat `docs/PERUBAHAN_V6.md` bagian 9).
- Ini simulasi: kekakuan daun, kekuatan batang dan gesekan pad adalah tebakan, belum divalidasi di robot nyata.
