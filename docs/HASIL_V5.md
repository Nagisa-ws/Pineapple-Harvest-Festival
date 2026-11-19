# Hasil terukur v5

Semua angka di bawah berasal dari run yang benar-benar dijalankan di mesin pengembangan (Linux, 2 inti CPU, tanpa GPU).
Jumlah episode kecil, jadi angka persentase punya ketidakpastian besar; anggap sebagai pemeriksaan fungsi, bukan
statistik akhir.

## Pretraining: pemilih T2 diffusion
Dataset 5.399 sampel (900 adegan realistis × 6 pandangan; 4.860 latih / 539 validasi). Skor = jarak garis T1–T2 ke titik
daun sebenarnya + 0,5 × jarak T2 ke daun (meter, makin besar makin aman), rata-rata pada data validasi:

| Pemilih | Skor validasi |
|---|---|
| Oracle (kandidat terbaik menurut ground truth) | 0,1081 |
| **Diffusion, mode "proposal" (dipakai)** | **0,1051** |
| Heuristik kamera (skor dari titik daun yang terlihat) | 0,1053 |
| Diffusion, suara saja | 0,1038 |
| Acak | 0,0718 |

Diffusion mendekati oracle (97 %) dan jauh di atas acak, setara dengan heuristik kamera. Kandidat pertama diffusion
sama dengan kandidat terbaik ground truth pada 86 % sampel validasi (`top1_hit`).

Demo rig (`outputs/pretrain/rig_demo.html`, 3 tanaman realistis): 9 komputasi dimulai, 3 dibatalkan karena kamera
bergerak, 6 selesai; galat pusat buah 0,2–1,1 cm, galat reproyeksi 0,2–2,4 px, jarak garis-daun hasil kamera vs
sebenarnya selisih ≤ 0,7 cm; satu keputusan 135–206 ms.

## Training (adegan ringan)
| Misi | Langkah | Waktu | Evaluasi periodik (adegan ringan) | Bobot awal |
|---|---|---|---|---|
| Misi 1 | 30.000 | 26 menit | terbaik 4/6 episode (langkah 20.000) | Misi 3 referensi (abc_SAC_xyzBOT) |
| Misi 2 | 20.000 | 23 menit | 1/4 episode, 3/8 buah (stabil sepanjang training) | Misi 1 |
| Misi 3 | 3.000 | 40 menit | episode training terakhir 4/4 buah; evaluasi periodik 2/4 buah | Misi 2 |

Misi 3 hanya mendapat 3.000 langkah: satu episode ringan Misi 3 (2 × 2 tanaman) memakan ~10 menit di mesin ini.
`checkpoints/<misi>/train_log.csv` dan `eval_log.csv` berisi setiap episode.

## Evaluasi realistis (training ringan → evaluasi realistis)
| Misi | Policy | Episode | Sukses | Buah di keranjang | Kegagalan | Kerusakan daun | Galat estimasi buah (median) |
|---|---|---|---|---|---|---|---|
| Misi 1 | `mission1_v1.onnx` | 6 | **6/6** | 6/6 | – | 0 | 0,50 cm |
| Misi 2 | `mission2_v1.onnx` | 4 | 2/4 | 5/8 | buah kedua tidak tercapai (`carry_infeasible`), pandangan dekat kehilangan buah (`perception_lost`) | 2 | 0,84 cm |
| Misi 3 | `mission3_v1.onnx` | 1 baris pendek (10 tanaman, 3 m) | – | **7/10** | 3 buah tidak terpanen; 4 kejadian "lewati" (lunak): 2 `perception_lost`, 1 `route_blocked`, 1 `fruit_knocked_off` | 0 | 0,73 cm |

Tidak ada daun yang tercabut di Misi 1–3 realistis, dan tidak ada kegagalan keras (tabrakan, jatuh ke tanah, waktu
habis) di Misi 3.

### Misi 3 realistis secara rinci (`outputs/eval_xgh/final10/result.json`)
- Ladang realistis 2 × 5 tanaman (jarak 50 cm, zig-zag 25 cm, miring 0–15°, buah 60–70 cm), seed 990000.
- Kendaraan maju 10 kali, total 3,03 m, geser maksimum saat berhenti 1,6 mm; lengan di pose default setiap kali maju.
- Tanaman interaktif maksimum **3** sepanjang run (18 kali tukar LOD); keranjang: 4 + 3 buah.
- 11 percobaan, 8 tercengkeram, 7 lepas dari batang dan masuk keranjang.
- 525 s waktu simulasi, 38 menit waktu nyata (CPU).
- Ladang penuh 15 m (2 × 30 tanaman) bisa dijalankan dengan `scripts/eval_xgh_sections.py run` (6 seksi, bisa
  dilanjutkan); tidak dijalankan penuh di sini karena perkiraan waktunya ±4 jam di mesin 2 inti.

Pemeriksaan adegan ringan dengan controller referensi (sebelum training Misi 3, seed 302, 2 × 4 tanaman): 7/8 buah.

## Catatan yang jujur
- Sebelum perbaikan terakhir, Misi 1 realistis 5/6 dan Misi 2 punya 2 tabrakan pergelangan dengan platform saat fase
  approach. Perbaikannya: margin jarak 1 cm untuk lengan vs dirinya sendiri / platform / keranjang di pemeriksa rute,
  dan **safety shield** di antara policy dan servo (perintah yang akan menabrak diganti aksi referensi; bila tetap
  menabrak, lengan menahan posisi; tertahan 50 langkah = `route_blocked`). Setelah itu tabrakan hilang pada seed yang
  sama, tetapi dua episode Misi 2 gagal di tahap lain.
- Daun tercabut dimodelkan sebagai tekukan sendi pangkal daun > 66°; angka ini tebakan.
- Training RL di sini pendek; sebagian besar keberhasilan datang dari controller referensi + perencana (policy residual
  hanya mengoreksi gerakan).
