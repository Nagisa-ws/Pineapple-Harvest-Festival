# Urutan pipeline v5: Pretraining → Misi 1 → Misi 2 → Misi 3

Panah `→` berarti "memanggil / memakai". Semua perintah dijalankan dari folder proyek.

## 0. Aset realistis (sekali saja)
```
python scripts/make_xgh_assets.py --src /path/ke/mesh_pineapple   # folder hasil unzip mesh_pineapple.zip
 → assets_xgh/   mesh UR10e (obj asli), gripper U, XGH505H, 5 jenis nanas (buah, mahkota, daun per helai + kelas warna),
                 tanah, keranjang; summary.json berisi pemeriksaan mesh (0 masalah = tidak ada mesh rusak)
```
Folder `assets_xgh/` (~80 MB) tidak ikut di zip skrip; perintah di atas membuatnya ulang dari folder mesh asli
Anda (`mesh_pineapple`: `pineapple_field/`, `xgh505h_ur10e/`, ...).

## 1. Pretraining: rig kamera + diffusion policy pemilih T2
```
configs/pretrain.yaml (+ common.yaml)
 → pretrain/make_t2_dataset.py        900 adegan × 6 pandangan, mesh realistis (buah, daun, mahkota)
     → pinesim/rig.py  StargazeRig     kamera di tiang, tanaman realistis miring 0–15°, buah 60–70 cm
     → pinesim/vision/perception.py   segmentasi HSV → point cloud → sumbu buah (fit silinder) → shoulder per pandangan
     → pinesim/decision/window.py     window (kotak pembatas shoulder) → kandidat T2 per sel 8×8 → T1 = T2 + 20 cm
                                       ke luar → skor ground truth (jarak garis T1–T2 ke titik daun)
     → outputs/pretrain/t2_data_{0,1}.npz
 → pretrain/train_t2_diffusion.py     DDPM (T=50, DDIM 10 langkah) atas fitur window → models/t2_diffusion.pt
                                       + outputs/pretrain/t2_diffusion_eval.json (perbandingan dengan heuristik/acak/oracle)
 → pretrain/stargaze_viewer.py        demo/slider: tiang bergerak → komputasi dibatalkan; diam → hitung
                                       (lampu: merah = bergerak, kuning = menunggu diam, biru = menghitung, hijau = selesai)
 → pretrain/eval_rig.py               statistik kamera vs ground truth
```
Pemilih T2 (`pinesim/decision`) dipakai ulang di semua misi (`perception.decision: diffusion` di common.yaml).

## 2. Misi 1 (platform, satu tanaman)
```
configs/mission1.yaml → scripts/train_mission.py --mission mission1
 → pinesim/rl/train.py   init = third_party/abc_mission3
 → pinesim/envs/harvest_env.py  HarvestEnv (adegan ringan `platform`)
     pose default → scan → look → keputusan T1/T2 (diffusion) → ujung jari ke T1 → lurus ke T2 → buka →
     maju beberapa cm → cengkeram → puntir → cabut → keranjang → pose default
 → checkpoints/mission1/ → models/mission1_v1.onnx
evaluasi realistis: scripts/eval_realistic.py --mission mission1   (adegan `platform_detail`, envs/platform_detail.py)
```

## 3. Misi 2 (platform, tanaman + satu tetangga)
```
configs/mission2.yaml → scripts/train_mission.py --mission mission2   (init = checkpoints/mission1)
 → HarvestEnv yang sama, tetangga 42–55 cm, buah terdekat dulu, pose default setelah tiap sesi
evaluasi realistis: scripts/eval_realistic.py --mission mission2
```

## 4. Misi 3 (XGH505H, gabungan Misi 3 + 4 lama)
```
configs/mission3.yaml → scripts/train_mission.py --mission mission3   (init = checkpoints/mission2)
 → pinesim/envs/xgh_env.py  XghHarvestEnv (adegan ringan `xgh_light`)
     → pinesim/envs/vehicle_env.py   maju hanya saat lengan di pose default; satu buah = satu sesi
 → checkpoints/mission3/ → models/mission3_v1.onnx
evaluasi realistis: scripts/eval_xgh.py (cepat) atau scripts/eval_xgh_sections.py run / merge (ladang 14,75 m, 6 seksi)
 → XghEvalEnv: ladang 2 × 30 tanaman (jarak 50 cm, zig-zag 25 cm), hanya 3 tanaman interaktif (pinesim/lod.py)
```

## 5. Visualisasi
`pinesim/viz/xgh_recorder.py` merekam semua adegan (ringan/realistis) dan `export_html` membuat HTML Meshcat mandiri:
panel kamera (bebas, ikuti kendaraan, kamera pojok, pandangan atas, **kamera gripper** dengan bounding box, window,
shoulder, kandidat T2, T1, T2 dan kotak info), lampu status berkedip, panel Animations untuk memutar/menggeser waktu.
`scripts/visualize_meshcat.py` merekam satu episode misi mana pun.

## Skrip rangkaian
- `scripts/run_curriculum.sh [m1] [m2] [m3]`: aset → dataset T2 → diffusion → training M1 → M2 → M3 (lewati yang sudah ada).
- `scripts/run_final_eval.sh [n1] [n2]`: demo pretraining + evaluasi realistis M1, M2, M3 dengan HTML.
