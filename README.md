# Robot Panen Nanas v6: panen < 40 s per buah, satu baris penuh, baris tetangga diam (MuJoCo, SAC residual, Drake Meshcat)

UR10e + gripper U (jari tertutup 11,65 cm, buka 16,65 cm) + kamera pergelangan. Tanaman realistis (5 jenis mesh
nanas, daun per helai hijau/oranye, mahkota hijau tua, buah kuning) dengan buah 60–70 cm dari tanah dan tanaman miring
0–15° ke segala arah. Visualisasi: HTML Drake Meshcat mandiri (buka di browser, tidak perlu menjalankan simulasi).

**Yang baru di v6** (rinci: `docs/PERUBAHAN_V6.md`, angka: `docs/HASIL.md`):
- observasi → keranjang rata-rata jauh di bawah 40 s per buah (survei berhenti lebih awal, satu pandangan dekat pada
  pose ideal, gerak sendi berwaktu, memori buah di Misi 3);
- gripper tidak menjatuhkan buah: pembatas kecepatan dekat buah yang terlihat, pemeriksaan jalur terhadap buah, mahkota
  dan batang, maju dengan penjaga gaya, kamera gripper memastikan posisi buah di T1;
- cengkeram di bagian buah terlebar (taktil), puntir dua arah, tarik aksial, pola peletakan di keranjang;
- Misi 3: baris ladang diam di kiri dan kanan, jarak antar tepi ladang bisa diatur (`field.side_rows.edge_gap`,
  awal 0,50 m), satu baris 2 × 30 tanaman selesai dalam satu run;
- waktu per buah (gerak + komputasi) dilaporkan di setiap evaluasi dan di HTML.

| Tahap | Isi | Training / evaluasi |
|---|---|---|
| **Pretraining** | Rig kamera di tiang mengelilingi tanaman realistis; diffusion policy memilih T2. | adegan realistis |
| **Misi 1** | Satu tanaman di depan platform: survei → pandangan dekat → T1/T2 → cengkeram → puntir → keranjang. | training ringan, evaluasi realistis |
| **Misi 2** | Misi 1 + satu tetangga (42–55 cm). | training ringan, evaluasi realistis |
| **Misi 3** | XGH505H di ladang 2 × 30 tanaman (15 m) + baris tetangga diam kiri/kanan; hanya 3 tanaman interaktif. | training ringan, evaluasi realistis |

## Status terukur (evaluasi realistis, seed uji; rincian di `docs/HASIL.md`)
| Tahap | Policy | Hasil |
|---|---|---|
| Misi 1 (30 episode) | `mission1_v1.onnx` | 29/30 episode; observasi → keranjang rata-rata 26,3 s (terlama 36,8 s) |
| Misi 2 (20 episode) | `mission2_v1.onnx` | 18/20 episode, 37/40 buah; rata-rata 21,9 s (terlama 30,4 s) |
| Misi 3, satu baris penuh 2 × 30 tanaman | controller referensi | 59/60 buah, baris selesai; rata-rata 23,2 s (terlama 34,4 s), 27,4 s per buah termasuk jalan |
| Misi 3, 20 tanaman pertama | `mission3_v1.onnx` | 20/20 buah; rata-rata 24,4 s |
| Misi 3, ladang lain (seed 2027, 20 tanaman) | controller referensi | 19/20 buah; rata-rata 23,1 s |

Tidak ada buah tersenggol jatuh dan tidak ada daun tercabut di evaluasi akhir. Policy residual SAC **tidak** melampaui
controller referensi (jumlah buah sama); peningkatan v6 berasal dari pipeline persepsi/perencanaan/cengkeraman.

## Mengatur baris ladang kiri/kanan (Misi 3)
`configs/common.yaml`:
```yaml
field:
  side_rows:
    enabled: true
    edge_gap: 0.50      # jarak antar TEPI ladang (m): tepi = garis tengah baris tanaman terluar sebuah bedengan
    beds_per_side: 1    # jumlah bedengan tetangga per sisi
    extra_plants: 1     # tanaman tambahan di depan/belakang baris
```
atau per evaluasi: `python scripts/eval_xgh.py --edge-gap 0.80` (`--edge-gap 0` mematikannya). Baris tetangga adalah
mesh diam: tanpa sendi, tanpa kontak, tidak pernah menjadi target.

Jumlah bedengan tetangga per sisi: `beds_per_side: 3` (3 di kiri + 3 di kanan), atau per evaluasi `--side-beds 3`.
Catatan: baris tetangga TERLIHAT oleh kamera robot, jadi jalannya panen bisa sedikit berbeda dari run dengan 1 bedengan.

## Pemandangan di HTML Misi 3 (kubah foto 360°, tanah dilebarkan, batu, pohon, tunggul, gubuk)
Hiasan ini hanya ada di HTML Meshcat. Ia bukan bagian model MuJoCo dan tidak terlihat kamera robot, sehingga fisika,
persepsi, waktu dan hasil evaluasi sama persis dengan atau tanpa hiasan. Kode: `pinesim/viz/scenery.py`, aset:
`assets_xgh/scenery/`.

```bash
python scripts/eval_xgh.py --tag baris_penuh --every 6 --meters-per-part 8 --max-frames-per-part 8000   # HTML dengan pemandangan
python scripts/eval_xgh.py --tag baris_penuh --no-scenery ...                                           # tanpa pemandangan
python scripts/eval_xgh.py --tag baris_penuh --html-only --meters-per-part 8 --max-frames-per-part 8000 # buat ulang HTML dari
                                                                                                        # recording.npz (tanpa simulasi ulang)
python scripts/make_scenery_assets.py --panorama /path/foto_360.hdr     # ganti foto kubah (.hdr / .exr / .jpg / .png, rasio 2:1)
python scripts/make_scenery_assets.py --panorama foto.hdr --radius 50 --exposure 1.2   # jari-jari kubah, kecerahan foto
```
Pohon, tunggul dan gubuk adalah mesh milik pengguna yang sudah dikonversi (satu zip sumber = satu file glTF) di
`assets_xgh/scenery/models/`; daftarnya ada di `models.json` (nama, jenis `tree` / `stump` / `hut`, ukuran, zip asal).
Menghapus satu entri dari `models.json` mengeluarkan model itu dari pemandangan. Tanpa folder `models/` dipakai pohon dan
gubuk low-poly bawaan (`tree.gltf`, `hut.gltf`). Batu selalu `rock.gltf`.

Pengaturan opsional (tambahkan blok ini ke `configs/common.yaml`; tanpa blok ini nilai di bawah yang dipakai):
```yaml
scenery:
  enabled: true
  seed: 5                  # seed lain = susunan batu / pohon lain
  yaw_deg: 0.0             # memutar foto mengelilingi ladang
  n_rocks: 34
  rock_scale: [0.35, 1.6]  # rentang ukuran batu (x batu 1,1 m)
  n_trees: 22              # semua model berjenis "tree" dipakai bergiliran
  tree_scale: [0.8, 1.25]  # rentang ukuran pohon (x ukuran model, 6-9 m)
  n_stumps: 8              # tunggul (model berjenis "stump")
  stump_scale: [0.7, 1.2]
  margin_side: 1.6         # m dari baris tanaman terluar ke batu terdekat
  margin_ends: 4.0         # m bebas di depan / belakang ladang (jalur kendaraan)
  tree_margin: 4.5         # pohon lebih jauh sekian meter daripada batu
  hut: true
  hut_xy: null             # [x, y] gubuk; null = di samping tengah ladang, sisi +y
  hut_yaw_deg: null
```

## Instalasi
```bash
pip install -r requirements.txt
export MUJOCO_GL=egl                  # Linux tanpa layar
python scripts/make_xgh_assets.py --src /path/ke/mesh_pineapple   # membuat assets_xgh/ (~80 MB) dari folder zip mesh_pineapple Anda - wajib sekali
```
v6 diuji di Linux, Python 3.13, MuJoCo 4.0.1, Drake 1.58, SB3 2.9.2, 2 inti CPU tanpa GPU (v5: MuJoCo 3.17).

## Perintah
```bash
# Pretraining
python pretrain/make_t2_dataset.py --scenes 450 --seed0 0 --out outputs/pretrain/t2_data_0.npz
python pretrain/make_t2_dataset.py --scenes 450 --seed0 1 --out outputs/pretrain/t2_data_1.npz
python pretrain/train_t2_diffusion.py                                  # -> models/t2_diffusion.pt
python pretrain/stargaze_viewer.py --viewer meshcat                    # slider di browser (http://localhost:7000)
python pretrain/stargaze_viewer.py --viewer mujoco                     # MuJoCo viewer
python pretrain/stargaze_viewer.py --viewer demo --html outputs/pretrain/rig_demo.html

# Training (bobot berpindah: abc_mission3 -> M1 -> M2 -> M3), ekspor ONNX otomatis ke models/
python scripts/train_mission.py --mission mission1 --timesteps 60000 --n-envs 1
python scripts/train_mission.py --mission mission2 --timesteps 30000 --n-envs 1
python scripts/train_mission.py --mission mission3 --timesteps 12000 --n-envs 1
scripts/run_curriculum.sh                                              # semuanya berurutan

# Resume Training
python scripts/train_mission.py --mission mission1 --resume --timesteps 20000 --n-envs 4 #--timesteps adalah timesteps tambahan yang diinginkan

# Evaluasi realistis + HTML Meshcat (seed uji; --record N = N episode pertama direkam, --record-failures = episode gagal direkam)
python scripts/eval_realistic.py --mission mission1 --onnx models/mission1_v1.onnx --seeds 940000 30 --record 2 --tag onnx --record-failures
python scripts/eval_realistic.py --mission mission1 --seeds 940000 30 --record 0 --tag referensi          # controller referensi saja
python scripts/eval_realistic.py --mission mission2 --onnx models/mission2_v1.onnx --seeds 950000 20 --record 2 --tag onnx --record-failures
python scripts/eval_xgh.py --tag ref_baris_penuh --every 6 --meters-per-part 8 --max-frames-per-part 8000   # satu baris penuh 2 x 30 tanaman
python scripts/eval_xgh.py --onnx models/mission3_v1.onnx --tag baris_penuh_onnx --every 6 --meters-per-part 8 --max-frames-per-part 8000  # satu baris penuh 2 x 30 tanaman dengan SAC
python scripts/eval_xgh.py --onnx models/mission3_v1.onnx --section 0 20 --tag onnx_seksi                   # 20 tanaman pertama
python scripts/eval_xgh.py --field-seed 2027 --plants-per-row 10 --tag ladang_lain                          # ladang lain (tata letak baru)
python scripts/eval_xgh.py --edge-gap 0.80 --section 0 6 --tag gap80                                        # jarak tepi ladang 0,8 m
scripts/run_final_eval.sh                                              # evaluasi akhir lengkap

# Evaluasi cepat di adegan ringan / satu rekaman
python scripts/eval_mission.py --mission mission1 --seeds 900000 10 --onnx models/mission1_v1.onnx
python scripts/visualize_meshcat.py --mission mission2 --seed 910000 --html outputs/m2_light.html

# Tes
python -m pytest -q tests/test_core.py tests/test_mission3.py tests/test_v6.py
python -m pytest -q tests/test_missions_slow.py      # beberapa menit
```
Colab: `notebooks/colab_meshcat_eval.ipynb`.

## Membaca HTML Meshcat
- Panel **Kamera** (kiri bawah): `1` bebas (orbit), `2` pandangan atas, `3` **kamera gripper**; di Misi 3 juga ikuti
  kendaraan dan kamera pojok. Panel **Animations** (kanan atas, "Open Controls"): play/pause, geser waktu.
- Kamera gripper menampilkan bounding box buah (biru), window (putih), shoulder (titik magenta), kandidat T2 (cyan), T1
  (hijau), T2 (merah) dan kotak info keputusan; overlay tampil sejak keputusan dibuat sampai buah dicabut.
- Lampu status berkedip: kuning = observasi (mencari T1/T2), biru = aksi panen, hijau = sukses/standby, merah = gagal.

## Struktur
`configs/` (satu sumber kebenaran parameter), `pinesim/` (model, tanaman, persepsi, keputusan T2 + diffusion, rig,
perencana, env, RL, LOD, perekam Meshcat, vendor), `pretrain/`, `scripts/`, `tests/`, `notebooks/`,
`third_party/abc_mission3/` (bobot awal Misi 1), `docs/`.

## Transfer bobot
Semua misi memakai observasi 63 nilai dan aksi 6-D yang sama. Misi 1 mulai dari Misi 3 referensi (abc_SAC_xyzBOT):
input yang konstan di sumber di-nol-kan, normalisasi diisi ulang, output residual mulai ~0 (= controller referensi).
Misi 1 → 2 → 3: policy, critic dan normalisasi disalin utuh. `checkpoints/<misi>/init_info.json` mencatat detailnya.

## Catatan kejujuran
- Robot tidak diberi posisi tanaman: semua target berasal dari kamera (ground truth hanya untuk penilaian dan untuk
  label dataset diffusion).
- Buah lepas karena batang patah secara fisik (pegas tarik/puntir melewati ambang), tanpa teleport atau weld.
- Ukuran tanaman, kekakuan daun, kekuatan batang dan batas "daun tercabut" (1,15 rad) adalah tebakan.
- Training dilakukan dengan anggaran kecil di mesin 2 inti; angka di `docs/HASIL.md` adalah apa yang benar-benar terukur.
- v6: jumlah episode evaluasi kecil (30 / 20 / 60 buah) dan semuanya simulasi; batasan pengukuran ada di akhir
  `docs/HASIL.md`. Checkpoint v5 lama Anda dipindahkan ke `arsip_v5/` (tidak dihapus).
