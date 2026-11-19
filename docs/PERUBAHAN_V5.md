# Perubahan v5 dan asumsi yang dipakai

## Pretraining
- Rig memakai **mesh realistis** (buah, daun per helai, mahkota) dari 5 jenis nanas; tanaman miring 0–15° ke segala
  arah horizontal, pusat buah 60–70 cm dari tanah.
- **Shoulder** = permukaan buah yang terlihat kamera dari ketinggian pusat buah sampai pangkal mahkota. Piksel buah yang
  tertutup daun tidak dihitung (segmentasi warna hanya melihat piksel kuning yang benar-benar tampak).
- **Window** = kotak pembatas shoulder (persentil 2–98 di gambar). T1 dan T2 wajib berada di dalam window
  (T1 diproyeksikan ke gambar dan harus jatuh di dalam window + 6 px).
- Banyak kandidat T2: window dibagi grid 8×8, tiap sel yang berisi ≥3 piksel shoulder memberi satu kandidat (median
  titik 3D sel itu). Untuk tiap kandidat dibuat beberapa T1 (elevasi −10 … 40°).
- **Pemilih T2 = diffusion policy** (`pinesim/decision/diffusion.py`): DDPM yang dilatih meniru kandidat terbaik
  menurut ground truth (jarak garis T1–T2 terhadap titik daun + 0,5 × jarak titik T2 ke daun). Saat dipakai, 24
  sampel diffusion "memilih" sel; sel yang mendapat ≥12 % suara diurutkan dengan skor kamera, sisanya menyusul.
  Planner memvalidasi kandidat sesuai urutan dan mengambil yang pertama yang layak.
- Cara kerja kamera tetap: tiang/lengan bergerak → komputasi dihentikan; diam → komputasi berjalan.

## Misi 1–3
- **Daun tersentuh tidak mengakhiri episode.** Sentuhan dan tekukan daun tetap dihukum (−5 per kejadian tekukan
  pangkal > 0,9 rad) dan dihindari (penjaga garis mundur lalu mencoba garis T1–T2 berikutnya bila tekukan mendekati
  batas). Episode/sesi berakhir bila **daun tercabut**.
- **Asumsi "daun tercabut"**: simulasi tidak memodelkan akar sobek, jadi daun dianggap tercabut bila sendi pangkalnya
  menekuk lebih dari **1,15 rad (66°)** (`limits.leaf_uproot_rad`). Angka ini perkiraan; bisa diubah di config.
- Sesi selesai sukses bila nanas masuk keranjang. Training di adegan ringan, evaluasi di adegan realistis.
- Pose default robot sesuai gambar (q = [0, −1,27, 2,76, 1,658, −1,571, 0]) di awal dan akhir tiap sesi; di
  kendaraan pan bahu 180° (lengan menghadap depan di atas kanopi). Gripper tertutup sejak awal sesi.
- **Ujung jari tertutup = titik acuan**: ujung jari ke T1, lalu ke T2. Di T2: buka, maju beberapa cm (sampai jari
  mengapit sumbu buah), cengkeram, puntir, cabut, taruh di keranjang (gerak ke keranjang dipercepat).
- Misi 2: tetangga pada jarak 42–55 cm (sedikit lebih lebar), buah terdekat diprioritaskan, konsep celah terluas tetap.
- Misi 3 = gabungan Misi 3 dan 4 lama di XGH505H. Ladang realistis 2 baris × 30 tanaman, **jarak 50 cm, zig-zag
  25 cm** (sebelumnya 30/15). Hanya **3 tanaman interaktif**:
  - awal: tanaman target + 1 terdekat;
  - setelah ≥1 buah masuk keranjang: tanaman target + tanaman yang baru dipanen + target berikutnya yang terdekat.
  Tanaman lain hanya mesh statis (tetap tampil utuh), sehingga simulasi ringan.
- Warna: bak kendaraan hijau, rangka perak, track hitam; lengan putih, servo gripper perak, jari perak + bantalan
  hitam; daun campuran hijau dan oranye, mahkota hijau tua, buah kuning.
- **Semua mesh utuh**: UR10e memakai obj asli tanpa decimation, kendaraan tanpa decimation, mesh nanas dibersihkan dari
  segitiga cacat (`summary.json` → `mesh_checks`: 0 masalah).

## Fitur
- Kamera gripper di HTML menampilkan bounding box (biru), window (putih), shoulder (titik magenta), kandidat T2 (cyan),
  T1 (hijau), T2 (merah) dan kotak info keputusan (pemilih, jumlah kandidat, persen buah terlihat, jarak garis-daun).
- Lampu berkedip 2 Hz di pojok kendaraan yang berseberangan dengan kamera kendaraan, dekat lengan (di platform: di
  tiang dekat dasar lengan): biru = aksi panen, kuning = observasi (mencari T1/T2), hijau = sukses/standby,
  merah = gagal (buah jatuh, daun tercabut, eksekusi gagal).

## Penyesuaian teknis yang perlu diketahui
- **Kamera pergelangan dipindah** ke samping jari (`pos_in_tool [0, 0.12, 0.12]`): jari U yang tertutup menghalangi
  pandangan kamera di posisi lama.
- **Keranjang diberi warna biru**: keranjang oranye terbaca sebagai daun oranye oleh segmentasi.
- Gripper U (11,65–16,65 cm) dipakai di semua misi (dulu hanya di Misi 4), dengan kedua jari dikopel.
- Kontak daun dibuat lebih lunak dan daun adegan ringan sedikit lebih pendek agar daun tidak memaksa jari terbuka saat
  menarik buah.
- Kendaraan hanya maju bila lengan di pose default (bila pose default tidak tercapai tanpa tabrakan: pose kerja).
- Dasar lengan dan link bahu dikecualikan dari deteksi kontak (bersentuhan secara geometris di mesh asli).
- **Safety shield** di fase yang dikendalikan policy (approach, garis, retreat, lift, transport): perintah sendi yang
  akan menabrak lengan sendiri, platform, kendaraan, keranjang atau lantai diganti aksi referensi; bila tetap
  menabrak, lengan menahan posisi; tertahan 50 langkah → kegagalan `route_blocked` (lunak di Misi 3: buah dilewati).
- Pemeriksa rute memakai **margin 1 cm** untuk lengan atas/lengan bawah vs pergelangan/gripper dan lengan vs
  platform/keranjang (servo tertinggal dari jalur yang direncanakan; tanpa margin terjadi tabrakan). Gerakan terskrip
  yang lambat boleh mencoba lagi tanpa margin bila tidak ada jalur lain.
- Gerakan kamera (scan/look) menjauhi buah lain **beserta mahkotanya** (mahkota tetangga tersenggol jari → batang patah).
- Tanaman ringan (training): tidak ada daun lebih tinggi dari **0,86 m** (setelah kemiringan), karena XGH505H melintas
  di atas baris dengan dasar lengan 0,91 m dari tanah (daun 1,02 m terjepit ke lengan saat kendaraan maju). Tanaman
  realistis sudah ≤ 0,82 m.
- Lampu status juga ada di platform Misi 1–2 (pojok atas tiang, dekat dasar lengan).
- Ladang training Misi 3 (adegan ringan) dipersingkat menjadi 2 × 2 tanaman agar satu episode ~10 menit di mesin 2 inti.
