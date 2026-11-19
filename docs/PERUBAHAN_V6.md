# Perubahan v6

Tujuan v6 (permintaan pengguna): dari observasi sampai buah di keranjang rata-rata < 40 detik, < 40 detik per buah,
satu baris ladang selesai, tingkat keberhasilan panen > 85 %, gripper tidak boleh menabrak buah sampai jatuh, kamera
gripper dipakai untuk memastikan posisi buah pada pose ideal, dan baris ladang diam di kiri-kanan pada Misi 3.

Angka terukur ada di `docs/HASIL.md`. Dokumen ini menjelaskan APA yang diubah dan MENGAPA (setiap perubahan berasal
dari kegagalan yang terukur, bukan dari tebakan).

## 1. Dari mana waktu dan kegagalan v5 berasal (diukur sebelum mengubah apa pun)
| Pengukuran v5 (controller referensi, adegan realistis) | Hasil |
|---|---|
| Misi 1 | 8/12 episode, ±75 s per buah |
| Misi 2 | 2/6 episode |
| Misi 3 (evaluasi lama) | 37/60 buah, ±95 s per buah pada seksi yang baik |

Waktu habis di: survei 8 pandangan setiap buah, 3 pandangan dekat setiap buah, gerak antar-pose yang lambat dan
berhenti di setiap titik, jeda "settle" yang panjang, dan (Misi 3) survei ulang sebelum setiap buah. Kegagalan hampir
semuanya terjadi di bagian yang di-skrip (persepsi, perencanaan, cengkeraman, puntir, angkut), bukan di fase yang
dikendalikan policy RL. Karena itu perbaikan utama ada di pipeline, dan RL diperlakukan sebagai koreksi kecil.

## 2. Observasi lebih cepat
- **Survei berhenti lebih awal** (`planning.scan.early_stop`): begitu jumlah buah yang dibutuhkan terlihat dengan
  cukup titik, pandangan survei berikutnya dilewati (Misi 1: biasanya 1–2 dari 8 pandangan).
- **Satu pandangan dekat pada pose ideal**: gripper dibawa ke pose ideal (0,55 m dari buah, elevasi 30°) dan keputusan
  window/shoulder/T1/T2 dibuat dari pandangan itu. Pandangan tambahan hanya diambil bila keputusan atau rute gagal.
- **Kamera gripper di T1** (`harvest.t1_verify`): sesaat sebelum masuk garis T1→T2, kamera gripper mendeteksi ulang
  buah; T2, akhir maju dan titik cengkeram digeser (≤ 3 cm) dan rutenya divalidasi ulang.
- **Memori buah di Misi 3**: posisi buah disimpan di kerangka dunia (odometri kendaraan). Survei hanya dilakukan bila
  memori kosong atau buah yang diingat tidak ditemukan lagi, bukan sebelum setiap buah.
- **Hanya deteksi yang yakin menghentikan survei** (`planning.scan.stop_confidence: 0.85`, Misi 1–2): deteksi lemah
  (sedikit titik, radius di batas fit) ternyata ujung daun kuning (36–60 cm dari buah mana pun) atau buah di balik
  daun yang posisinya meleset 3–8 cm. Bila deteksi lebih banyak daripada jumlah buah yang ada, yang paling tidak yakin
  dibuang. Sebelumnya pandangan dekat menuju deteksi palsu menyenggol buah asli yang belum terlihat.
- Titik kerja Misi 3 dipindah ke 1,00 m di depan base (`reach_window.dx_target`): pada jarak lama pose pandang ideal
  tidak terjangkau tanpa menyentuh kendaraan, sehingga banyak pandangan dilewati dan buah pernah tersenggol.

## 3. Gerak lebih cepat tetapi aman untuk buah
- **Lintasan sendi trapesium berwaktu** (`_joint_trajectory`): percepatan/perlambatan halus, tidak berhenti di setiap
  titik antara, melambat di tikungan.
- **Pembatas kecepatan dekat buah** (`policy.fruit_governor`): kecepatan tangan dibatasi berdasarkan jarak ke buah yang
  TERLIHAT kamera (0,06 m/s pada 2 cm, 0,60 m/s di atas 25 cm). Berlaku untuk aksi referensi maupun aksi gabungan
  dengan residual RL, dan untuk gerak sendi (`planning.joint_speed_near_fruit`).
- **Pemeriksaan jalur terhadap buah**: setiap gerak antar-pose diperiksa terhadap semua buah yang terlihat, termasuk
  mahkota buah target sendiri; putaran puntir diperiksa terhadap buah tetangga (`harvest.twist_fruit_margin`).
- **Batang tanaman sebagai rintangan**: batang di bawah buah lain dan batang kosong bekas buah yang sudah dipanen
  dijauhi saat puntir (pergelangan pernah menekan batang tetangga 530 N), begitu juga batang target oleh pergelangan
  dan rumah gripper (`harvest.twist_stalk_margin`).
- **Penjaga daun pada gerak pandang** (`limits.move_guard_frac`): daun yang belum terlihat kamera dan hampir tercabut
  saat lengan menuju pose pandang → lengan berhenti, kembali lewat jalur yang sama, pandangan itu dilewati.
- **Maju dengan penjaga gaya** (`_open_and_advance`): maju 2,5 mm per langkah; bila rahang menyentuh buah dengan gaya
  > 12 N, tangan mundur, geser ke samping, lalu maju lagi (maks. 3 koreksi).
- Servo lebih responsif (`control.servo_rate: 10`), jeda settle dipendekkan, kecepatan kendaraan 0,45 m/s.

## 4. Cengkeram, puntir, cabut
- **Cengkeram di bagian buah terlebar** + **cengkeram ulang taktil**: posisi kontak pad dibandingkan dengan garis
  tengah pad; bila meleset > 1 cm tangan dilepas sedikit, digeser, dicengkeram lagi (maks. 2 kali). Sebelumnya buah
  dicengkeram di bahu (di atas titik berat) dan lepas seperti bandul saat diangkat.
- Cengkeram ulang **harus konvergen**: bila setelah digeser kontak pad tidak mendekati garis tengah, tangan kembali ke
  tinggi cengkeram rencana (jari berengsel pada buah besar/miring menyentuh dekat ujung pad; koreksi yang salah
  meninggalkan buah 3,8 cm dari tengah pad dan buah lepas saat batang patah). Cengkeram ulang juga dilakukan setelah
  puntir yang gagal, sebelum tarik aksial.
- **Puntir dua arah**: arah terbaik dipilih dari IK + kontak + jarak ke buah tetangga; bila berhenti (batas sendi, daun)
  arah lain dicoba. Setelah batang patah: diam 5 langkah, angkat 6 cm, lalu puntiran dikembalikan dengan halus.
- **Tarik aksial** (`harvest.pull_up`) bila puntir tidak mematahkan batang atau tidak mungkin dilakukan.
- **Angkut**: tinggi angkat tambahan yang tidak dipaksa ke tinggi keranjang (`lift_heights_low`), alasan penolakan
  rute dicatat, penjaga daun saat menarik keluar (berhenti, rencana ulang dengan arah lain).
- **Pola peletakan di keranjang** (`basket.drop_pattern`): buah dilepas di tempat berbeda di lantai keranjang. Dulu
  semua buah dilepas di tengah, sehingga buah ke-5 berdiri di atas buah sebelumnya, di atas bibir keranjang
  (`basket_miss`, terukur tepat pada buah ke-5 dan ke-8 setiap keranjang).

## 5. Daun dan pemulihan
- **Penjaga daun prediktif**: tekukan pangkal daun + laju tekukan × 5 langkah dibandingkan dengan batas. Dipakai saat
  maju, puntir, tarik keluar dan angkut. Bila terpicu di garis: mundur ke T1, ambil pandangan kamera baru, rencana baru.
- **Shield**: gerak yang tidak mengurangi jarak aman tetap diizinkan di dalam margin (dulu lengan membeku 9 mm dari
  braket lampu kendaraan); tertahan 10 langkah → rencana ulang.
- **Pemulihan setelah melewati buah (Misi 3)**: mundur terperiksa, pulang ke pose default, lanjut ke buah berikut;
  satu buah gagal tidak lagi mengakhiri baris.
- Semua buah yang masih di tanaman (termasuk yang dilewati) tetap diingat sebagai rintangan.

## 6. Baris ladang di kiri dan kanan (Misi 3)
`field.side_rows` di `configs/common.yaml`:
```yaml
field:
  side_rows:
    enabled: true
    edge_gap: 0.50        # jarak antar TEPI ladang (m) - ubah di sini
    beds_per_side: 1
    extra_plants: 1
```
Baris tetangga adalah mesh diam: tanpa sendi, tanpa kontak (contype = conaffinity = 0), tidak masuk persepsi sebagai
target. Jarak dapat juga diberikan saat evaluasi: `python scripts/eval_xgh.py --edge-gap 0.8`.

## 7. Waktu yang dilaporkan
`pinesim/timing.py`. Per buah: `observe_to_basket = (t_placed − t_start) + compute_s`, yaitu waktu simulasi dari awal
observasi buah itu sampai buah terverifikasi di keranjang, DITAMBAH waktu komputasi nyata (persepsi, keputusan,
perencanaan) yang diukur dengan jam dinding di mesin 2 inti ini. `s_per_fruit = (waktu simulasi seluruh episode +
komputasi) / buah di keranjang` memasukkan juga perjalanan kendaraan, pulang ke pose default dan waktu yang terbuang
pada buah yang gagal.

Komputasi Misi 3 turun dari ±32 s menjadi ±3 s per buah: uji kontak dengan `mj_kinematics + mj_collision`, cache
pemeriksaan segmen, kandidat gerak bertahap, titik daun dibatasi, voxel/klaster yang lebih cepat.

## 8. RL (SAC residual)
- Reward: penalti gagal seragam sekali per kegagalan, penalti jarak kecil, bonus sukses, penalti shield.
- Evaluasi periodik memakai seed VALIDASI tetap dan, sebelum training, menjalankan **controller referensi saja**
  (`eval_log.csv` baris `reference`, `baseline_reference.json`). Policy residual hanya berguna bila melampaui baris itu.
- **Watchdog residual**: bila tidak ada kemajuan 60 langkah atau shield terpicu 3 kali dalam satu fase, residual
  dimatikan untuk fase itu (kembali ke referensi).
- **Residual hanya di fase approach/line** (`policy.residual_phases`): dengan buah di tangan setiap penyimpangan dari
  rute hanya merugikan (terukur: daun tercabut saat menarik keluar, lengan diarahkan ke samping lengan bawahnya saat
  angkut sampai shield memblokir).
- **Mundur ke awal fase**: bila shield tetap terblokir setelah residual menggerakkan lengan, lengan kembali lewat
  postur yang sudah dilaluinya ke awal fase dan buah itu diselesaikan controller referensi saja.
- Checkpoint yang diekspor ke ONNX adalah yang terbaik pada seed validasi, bukan yang terakhir.
- Hasil jujurnya ada di `docs/HASIL.md`: residual tidak melampaui controller referensi.

## 9. Asumsi dan batasan
- Penjaga daun memakai tekukan pangkal daun dari simulator (sama seperti penjaga garis v5); di robot nyata ini perlu
  sensor gaya/torsi atau estimasi dari kamera.
- Cengkeram ulang memakai posisi kontak pad (diasumsikan tersedia dari pad taktil).
- Diuji dengan MuJoCo 4.0.1 (proyek asli: 3.17).
- Waktu komputasi diukur di mesin 2 inti tanpa GPU; di komputer lain angkanya berbeda.
- Kekakuan daun, kekuatan batang dan batas "daun tercabut" tetap tebakan seperti di v5.
