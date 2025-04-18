
# Exercise Tracker with YOLO Pose Detection

Aplikasi ini mengintegrasikan deteksi pose menggunakan **YOLO** dengan antarmuka pengguna grafis (GUI) berbasis **PyQt**. Aplikasi ini dapat menghitung jumlah squat dan durasi plank secara real-time menggunakan kamera laptop.

## Fitur Utama
- **Deteksi Squat dan Plank** menggunakan model YOLO.
- **Real-time Camera Feed** dengan overlay label yang menunjukkan jumlah squat atau durasi plank.
- **Histori Aktivitas**: Menyimpan data nama, mode (squat/plank), jumlah squat, dan durasi plank untuk ditampilkan di tab "History".

## Prasyarat
Sebelum menjalankan aplikasi ini, pastikan Anda telah menginstal beberapa dependensi yang diperlukan.

### 1. **Instalasi Python**
Pastikan Python 3.7 atau yang lebih baru terinstal di sistem Anda. Anda dapat mengunduhnya di [https://www.python.org/downloads/](https://www.python.org/downloads/).

### 2. **Instalasi Dependensi**
Buka terminal (atau Command Prompt di Windows) dan jalankan perintah berikut untuk menginstal dependensi yang diperlukan:

```bash
pip install -r requirements.txt
```

### 3. **Instalasi YOLO Model**
Pastikan model YOLO yang digunakan (misalnya `best_yolo8pose.pt`) sudah tersedia di folder `models/`.


---

## **Membangun Aplikasi (Build)**
Jika Anda ingin membangun aplikasi ini menjadi executable yang dapat dijalankan di sistem Anda, ikuti langkah-langkah berikut:

### 1. **Membangun untuk Windows (menggunakan PyInstaller)**
Untuk membangun aplikasi GUI menjadi file executable (`.exe`), Anda dapat menggunakan **PyInstaller**. Pastikan Anda sudah menginstal PyInstaller dengan menjalankan:

```bash
pip install pyinstaller
```

### 2. **Menyiapkan Build**
Jalankan perintah berikut untuk membangun aplikasi:

```bash
python -m PyInstaller --noconfirm --windowed --onefile --exclude-module Tkinter --exclude-module test src/main.py --add-data "models/best_yolo8pose.pt;models"

```

- **`--onefile`**: Membuat satu file executable.
- **`--windowed`**: Menjaga aplikasi tetap berjalan tanpa membuka jendela terminal (opsional, cocok untuk GUI).

Setelah build selesai, file executable akan tersedia di folder `dist/` dalam proyek Anda.

### 3. **Mengeksekusi Aplikasi**
Di dalam folder `dist/`, Anda akan menemukan file `main.exe` yang dapat dijalankan pada Windows.


---

## **Menjalankan Aplikasi**
Jika Anda hanya ingin menjalankan aplikasi tanpa melakukan build, cukup jalankan perintah berikut dari direktori proyek:

```bash
python main.py
```

Aplikasi akan membuka antarmuka pengguna dan menunggu input untuk mulai mendeteksi squat dan plank.


---

## **Struktur Proyek**
Berikut adalah struktur proyek aplikasi ini:

```
/GUI
│
├── /build/              # Folder sementara untuk build aplikasi
├── /dist/               # Folder hasil build (.exe)
├── /models/             # Folder tempat menyimpan model YOLO (.pt)
│   └── best_yolo8pose.pt
├── /src/                # Sumber kode aplikasi
│   └── main.py          # File utama untuk menjalankan aplikasi
│
├── requirements.txt     # Daftar dependensi
├── main.spec            # File konfigurasi PyInstaller
├── README.md            # Dokumen ini
└── .gitignore           # Mengabaikan file tertentu dari Git
```

---

## **Masalah Umum**
Jika Anda mengalami masalah saat menjalankan aplikasi, berikut beberapa solusi umum:
- Pastikan Python dan semua dependensi diinstal dengan benar.
- Jika terjadi error terkait model YOLO, pastikan model yang tepat telah diunduh dan disimpan di folder `models/`.


---

Jika Anda membutuhkan bantuan lebih lanjut atau ingin menyumbang pada proyek ini, buka masalah atau ajukan pull request di [GitHub repository](#).
