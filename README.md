# Exercise Tracker with YOLO Pose Detection

Aplikasi ini mengintegrasikan deteksi pose menggunakan YOLO dengan antarmuka pengguna grafis (GUI) berbasis PyQt. Aplikasi ini dapat menghitung jumlah squat dan durasi plank secara real-time menggunakan kamera laptop.

## Fitur Utama
- **Deteksi Squat dan Plank** menggunakan model YOLO.
- **Real-time Camera Feed** dengan overlay label yang menunjukkan jumlah squat atau durasi plank.
- **Histori Aktivitas**: Menyimpan data nama, mode (squat/plank), jumlah squat, dan durasi plank untuk ditampilkan di tab "History".

## Prasyarat
Sebelum menjalankan aplikasi ini, pastikan Anda telah menginstal beberapa dependensi yang diperlukan.

### 1. **Instalasi Python**
Pastikan Python 3.7 atau yang lebih baru terinstal di sistem Anda. Anda dapat mengunduhnya di [https://www.python.org/downloads/](https://www.python.org/downloads/).

### 2. **Instalasi Dependensi**
Buka terminal (atau Command Prompt di Windows) dan jalankan perintah berikut untuk menginstal dependensi yang diperlukan.

```bash
pip install -r requirements.txt
