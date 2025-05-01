# SIBI Gesture Recognition

Sistem pengenalan gesture tangan bahasa isyarat Indonesia (SIBI) berbasis CNN yang dapat digunakan untuk klasifikasi real-time menggunakan webcam. Dataset mencakup huruf A–Y (kecuali J dan Z).

## Fitur
- Arsitektur CNN yang dioptimasi untuk klasifikasi gesture tangan
- Augmentasi data dan pelatihan menggunakan Keras
- Implementasi real-time menggunakan OpenCV dan MediaPipe
- Visualisasi metrik evaluasi dengan confusion matrix dan classification report

## Struktur Direktori
- `/train.py`: Training CNN model
- `/real_time.py`: Real-time inference with webcam
- `models/`: Model hasil pelatihan
- `dataset/`: Folder berisi data train/validation
-  `dataset/`:Download dataset di kaggle "https://www.kaggle.com/datasets/alvinbintang/sibi-dataset" dan bagi menjadi 2 folder:
A. Train (berisikan 80 % dari total gambar) 
B. Validation (Berisikan 20% dari total gambar)

## Cara Menjalankan
```bash
# Pelatihan model
python train.py

# Real-time inference
python real_time.py
