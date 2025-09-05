# TA---Sentimen-analysis-and-Topic-Modelling-using-IndoBERT-and-BERTopic
Analisis sentimen &amp; topik komentar TikTok ibu hamil di Indonesia dengan IndoBERT, BERTopic, &amp; fuzzy match string. Fokus pada kesehatan mental ibu hamil melalui klasifikasi sentimen, ekstraksi topik, dan preprocessing teks.

# Analisis Sentimen & Topik Komentar TikTok Ibu Hamil

[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![Transformers](https://img.shields.io/badge/transformers-4.43.0-green.svg)](https://huggingface.co/transformers/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.x-red.svg)](https://pytorch.org/)

> Analisis sentimen & topik komentar TikTok ibu hamil di Indonesia dengan **IndoBERT**, **BERTopic**, dan **fuzzy match string**. Fokus pada kesehatan mental ibu hamil melalui klasifikasi sentimen, ekstraksi topik, dan pemetaan aspek.

---

## 📂 Struktur Repo

├── IndoBERT_Analisis Sentimen Mental Ibu Hamil dari data Tiktok.ipynb
├── BERTopic dan Fuzzy Match String.ipynb
└── README.md


---

## 📊 Dataset
- Sumber: Komentar TikTok publik (2022–2024), diperoleh via **Apify TikTok Comments Scraper**:contentReference[oaicite:0]{index=0}.
- Dataset berlabel: 4.000 komentar manual + 6.789 pseudo-label → total ±10.789 komentar:contentReference[oaicite:1]{index=1}.
- Label: `0 = Negatif`, `1 = Positif`.
- Tidak disertakan dalam repo karena alasan privasi.

---

## ⚙️ Dependensi
- Python 3.10 (Google Colab, GPU T4 disarankan):contentReference[oaicite:2]{index=2}
- Core libs:
  - `transformers==4.43.0`, `torch`, `sentencepiece`
  - `bertopic`, `hdbscan`, `umap-learn`, `gensim`, `sentence-transformers`
  - `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `wordcloud`
  - `rapidfuzz`
  - `tqdm`, `re`, `string`, `collections`

---

## 🚀 Cara Menjalankan
1. Clone repo dan buka di **Google Colab**.
2. Install dependensi:
   ```bash
   pip install transformers==4.43.0 torch bertopic hdbscan umap-learn gensim sentence-transformers rapidfuzz

Siapkan dataset CSV dengan kolom text, sentiment_label.

Jalankan notebook:

IndoBERT_Analisis Sentimen...ipynb → training & evaluasi model sentimen.

BERTopic dan Fuzzy Match String.ipynb → topik modeling + pemetaan aspek.

Hasil visualisasi (bar chart, wordcloud, intertopic map) otomatis tersimpan sebagai PNG.

🧠 Metodologi
1. Preprocessing

Penghapusan kolom tidak relevan, drop null, case folding, cleaning simbol.

Normalisasi slang dengan Kamus Alay (contoh: “promil” → “program hamil”).

Stopword removal & balancing class menggunakan class weight.

2. Klasifikasi Sentimen (IndoBERT)

Model: indobenchmark/indobert-base-p2.

Skema validasi: Stratified K-Fold (5).

Metrik utama: Macro-F1, akurasi.

Hasil: Macro-F1 ≈ 0.94 (terbaik di antara baseline).

3. Topic Modeling (BERTopic)

Embedding: firqaaa/indo-sentence-bert-base.

Dimensi UMAP: 5, clustering dengan HDBSCAN.

Evaluasi: coherence c_v.

Topik positif → dukungan sosial, coping.

Topik negatif → distress emosi, konflik pasangan.

4. Pemetaan Aspek (Fuzzy Match + Embedding)

Kombinasi cosine similarity (0.6) + fuzzy ratio (0.4).

Implementasi dengan rapidfuzz.partial_ratio + IndoBERT embedding.

Threshold = 0.35 → label aspek (dukungan sosial, coping, kesehatan, pasangan, spiritual).

📈 Hasil Utama

65% komentar → sentimen negatif; 35% positif.

5 fitur aplikasi direkomendasikan: coping interaktif, konten spiritual, forum peer-support, toolkit pasangan, micro-CBT.

Visualisasi:

Bar chart distribusi sentimen (±10.789 data).

WordCloud per aspek-sentimen.

Intertopic distance map.
