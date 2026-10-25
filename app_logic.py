"""Pengelolaan jawaban dan nilai sementara untuk tampilan Streamlit.

Modul ini hanya menggabungkan hasil penilaian. Rumus cosine dan skor
berbobot tetap berada di engine.py.
"""
from __future__ import annotations

import pandas as pd


def hasil_masih_valid(item: dict | None, jawaban_sekarang: str) -> bool:
    """Nilai hanya sah jika berasal dari teks jawaban yang belum berubah."""
    return bool(item and item.get("jawaban") == jawaban_sekarang)


def jumlah_terisi(bank: pd.DataFrame, answers: dict[str, str]) -> int:
    return sum(bool(str(answers.get(str(i), "")).strip()) for i in range(len(bank)))


def buat_rekap_lengkap(
    bank: pd.DataFrame, answers: dict[str, str], nilai_sesi: dict[str, dict]
) -> pd.DataFrame:
    """Soal yang belum diperiksa tetap terlihat, tanpa nilai buatan (NaN)."""
    rows = []
    for i, row in bank.reset_index(drop=True).iterrows():
        key = str(i)
        jawaban = str(answers.get(key, ""))
        saved = nilai_sesi.get(key)
        valid = hasil_masih_valid(saved, jawaban)
        result = saved["result"] if valid else None
        rows.append({
            "Kode Soal": str(row["Kode_Soal"]),
            "Pertanyaan": str(row["Pertanyaan"]),
            "Jawaban Siswa": jawaban,
            "Preprocessed": result["Preprocessed"] if result else "",
            "Kunci Guru": str(row["Kunci Guru"]),
            "Cosine Similarity": result["Cosine Similarity"] if result else float("nan"),
            "Cosine (%)": result["Cosine (%)"] if result else float("nan"),
            "Bobot": float(row["Bobot"]),
            "Nilai Prediksi": result["Nilai Prediksi"] if result else float("nan"),
            "Status": "Sudah diperiksa" if valid else ("Belum diperiksa" if jawaban.strip() else "Belum dijawab"),
        })
    return pd.DataFrame(rows)


def statistik_rekap(rekap: pd.DataFrame) -> dict:
    assessed = rekap["Status"].eq("Sudah diperiksa")
    return {
        "diperiksa": int(assessed.sum()),
        "dijawab": int(rekap["Jawaban Siswa"].astype(str).str.strip().ne("").sum()),
        "jumlah": len(rekap),
        "total": float(rekap["Nilai Prediksi"].sum()),
        "maksimum": float(rekap["Bobot"].sum()),
        "maksimum_diperiksa": float(rekap.loc[assessed, "Bobot"].sum()),
    }
