"""Mesin penilaian yang mereplikasi Tahap 2, 3, 5, dan 6 dari riset_2.py.

Model untuk cosine: encoder BERT hasil fine-tuning Tahap 4, BUKAN regressor.
Kunci Guru di Tahap 5 digunakan sebagai teks mentah; hanya Jawaban Siswa
menggunakan preprocessing Tahap 2. Jangan mengubah salah satu diam-diam.
"""
from __future__ import annotations

import io
import math
import os
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

MODEL_NAME = "indobenchmark/indobert-base-p1"
MAX_LENGTH = 256
BATCH_SIZE = 8
KAMUS_URL = "https://raw.githubusercontent.com/evrintobing17/NormalisasiKata/master/kamus_alay.csv"
KBBA_URL = "https://raw.githubusercontent.com/evrintobing17/NormalisasiKata/master/kbba.txt"
REQUIRED_BANK = ["Kode_Soal", "Pertanyaan", "Kunci Guru", "Bobot"]
BOBOT_VALID = (0.5, 1.0, 2.0)


class KonfigurasiError(RuntimeError):
    """Informasi yang harus diperbaiki pemilik deployment."""


def tambah_kamus(kamus: dict, kata: object, baku: object) -> None:
    # Sesuai riset_2.py: str(...).strip().lower(), isalpha() keduanya.
    if pd.isna(kata) or pd.isna(baku):
        return
    kata, baku = str(kata).strip().lower(), str(baku).strip().lower()
    if kata.isalpha() and baku.isalpha():
        kamus[kata] = baku


def buat_kamus_dari_teks(alay_text: str, kbba_text: str) -> dict[str, str]:
    kamus: dict[str, str] = {}
    alay = pd.read_csv(io.StringIO(alay_text), header=None)
    if alay.shape[1] < 2:
        raise KonfigurasiError("Format kamus_alay.csv berubah (kolom tidak lengkap).")
    for _, row in alay.iloc[:, :2].iterrows():
        tambah_kamus(kamus, row.iloc[0], row.iloc[1])
    for line in kbba_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if "\t" in line:
            parts = line.split("\t", 1)
        elif ":" in line:
            parts = line.split(":", 1)
        else:
            continue
        tambah_kamus(kamus, parts[0], parts[1])
    if not kamus:
        raise KonfigurasiError("Kamus normalisasi kosong. Periksa file GitHub.")
    return kamus


def ambil_kamus() -> dict[str, str]:
    """Unduh kamus sama dengan eksperimen, atau gunakan salinan lokal yang sama."""
    local_alay = Path("data/kamus_alay.csv")
    local_kbba = Path("data/kbba.txt")
    if local_alay.exists() and local_kbba.exists():
        alay_text = local_alay.read_bytes().decode("latin-1")
        kbba_text = local_kbba.read_text(encoding="utf-8")
    else:
        try:
            response_alay = requests.get(KAMUS_URL, timeout=30)
            response_alay.raise_for_status()
            response_kbba = requests.get(KBBA_URL, timeout=30)
            response_kbba.raise_for_status()
            alay_text = response_alay.content.decode("latin-1")
            kbba_text = response_kbba.text
        except requests.RequestException as err:
            raise KonfigurasiError(
                "Kamus GitHub tidak bisa diakses. Simpan kamus_alay.csv dan kbba.txt "
                "dari repo NormalisasiKata ke folder data/ aplikasi."
            ) from err
    return buat_kamus_dari_teks(alay_text, kbba_text)


def preprocess_tanpa_regex(jawaban: object, kamus: dict[str, str]) -> dict[str, str]:
    """Sesuai persis fungsi case_folding, cleaning, normalisasi Tahap 2 riset_2."""
    asli = "" if pd.isna(jawaban) else str(jawaban)
    case_folding = asli.lower()
    cleaning = case_folding.replace("http://", " ").replace("https://", " ")
    cleaning = "".join(c if (c.isalnum() or c.isspace()) else " " for c in cleaning)
    cleaning = " ".join(cleaning.split()).strip()
    normalisasi = " ".join(kamus.get(kata, kata) for kata in cleaning.split())
    return {
        "Asli": asli,
        "Case Folding": case_folding,
        "Cleaning": cleaning,
        "Normalisasi": normalisasi,
        "Preprocessed": normalisasi,
    }


def validasi_bank_soal(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        raise ValueError("Bank soal kosong. Isi minimal satu baris soal.")
    missing = [k for k in REQUIRED_BANK if k not in data.columns]
    if missing:
        raise ValueError("Kolom bank soal belum lengkap: " + ", ".join(missing))
    bank = data[REQUIRED_BANK].copy()
    for col in ("Kode_Soal", "Pertanyaan", "Kunci Guru"):
        if bank[col].isna().any() or bank[col].astype(str).str.strip().eq("").any():
            raise ValueError(f"Kolom '{col}' tidak boleh kosong.")
        bank[col] = bank[col].astype(str).str.strip()
    bank["Bobot"] = pd.to_numeric(bank["Bobot"], errors="coerce")
    if bank["Bobot"].isna().any() or (~bank["Bobot"].isin(BOBOT_VALID)).any():
        raise ValueError("Bobot harus salah satu dari 0.5, 1.0, atau 2.0.")
    if bank["Kode_Soal"].duplicated().any():
        raise ValueError("Kode_Soal harus unik untuk setiap pertanyaan.")
    if len(bank) > 60:
        raise ValueError("Maksimal 60 soal per sesi untuk mencegah beban server berlebihan.")
    return bank.reset_index(drop=True)


def ambil_soal_dari_dataset_asli(data: pd.DataFrame) -> pd.DataFrame:
    """Dataset Biologi: otomatis 1.258 baris jawaban -> 37 soal unik.

    Yang dipakai hanya Kode_Soal, Pertanyaan, Kunci Guru, Bobot.
    Nama, Absen, Kelas, Jawaban Siswa, dan Nilai siswa lama TIDAK disalin.
    Tidak mengubah dataset asli dan menolak kunci/bobot yang inkonsisten.
    """
    missing = [kol for kol in REQUIRED_BANK if kol not in data.columns]
    if missing:
        raise ValueError("Kolom dataset tidak lengkap: " + ", ".join(missing))
    soal = data[REQUIRED_BANK].copy()
    if soal.empty:
        raise ValueError("Dataset Biologi kosong.")

    def normalisasi_kode(kode):
        if pd.isna(kode):
            return ""
        if isinstance(kode, (int, float, np.integer, np.floating)) and float(kode).is_integer():
            return str(int(kode))
        return str(kode).strip()

    soal["Kode_Soal"] = soal["Kode_Soal"].map(normalisasi_kode)
    for kol in ("Pertanyaan", "Kunci Guru"):
        soal[kol] = soal[kol].fillna("").astype(str).str.strip()
    soal["Bobot"] = pd.to_numeric(soal["Bobot"], errors="coerce")

    for kol in REQUIRED_BANK:
        if soal[kol].isna().any() or (kol != "Bobot" and soal[kol].astype(str).str.strip().eq("").any()):
            raise ValueError(f"Dataset memiliki nilai kosong pada '{kol}'.")
    if not soal["Bobot"].isin(BOBOT_VALID).all():
        raise ValueError("Dataset memiliki Bobot di luar 0.5, 1.0, atau 2.0.")

    # Satu kode soal harus menunjuk ke pertanyaan, kunci, dan bobot yang sama.
    untuk_cek = soal.groupby("Kode_Soal", sort=False)[["Pertanyaan", "Kunci Guru", "Bobot"]].nunique(dropna=False)
    bermasalah = untuk_cek[(untuk_cek > 1).any(axis=1)].index.tolist()
    if bermasalah:
        raise ValueError("Data soal tidak konsisten pada Kode_Soal: " + ", ".join(bermasalah[:10]))

    bank = soal.drop_duplicates(subset="Kode_Soal", keep="first")
    return validasi_bank_soal(bank)


def baca_bank_soal(file_or_bytes: str | Path | bytes | io.BytesIO) -> pd.DataFrame:
    """Terima dataset Biologi asli, bank soal 4 kolom, atau CSV.

    Dataset asli dibaca dari sheet 'Biologi' jika tersedia; kolom identitas
    siswa lama tidak pernah dikembalikan ke aplikasi.
    """
    if isinstance(file_or_bytes, bytes):
        file_or_bytes = io.BytesIO(file_or_bytes) if file_or_bytes[:2] == b"PK" else io.StringIO(file_or_bytes.decode("utf-8-sig"))
    source_name = str(file_or_bytes).lower() if isinstance(file_or_bytes, (str, Path)) else ""
    if source_name.endswith(".json"):
        data = pd.read_json(file_or_bytes, orient="records")
    elif source_name.endswith(".csv") or isinstance(file_or_bytes, io.StringIO):
        data = pd.read_csv(file_or_bytes)
    else:
        with pd.ExcelFile(file_or_bytes, engine="openpyxl") as excel:
            sheet = "Biologi" if "Biologi" in excel.sheet_names else excel.sheet_names[0]
            data = pd.read_excel(excel, sheet_name=sheet)
    # File dataset asli berisi jawaban banyak siswa untuk setiap soal.
    if ("Jawaban Siswa" in data.columns or "Absen" in data.columns or "Nama" in data.columns):
        return ambil_soal_dari_dataset_asli(data)
    return validasi_bank_soal(data)


def nilai_dari_cosine(cosine: float, bobot: float) -> float:
    """Tahap 6: np.clip(Cosine Similarity * Bobot, 0, Bobot)."""
    c, b = float(cosine), float(bobot)
    if not np.isfinite(c) or not np.isfinite(b) or b <= 0:
        raise ValueError("Cosine atau Bobot tidak valid.")
    return float(np.clip(c * b, 0, b))


def dapatkan_checkpoint_path(
    repo: str = "", token: str = "", filename: str = "model.pth"
) -> Path:
    """Cari checkpoint lokal; jika tidak ada, coba repo HF yang disediakan pemilik."""
    for name in ("models/model.pth", "IndoBERT_Essay_Scoring/model.pth"):
        candidate = Path(name)
        if candidate.is_file():
            return candidate
    if repo:
        from huggingface_hub import hf_hub_download
        try:
            return Path(hf_hub_download(repo_id=repo, filename=filename, token=token or None))
        except Exception as exc:
            raise KonfigurasiError(
                f"Checkpoint tidak bisa diambil dari repositori '{repo}': {exc}"
            ) from exc
    raise KonfigurasiError(
        "File fine-tuning model.pth belum ada. Unduh dari Colab Tahap 4 "
        "dan simpan di models/model.pth, atau atur HF_MODEL_REPO di Streamlit secrets. "
        "Aplikasi tidak akan memberi skor dari bobot model yang belum dilatih."
    )


class MesinPenilai:
    def __init__(self, checkpoint: Path):
        # Impor lazy supaya layar setup tetap dapat dibuka tanpa model.
        import torch
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        # Semua parameter encoder ditimpa checkpoint strict=True.
        # from_config menghindari unduhan bobot IndoBERT dasar (~500 MB) yang tak diperlukan.
        self.model = AutoModel.from_config(AutoConfig.from_pretrained(MODEL_NAME))
        try:
            state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
        except Exception as exc:
            raise KonfigurasiError(f"Checkpoint tidak dapat dibaca: {exc}") from exc
        if not isinstance(state, dict):
            raise KonfigurasiError("Isi checkpoint harus state_dict dari Tahap 4.")
        bert_state = {key.removeprefix("bert."): value
                      for key, value in state.items() if key.startswith("bert.")}
        if not bert_state:
            raise KonfigurasiError("Checkpoint tidak memiliki key 'bert.' dari riset_2.")
        try:
            self.model.load_state_dict(bert_state, strict=True)
        except Exception as exc:
            raise KonfigurasiError(f"Arsitektur checkpoint tidak cocok: {exc}") from exc
        del state, bert_state
        self.model = self.model.to(self.device).eval()

    def embed(self, teks: list[str]) -> np.ndarray:
        semua = []
        for start in range(0, len(teks), BATCH_SIZE):
            batch = teks[start : start + BATCH_SIZE]
            tokens = self.tokenizer(
                batch,
                padding="max_length", truncation=True, max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            with self.torch.no_grad():
                output = self.model(
                    input_ids=tokens["input_ids"].to(self.device),
                    attention_mask=tokens["attention_mask"].to(self.device),
                )
            semua.append(output.last_hidden_state[:, 0, :].cpu().numpy())
        return np.vstack(semua)

    def cosine(self, teks_siswa: list[str], kunci_asli: list[str]) -> np.ndarray:
        # Tahap 5: kedua teks dipisah, embedding CLS, cosine per pasangan.
        emb_siswa = self.embed(teks_siswa)
        emb_kunci = self.embed(kunci_asli)
        similarities = []
        for a, b in zip(emb_siswa, emb_kunci):
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            similarity = 0.0 if (na == 0 or nb == 0) else float(np.dot(a, b) / (na * nb))
            similarities.append(similarity)
        return np.array(similarities, dtype=float)


def penilaian(
    bank: pd.DataFrame,
    jawaban: list[str],
    kamus: dict[str, str],
    model: MesinPenilai,
) -> tuple[pd.DataFrame, float]:
    if len(bank) != len(jawaban):
        raise ValueError("Jumlah jawaban tidak sama dengan jumlah soal.")
    mulai = time.perf_counter()
    hasil_preproses = [preprocess_tanpa_regex(s, kamus) for s in jawaban]
    student_text = [row["Preprocessed"] for row in hasil_preproses]
    kunci_raw = bank["Kunci Guru"].fillna("").astype(str).tolist()
    cosine = model.cosine(student_text, kunci_raw)
    if len(cosine) != len(bank):
        raise RuntimeError("Jumlah hasil cosine tidak cocok dengan jumlah soal.")

    output = []
    for i, row in bank.reset_index(drop=True).iterrows():
        sim = float(cosine[i])
        # Aturan UI saja: jawaban yang betul-betul kosong mendapat nol.
        # Tahap 6 historis tidak menguji input kosong pada aplikasi publik.
        kosong = not str(jawaban[i]).strip()
        nilai = 0.0 if kosong else nilai_dari_cosine(sim, float(row["Bobot"]))
        output.append({
            "Kode Soal": row["Kode_Soal"],
            "Pertanyaan": row["Pertanyaan"],
            "Jawaban Siswa": str(jawaban[i]),
            "Preprocessed": student_text[i],
            "Kunci Guru": str(row["Kunci Guru"]),
            "Cosine Similarity": sim,
            "Cosine (%)": float(np.clip(sim * 100, -100, 100)),
            "Bobot": float(row["Bobot"]),
            "Nilai Prediksi": nilai,
        })
    elapsed = time.perf_counter() - mulai
    return pd.DataFrame(output), elapsed


def ringkasan_nilai(hasil: pd.DataFrame) -> dict:
    total = float(hasil["Nilai Prediksi"].sum())
    maksimum = float(hasil["Bobot"].sum())
    return {
        "total": total,
        "maksimum": maksimum,
        "persentase": (total / maksimum * 100.0) if maksimum else 0.0,
        "rerata_cosine": float(hasil["Cosine Similarity"].mean()),
        "terjawab": int(hasil["Jawaban Siswa"].astype(str).str.strip().ne("").sum()),
        "jumlah": int(len(hasil)),
    }


def sanitasi_excel(data: pd.DataFrame) -> pd.DataFrame:
    """Hindari formula injection dari isian siswa dalam file ekspor."""
    clone = data.copy()
    for col in clone.select_dtypes(include="object").columns:
        clone[col] = clone[col].map(
            lambda value: "'" + value if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")) else value
        )
    return clone


def buat_excel_hasil(identitas: dict, hasil: pd.DataFrame, durasi: float, show_answer_key: bool = False) -> bytes:
    """Unduhan Excel dengan dua sheet (ringkasan dan rincian)."""
    summary = ringkasan_nilai(hasil)
    info = pd.DataFrame([
        {"Keterangan": "Nama siswa", "Nilai": identitas["nama"]},
        {"Keterangan": "Nomor absen", "Nilai": identitas["absen"]},
        {"Keterangan": "Kelas", "Nilai": identitas["kelas"]},
        {"Keterangan": "Metode", "Nilai": "IndoBERT dan Cosine Similarity"},
        {"Keterangan": "Rumus", "Nilai": "clip(Cosine Similarity × Bobot, 0, Bobot)"},
        {"Keterangan": "Total nilai", "Nilai": round(summary["total"], 6)},
        {"Keterangan": "Skor maksimal", "Nilai": round(summary["maksimum"], 6)},
        {"Keterangan": "Nilai persentase", "Nilai": round(summary["persentase"], 4)},
        {"Keterangan": "Jumlah dijawab", "Nilai": summary["terjawab"]},
        {"Keterangan": "Jumlah soal", "Nilai": summary["jumlah"]},
        {"Keterangan": "Jumlah diperiksa", "Nilai": int(hasil["Status"].eq("Sudah diperiksa").sum()) if "Status" in hasil.columns else summary["jumlah"]},
        {"Keterangan": "Waktu inferensi (detik)", "Nilai": round(durasi, 4)},
    ])
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        sanitasi_excel(info).to_excel(writer, sheet_name="Ringkasan", index=False)
        export_detail = hasil if show_answer_key else hasil.drop(columns=["Kunci Guru"], errors="ignore")
        sanitasi_excel(export_detail).to_excel(writer, sheet_name="Rincian Nilai", index=False)
        for sheet in writer.book.worksheets:
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
            for cell in sheet[1]:
                cell.font = __import__("openpyxl").styles.Font(color="FFFFFF", bold=True)
                cell.fill = __import__("openpyxl").styles.PatternFill("solid", fgColor="876BAA")
            for column in sheet.columns:
                letter = column[0].column_letter
                max_width = min(max(len(str(cell.value or "")) for cell in column) + 3, 46)
                sheet.column_dimensions[letter].width = max(14, max_width)
    return buffer.getvalue()
