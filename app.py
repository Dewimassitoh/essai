"""Esai Cerdas — antarmuka siswa sederhana dengan penilaian IndoBERT."""
from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

import pandas as pd
import streamlit as st

from app_logic import buat_rekap_lengkap, jumlah_terisi, statistik_rekap, hasil_masih_valid
from engine import (
    KonfigurasiError, MesinPenilai, ambil_kamus, baca_bank_soal,
    buat_excel_hasil, dapatkan_checkpoint_path, penilaian,
)

st.set_page_config(
    page_title="Esai Cerdas — Penilaian Esai",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
:root { color-scheme:light; }
html, body, [class*="css"] {font-family:'DM Sans',sans-serif;}
[data-testid="stAppViewContainer"] {background:#fcfaf8;color:#393348;}
[data-testid="stHeader"] {background:transparent;}
[data-testid="stSidebar"] {background:#f7f2f5;border-right:1px solid #eee6ef;}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {color:#756b7e;}
[data-testid="stSidebar"] [data-testid="stSidebarContent"] {padding-top:2.1rem;}
.block-container {max-width:1020px;padding:2.0rem 2.0rem 4.5rem;}
h1,h2,h3 {font-family:'Plus Jakarta Sans',sans-serif!important;letter-spacing:-.045em;color:#352d42!important;}
h1 {font-size:2.65rem!important;line-height:1.23!important;}
h2 {font-size:1.50rem!important;}
h3 {font-size:1.16rem!important;}
[data-testid="stCaptionContainer"], small {color:#8a8090;}
.stButton > button {border-radius:13px;min-height:2.65rem;box-shadow:none;transition:background .15s ease;}
.stButton > button[kind="primary"], [data-testid="stFormSubmitButton"] button[kind="primary"] {background:#896caf!important;border:1px solid #896caf!important;color:#fff!important;font-weight:700!important;box-shadow:0 6px 18px #967bb632!important;}
.stButton > button[kind="primary"]:hover {background:#745696!important;border-color:#745696!important;}
.stButton > button[kind="secondary"] {background:#fff;border:1px solid #e4dbe8;color:#574663;}
[data-testid="stDownloadButton"] button {border-radius:12px;background:#f3edf8;color:#634c7e;border:1px solid #e4dbea;font-weight:700;}
[data-testid="stTextInput"] input,[data-testid="stTextArea"] textarea {background:#fff!important;border:1px solid #e3dce7!important;color:#3c3448!important;border-radius:12px!important;box-shadow:none!important;}
[data-testid="stTextInput"] input:focus,[data-testid="stTextArea"] textarea:focus {border-color:#9b81bb!important;box-shadow:0 0 0 2px #b69ac733!important;}
[data-testid="stSelectbox"] [data-baseweb="select"] {border-radius:12px;}
[data-testid="stMetric"] {background:#fff;border:1px solid #eee6ed;border-radius:16px;padding:14px 17px;min-height:108px;}
[data-testid="stMetricValue"] {color:#59436f;font-size:1.55rem;font-weight:800;}
[data-testid="stAlert"] {border-radius:14px;}
[data-testid="stProgress"] > div > div > div > div {background:#9476b3!important;}
.stTabs [data-baseweb="tab-list"] {gap:12px;border-bottom:1px solid #eee6ed;}
.stTabs [data-baseweb="tab"] {font-weight:700;color:#817487;padding:12px 14px;background:transparent;}
.stTabs [aria-selected="true"] {color:#8060a3!important;}
.stTabs [data-baseweb="tab-highlight"] {background:#9576b2!important;}
hr {border-color:#eee8ee!important;}
.hero {background:linear-gradient(117deg,#f1eaf7 0%,#fbf5f5 63%,#fcf6ef 100%);border:1px solid #eee3ee;border-radius:24px;padding:37px 40px;margin:0 0 26px;}
.hero-kicker {font-size:.76rem;color:#9575ac;font-weight:800;letter-spacing:.12em;text-transform:uppercase;margin:0 0 12px;}
.hero h1 {margin:0 0 13px;font-size:2.65rem;}
.hero p {color:#72687b;line-height:1.65;max-width:640px;margin:0;font-size:1rem;}
.soft-pill {display:inline-block;margin-top:19px;padding:7px 14px;border-radius:999px;border:1px solid #dfcfe7;background:#ffffff99;color:#735486;font-size:.8rem;font-weight:700;}
.side-brand {display:flex;align-items:center;gap:10px;margin:5px 0 5px;}
.side-logo {background:#e5d9ed;color:#785898;padding:8px 12px;border-radius:13px;font-size:1.25rem;font-weight:800;}
.side-title {font-family:'Plus Jakarta Sans',sans-serif;color:#463551;font-weight:800;font-size:1.32rem;letter-spacing:-.04em;}
.side-sub {color:#948698;font-size:.84rem;margin-left:2px;margin-bottom:22px;}
.side-line {height:1px;background:#e9e0e9;margin:19px 0;}
.side-meta {color:#84798c;font-size:.82rem;line-height:1.8;}
.section-note {font-size:.77rem;font-weight:800;letter-spacing:.12em;text-transform:uppercase;color:#a18baa;margin:20px 0 9px;}
.paper-card {background:#fff;border:1px solid #ede6ec;border-radius:18px;padding:26px 26px 18px;box-shadow:0 8px 25px #6f58710c;margin:8px 0 15px;}
.question-label {font-size:.75rem;font-weight:800;letter-spacing:.12em;text-transform:uppercase;color:#9c7cb8;margin-bottom:11px;}
.question-title {font-family:'Plus Jakarta Sans',sans-serif;font-size:1.28rem;line-height:1.62;color:#383040;font-weight:700;margin:3px 0 13px;}
.muted {font-size:.9rem;color:#8a7c8a;line-height:1.6;}
.score-result {background:#f5effa;border:1px solid #e4d4ee;padding:18px 21px;border-radius:15px;margin:11px 0;}
.score-result strong {color:#5f447c;}
.score-top {font-size:.84rem;font-weight:700;color:#8a699c;letter-spacing:.035em;}
.score-number {font-family:'Plus Jakarta Sans',sans-serif;font-size:1.8rem;font-weight:800;color:#674c88;margin-top:5px;}
[data-testid="stVerticalBlockBorderWrapper"] {background:#fff;border-color:#ece4ed!important;border-radius:18px!important;}
.guide-step {background:#fff;border:1px solid #ede6ec;border-radius:15px;padding:18px 20px;margin:11px 0;}
.guide-step b {color:#654d78;}
.footer {text-align:center;color:#b0a2b2;border-top:1px solid #eee5ec;padding-top:20px;margin-top:36px;font-size:.82rem;}
@media(max-width:850px){.block-container{padding:1.1rem .9rem 3rem}.hero{padding:25px 23px}.hero h1{font-size:1.95rem!important}.paper-card{padding:19px 15px}.question-title{font-size:1.08rem}}
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)


def setting(name: str, default: str = "") -> str:
    env = os.environ.get(name, "")
    if env:
        return str(env)
    try:
        return str(st.secrets.get(name, default))
    except (FileNotFoundError, KeyError):
        return default


@st.cache_resource(show_spinner=False)
def load_model(repo: str, token: str, filename: str) -> MesinPenilai:
    return MesinPenilai(dapatkan_checkpoint_path(repo, token, filename))


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def load_kamus() -> dict[str, str]:
    return ambil_kamus()


def bank_default():
    # soal.xlsx versi guru diprioritaskan dari dataset sumber dan bank contoh.
    preferred = (
        Path("data/soal.xlsx"), Path("data/Dataset Biologi.xlsx"),
        Path("data/soal_publik.json"), Path("data/soal.csv"),
        Path("data/contoh_bank_soal.xlsx"),
    )
    for path in preferred:
        if path.is_file():
            return baca_bank_soal(path), path.name == "contoh_bank_soal.xlsx"
    # Untuk repositori kode publik, bank soal dapat disimpan secara privat
    # dan diambil di server (bukan dikirim ke browser siswa).
    bank_repo = setting("HF_BANK_REPO")
    if bank_repo:
        from huggingface_hub import hf_hub_download
        private_path = hf_hub_download(
            repo_id=bank_repo,
            repo_type="dataset",
            filename=setting("HF_BANK_FILENAME", "soal.xlsx"),
            token=setting("HF_TOKEN") or None,
        )
        return baca_bank_soal(private_path), False
    return None, False


def signature(bank: pd.DataFrame) -> str:
    return hashlib.sha256(bank.to_json(orient="records",force_ascii=False).encode()).hexdigest()


def reset_sesi(kosongkan_identitas: bool = False) -> None:
    for k in list(st.session_state):
        if k.startswith("jawaban_"):
            del st.session_state[k]
    st.session_state["nilai_per_soal"] = {}
    st.session_state["durasi_terakhir"] = 0.0
    st.session_state["soal_aktif"] = 0
    if kosongkan_identitas:
        for k in ("nama_peserta", "absen_peserta", "kelas_peserta"):
            if k in st.session_state:
                del st.session_state[k]


def pindah_soal(jarak: int, batas: int) -> None:
    st.session_state["soal_aktif"] = max(0, min(batas - 1, st.session_state.get("soal_aktif", 0) + jarak))


def simpan_jawaban(i: int) -> None:
    """Simpan jawaban di state aplikasi, terpisah dari state widget sementara."""
    st.session_state[f"jawaban_{i}"] = st.session_state.get(f"jawaban_input_{i}", "")


def nilai_terisi(bank: pd.DataFrame, indices: list[int], daftar_jawaban: dict[str, str]):
    pilihan = bank.iloc[indices].reset_index(drop=True)
    jawaban = [daftar_jawaban[str(i)] for i in indices]
    repo = setting("HF_MODEL_REPO")
    token = setting("HF_TOKEN")
    filename = setting("HF_MODEL_FILENAME", "model.pth")
    with st.spinner("Sedang memeriksa jawaban..."):
        kamus = load_kamus()
        mesin = load_model(repo, token, filename)
        hasil, durasi = penilaian(pilihan, jawaban, kamus, mesin)
    for j, i in enumerate(indices):
        st.session_state["nilai_per_soal"][str(i)] = {
            "jawaban": daftar_jawaban[str(i)],
            "result": hasil.iloc[j].to_dict(),
        }
    st.session_state["durasi_terakhir"] = durasi
    return len(indices)


def kumpulkan_jawaban(bank: pd.DataFrame) -> dict[str, str]:
    return {str(i):str(st.session_state.get(f"jawaban_{i}", "")) for i in range(len(bank))}


def kartu_nilai(row: dict) -> None:
    cos = float(row["Cosine Similarity"])
    bobot = float(row["Bobot"])
    nilai = float(row["Nilai Prediksi"])
    st.markdown(
        f'<div class="score-result"><div class="score-top">HASIL PEMERIKSAAN SOAL</div>'
        f'<div class="score-number">{nilai:.3f} <span style="font-size:.96rem;font-weight:600">/ {bobot:g}</span></div>'
        f'<div class="muted">Cosine Similarity: <strong>{cos:.4f}</strong> · Bobot: {bobot:g}</div></div>',
        unsafe_allow_html=True,
    )
    with st.expander("Lihat cara nilai dihitung"):
        st.write(f"Cosine Similarity × Bobot = {cos:.4f} × {bobot:g}")
        st.write(f"Hasil dibatasi antara 0 dan {bobot:g}, sehingga nilai soal ini menjadi **{nilai:.4f}**.")


with st.sidebar:
    st.markdown(
        '<div class="side-brand"><div class="side-logo">✦</div><div class="side-title">Esai Cerdas</div></div>'
        '<div class="side-sub">Belajar dan periksa esai dengan nyaman.</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="side-line"></div>', unsafe_allow_html=True)
    st.markdown('**Ruang belajarmu**')
    st.caption("Biologi · Penilaian esai")
    # Fitur guru opsional; tidak menambah keramaian bagi siswa.
    uploaded = None
    admin_password = setting("ADMIN_PASSWORD")
    if admin_password:
        with st.expander("Pengaturan guru"):
            typed = st.text_input("Sandi guru", type="password", key="sandi_guru")
            if typed and typed == admin_password:
                uploaded = st.file_uploader("Unggah bank soal", type=["xlsx","csv"], key="bank_upload")
            elif typed:
                st.error("Sandi tidak sesuai.")

try:
    bank, demo = bank_default()
    if uploaded:
        bank = baca_bank_soal(uploaded.getvalue())
        demo = False
except Exception as exc:
    bank, demo = None, False
    st.error("Bank soal belum dapat dibuka: " + str(exc))

if bank is not None:
    sig = signature(bank)
    if st.session_state.get("tanda_bank") != sig:
        reset_sesi()
        st.session_state["tanda_bank"] = sig
    st.session_state.setdefault("nilai_per_soal", {})
    st.session_state.setdefault("soal_aktif", 0)
    st.session_state.setdefault("durasi_terakhir", 0.0)
    answers = kumpulkan_jawaban(bank)
    rekap = buat_rekap_lengkap(bank, answers, st.session_state["nilai_per_soal"])
    stats = statistik_rekap(rekap)
    with st.sidebar:
        st.markdown('<div class="side-line"></div>', unsafe_allow_html=True)
        st.caption("PROGRES PENGERJAAN")
        st.progress(stats["diperiksa"] / max(1, len(bank)))
        st.markdown(f'<div class="side-meta"><b>{stats["diperiksa"]} dari {len(bank)} soal</b> sudah diperiksa</div>', unsafe_allow_html=True)
        st.button("Mulai sesi baru", key="reset_button", on_click=reset_sesi, args=(True,), use_container_width=True)
        st.markdown('<div class="side-line"></div>', unsafe_allow_html=True)
        st.caption("Kerjakan dengan tenang, satu soal demi satu soal. ✨")

st.markdown('''<div class="hero">
<div class="hero-kicker">✦ Selamat datang di Esai Cerdas</div>
<h1>Penilaian Esai Otomatis</h1>
<p>Tulis jawabanmu, periksa nilainya, dan lihat perkembanganmu dengan mudah. Kamu bisa mengerjakan satu soal dahulu atau beberapa soal sekaligus.</p>
<span class="soft-pill">✦ Biologi</span>
</div>''', unsafe_allow_html=True)

if bank is None:
    st.info("Soal belum tersedia. Silakan hubungi pengelola aplikasi.")
    st.stop()

model_ready = bool(Path("models/model.pth").is_file() or Path("IndoBERT_Essay_Scoring/model.pth").is_file() or setting("HF_MODEL_REPO"))
if not model_ready:
    st.info("Mesin penilaian belum siap. Kamu tetap bisa menulis jawaban; pemeriksaan tersedia setelah pengelola menyiapkan sistem.")
if demo:
    st.caption("Mode contoh soal aktif.")

kerjakan, hasil_tab, panduan = st.tabs(["✏️ Kerjakan Esai", "📋 Hasil Nilai", "📖 Panduan"])

with kerjakan:
    st.markdown('<div class="section-note">01 / Identitas peserta</div>',unsafe_allow_html=True)
    col_nama, col_absen, col_kelas = st.columns([2, 1, 1.2])
    with col_nama:
        st.text_input("Nama lengkap", key="nama_peserta", placeholder="Tulis nama lengkapmu")
    with col_absen:
        st.text_input("No. absen", key="absen_peserta", placeholder="Contoh: 12",max_chars=10)
    with col_kelas:
        st.text_input("Kelas", key="kelas_peserta", placeholder="Contoh: XI IPA 1")

    st.markdown('<div class="section-note">02 / Mulai mengerjakan</div>',unsafe_allow_html=True)
    a, b = st.columns([2.6, 1])
    with a:
        st.selectbox(
            "Pilih nomor soal", options=range(len(bank)),key="soal_aktif",
            format_func=lambda i: f"Soal {i+1:02d}  ·  Kode {bank.iloc[i]['Kode_Soal']}",
        )
    with b:
        st.metric("Sudah dijawab",f"{jumlah_terisi(bank, answers)} / {len(bank)}")

    i = int(st.session_state["soal_aktif"])
    soal = bank.iloc[i]
    with st.container(border=True):
        st.markdown(f'<div class="question-label">SOAL {i+1:02d} &nbsp; · &nbsp; BOBOT {float(soal["Bobot"]):g}</div>', unsafe_allow_html=True)
        # Teks pertanyaan dibaca melalui Markdown Streamlit, bukan HTML mentah.
        st.markdown(f"### {soal['Pertanyaan']}")
        input_key = f"jawaban_input_{i}"
        if input_key not in st.session_state:
            st.session_state[input_key] = st.session_state.get(f"jawaban_{i}", "")
        st.text_area(
            "Jawabanmu",
            key=input_key,
            height=170,
            placeholder="Tulis jawabanmu di sini...",
            on_change=simpan_jawaban,
            args=(i,),
        )
        left, middle, right = st.columns([1, 1.65, 1])
        with left:
            st.button("← Sebelumnya",key=f"prev_{i}",disabled=i==0,on_click=pindah_soal,args=(-1,len(bank)),use_container_width=True)
        with middle:
            pressed_single = st.button("✦ Periksa soal ini",key=f"check_{i}",type="primary",use_container_width=True)
        with right:
            st.button("Berikutnya →",key=f"next_{i}",disabled=i==len(bank)-1,on_click=pindah_soal,args=(1,len(bank)),use_container_width=True)

        current_answer = str(st.session_state.get(f"jawaban_{i}",""))
        if pressed_single:
            if not current_answer.strip():
                st.warning("Isi jawaban soal ini terlebih dahulu, ya.")
            elif not model_ready:
                st.warning("Mesin penilaian belum siap. Silakan hubungi guru atau pengelola aplikasi.")
            else:
                try:
                    nilai_terisi(bank, [i], kumpulkan_jawaban(bank))
                    st.success("Jawaban berhasil diperiksa!")
                except KonfigurasiError:
                    st.error("Pemeriksaan belum tersedia. Silakan hubungi pengelola aplikasi.")
                except Exception as exc:
                    st.error(f"Jawaban belum berhasil diperiksa: {exc}")

        item = st.session_state["nilai_per_soal"].get(str(i))
        if hasil_masih_valid(item,current_answer):
            kartu_nilai(item["result"])
        elif item is not None and current_answer.strip():
            st.caption("Jawaban ini sudah diubah. Klik **Periksa soal ini** untuk mendapatkan nilai terbaru.")
    st.divider()
    st.markdown("**Mau memeriksa beberapa soal sekaligus?**")
    st.caption("Tombol ini hanya memeriksa soal yang sudah diisi. Jawaban kosong akan tetap berstatus belum dijawab.")
    bulk = st.button("Periksa semua jawaban terisi",key="check_all",use_container_width=True,disabled=not model_ready)
    if bulk:
        isian = kumpulkan_jawaban(bank)
        indices=[k for k in range(len(bank)) if isian[str(k)].strip()]
        if not indices:
            st.warning("Belum ada jawaban yang diisi.")
        else:
            try:
                count = nilai_terisi(bank,indices,isian)
                st.success(f"Berhasil memeriksa {count} jawaban. Buka tab **Hasil Nilai** untuk melihat rekapnya.")
            except KonfigurasiError:
                st.error("Pemeriksaan belum tersedia. Silakan hubungi pengelola aplikasi.")
            except Exception as exc:
                st.error(f"Pemeriksaan belum berhasil: {exc}")

with hasil_tab:
    # Data disusun ulang pada setiap rerun agar nilai yang sudah diedit tidak ikut tercatat.
    answers = kumpulkan_jawaban(bank)
    rekap = buat_rekap_lengkap(bank,answers,st.session_state["nilai_per_soal"])
    stats = statistik_rekap(rekap)
    st.markdown("## Hasil Penilaian")
    if stats["diperiksa"] == 0:
        st.info("Belum ada nilai yang diperiksa. Isi jawaban di tab **Kerjakan Esai**, lalu klik **Periksa soal ini**.")
    else:
        nama = st.session_state.get("nama_peserta","").strip() or "Peserta"
        kelas = st.session_state.get("kelas_peserta","").strip()
        st.caption("Nama: " + nama + (" · Kelas: " + kelas if kelas else ""))
        c1,c2,c3=st.columns(3)
        c1.metric("Nilai terkumpul",f"{stats['total']:.3f} / {stats['maksimum']:g}")
        c2.metric("Soal diperiksa",f"{stats['diperiksa']} / {stats['jumlah']}")
        c3.metric("Waktu proses terakhir",f"{float(st.session_state['durasi_terakhir']):.2f} dtk")
        if stats["diperiksa"] < stats["jumlah"]:
            st.caption("Ini **nilai sementara**: hanya soal berstatus sudah diperiksa yang dihitung. Soal lain belum diberi nilai.")
        else:
            st.success("Seluruh soal telah diperiksa.")
        st.progress(stats["diperiksa"] / max(1,stats["jumlah"]))
        st.markdown("### Rincian per soal")
        cols=["Kode Soal","Pertanyaan","Status","Cosine Similarity","Bobot","Nilai Prediksi"]
        st.dataframe(
            rekap[cols],use_container_width=True,hide_index=True,
            column_config={
                "Cosine Similarity": st.column_config.NumberColumn(format="%.4f"),
                "Bobot": st.column_config.NumberColumn(format="%.1f"),
                "Nilai Prediksi": st.column_config.NumberColumn(format="%.4f"),
            },
        )
        st.markdown("### Simpan hasil")
        st.caption("File Excel berisi nilai, status pemeriksaan, dan jawaban siswa. Soal yang belum diperiksa tidak diberi angka nilai.")
        identitas={
            "nama":st.session_state.get("nama_peserta","").strip() or "Belum diisi",
            "absen":st.session_state.get("absen_peserta","").strip() or "Belum diisi",
            "kelas":st.session_state.get("kelas_peserta","").strip() or "Belum diisi",
        }
        try:
            excel=buat_excel_hasil(
                identitas,rekap,float(st.session_state["durasi_terakhir"]),
                show_answer_key=False,
            )
            shortname=re.sub(r"[^a-zA-Z0-9_-]","_",identitas["nama"])[:25]
            st.download_button(
                "⬇ Unduh hasil (.xlsx)",data=excel,
                file_name=f"hasil_esai_{shortname}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        except Exception as exc:
            st.error(f"File Excel belum dapat dibuat: {exc}")

with panduan:
    st.markdown("## Panduan Penggunaan")
    st.caption("Langkah sederhana untuk mulai mengerjakan dan melihat nilaimu.")
    st.markdown('<div class="guide-step"><b>01 · Isi identitas</b><div class="muted">Masukkan nama, nomor absen, dan kelas pada tab Kerjakan Esai.</div></div>',unsafe_allow_html=True)
    st.markdown('<div class="guide-step"><b>02 · Pilih dan jawab soal</b><div class="muted">Pilih nomor soal, lalu ketik jawaban pada kotak yang tersedia. Kamu bebas berpindah soal.</div></div>',unsafe_allow_html=True)
    st.markdown('<div class="guide-step"><b>03 · Periksa jawaban</b><div class="muted">Klik <b>Periksa soal ini</b> untuk melihat nilai satu soal. Kalau ingin memeriksa banyak soal sekaligus, klik <b>Periksa semua jawaban terisi</b>.</div></div>',unsafe_allow_html=True)
    st.markdown('<div class="guide-step"><b>04 · Lihat hasil</b><div class="muted">Buka tab Hasil Nilai untuk melihat nilai yang sudah diperiksa dan ringkasannya. Kamu juga bisa mengunduh hasil ke Excel.</div></div>',unsafe_allow_html=True)
    st.markdown('<div class="guide-step"><b>05 · Perbaiki jawaban bila perlu</b><div class="muted">Kalau jawaban diubah, nilainya perlu diperiksa ulang. Klik Mulai sesi baru pada sidebar bila ingin mengerjakan dari awal.</div></div>',unsafe_allow_html=True)
    with st.expander("Bagaimana nilai dihitung?"):
        st.write(
            "Sistem membandingkan jawabanmu dengan jawaban acuan menggunakan "
            "IndoBERT dan Cosine Similarity. Semakin mirip maknanya, "
            "umumnya semakin tinggi skor kemiripannya."
        )
        st.latex(r"\text{Nilai Soal} = \text{Cosine Similarity} \times \text{Bobot}")
        st.caption(
            "Hasil nilai dibatasi antara 0 dan bobot soal, "
            "jadi tidak bisa negatif atau melebihi nilai maksimum. "
            "Guru tetap dapat meninjau jawaban jika diperlukan."
        )
    st.info("Tips: tulis jawaban dengan kalimat yang jelas dan lengkap, gunakan kata-katamu sendiri, dan periksa kembali sebelum melanjutkan.")

st.markdown('<div class="footer">✦ Esai Cerdas · Dibuat agar belajar terasa lebih nyaman.</div>',unsafe_allow_html=True)