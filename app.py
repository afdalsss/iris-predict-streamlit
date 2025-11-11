import streamlit as st
import pickle
import numpy as np
import Orange

# --- Konfigurasi ---
st.set_page_config(page_title="Prediksi Iris", page_icon="🌸", layout="centered")
st.title("🌸 Prediksi Kategori Bunga Iris")
st.caption("Model: Random Forest (Orange3) | Dataset: Iris")

# --- Load model ---
model_file = "model_iris.pkcls"
try:
    with open(model_file, "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("❌ File model tidak ditemukan. Pastikan 'model_iris.pkcls' ada di folder yang sama.")
    st.stop()

# --- Ambil nama fitur dari domain model ---
feature_names = [attr.name for attr in model.domain.attributes]

# --- Input fitur ---
st.sidebar.header("Masukkan Nilai Fitur")
inputs = []
for feature in feature_names:
    val = st.sidebar.number_input(
        f"{feature.title()} (cm)",
        min_value=0.0,
        max_value=10.0,
        value=5.0,
        step=0.1,
    )
    inputs.append(val)

# --- Prediksi ---
if st.button("🔍 Prediksi"):
    try:
        # 1) Buat domain hanya berisi atribut (fitur) — tanpa kelas
        domain_input = Orange.data.Domain(model.domain.attributes)

        # 2) Buat Table dari numpy (fitur saja)
        X = np.array([inputs])
        table_features = Orange.data.Table.from_numpy(domain=domain_input, X=X)

        # 3) Transformasi ke domain model (menambahkan class var yang kosong)
        #    Ini menjaga struktur kolom sama persis dengan yang model harapkan.
        data_for_model = Orange.data.Table(model.domain, table_features)

        # 4) Panggil model -> hasil bisa bermacam tipe
        raw_pred = model(data_for_model)

        # raw_pred bisa:
        # - Orange.data.Table/Row/Value object dengan .value
        # - numpy array / numpy.float64 (indeks kelas)
        # - list-like berisi angka / objek
        # Kita ambil elemen pertama (karena 1 sampel input)
        pred0 = None
        # Jika raw_pred adalah Orange Table-like, index 0 kemungkinan memberi Value/Row
        try:
            pred0 = raw_pred[0]
        except Exception:
            # jika tidak bisa di-index, raw_pred mungkin sudah scalar
            pred0 = raw_pred

        # 5) Tentukan label akhir
        label = None
        # a) Jika objek punya attribute 'value' (Orange Value), gunakan itu
        if hasattr(pred0, "value"):
            label = pred0.value
        else:
            # b) Bisa berupa angka (indeks kelas) atau string label
            #    Jika ada class_var di domain, coba map indeks -> nama kelas
            class_var = getattr(model.domain, "class_var", None)
            if class_var is not None and getattr(class_var, "values", None):
                try:
                    # jika pred0 adalah array-like ambil nilai numeriknya
                    idx = int(np.asarray(pred0).item())
                    # Proteksi: pastikan idx berada di rentang
                    values = list(class_var.values)
                    if 0 <= idx < len(values):
                        label = values[idx]
                    else:
                        # bukan index -> tampilkan langsung
                        label = str(pred0)
                except Exception:
                    # bukan angka -> tampilkan string representasi
                    label = str(pred0)
            else:
                # tidak ada class_var -> tampilkan representasi pred0
                label = str(pred0)

        # Tampilkan hasil
        st.success(f"Hasil Prediksi: **{label}** 🌼")

        st.write("### Nilai Input")
        st.table({
            "Fitur": feature_names,
            "Nilai (cm)": inputs
        })

    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")
        # tampilkan sedikit info debug jika perlu (bisa dikomentari)
        st.debug = getattr(st, "debug", None)
        # optional: tampilkan tipe raw_pred supaya lebih mudah debug
        try:
            st.write("Debug info (tipe raw_pred):", type(raw_pred))
        except Exception:
            pass

# Footer
st.markdown("""
---
💡 **Petunjuk:**
- Pastikan `model_iris.pkcls` dibuat dari Orange dan berada di folder yang sama.  
- Isi empat fitur (sepal/petal) di sidebar lalu klik **Prediksi**.
""")
