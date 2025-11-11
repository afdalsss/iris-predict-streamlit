import Orange
import pickle

# --- 1. Load dataset bawaan Iris ---
data = Orange.data.Table("iris")

# --- 2. Buat model Random Forest (sesuai workflow di Orange) ---
model = Orange.classification.RandomForestLearner(n_estimators=10)
classifier = model(data)

# --- 3. Simpan model ke file pkcls ---
model_file = "model_iris.pkcls"
with open(model_file, "wb") as f:
    pickle.dump(classifier, f)

print(f"✅ Model berhasil dibuat dan disimpan sebagai: {model_file}")
