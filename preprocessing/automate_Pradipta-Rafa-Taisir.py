import os
import pandas as pd

RAW_FILE = "car_prices.csv"
OUTPUT_FILE = "car_prices_clean.csv"

def load_dataset():
    if not os.path.exists(RAW_FILE):
        raise FileNotFoundError(f"File tidak ditemukan: {RAW_FILE}")
    return pd.read_csv(RAW_FILE, engine="python", on_bad_lines="skip")

def clean_dataset(df):
    # Hapus duplikasi
    df = df.drop_duplicates()

    important_cols = ["sellingprice", "odometer", "make", "model", "year"]

    df = df.dropna(subset=important_cols)

    # Isi missing value lain otomatis
    df = df.fillna(method="ffill").fillna(method="bfill")

    # Convert tipe data numerik
    numeric_cols = ["sellingprice", "odometer", "year", "mmr", "condition"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Format tanggal
    if "saledate" in df.columns:
        df["saledate"] = pd.to_datetime(df["saledate"], errors="coerce")

    return df

def save_dataset(df):
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Dataset bersih berhasil disimpan sebagai: {OUTPUT_FILE}")

def main():
    print("=== Memuat dataset ===")
    df = load_dataset()
    
    print("=== Membersihkan dataset ===")
    df_clean = clean_dataset(df)
    
    print("=== Menyimpan dataset ===")
    save_dataset(df_clean)

    print("=== Selesai! ===")

if __name__ == "__main__":
    main()