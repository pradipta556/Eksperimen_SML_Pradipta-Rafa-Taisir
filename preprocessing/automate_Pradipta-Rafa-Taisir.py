import sys
import pandas as pd
import os

def load_dataset(input_path):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File tidak ditemukan: {input_path}")
    return pd.read_csv(input_path, engine="python", on_bad_lines="skip")

def clean_dataset(df):
    # Hapus duplikasi
    df = df.drop_duplicates()

    important_cols = ["sellingprice", "odometer", "make", "model", "year"]
    df = df.dropna(subset=important_cols)

    df = df.fillna(method="ffill").fillna(method="bfill")

    # Convert ke numeric
    numeric_cols = ["sellingprice", "odometer", "year", "mmr", "condition"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Format tanggal
    if "saledate" in df.columns:
        df["saledate"] = pd.to_datetime(df["saledate"], errors="coerce")

    return df

def save_dataset(df, output_path):
    df.to_csv(output_path, index=False)
    print(f"Dataset bersih berhasil disimpan sebagai: {output_path}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python automate.py <input.csv> <output.csv>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    print("=== Memuat dataset ===")
    df = load_dataset(input_path)

    print("=== Membersihkan dataset ===")
    df_clean = clean_dataset(df)

    print("=== Menyimpan dataset ===")
    save_dataset(df_clean, output_path)

    print("=== Selesai! ===")

if __name__ == "__main__":
    main()
