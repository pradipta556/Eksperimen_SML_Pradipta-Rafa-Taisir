import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_dataset(path):
    print("Loading dataset...")
    df = pd.read_csv(path)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def preprocess_data(df):

    print("Starting preprocessing...")

    df_clean = df.copy()

    df_clean.drop_duplicates(inplace=True)

    df_clean.dropna(inplace=True)

    categorical_cols = ['Job_Title', 'Education_Level', 'Risk_Category']
    numerical_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns.tolist()

    print("Categorical columns:", categorical_cols)
    print("Numerical columns:", numerical_cols)

    Q1 = df_clean[numerical_cols].quantile(0.25)
    Q3 = df_clean[numerical_cols].quantile(0.75)
    IQR = Q3 - Q1

    df_clean = df_clean[~(
        (df_clean[numerical_cols] < (Q1 - 1.5 * IQR)) |
        (df_clean[numerical_cols] > (Q3 + 1.5 * IQR))
    ).any(axis=1)]

    print("Shape after outlier removal:", df_clean.shape)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(drop='first'), categorical_cols)
        ]
    )

    print("Fitting and transforming data...")
    processed_array = preprocessor.fit_transform(df_clean)

    encoded_cols = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
    final_cols = numerical_cols + list(encoded_cols)

    df_processed = pd.DataFrame(processed_array, columns=final_cols)

    print("Preprocessing completed.")
    print("Final shape:", df_processed.shape)

    return df_processed

def save_dataset(df, output_path):
    df.to_csv(output_path, index=False)
    print(f"Saved preprocessed dataset to: {output_path}")

def automate():
    raw_path = "../AI_Impact_on_Jobs_2030_raw/AI_Impact_on_Jobs_2030.csv"
    output_path = "./AI_Impact_on_Jobs_2030_preprocessing.csv"

    df = load_dataset(raw_path)
    df_processed = preprocess_data(df)
    save_dataset(df_processed, output_path)

if __name__ == "__main__":
    automate()
