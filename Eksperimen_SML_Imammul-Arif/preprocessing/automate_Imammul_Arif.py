import pandas as pd
from sklearn.preprocessing import StandardScaler

def run_preprocessing(input_path, output_path):
    df = pd.read_csv(input_path)
    # Handling missing values
    df['ca'] = df['ca'].fillna(df['ca'].mode()[0])
    df['thal'] = df['thal'].fillna(df['thal'].mode()[0])
    # Save processed data
    df.to_csv(output_path, index=False)
    print(f"Otomatisasi selesai. File disimpan di {output_path}")

if __name__ == "__main__":
    run_preprocessing("Eksperimen_SML_Imammul_Arif/dataset_heart_raw/heart_disease_raw.csv", 
                      "Eksperimen_SML_Imammul_Arif/preprocessing/heart_disease_preprocessed.csv")