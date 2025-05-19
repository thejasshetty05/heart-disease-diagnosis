import pandas as pd

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def clean_data(df):
    df = df.dropna()
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    df['sex'] = df['sex'].map({0: 'female', 1: 'male'})
    df['target'] = df['target'].map({0: 'no_disease', 1: 'disease'})
    return df

def save_cleaned_data(df, output_path):
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    input_path = r"data/heart.csv"
    output_path = r"data/cleaned_heart.csv"

    print("Loading data...")
    data = load_data(input_path)

    print("Cleaning data...")
    cleaned_data = clean_data(data)

    print("Saving cleaned data...")
    save_cleaned_data(cleaned_data, output_path)

    print("Data cleaning complete!")
