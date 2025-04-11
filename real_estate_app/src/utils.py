from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):
    target_col = 'price'

    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found. Available columns: {df.columns.tolist()}")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Label encode all object (categorical) columns
    for col in X.select_dtypes(include='object').columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    return train_test_split(X, y, test_size=0.2, random_state=42)
