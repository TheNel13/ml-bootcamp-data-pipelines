import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_placement_data():
    return pd.read_csv(
        "data/ppipeline/Placement_Data_Full_Class.csv"
    )

def create_target(df):
    df = df.copy()
    df["placed"] = (df["status"] == "Placed").astype(int)
    return df

def clean_data(df):
    return df.drop(columns=["status", "salary", "sl_no"], errors="ignore")

def encode_and_scale(df, target):
    X = df.drop(columns=[target])
    y = df[target]

    X = pd.get_dummies(X, drop_first=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

def split_data(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_tune, X_test, y_tune, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    return X_train, X_tune, X_test, y_train, y_tune, y_test

def placement_pipeline():
    df = load_placement_data()
    df = create_target(df)
    df = clean_data(df)

    print("Placement prevalence:", df["placed"].mean())

    X, y = encode_and_scale(df, "placed")
    return split_data(X, y)