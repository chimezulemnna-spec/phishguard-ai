from src.preprocessing import load_and_preprocess
from src.url_features import extract_url_features
from train_models import train_models
from src.evaluate import evaluate_model

def main():
    # 1️⃣ Load dataset
    df = load_and_preprocess("data/phishing_urls.csv", "data/legitimate_urls.csv")

    # 2️⃣ Remove duplicates
    if 'url' in df.columns:
        df = df.drop_duplicates(subset=['url'])

    # 3️⃣ Extract numeric features
    df = extract_url_features(df)

    # 4️⃣ Drop the raw 'url' column and other non-numeric columns
    cols_to_drop = ['url', 'phish_detail_url', 'submission_time', 'verified', 
                    'verification_time', 'online', 'target', 'Unnamed: 0']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # 5️⃣ Define feature columns
    feature_cols = [col for col in df.columns if col != 'label']

    # 6️⃣ Train models (returns rf, xgb, voting, X_train, X_test, y_train, y_test)
    rf, xgb, voting, X_train, X_test, y_train, y_test = train_models(df, feature_cols)

    # 7️⃣ Evaluate all models
    print("\n--- Random Forest Evaluation ---")
    evaluate_model(rf, X_test, y_test)

    print("\n--- XGBoost Evaluation ---")
    evaluate_model(xgb, X_test, y_test)

    print("\n--- Voting Ensemble Evaluation ---")
    evaluate_model(voting, X_test, y_test)

if __name__ == "__main__":
    main()