import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score, f1_score
import xgboost as xgb

def load_and_clean_data(data_path: str = 'data.csv') -> pd.DataFrame:
    """Load data and standardize column names and target variable."""
    df = pd.read_csv(data_path, sep=';')
    
    # Standardize column names to snake_case
    df.columns = (
        df.columns.str.lower()
        .str.strip()
        .str.replace(' ', '_')
        .str.replace('(', '', regex=False)
        .str.replace(')', '', regex=False)
        .str.replace('/', '_', regex=False)
        .str.replace("'", '', regex=False)
    )
    
    # Map target: Dropout -> 1, Graduate / Enrolled -> 0
    df['target'] = df['target'].map({'Dropout': 1, 'Graduate': 0, 'Enrolled': 0})
    return df

def train_model(data_path: str = 'data.csv', output_file: str = 'model.bin'):
    print(f"Loading data from {data_path}...")
    df = load_and_clean_data(data_path)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns. Base dropout rate: {df['target'].mean():.3f}")

    # Split for validation check: 80% train, 20% validation
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['target'])
    
    y_train = train_df['target'].values
    y_val = val_df['target'].values
    
    train_features = train_df.drop(columns=['target']).to_dict(orient='records')
    val_features = val_df.drop(columns=['target']).to_dict(orient='records')
    
    # Vectorize
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(train_features)
    X_val = dv.transform(val_features)
    
    feature_names = dv.get_feature_names_out().tolist()
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
    
    # XGBoost hyperparameters selected from notebook tuning
    params = {
        'eta': 0.1,
        'max_depth': 3,
        'min_child_weight': 3,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'seed': 42,
        'verbosity': 0
    }
    
    print("Training XGBoost model...")
    model = xgb.train(params, dtrain, num_boost_round=100)
    
    # Validation metrics
    val_proba = model.predict(dval)
    val_pred = (val_proba >= 0.5).astype(int)
    
    auc = roc_auc_score(y_val, val_proba)
    f1 = f1_score(y_val, val_pred)
    
    print(f"Validation Results:")
    print(f"  ROC-AUC:  {auc:.4f}")
    print(f"  F1-Score: {f1:.4f}")

    # Retrain on full dataset
    print("\nRetraining final model on the entire dataset...")
    y_full = df['target'].values
    full_features = df.drop(columns=['target']).to_dict(orient='records')
    
    dv_full = DictVectorizer(sparse=False)
    X_full = dv_full.fit_transform(full_features)
    dfull = xgb.DMatrix(X_full, label=y_full, feature_names=dv_full.get_feature_names_out().tolist())
    
    final_model = xgb.train(params, dfull, num_boost_round=100)
    
    # Save model and vectorizer
    with open(output_file, 'wb') as f_out:
        pickle.dump((dv_full, final_model), f_out)
    
    print(f"Final model successfully exported to '{output_file}'!")

if __name__ == '__main__':
    train_model()
