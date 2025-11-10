"""
CognitiveAI - Prédiction du profil cognitif et risque de déclin
Optimisé pour RTX 3060 (6 Go VRAM)
Dataset: NACC (investigator_nacc71.csv)
"""

import os
import argparse
import numpy as np
import pandas as pd
import pickle
import warnings
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, recall_score, f1_score, 
                           classification_report, confusion_matrix)
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.inspection import permutation_importance

warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
class Config:
    # Paths
    DATA_PATH = "investigator_nacc71.csv"
    MODEL_PATH = "cognitiveai_model.pt"
    SCALER_PATH = "scaler.pkl"
    ENCODER_PATH = "label_encoders.pkl"
    
    # Mots-clés pour sélection automatique des colonnes
    KEYWORDS = [
        'age', 'sex', 'educ', 'marist', 'mmse', 'moca',
        'memory', 'lang', 'exec', 'visu', 'hypert', 'diabet',
        'stroke', 'bmi', 'smoke', 'cogn', 'dx', 'naccudsd'
    ]
    
    # Hyperparamètres
    BATCH_SIZE = 128
    EPOCHS = 100
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    PATIENCE = 10
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    # Architecture
    HIDDEN_DIMS = [256, 128, 64]
    DROPOUT = 0.3
    
    # GPU
    NUM_WORKERS = 2
    PIN_MEMORY = True

config = Config()

# ==================== DATASET ====================
class CognitiveDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ==================== MODÈLE ====================
class CognitiveNet(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.3):
        super(CognitiveNet, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# ==================== PREPROCESSING ====================
def load_and_select_features(csv_path, keywords):
    """Charge le CSV et sélectionne automatiquement les colonnes pertinentes"""
    print(f"📂 Chargement de {csv_path}...")
    
    # Lecture avec dtype optimisé
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"   Dataset: {df.shape[0]:,} lignes × {df.shape[1]:,} colonnes")
    
    # Sélection des colonnes pertinentes
    selected_cols = []
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in keywords):
            selected_cols.append(col)
    
    print(f"   🎯 {len(selected_cols)} colonnes sélectionnées automatiquement")
    
    df_selected = df[selected_cols].copy()
    
    # Affichage des premières colonnes sélectionnées
    print(f"   Exemples: {', '.join(selected_cols[:10])}...")
    
    return df_selected

def clean_data(df):
    """Nettoie les données (codes manquants, NaN)"""
    print("\n🧹 Nettoyage des données...")
    
    # Remplacer les codes manquants par NaN
    df = df.replace([-4, -9, 88, 99, 888, 999], np.nan)
    df = df.replace(['', ' ', 'NA', 'N/A'], np.nan)
    
    # Supprimer colonnes avec >80% de valeurs manquantes
    threshold = 0.8
    missing_pct = df.isnull().sum() / len(df)
    cols_to_keep = missing_pct[missing_pct < threshold].index.tolist()
    df = df[cols_to_keep]
    
    print(f"   ✓ Colonnes conservées: {len(cols_to_keep)}")
    print(f"   ✓ Valeurs manquantes: {df.isnull().sum().sum():,}")
    
    return df

def prepare_target_variable(df):
    """Prépare la variable cible (diagnostic)"""
    print("\n🎯 Préparation de la variable cible...")
    
    # Recherche de la colonne de diagnostic
    target_col = None
    for col in df.columns:
        col_lower = col.lower()
        if 'dx' in col_lower or 'naccudsd' in col_lower or 'cogn' in col_lower:
            if df[col].nunique() > 1 and df[col].nunique() < 10:
                target_col = col
                break
    
    if target_col is None:
        # Créer une variable cible synthétique basée sur MMSE si disponible
        mmse_cols = [c for c in df.columns if 'mmse' in c.lower()]
        if mmse_cols:
            print("   ⚠️ Pas de colonne diagnostic trouvée, création basée sur MMSE...")
            mmse_col = mmse_cols[0]
            df['target'] = pd.cut(df[mmse_col], bins=[0, 20, 24, 30], labels=[2, 1, 0])
            target_col = 'target'
        else:
            raise ValueError("Impossible de trouver ou créer une variable cible")
    
    # Nettoyer la cible
    df = df.dropna(subset=[target_col])
    
    # Encoder la cible
    le_target = LabelEncoder()
    y = le_target.fit_transform(df[target_col])
    
    print(f"   ✓ Variable cible: {target_col}")
    print(f"   ✓ Classes: {le_target.classes_}")
    print(f"   ✓ Distribution: {np.bincount(y)}")
    
    # Retirer la cible des features
    X = df.drop(columns=[target_col])
    
    return X, y, le_target

def preprocess_features(X_train, X_test, y_train):
    """Prétraite les features (imputation, encodage, normalisation)"""
    print("\n⚙️ Prétraitement des features...")
    
    # Séparer colonnes numériques et catégorielles
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
    
    print(f"   Numériques: {len(numeric_cols)}, Catégorielles: {len(categorical_cols)}")
    
    # Imputation des valeurs manquantes pour les numériques
    imputer_num = SimpleImputer(strategy='median')
    try:
        X_train_num_imputed = imputer_num.fit_transform(X_train[numeric_cols])
        X_test_num_imputed = imputer_num.transform(X_test[numeric_cols])

        # Vérifier que la forme retournée correspond bien au nombre de colonnes
        if X_train_num_imputed.ndim != 2 or X_train_num_imputed.shape[1] != len(numeric_cols):
            print("Warning: unexpected shape from imputer for numeric columns — falling back to median fillna per-column")
            # Fallback: remplir par la médiane en gardant les colonnes d'origine
            X_train_num = X_train[numeric_cols].fillna(X_train[numeric_cols].median())
            X_test_num = X_test[numeric_cols].fillna(X_train[numeric_cols].median())
        else:
            # Reconstruire les DataFrames numériques (conserver index/colonnes)
            X_train_num = pd.DataFrame(X_train_num_imputed, columns=numeric_cols, index=X_train.index)
            X_test_num = pd.DataFrame(X_test_num_imputed, columns=numeric_cols, index=X_test.index)

    except Exception as e:
        # En cas d'erreur (ex: types ou décodage), fallback sûr
        print(f"Warning: imputer failed ({e}), using per-column median fillna as fallback")
        X_train_num = X_train[numeric_cols].fillna(X_train[numeric_cols].median())
        X_test_num = X_test[numeric_cols].fillna(X_train[numeric_cols].median())
    
    # Encodage des variables catégorielles
    label_encoders = {}
    X_train_cat_list = []
    X_test_cat_list = []
    
    for col in categorical_cols:
        le = LabelEncoder()
        
        # Gérer les valeurs manquantes
        train_col = X_train[col].fillna('missing').astype(str)
        test_col = X_test[col].fillna('missing').astype(str)
        
        # Fit sur train
        le.fit(train_col)
        
        # Transform train
        train_encoded = le.transform(train_col)
        
        # Transform test avec gestion des nouvelles catégories
        test_encoded = []
        for val in test_col:
            if val in le.classes_:
                test_encoded.append(le.transform([val])[0])
            else:
                # Assigner à 'missing' si nouvelle catégorie
                test_encoded.append(le.transform(['missing'])[0])
        
        X_train_cat_list.append(pd.Series(train_encoded, index=X_train.index, name=col))
        X_test_cat_list.append(pd.Series(test_encoded, index=X_test.index, name=col))
        label_encoders[col] = le
    
    # Combiner numériques et catégorielles
    if categorical_cols:
        X_train_cat = pd.concat(X_train_cat_list, axis=1)
        X_test_cat = pd.concat(X_test_cat_list, axis=1)
        X_train_combined = pd.concat([X_train_num, X_train_cat], axis=1)
        X_test_combined = pd.concat([X_test_num, X_test_cat], axis=1)
    else:
        X_train_combined = X_train_num
        X_test_combined = X_test_num
    
    # 1. Imputation des valeurs manquantes d'abord
    print("\n🔍 Imputation des valeurs manquantes...")
    
    # Traiter les colonnes numériques et catégorielles séparément
    numeric_cols = X_train_combined.select_dtypes(include=[np.number]).columns
    categorical_cols = X_train_combined.select_dtypes(exclude=[np.number]).columns
    
    # Identifier les colonnes avec trop de valeurs manquantes (>50%)
    missing_ratio = X_train_combined.isna().mean()
    cols_to_drop = missing_ratio[missing_ratio > 0.5].index
    if len(cols_to_drop) > 0:
        print(f"   ⚠️ Suppression de {len(cols_to_drop)} colonnes avec >50% de NaN:")
        print("   " + ", ".join(cols_to_drop[:5]) + "..." if len(cols_to_drop) > 5 else "   " + ", ".join(cols_to_drop))
        X_train_combined = X_train_combined.drop(columns=cols_to_drop)
        X_test_combined = X_test_combined.drop(columns=cols_to_drop)
        
        # Mettre à jour les listes de colonnes
        numeric_cols = X_train_combined.select_dtypes(include=[np.number]).columns
        categorical_cols = X_train_combined.select_dtypes(exclude=[np.number]).columns
    
    # Pour les colonnes numériques : imputation en deux étapes
    X_train_num = X_train_combined[numeric_cols].copy()
    X_test_num = X_test_combined[numeric_cols].copy()
    
    # 1. Remplacer les valeurs extrêmes par NaN
    for col in numeric_cols:
        q1 = X_train_num[col].quantile(0.01)
        q3 = X_train_num[col].quantile(0.99)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        
        # Remplacer les valeurs extrêmes par NaN
        X_train_num.loc[X_train_num[col] < lower, col] = np.nan
        X_train_num.loc[X_train_num[col] > upper, col] = np.nan
        X_test_num.loc[X_test_num[col] < lower, col] = np.nan
        X_test_num.loc[X_test_num[col] > upper, col] = np.nan
    
    # 2. Imputation avec la médiane
    train_medians = X_train_num.median()
    X_train_num = X_train_num.fillna(train_medians)
    X_test_num = X_test_num.fillna(train_medians)
    
    # Pour les colonnes catégorielles : utiliser le mode et une catégorie 'UNKNOWN' pour les NaN
    if len(categorical_cols) > 0:
        X_train_cat = X_train_combined[categorical_cols].fillna('UNKNOWN')
        X_test_cat = X_test_combined[categorical_cols].fillna('UNKNOWN')
        
        # Recombiner les colonnes numériques et catégorielles
        X_train_imputed = pd.concat([X_train_num, X_train_cat], axis=1)
        X_test_imputed = pd.concat([X_test_num, X_test_cat], axis=1)
    else:
        X_train_imputed = X_train_num
        X_test_imputed = X_test_num
    
    # Vérification finale
    train_na = X_train_imputed.isna().sum().sum()
    test_na = X_test_imputed.isna().sum().sum()
    if train_na > 0 or test_na > 0:
        raise ValueError(f"Il reste des NaN après imputation: train={train_na}, test={test_na}")
    
    print(f"   ✓ Données nettoyées et imputées: {X_train_imputed.shape}")
    print(f"   ✓ Colonnes restantes: {len(X_train_imputed.columns)}")
    
    # 2. Normalisation des données imputées
    print("📊 Normalisation des features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)
    
    # 3. Sélection des meilleures features
    print("\n🎯 Sélection des features les plus importantes...")
    selector = SelectKBest(score_func=f_classif, k=30)  # garder les 30 meilleures features
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)
    
    # Afficher les features sélectionnées et leurs scores
    feature_names = np.array(X_train_combined.columns)
    selected_mask = selector.get_support()
    selected_features = feature_names[selected_mask]
    feature_scores = pd.DataFrame({
        'Feature': feature_names,
        'Score': selector.scores_,
        'Selected': selected_mask
    }).sort_values('Score', ascending=False)
    
    print("\nTop 10 features sélectionnées:")
    print(feature_scores[feature_scores['Selected']].head(10).to_string(index=False))

    # Convertir en float32 et s'assurer qu'il n'y a pas de NaN
    X_train_selected = X_train_selected.astype(np.float32)
    X_test_selected = X_test_selected.astype(np.float32)

    # Vérification finale
    if np.isnan(X_train_selected).any() or np.isnan(X_test_selected).any():
        raise ValueError("Des NaN persistent après le prétraitement!")

    # Count problematic values before sanitizing
    nan_train = np.isnan(X_train_scaled).sum()
    nan_test = np.isnan(X_test_scaled).sum()
    inf_train = np.isinf(X_train_scaled).sum()
    inf_test = np.isinf(X_test_scaled).sum()
    if nan_train or nan_test or inf_train or inf_test:
        print(f"   ⚠️ Detected numeric issues before sanitization: nan_train={nan_train}, nan_test={nan_test}, inf_train={inf_train}, inf_test={inf_test}")

    # Replace NaN with 0 and infinite with large finite values
    X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=1e6, neginf=-1e6)
    X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=1e6, neginf=-1e6)

    # Clip extreme values that could destabilize training
    clip_val = 1e6
    X_train_scaled = np.clip(X_train_scaled, -clip_val, clip_val)
    X_test_scaled = np.clip(X_test_scaled, -clip_val, clip_val)

    print(f"   ✓ Shape finale: {X_train_selected.shape}")
    print(f"   ✓ Réduction de dimension: {X_train_combined.shape[1]} → {X_train_selected.shape[1]} features")

    return X_train_selected, X_test_selected, scaler, label_encoders

# ==================== ENTRAÎNEMENT ====================
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def train_model(model, train_loader, val_loader, device, config, class_weights=None):
    """Entraîne le modèle avec mixed precision et early stopping"""
    print("\n🚀 Début de l'entraînement...")
    
    # Loss pondérée pour gérer le déséquilibre des classes
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, 
                           weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                     factor=0.5, patience=5)
    
    # Mixed precision training
    scaler = GradScaler()
    early_stopping = EarlyStopping(patience=config.PATIENCE)
    
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0
    
    for epoch in range(config.EPOCHS):
        # Training
        model.train()
        train_loss = 0
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.EPOCHS}')
        
        for X_batch, y_batch in train_bar:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
            
                # Check for NaN/inf in outputs before backward
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    print("Error: model outputs contain NaN or Inf. Dumping diagnostics:")
                    print(f"  X_batch: nan={torch.isnan(X_batch).sum().item()}, inf={(~torch.isfinite(X_batch)).sum().item()}")
                    print(f"  y_batch unique: {torch.unique(y_batch)}")
                    return history

                if torch.isnan(loss) or not torch.isfinite(loss):
                    print(f"Error: loss is NaN or Inf (loss={loss}). Dumping diagnostics:")
                    print(f"  outputs min/max: {torch.min(outputs).item()}/{torch.max(outputs).item()}")
                    print(f"  X_batch stats: min={torch.min(X_batch).item()}, max={torch.max(X_batch).item()}, mean={torch.mean(X_batch).item()}")
                    print(f"  y_batch unique: {torch.unique(y_batch)}")
                    return history

                scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            train_bar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                with autocast():
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(all_labels, all_preds)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, "
              f"Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.4f}")
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), config.MODEL_PATH)
            print(f"   ✓ Modèle sauvegardé (acc={val_acc:.4f})")
        
        # Early stopping
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print(f"⏹️ Early stopping à l'epoch {epoch+1}")
            break
    
    return history

def evaluate_model(model, test_loader, device, le_target):
    """Évalue le modèle sur le set de test"""
    print("\n📊 Évaluation du modèle...")
    
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            
            with autocast():
                outputs = model(X_batch)
            
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Métriques
    acc = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    print(f"\n{'='*50}")
    print(f"   Accuracy:  {acc:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    print(f"{'='*50}\n")
    
    # Classification report
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, 
                               target_names=le_target.classes_.astype(str)))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=le_target.classes_,
               yticklabels=le_target.classes_)
    plt.title('Matrice de Confusion')
    plt.ylabel('Vraie classe')
    plt.xlabel('Classe prédite')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    print("✓ Matrice de confusion sauvegardée: confusion_matrix.png")
    
    return acc, recall, f1

def plot_training_history(history):
    """Visualise l'historique d'entraînement"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Évolution de la Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy
    axes[1].plot(history['val_acc'], label='Val Accuracy', color='green')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Évolution de l\'Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300)
    print("✓ Historique d'entraînement sauvegardé: training_history.png")

def compute_feature_importance(model, X_test, y_test, feature_names, device):
    """Calcule l'importance des features via permutation"""
    print("\n🔍 Calcul de l'importance des features...")

    from sklearn.base import BaseEstimator

    class ModelWrapper(BaseEstimator):
        """Adapter le modèle PyTorch au format sklearn"""
        def __init__(self, model, device):
            self.model = model
            self.device = device

        def fit(self, X, y=None):
            # Rien à entraîner, mais nécessaire pour sklearn
            return self

        def predict(self, X):
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).to(self.device)
                with autocast():
                    outputs = self.model(X_tensor)
                preds = torch.argmax(outputs, dim=1)
            return preds.cpu().numpy()

        def score(self, X, y):  # ✅ Ajout obligatoire
            """Retourne la précision (accuracy)"""
            preds = self.predict(X)
            return accuracy_score(y, preds)

    wrapper = ModelWrapper(model, device)

    # Échantillon pour accélérer le calcul
    sample_size = min(5000, len(X_test))
    indices = np.random.choice(len(X_test), sample_size, replace=False)
    X_sample = X_test[indices]
    y_sample = y_test[indices]

    # Calcul de l'importance avec scoring explicite
    result = permutation_importance(
        wrapper.fit(X_sample, y_sample),
        X_sample, y_sample,
        scoring="accuracy",        # ✅ explicite
        n_repeats=10,
        random_state=42,
        n_jobs=-1
    )

    importance_df = (
        pd.DataFrame({
            "feature": feature_names,
            "importance": result.importances_mean
        })
        .sort_values("importance", ascending=False)
    )

    print("\nTop 15 features les plus importantes:")
    print(importance_df.head(15).to_string(index=False))

    plt.figure(figsize=(12, 8))
    top = importance_df.head(20)
    plt.barh(range(len(top)), top["importance"])
    plt.yticks(range(len(top)), top["feature"])
    plt.xlabel("Importance moyenne (Permutation)")
    plt.title("Top 20 Features - Permutation Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=300)
    print("✓ Importance des features sauvegardée: feature_importance.png")

    return importance_df

    """Calcule l'importance des features via permutation"""
    print("\n🔍 Calcul de l'importance des features...")

    from sklearn.base import BaseEstimator

    class ModelWrapper(BaseEstimator):
        """Adapter le modèle PyTorch au format sklearn"""
        def __init__(self, model, device):
            self.model = model
            self.device = device

        def fit(self, X, y=None):
            # rien à entraîner, mais nécessaire pour sklearn
            return self

        def predict(self, X):
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).to(self.device)
                with autocast():
                    outputs = self.model(X_tensor)
                preds = torch.argmax(outputs, dim=1)
            return preds.cpu().numpy()

    wrapper = ModelWrapper(model, device)

    # Échantillon pour accélérer le calcul
    sample_size = min(5000, len(X_test))
    indices = np.random.choice(len(X_test), sample_size, replace=False)
    X_sample = X_test[indices]
    y_sample = y_test[indices]

    result = permutation_importance(
        wrapper.fit(X_sample, y_sample),  # ✅ appel à fit ajouté
        X_sample, y_sample,
        n_repeats=10, random_state=42, n_jobs=-1
    )

    importance_df = (
        pd.DataFrame({"feature": feature_names,
                      "importance": result.importances_mean})
        .sort_values("importance", ascending=False)
    )

    print("\nTop 15 features les plus importantes:")
    print(importance_df.head(15).to_string(index=False))

    plt.figure(figsize=(12, 8))
    top = importance_df.head(20)
    plt.barh(range(len(top)), top["importance"])
    plt.yticks(range(len(top)), top["feature"])
    plt.xlabel("Importance moyenne (Permutation)")
    plt.title("Top 20 Features - Permutation Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=300)
    print("✓ Importance des features sauvegardée: feature_importance.png")

    return importance_df

    """Calcule l'importance des features via permutation"""
    print("\n🔍 Calcul de l'importance des features...")
    
    # Wrapper pour sklearn
    class ModelWrapper:
        def __init__(self, model, device):
            self.model = model
            self.device = device
        
        def predict(self, X):
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).to(self.device)
                with autocast():
                    outputs = self.model(X_tensor)
                preds = torch.argmax(outputs, dim=1)
            return preds.cpu().numpy()
    
    wrapper = ModelWrapper(model, device)
    
    # Permutation importance (sur un échantillon pour accélérer)
    sample_size = min(5000, len(X_test))
    indices = np.random.choice(len(X_test), sample_size, replace=False)
    X_sample = X_test[indices]
    y_sample = y_test[indices]
    
    result = permutation_importance(wrapper, X_sample, y_sample, 
                                   n_repeats=10, random_state=42, n_jobs=-1)
    
    # Top features
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': result.importances_mean
    }).sort_values('importance', ascending=False)
    
    print("\nTop 15 features les plus importantes:")
    print(importance_df.head(15).to_string(index=False))
    
    # Visualisation
    plt.figure(figsize=(12, 8))
    top_features = importance_df.head(20)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance')
    plt.title('Top 20 Features - Permutation Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300)
    print("✓ Importance des features sauvegardée: feature_importance.png")
    
    return importance_df

# ==================== MODE INFÉRENCE ====================
def predict_new_patients(model_path, scaler_path, encoders_path, input_csv, device):
    """Prédit le profil cognitif de nouveaux patients"""
    print(f"\n🔮 Mode Inférence - {input_csv}")
    
    # Charger le modèle
    checkpoint = torch.load(model_path, map_location=device)
    input_dim = checkpoint['network.0.weight'].shape[1]
    output_dim = checkpoint[list(checkpoint.keys())[-1]].shape[0]
    
    model = CognitiveNet(input_dim, config.HIDDEN_DIMS, output_dim, config.DROPOUT)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    
    # Charger scaler et encoders
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(encoders_path, 'rb') as f:
        label_encoders = pickle.load(f)
    
    # Charger les données
    df = pd.read_csv(input_csv)
    print(f"   {len(df)} nouveaux patients à prédire")
    
    # Prétraitement (simplifié)
    X = df.select_dtypes(include=[np.number]).fillna(0)
    X_scaled = scaler.transform(X)
    
    # Prédiction
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_scaled).to(device)
        with autocast():
            outputs = model(X_tensor)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(outputs, dim=1)
    
    # Résultats
    df['predicted_class'] = preds.cpu().numpy()
    df['confidence'] = probs.max(dim=1)[0].cpu().numpy()
    
    output_path = 'predictions.csv'
    df.to_csv(output_path, index=False)
    print(f"✓ Prédictions sauvegardées: {output_path}")
    
    return df

# ==================== MAIN ====================
def main(args):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"🧠 CognitiveAI - Prédiction du déclin cognitif")
    print(f"{'='*60}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} Go")
    print(f"{'='*60}\n")
    
    # Mode inférence
    if args.predict:
        predict_new_patients(config.MODEL_PATH, config.SCALER_PATH, 
                           config.ENCODER_PATH, args.predict, device)
        return
    
    # ===== PIPELINE D'ENTRAÎNEMENT =====
    
    # 1. Chargement et sélection
    df = load_and_select_features(config.DATA_PATH, config.KEYWORDS)
    
    # 2. Nettoyage
    df = clean_data(df)
    
    # 3. Préparation cible
    X, y, le_target = prepare_target_variable(df)
    
    # 4. Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE,
        stratify=y
    )
    print(f"\n📊 Split: Train={len(X_train):,}, Test={len(X_test):,}")
    
    # 5. Prétraitement
    X_train_scaled, X_test_scaled, scaler, label_encoders = preprocess_features(
        X_train.copy(), X_test.copy(), y_train
    )
    
    # Sauvegarde des preprocessors
    with open(config.SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    with open(config.ENCODER_PATH, 'wb') as f:
        pickle.dump(label_encoders, f)
    print(f"✓ Scaler et encoders sauvegardés")
    
    # 6. DataLoaders
    train_dataset = CognitiveDataset(X_train_scaled, y_train)
    test_dataset = CognitiveDataset(X_test_scaled, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,
                             shuffle=True, num_workers=config.NUM_WORKERS,
                             pin_memory=config.PIN_MEMORY)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE,
                            shuffle=False, num_workers=config.NUM_WORKERS,
                            pin_memory=config.PIN_MEMORY)
    
    # Split validation
    val_size = int(0.2 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    train_loader_split = DataLoader(train_subset, batch_size=config.BATCH_SIZE,
                                   shuffle=True, num_workers=config.NUM_WORKERS,
                                   pin_memory=config.PIN_MEMORY)
    val_loader = DataLoader(val_subset, batch_size=config.BATCH_SIZE,
                          shuffle=False, num_workers=config.NUM_WORKERS,
                          pin_memory=config.PIN_MEMORY)
    
    # 7. Modèle
    input_dim = X_train_scaled.shape[1]
    output_dim = len(np.unique(y))
    
    model = CognitiveNet(input_dim, config.HIDDEN_DIMS, output_dim, config.DROPOUT)
    model.to(device)
    
    print(f"\n🏗️ Architecture du modèle:")
    print(f"   Input: {input_dim} features")
    print(f"   Hidden: {config.HIDDEN_DIMS}")
    print(f"   Output: {output_dim} classes")
    print(f"   Paramètres: {sum(p.numel() for p in model.parameters()):,}")

    # Calculer les poids pour le rééquilibrage des classes
    print("\n⚖️ Calcul des poids pour rééquilibrage des classes...")
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weights = torch.FloatTensor(class_weights).to(device)
    print(f"   Poids: {dict(zip(range(len(class_weights)), class_weights.cpu().numpy()))}")
    
    # 8. Entraînement
    history = train_model(model, train_loader_split, val_loader, device, config, class_weights=class_weights)
    
    # 9. Visualisation historique
    plot_training_history(history)
    
    # 10. Chargement du meilleur modèle
    model.load_state_dict(torch.load(config.MODEL_PATH))
    
    # 11. Évaluation
    acc, recall, f1 = evaluate_model(model, test_loader, device, le_target)
    
    # 12. Feature importance
    feature_names = X_train.columns.tolist()
    importance_df = compute_feature_importance(model, X_test_scaled, y_test, 
                                              feature_names, device)
    
    print(f"\n{'='*60}")
    print("✅ Entraînement terminé avec succès!")
    print(f"   Modèle: {config.MODEL_PATH}")
    print(f"   Accuracy: {acc:.4f}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CognitiveAI')
    parser.add_argument('--predict', type=str, help='Fichier CSV pour inférence')
    args = parser.parse_args()
    
    main(args)