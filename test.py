#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Execute file for Credit Card Fraud Detection Project.
This script runs the DNN model using 5-fold cross-validation on the original imbalanced dataset,
training on 4 folds and testing on 1 fold, averaging results across folds, and generating visualizations.
"""
import sys
import os
# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from model.dnn_model import CreditCardFraudDNN
from preprocessing.preprocess import CreditCardFraudPreprocessor

# Set the paths
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = BASE_DIR.parent / "outputs"
DNN_OUTPUT_DIR = OUTPUT_DIR / "dnn_model"

def main():
    """Main execution function"""
    print("\n" + "="*70)
    print("CREDIT CARD FRAUD DETECTION - DNN MODEL WITH 5-FOLD CROSS-VALIDATION")
    print("="*70)
    
    # Load and preprocess data
    print("\nLoading and preprocessing data...")
    preprocessor = CreditCardFraudPreprocessor(download_data=True)
    X, y = preprocessor.preprocess()
    
    # Initialize 5-fold cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Lists to store metrics across folds
    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1_score': [],
        'roc_auc': [],
        'pr_auc': [],
        'tp': [],
        'fp': [],
        'fn': [],
        'tn': [],
        'tpr': [],
        'fpr': [],
        'tnr': [],
        'fnr': [],
        'best_threshold': []
    }
    
    # Perform 5-fold cross-validation
    fold = 1
    for train_index, test_index in skf.split(X, y):
        print(f"\n{'='*50}")
        print(f"Fold {fold}")
        print(f"{'='*50}")
        
        # Split data
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        print(f"Training set size: {X_train.shape[0]} samples")
        print(f"Testing set size: {X_test.shape[0]} samples")
        print(f"Fraud cases in test set: {sum(y_test)}")
        
        # Apply SMOTE to training data
        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        print(f"After SMOTE, training set size: {X_train_smote.shape[0]} samples")
        print(f"Class distribution after SMOTE: {np.bincount(y_train_smote)}")
        
        # Initialize and train model
        nn_model = CreditCardFraudDNN(output_dir=str(DNN_OUTPUT_DIR), validation_split=0.2)
        nn_model.train_model(X_train_smote, y_train_smote, epochs=100, batch_size=32, verbose=1)
        
        # Evaluate model
        fold_metrics = nn_model.evaluate_model(X_test, y_test, fold=fold)
        
        # Store metrics
        for key in metrics:
            metrics[key].append(fold_metrics[key])
        
        fold += 1
    
    # Calculate and print average metrics
    print("\n" + "="*70)
    print("AVERAGE PERFORMANCE ACROSS 5 FOLDS")
    print("="*70)
    for key in metrics:
        avg = np.mean(metrics[key])
        std = np.std(metrics[key])
        print(f"{key.replace('_', ' ').title()}: {avg:.4f} ± {std:.4f}")

if __name__ == "__main__":
    main()
