
import numpy as np
import pandas as pd
import os
import glob
import warnings
warnings.filterwarnings('ignore')

from scipy import signal
from scipy.signal import welch
from scipy.stats import skew, kurtosis, entropy
import pywt

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, classification_report, 
                           confusion_matrix, f1_score, recall_score, precision_score,
                           mean_absolute_error, mean_squared_error, r2_score,
                           roc_auc_score, roc_curve)

from xgboost import XGBClassifier, XGBRegressor
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN

import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds for reproducibility
np.random.seed(42)


# CONFIGURATION - ADAPTED TO  DATASET


BASE_PATH = "C:/Users/aliak/OneDrive/Desktop/ClassTask/dataset"
FS = 200  

# Updated to match actual directory names
ATTACHMENT_MAGNITUDES = {
    'Attachment1': 4.2,
    'Attachment2': 5.0,
    'Attachment3': 6.0,
    'Attachment4': 6.4,
    'Attachment5': 7.0,
    'Attachment6': 7.4,
    'Attachment7': 8.0,
}


# DATA LOADING UTILITIES 

def load_signal_file(file_path):
    """Load a single signal file (txt or csv)."""
    try:
        if file_path.endswith('.txt'):
            # Try different delimiters for txt files
            try:
                data = np.loadtxt(file_path)
            except:
                # Try with different delimiters
                try:
                    data = pd.read_csv(file_path, sep=r'\s+', header=None).values.flatten()
                except:
                    data = pd.read_csv(file_path, delimiter=',', header=None).values.flatten()
        elif file_path.endswith('.csv'):
            # Try different CSV formats
            try:
                data = pd.read_csv(file_path, header=None).values.flatten()
            except:
                data = pd.read_csv(file_path).values.flatten()
        elif file_path.endswith('.xls') or file_path.endswith('.xlsx'):
            data = pd.read_excel(file_path).values.flatten()
        else:
            print(f"Unsupported file format: {file_path}")
            return None
        
        # Ensure it's a 1D array
        if len(data.shape) > 1:
            data = data.flatten()
        
        # Remove any NaN values
        data = data[~np.isnan(data)]
        
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def load_attachment_signals(attachment_path):
    """Load all station signals from an attachment folder."""
    signals = []
    
    if not os.path.exists(attachment_path):
        print(f"Attachment path does not exist: {attachment_path}")
        return signals
    
    # Look for all data files
    txt_files = glob.glob(os.path.join(attachment_path, "*.txt"))
    csv_files = glob.glob(os.path.join(attachment_path, "*.csv"))
    xls_files = glob.glob(os.path.join(attachment_path, "*.xls"))
    xlsx_files = glob.glob(os.path.join(attachment_path, "*.xlsx"))
    all_files = txt_files + csv_files + xls_files + xlsx_files
    
    if not all_files:
        print(f"  No data files found directly in {os.path.basename(attachment_path)}")
    else:
        for file_path in all_files:
            signal_data = load_signal_file(file_path)
            if signal_data is not None and len(signal_data) > 0:
                signals.append(signal_data)
    
    return signals


# FEATURE EXTRACTION (ENHANCED WITH DISCRIMINATIVE FEATURES)


class SeismicFeatureExtractor:
    """Extract features from seismic waveforms with focus on discrimination."""
    
    def __init__(self, fs=200):
        self.fs = fs
        self.feature_names = []
        
    def preprocess_signal(self, x):
        """Basic preprocessing for seismic signals."""
        if len(x) < 10:
            return np.zeros(1000)
        
        # Remove DC offset
        x = x - np.mean(x)
        
        # Bandpass filter (0.5-20 Hz for seismic signals)
        nyquist = self.fs / 2
        low = 0.5 / nyquist
        high = 20.0 / nyquist
        b, a = signal.butter(4, [low, high], btype='band')
        x = signal.filtfilt(b, a, x)
        
        # Normalize
        max_val = np.max(np.abs(x))
        if max_val > 0:
            x = x / max_val
        
        return x
    
    def extract_discriminative_features(self, signal_data):
        """Extract features that specifically help distinguish natural vs non-natural events."""
        features = {}
        
        # Skip if signal is too short
        if len(signal_data) < 100:
            signal_data = np.zeros(1000)
        
        # Preprocess
        signal_clean = self.preprocess_signal(signal_data)
        
        # TIME DOMAIN DISCRIMINATIVE FEATURES 
        features['mean'] = np.mean(signal_clean)
        features['std'] = np.std(signal_clean)
        features['variance'] = np.var(signal_clean)
        features['skewness'] = skew(signal_clean) if len(signal_clean) > 0 else 0
        features['kurtosis'] = kurtosis(signal_clean) if len(signal_clean) > 0 else 0
        
        # Amplitude features (explosions often have higher initial amplitude)
        abs_signal = np.abs(signal_clean)
        features['max_amp'] = np.max(abs_signal)
        features['rms'] = np.sqrt(np.mean(signal_clean**2))
        
        # Peak-to-RMS ratio (higher for impulsive sources like explosions)
        features['peak_to_rms'] = features['max_amp'] / (features['rms'] + 1e-8)
        
        # Signal complexity features
        features['zero_crossings'] = np.sum(np.diff(np.sign(signal_clean)) != 0)
        features['zero_crossing_rate'] = features['zero_crossings'] / len(signal_clean)
        
        # Energy distribution (explosions often have energy concentrated early)
        quarter_point = len(signal_clean) // 4
        half_point = len(signal_clean) // 2
        energy_first_quarter = np.sum(signal_clean[:quarter_point]**2)
        energy_second_quarter = np.sum(signal_clean[quarter_point:half_point]**2)
        energy_total = np.sum(signal_clean**2) + 1e-8
        
        features['energy_ratio_first_quarter'] = energy_first_quarter / energy_total
        features['energy_ratio_second_quarter'] = energy_second_quarter / energy_total
        features['energy_concentration'] = energy_first_quarter / (energy_total - energy_first_quarter + 1e-8)
        
        # FREQUENCY DOMAIN DISCRIMINATIVE FEATURES 
        if len(signal_clean) >= 256:
            nperseg = min(256, len(signal_clean)//4)
            freqs, psd = welch(signal_clean, fs=self.fs, nperseg=nperseg)
            
            total_energy = np.sum(psd) + 1e-8
            features['spectral_centroid'] = np.sum(freqs * psd) / total_energy
            features['dominant_freq'] = freqs[np.argmax(psd)]
            
            # Natural earthquakes typically have more low-frequency content
            # Explosions typically have more high-frequency content
            bands = {
                'VLF': (0.5, 3),    # Very Low Frequency (earthquake dominant)
                'LF': (3, 10),      # Low Frequency  
                'MF': (10, 20),     # Medium Frequency
                'HF': (20, 50),     # High Frequency (explosion dominant)
            }
            
            for band_name, (low, high) in bands.items():
                mask = (freqs >= low) & (freqs <= high)
                if np.any(mask):
                    band_energy = np.sum(psd[mask])
                    features[f'{band_name}_energy'] = band_energy
                    features[f'{band_name}_energy_ratio'] = band_energy / total_energy
                else:
                    features[f'{band_name}_energy'] = 0
                    features[f'{band_name}_energy_ratio'] = 0
            
            # Key discriminative ratio: High/Low frequency ratio
            if 'HF_energy' in features and 'VLF_energy' in features:
                features['HF_VLF_ratio'] = features['HF_energy'] / (features['VLF_energy'] + 1e-8)
            
            # Spectral flatness (explosions often have flatter spectra)
            features['spectral_flatness'] = np.exp(np.mean(np.log(psd + 1e-10))) / np.mean(psd)
            
            # Spectral entropy (earthquakes typically have higher entropy)
            psd_norm = psd / np.sum(psd)
            features['spectral_entropy'] = entropy(psd_norm)
            
            # Spectral roll-off (85% energy frequency)
            features['spectral_rolloff'] = freqs[np.where(np.cumsum(psd) >= 0.85 * total_energy)[0][0]]
            
        else:
            # Default values for short signals
            for band in ['VLF', 'LF', 'MF', 'HF']:
                features[f'{band}_energy'] = 0
                features[f'{band}_energy_ratio'] = 0
            features['spectral_centroid'] = 0
            features['dominant_freq'] = 0
            features['spectral_flatness'] = 0
            features['spectral_entropy'] = 0
            features['spectral_rolloff'] = 0
            features['HF_VLF_ratio'] = 0
        
        # WAVELET FEATURES 
        if len(signal_clean) >= 64:
            try:
                coeffs = pywt.wavedec(signal_clean, 'db4', level=min(4, pywt.dwt_max_level(len(signal_clean), 'db4')))
                wavelet_energies = [np.sum(c**2) for c in coeffs]
                total_wavelet_energy = np.sum(wavelet_energies) + 1e-8
                
                # Low-level coefficients capture high frequencies (explosions)
                # High-level coefficients capture low frequencies (earthquakes)
                for i, energy in enumerate(wavelet_energies):
                    features[f'wavelet_energy_lvl{i}'] = energy
                    features[f'wavelet_energy_ratio_lvl{i}'] = energy / total_wavelet_energy
                
                # Key discriminative feature: High-frequency wavelet energy ratio
                if len(wavelet_energies) >= 2:
                    features['wavelet_HF_ratio'] = wavelet_energies[0] / (sum(wavelet_energies[1:]) + 1e-8)
                
                # Fill missing levels
                for i in range(len(wavelet_energies), 5):
                    features[f'wavelet_energy_lvl{i}'] = 0
                    features[f'wavelet_energy_ratio_lvl{i}'] = 0
                
            except:
                for i in range(5):
                    features[f'wavelet_energy_lvl{i}'] = 0
                    features[f'wavelet_energy_ratio_lvl{i}'] = 0
                features['wavelet_HF_ratio'] = 0
        else:
            for i in range(5):
                features[f'wavelet_energy_lvl{i}'] = 0
                features[f'wavelet_energy_ratio_lvl{i}'] = 0
            features['wavelet_HF_ratio'] = 0
        
        #  SHAPE-BASED DISCRIMINATIVE FEATURES 
        # Rise time (explosions typically have faster rise)
        abs_signal = np.abs(signal_clean)
        peak_idx = np.argmax(abs_signal)
        if peak_idx > 0:
            # Time from start to 90% of peak
            peak_val = abs_signal[peak_idx]
            threshold = 0.1 * peak_val
            rise_start = np.where(abs_signal[:peak_idx] <= threshold)[0]
            if len(rise_start) > 0:
                rise_time = (peak_idx - rise_start[-1]) / self.fs
                features['rise_time'] = rise_time
            else:
                features['rise_time'] = peak_idx / self.fs
        else:
            features['rise_time'] = 0
        
        # Decay time (explosions decay faster)
        if peak_idx < len(signal_clean) - 1:
            decay_threshold = 0.1 * peak_val
            decay_end = np.where(abs_signal[peak_idx:] <= decay_threshold)[0]
            if len(decay_end) > 0:
                decay_time = decay_end[0] / self.fs
                features['decay_time'] = decay_time
            else:
                features['decay_time'] = (len(signal_clean) - peak_idx) / self.fs
        else:
            features['decay_time'] = 0
        
        features['rise_decay_ratio'] = features['rise_time'] / (features['decay_time'] + 1e-8)
        
        # Store feature names
        self.feature_names = list(features.keys())
        
        return features
    
    def extract_features(self, signal_data):
        """Wrapper for backward compatibility."""
        return self.extract_discriminative_features(signal_data)


# TASK 1: NATURAL VS NON-NATURAL CLASSIFICATION (FIXED FOR IMBALANCE)


def perform_task1_fixed(base_path, attachments):
    """Perform Task 1 with comprehensive imbalance handling."""
    print("\n" + "="*80)
    print("TASK 1: Natural vs Non-natural Earthquake Classification")
    print("WITH ADVANCED IMBALANCE HANDLING")
    print("="*80)
    
    # Initialize feature extractor
    extractor = SeismicFeatureExtractor(fs=FS)
    
    X = []
    y = []
    event_info = []
    
    print("\nLoading natural earthquakes (Attachments 1-7):")
    
    # Process natural earthquakes
    for i in range(1, 8):
        att_name = f'Attachment{i}'
        if att_name in attachments:
            print(f"\nProcessing {att_name}...")
            attachment_info = attachments[att_name]
            station_signals = load_attachment_signals(attachment_info['path'])
            
            if station_signals:
                print(f"  Loaded {len(station_signals)} signals")
                
                # Process EACH station as a separate sample
                for j, signal_data in enumerate(station_signals):
                    features_dict = extractor.extract_features(signal_data)
                    features = list(features_dict.values())
                    X.append(features)
                    y.append(1)  # Natural earthquake
                
                event_info.append({
                    'name': att_name,
                    'type': 'natural',
                    'magnitude': ATTACHMENT_MAGNITUDES.get(att_name, 0),
                    'n_stations': len(station_signals)
                })
                
                print(f"   Added {len(station_signals)} samples from {att_name} (M{ATTACHMENT_MAGNITUDES.get(att_name, 0)})")
            else:
                print(f"   No signals loaded from {att_name}")
        else:
            print(f"   {att_name} not found in dataset")
    
    print("\n" + "-"*80)
    print("Loading non-natural events (Attachment 8):")
    
    # Process non-natural event
    if 'Attachment8' in attachments:
        att_name = 'Attachment8'
        attachment_info = attachments[att_name]
        station_signals = load_attachment_signals(attachment_info['path'])
        
        if station_signals:
            print(f"  Loaded {len(station_signals)} signals")
            
            # Process EACH station as a separate sample
            for j, signal_data in enumerate(station_signals):
                features_dict = extractor.extract_features(signal_data)
                features = list(features_dict.values())
                X.append(features)
                y.append(0)  # Non-natural event
            
            event_info.append({
                'name': att_name,
                'type': 'non-natural',
                'magnitude': None,
                'n_stations': len(station_signals)
            })
            
            print(f"  ✓ Added {len(station_signals)} samples from {att_name}")
        else:
            print(f"  ✗ No signals loaded from {att_name}")
    else:
        print("  ✗ Attachment8 not found in dataset")
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    print(f"\n" + "="*80)
    print("TASK 1 DATA SUMMARY")
    print("="*80)
    print(f"Total samples loaded: {len(X)}")
    print(f"Natural earthquake samples: {np.sum(y == 1)}")
    print(f"Non-natural event samples: {np.sum(y == 0)}")
    print(f"Feature vector size: {X.shape[1]}")
    
    print("\nEvent Details:")
    for info in event_info:
        mag_str = f"M{info['magnitude']}" if info['magnitude'] else "No magnitude"
        print(f"  {info['name']}: {info['type']}, Stations: {info['n_stations']}, {mag_str}")
    
    # ADVANCED IMBALANCE HANDLING 
    print("\n" + "="*80)
    print("CLASS IMBALANCE ANALYSIS & HANDLING")
    print("="*80)
    
    natural_count = np.sum(y == 1)
    non_natural_count = np.sum(y == 0)
    imbalance_ratio = natural_count / (non_natural_count + 1e-8)
    
    print(f"Natural samples: {natural_count}")
    print(f"Non-natural samples: {non_natural_count}")
    print(f"Imbalance ratio: {imbalance_ratio:.2f}:1")
    
    if imbalance_ratio > 3:
        print(f"\n  Severe class imbalance detected! Applying balancing techniques...")
        
        # Strategy 1: Apply SMOTE for oversampling minority class
        if non_natural_count >= 5:  # Need at least 5 samples for SMOTE
            print("  Applying SMOTE (Synthetic Minority Over-sampling Technique)...")
            smote = SMOTE(random_state=42, k_neighbors=min(5, non_natural_count-1))
            X_balanced, y_balanced = smote.fit_resample(X, y)
            print(f"  After SMOTE: {np.sum(y_balanced == 1)} natural, {np.sum(y_balanced == 0)} non-natural")
        else:
            print("  Too few non-natural samples for SMOTE, using class weights instead")
            X_balanced, y_balanced = X, y
            
        # Strategy 2: Use ensemble of balanced subsets
        print("  Creating balanced ensemble training subsets...")
        
    else:
        print("\n✓ Class imbalance is manageable. Using standard approach.")
        X_balanced, y_balanced = X, y
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_balanced)
    
    # Use STRATIFIED train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_balanced, test_size=0.3, random_state=42, stratify=y_balanced
    )
    
    print(f"\nTraining set: {X_train.shape}")
    print(f"  Natural: {np.sum(y_train == 1)}, Non-natural: {np.sum(y_train == 0)}")
    print(f"Test set: {X_test.shape}")
    print(f"  Natural: {np.sum(y_test == 1)}, Non-natural: {np.sum(y_test == 0)}")
    
    # TRAIN MODELS WITH IMBALANCE HANDLING 
    print("\n" + "="*80)
    print("TRAINING CLASSIFICATION MODELS")
    print("WITH IMBALANCE-SPECIFIC PARAMETERS")
    print("="*80)
    
    # Calculate class weights
    class_weight_ratio = np.sum(y_train == 1) / np.sum(y_train == 0)
    
    models = {
        'RandomForest_Balanced': RandomForestClassifier(
            n_estimators=200, 
            random_state=42, 
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            n_jobs=-1
        ),
        'RandomForest_CustomWeight': RandomForestClassifier(
            n_estimators=200, 
            random_state=42, 
            max_depth=10,
            class_weight={0: class_weight_ratio, 1: 1},  # Higher weight for minority
            n_jobs=-1
        ),
        'XGBoost_Balanced': XGBClassifier(
            n_estimators=200, 
            learning_rate=0.1, 
            random_state=42, 
            max_depth=5,
            scale_pos_weight=class_weight_ratio,
            eval_metric='logloss',
            use_label_encoder=False
        ),
    }
    
    results = {}
    for name, model in models.items():
        try:
            print(f"\nTraining {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='binary')
            recall = recall_score(y_test, y_pred, average='binary')
            precision = precision_score(y_test, y_pred, average='binary')
            
            # Calculate metrics specifically for minority class
            f1_minority = f1_score(y_test, y_pred, pos_label=0)
            recall_minority = recall_score(y_test, y_pred, pos_label=0)
            precision_minority = precision_score(y_test, y_pred, pos_label=0)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'f1': f1,
                'f1_minority': f1_minority,
                'recall': recall,
                'recall_minority': recall_minority,
                'precision': precision,
                'precision_minority': precision_minority,
                'predictions': y_pred,
                'predictions_proba': y_pred_proba,
                'scaler': scaler
            }
            
            print(f"  Overall Accuracy: {accuracy:.4f}")
            print(f"  Overall F1 Score: {f1:.4f}")
            print(f"  Non-natural (minority) F1: {f1_minority:.4f}")
            print(f"  Non-natural Recall: {recall_minority:.4f}")
            print(f"  Non-natural Precision: {precision_minority:.4f}")
            
            # Print detailed classification report
            print(f"\n  Classification Report:")
            report = classification_report(y_test, y_pred, target_names=['Non-natural', 'Natural'])
            for line in report.split('\n'):
                if line.strip():
                    print(f"    {line}")
                    
        except Exception as e:
            print(f"\nError training {name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not results:
        print("\nNo models were successfully trained!")
        return None, None, None
    
    # Select best model based on minority class F1 score
    best_model_name = max(results, key=lambda x: results[x]['f1_minority'])
    best_model = results[best_model_name]['model']
    
    print(f"\n" + "="*80)
    print(f"BEST MODEL: {best_model_name}")
    print(f"Best Non-natural F1 Score: {results[best_model_name]['f1_minority']:.4f}")
    
    # Confusion matrix for best model
    y_pred_best = results[best_model_name]['predictions']
    cm = confusion_matrix(y_test, y_pred_best)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=['Non-natural', 'Natural'],
               yticklabels=['Non-natural', 'Natural'],
               annot_kws={"size": 16})
    plt.title(f'Confusion Matrix - {best_model_name}\n(Non-natural F1: {results[best_model_name]["f1_minority"]:.3f})', 
              fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig('task1_confusion_matrix_fixed.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ROC Curve if probabilities are available
    if results[best_model_name]['predictions_proba'] is not None:
        fpr, tpr, thresholds = roc_curve(y_test, results[best_model_name]['predictions_proba'])
        roc_auc = roc_auc_score(y_test, results[best_model_name]['predictions_proba'])
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curve - {best_model_name}', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('task1_roc_curve.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # Feature importance
    if hasattr(best_model, 'feature_importances_'):
        print("\n" + "="*80)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*80)
        
        importance = best_model.feature_importances_
        
        # Create feature names
        feature_names = extractor.feature_names[:len(importance)]
        if len(feature_names) < len(importance):
            feature_names = feature_names + [f'feature_{i}' for i in range(len(feature_names), len(importance))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        print("\nTop 20 Most Important Features for Discrimination:")
        print(importance_df.head(20).to_string(index=False))
        
        # Plot feature importance
        plt.figure(figsize=(14, 10))
        top_n = min(25, len(importance_df))
        top_features = importance_df.head(top_n)
        
        colors = ['red' if 'HF' in feat or 'high' in feat.lower() else 
                 'blue' if 'VLF' in feat or 'low' in feat.lower() else 
                 'green' for feat in top_features['feature']]
        
        plt.barh(range(len(top_features)), top_features['importance'], color=colors)
        plt.yticks(range(len(top_features)), top_features['feature'], fontsize=10)
        plt.xlabel('Importance', fontsize=12)
        plt.title(f'Top {top_n} Features for Earthquake Classification\n(Red: High-freq features, Blue: Low-freq features)', 
                  fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('task1_feature_importance_fixed.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save feature importance to CSV
        importance_df.to_csv('task1_feature_importance.csv', index=False)
        print("\nFeature importance saved to 'task1_feature_importance.csv'")
    
    # CROSS-VALIDATION FOR ROBUSTNESS 
    print("\n" + "="*80)
    print("CROSS-VALIDATION FOR MODEL ROBUSTNESS")
    print("="*80)
    
    # Use stratified k-fold for imbalance
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    cv_scores = {}
    for name, model in list(models.items())[:2]:  # Test first 2 models
        try:
            # Cross-validation scores
            cv_accuracy = cross_val_score(model, X_scaled, y_balanced, cv=cv, scoring='accuracy', n_jobs=-1)
            cv_f1 = cross_val_score(model, X_scaled, y_balanced, cv=cv, scoring='f1', n_jobs=-1)
            
            cv_scores[name] = {
                'accuracy_mean': cv_accuracy.mean(),
                'accuracy_std': cv_accuracy.std(),
                'f1_mean': cv_f1.mean(),
                'f1_std': cv_f1.std()
            }
            
            print(f"\n{name} Cross-Validation:")
            print(f"  Accuracy: {cv_accuracy.mean():.4f} (±{cv_accuracy.std():.4f})")
            print(f"  F1 Score: {cv_f1.mean():.4f} (±{cv_f1.std():.4f})")
            
        except Exception as e:
            print(f"\nError in CV for {name}: {e}")
            continue
    
    #  ANALYSIS OF MISCLASSIFICATIONS 
    print("\n" + "="*80)
    print("ANALYSIS OF MISCLASSIFICATIONS")
    print("="*80)
    
    # Identify misclassified samples
    misclassified_indices = np.where(y_test != y_pred_best)[0]
    
    if len(misclassified_indices) > 0:
        print(f"\nNumber of misclassifications: {len(misclassified_indices)}")
        
        # Check if misclassifications are mostly minority class
        misclassified_labels = y_test[misclassified_indices]
        misclassified_predictions = y_pred_best[misclassified_indices]
        
        false_positives = np.sum((misclassified_labels == 0) & (misclassified_predictions == 1))
        false_negatives = np.sum((misclassified_labels == 1) & (misclassified_predictions == 0))
        
        print(f"  False Positives (Non-natural predicted as Natural): {false_positives}")
        print(f"  False Negatives (Natural predicted as Non-natural): {false_negatives}")
        
        if false_positives > 0:
            print("\n Some non-natural events are being misclassified as natural.")
            print("  This suggests the features may not capture key differences adequately.")
            print("  Consider adding more discriminative features or collecting more non-natural samples.")
    else:
        print("\n✓ Perfect classification on test set!")
    
    return results, X, y


# TASK 2: MAGNITUDE PREDICTION (UNCHANGED - USE ORIGINAL)


def perform_task2(base_path, attachments):
    """Perform Task 2: Magnitude Prediction using station-level data."""
    print("\n" + "="*80)
    print("TASK 2: Magnitude Prediction")
    print("="*80)
    
    # Initialize feature extractor
    extractor = SeismicFeatureExtractor(fs=FS)
    
    X = []
    y = []
    station_info = []
    
    print("\nLoading natural earthquakes for magnitude prediction (Attachments 1-7):")
    
    # Process ALL stations from each attachment
    for i in range(1, 8):
        att_name = f'Attachment{i}'
        if att_name in attachments:
            print(f"\nProcessing {att_name}...")
            attachment_info = attachments[att_name]
            station_signals = load_attachment_signals(attachment_info['path'])
            
            if station_signals:
                print(f"  Loaded {len(station_signals)} signals")
                
                # Process each station separately
                for j, signal_data in enumerate(station_signals):
                    features_dict = extractor.extract_features(signal_data)
                    features = list(features_dict.values())
                    X.append(features)
                    y.append(ATTACHMENT_MAGNITUDES[att_name])
                    
                    station_info.append({
                        'attachment': att_name,
                        'station': j+1,
                        'magnitude': ATTACHMENT_MAGNITUDES[att_name]
                    })
                
                print(f"   Added {len(station_signals)} station samples from {att_name} (M{ATTACHMENT_MAGNITUDES[att_name]})")
            else:
                print(f"   No signals loaded from {att_name}")
        else:
            print(f"   {att_name} not found in dataset")
    
    # Check if we have enough data
    if len(X) < 10:
        print(f"\nInsufficient data for magnitude prediction: {len(X)} station samples")
        return None, None
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    print(f"\n" + "="*80)
    print("TASK 2 DATA SUMMARY")
    print("="*80)
    print(f"Total station samples: {len(X)}")
    print(f"Magnitude range: {y.min():.1f} to {y.max():.1f}")
    print(f"Feature vector size: {X.shape[1]}")
    
    # Train magnitude prediction model
    print("\n" + "="*80)
    print("TRAINING MAGNITUDE PREDICTION MODEL")
    print("="*80)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Try multiple regression models
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5),
        'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, max_depth=3),
    }
    
    results = {}
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'predictions': y_pred,
                'scaler': scaler
            }
            
            print(f"\n{name}:")
            print(f"  MAE:  {mae:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  R²:   {r2:.4f}")
            
        except Exception as e:
            print(f"\nError training {name}: {e}")
            continue
    
    if not results:
        print("\nNo models were successfully trained!")
        return None, None
    
    # Select best model based on MAE
    best_model_name = min(results, key=lambda x: results[x]['mae'])
    best_model = results[best_model_name]['model']
    best_scaler = results[best_model_name]['scaler']
    
    print(f"\n" + "="*80)
    print(f"BEST MODEL: {best_model_name}")
    print(f"Best MAE: {results[best_model_name]['mae']:.4f}")
    print(f"Best R²: {results[best_model_name]['r2']:.4f}")
    
    # Plot predictions vs true values
    y_pred_best = results[best_model_name]['predictions']
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_best, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(y_test.min(), y_pred_best.min())
    max_val = max(y_test.max(), y_pred_best.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    plt.xlabel('True Magnitude', fontsize=12)
    plt.ylabel('Predicted Magnitude', fontsize=12)
    plt.title(f'Magnitude Prediction: {best_model_name}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Add error statistics to plot
    stats_text = f'MAE: {results[best_model_name]["mae"]:.3f}\n'
    stats_text += f'RMSE: {results[best_model_name]["rmse"]:.3f}\n'
    stats_text += f'R²: {results[best_model_name]["r2"]:.3f}\n'
    stats_text += f'Samples: {len(X_test)}'
    
    plt.text(0.05, 0.95, stats_text,
            transform=plt.gca().transAxes,
            verticalalignment='top',
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('task2_magnitude_prediction.png')
    plt.show()
    
    # Predict Attachment 9
    print("\n" + "="*80)
    print("PREDICTING MAGNITUDES FOR ATTACHMENT 9")
    print("="*80)
    
    if 'Attachment9' in attachments:
        att_name = 'Attachment9'
        attachment_info = attachments[att_name]
        station_signals = load_attachment_signals(attachment_info['path'])
        
        if station_signals:
            print(f"Loaded {len(station_signals)} signals from Attachment9")
            
            # Extract features for each station
            station_predictions = []
            station_features_list = []
            
            for j, signal_data in enumerate(station_signals):
                features_dict = extractor.extract_features(signal_data)
                features = list(features_dict.values())
                station_features_list.append(features)
            
            # Pad features to same length
            max_len = max(len(f) for f in station_features_list)
            station_features_padded = []
            for f in station_features_list:
                if len(f) < max_len:
                    f = f + [0] * (max_len - len(f))
                station_features_padded.append(f)
            
            # Predict for each station
            predictions = []
            for features in station_features_padded:
                features = np.array(features).reshape(1, -1)
                features_scaled = best_scaler.transform(features)
                pred = best_model.predict(features_scaled)[0]
                predictions.append(pred)
            
            # Calculate statistics
            predictions_array = np.array(predictions)
            avg_prediction = np.mean(predictions_array)
            std_prediction = np.std(predictions_array)
            median_prediction = np.median(predictions_array)
            
            print(f"\nAttachment 9 Magnitude Prediction Results:")
            print(f"  Number of stations: {len(predictions)}")
            print(f"  Mean prediction: {avg_prediction:.2f}")
            print(f"  Median prediction: {median_prediction:.2f}")
            print(f"  Standard deviation: {std_prediction:.2f}")
            print(f"  Minimum: {predictions_array.min():.2f}")
            print(f"  Maximum: {predictions_array.max():.2f}")
            print(f"  95% confidence interval: [{avg_prediction - 1.96*std_prediction:.2f}, {avg_prediction + 1.96*std_prediction:.2f}]")
            
            # Save predictions
            pred_df = pd.DataFrame({
                'station': range(1, len(predictions) + 1),
                'predicted_magnitude': predictions
            })
            pred_df.loc['Statistics'] = ['Mean', avg_prediction]
            pred_df.loc['Statistics2'] = ['Std Dev', std_prediction]
            pred_df.loc['Statistics3'] = ['Median', median_prediction]
            pred_df.to_csv('attachment9_predictions.csv', index=False)
            print("\nPredictions saved to 'attachment9_predictions.csv'")
            
            # Plot distribution of predictions
            plt.figure(figsize=(10, 6))
            plt.hist(predictions, bins=10, alpha=0.7, edgecolor='black')
            plt.axvline(avg_prediction, color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_prediction:.2f}')
            plt.axvline(median_prediction, color='green', linestyle='--', linewidth=2, label=f'Median: {median_prediction:.2f}')
            plt.xlabel('Predicted Magnitude', fontsize=12)
            plt.ylabel('Number of Stations', fontsize=12)
            plt.title('Distribution of Predicted Magnitudes for Attachment 9', fontsize=14)
            plt.legend(fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('attachment9_predictions_distribution.png')
            plt.show()
            
        else:
            print("   No signals loaded from Attachment9")
    else:
        print("   Attachment9 not found in dataset")
    
    return results, X


# TASK 3: RESERVOIR ATTRIBUTES MODELING (UNCHANGED - USE ORIGINAL)


def perform_task3_fixed(base_path, attachments):
    """Perform Task 3: Reservoir Attributes Modeling with proper Chinese column handling."""
    print("\n" + "="*80)
    print("TASK 3: Reservoir Attributes Modeling")
    print("="*80)
    
    # Check for reservoir data
    if 'ReservoirData' in attachments:
        print("\nLoading reservoir data from Excel file...")
        excel_path = attachments['ReservoirData']['path']
        
        try:
            # Read the Excel file
            df = pd.read_excel(excel_path)
            print(f"Successfully loaded {len(df)} rows from {os.path.basename(excel_path)}")
            
            # Rename Chinese columns to English for easier processing
            column_mapping = {
                '样本': 'sample_id',
                '库深/m': 'depth_m',
                '库容': 'capacity_10e6_m3',
                '断层类型': 'fault_type',
                '构造活动/基本烈度': 'tectonic_activity',
                '岩性': 'lithology',
                '震级': 'magnitude'
            }
            
            df_english = df.rename(columns=column_mapping)
            
            print(f"\nColumns (original -> English):")
            for cn, en in column_mapping.items():
                if cn in df.columns:
                    print(f"  {cn} -> {en}")
            
            print(f"\nFirst 5 rows (English columns):")
            print(df_english.head())
            
            df = df_english
            
        except Exception as e:
            print(f"Error reading Excel file: {e}")
            return None
    else:
        print("Reservoir data not found.")
        return None
    
    # Analyze the data
    print("\n" + "="*80)
    print("ANALYZING RESERVOIR DATA")
    print("="*80)
    
    # Basic statistics
    print("\n1. BASIC STATISTICS:")
    print(df.describe())
    
    # Prepare data for modeling
    print("\n2. PREPARING DATA FOR MODELING")
    
    # Encode categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns
    df_encoded = df.copy()
    
    for col in categorical_cols:
        encoder = LabelEncoder()
        df_encoded[f'{col}_encoded'] = encoder.fit_transform(df[col])
        print(f"  Encoded {col} -> {col}_encoded")
    
    # Select features (exclude original categorical, target, and sample_id)
    exclude_cols = list(categorical_cols) + ['magnitude', 'sample_id']
    feature_cols = [col for col in df_encoded.columns if col not in exclude_cols]
    
    print(f"\nSelected features: {feature_cols}")
    print(f"Target: magnitude")
    
    X = df_encoded[feature_cols].values
    y = df_encoded['magnitude'].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    print(f"\nTraining set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Train model
    print("\n3. TRAINING RESERVOIR MODEL")
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nModel Performance:")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²:   {r2:.4f}")
    
    # Feature importance
    importance = model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    print(f"\nFeature Importance:")
    print(importance_df.to_string(index=False))
    
    # Save results
    importance_df.to_csv('reservoir_feature_importance.csv', index=False)
    
    return {
        'model': model,
        'scaler': scaler,
        'performance': {'mae': mae, 'rmse': rmse, 'r2': r2},
        'feature_importance': importance_df,
    }


# DATASET SCANNING

def scan_and_load_dataset(base_path):
    """Scan and automatically identify dataset structure."""
    print("\n" + "="*80)
    print("SCANNING AND IDENTIFYING DATASET STRUCTURE")
    print("="*80)
    
    if not os.path.exists(base_path):
        print(f"ERROR: Base path does not exist: {base_path}")
        return {}
    
    print(f"Base path: {base_path}")
    
    # List all items in base path
    items = os.listdir(base_path)
    
    # Try to identify attachments
    attachments = {}
    
    for item in sorted(items):
        item_path = os.path.join(base_path, item)
        
        # Check for directories that look like attachments
        if os.path.isdir(item_path):
            if item.startswith('Attachment'):
                txt_count = len(glob.glob(os.path.join(item_path, "*.txt")))
                csv_count = len(glob.glob(os.path.join(item_path, "*.csv")))
                total_files = txt_count + csv_count
                
                if total_files > 0:
                    attachments[item] = {
                        'path': item_path,
                        'type': 'directory',
                        'txt_files': txt_count,
                        'csv_files': csv_count
                    }
        
        # Check for Excel files (likely reservoir data)
        elif os.path.isfile(item_path) and item.lower().endswith(('.xls', '.xlsx')):
            attachments['ReservoirData'] = {
                'path': item_path,
                'type': 'excel_file',
                'description': 'Reservoir attributes data'
            }
    
    # Print findings
    print(f"\nFound {len(attachments)} data sources:")
    for name, info in attachments.items():
        if info['type'] == 'directory':
            print(f"  {name}: {info['txt_files']} txt files, {info['csv_files']} csv files")
        else:
            print(f"   {name}: Excel file")
    
    return attachments


# MAIN EXECUTION


def main():
    """Main execution function."""
    print("=" * 80)
    print("EARTHQUAKE ANALYSIS PIPELINE - ADVANCED IMBALANCE HANDLING")
    print("=" * 80)
    print(f"Base Path: {BASE_PATH}")
    print("=" * 80)
    
    # Scan and identify dataset structure
    attachments = scan_and_load_dataset(BASE_PATH)
    
    if not attachments:
        print("\nERROR: No data found! Please check your BASE_PATH configuration.")
        print(f"Current BASE_PATH: {BASE_PATH}")
        return
    
    # Store results
    task_results = {}
    

    # TASK 1: Natural vs Non-natural Classification (FIXED)
    print("\n" + "=" * 80)
    print("STARTING TASK 1 (FIXED VERSION)")
    print("=" * 80)
    task1_results, X1, y1 = perform_task1_fixed(BASE_PATH, attachments)
    task_results['task1'] = task1_results
    
    
    # TASK 2: Magnitude Prediction
    
    print("\n" + "=" * 80)
    print("STARTING TASK 2")
    print("=" * 80)
    task2_results, X2 = perform_task2(BASE_PATH, attachments)
    task_results['task2'] = task2_results
    
    
    # TASK 3: Reservoir Attributes Modeling
    
    print("\n" + "=" * 80)
    print("STARTING TASK 3")
    print("=" * 80)
    task3_results = perform_task3_fixed(BASE_PATH, attachments)
    task_results['task3'] = task3_results
    
    
    # FINAL REPORT
    
    print("\n" + "=" * 80)
    print("FINAL SUMMARY REPORT")
    print("=" * 80)
    
    print("\n TASK 1: Classification Results (with imbalance handling)")
    print("-" * 50)
    if task1_results:
        # Find best model based on minority class F1
        best_f1_minority = max(r['f1_minority'] for r in task1_results.values())
        best_acc = max(r['accuracy'] for r in task1_results.values())
        print(f" Best Non-natural F1 Score: {best_f1_minority:.4f}")
        print(f" Best Accuracy: {best_acc:.4f}")
        if X1 is not None:
            print(f" Original samples: {X1.shape[0]}")
            print(f" Natural: {np.sum(y1 == 1)}, Non-natural: {np.sum(y1 == 0)}")
        print(f" Output files: task1_confusion_matrix_fixed.png, task1_feature_importance_fixed.png")
    else:
        print(" Task 1 not completed")
    
    print("\n TASK 2: Magnitude Prediction Results")
    print("-" * 50)
    if task2_results:
        best_mae = min(r['mae'] for r in task2_results.values())
        best_r2 = max(r['r2'] for r in task2_results.values())
        print(f" Best MAE: {best_mae:.4f}")
        print(f" Best R²: {best_r2:.4f}")
        if X2 is not None:
            print(f" Station samples: {X2.shape[0]}")
        print(f" Attachment 9 predictions generated")
        print(f" Output files: task2_magnitude_prediction.png, attachment9_predictions.csv")
    else:
        print("Task 2 not completed")
    
    print("\n TASK 3: Reservoir Analysis Results")
    print("-" * 50)
    if task3_results:
        print(f" Model MAE: {task3_results['performance']['mae']:.4f}")
        print(f" Model R²: {task3_results['performance']['r2']:.4f}")
        print(f"Feature importance analyzed")
        print(f" Output files: reservoir_feature_importance.csv")
    else:
        print(" Task 3 not completed")
    
    print("\n OUTPUT FILES GENERATED:")
    print("-" * 50)
    output_files = [
        'task1_confusion_matrix_fixed.png',
        'task1_feature_importance_fixed.png',
        'task1_feature_importance.csv',
        'task2_magnitude_prediction.png',
        'attachment9_predictions.csv',
        'attachment9_predictions_distribution.png',
        'reservoir_feature_importance.csv'
    ]
    
    for file in output_files:
        if os.path.exists(file):
            print(f"  {file}")
        else:
            print(f"  {file} (not generated)")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
   main()
