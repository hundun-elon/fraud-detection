import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, BatchNormalization, Layer,
    concatenate, Add, LayerNormalization, GaussianNoise
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, 
    auc, precision_recall_curve, precision_score, 
    recall_score, f1_score, accuracy_score
)

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class CreditCardFraudDNN:
    """
    Advanced Neural Network for Credit Card Fraud Detection with enhanced architecture
    """
    
    def __init__(self, output_dir='outputs/dnn_model', validation_split=0.2):
        """
        Initialize the fraud detection model.
        
        Args:
            output_dir: Directory to save model outputs
            validation_split: Proportion of training data to use for validation
        """
        self.output_dir = output_dir
        self.validation_split = validation_split
        self.model = None
        self.history = None
        self.best_threshold = 0.5
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    def focal_loss(self, y_true, y_pred, gamma=2.0, alpha=0.25):
        """
        Focal loss for handling class imbalance in fraud detection.
        
        Args:
            y_true: True labels
            y_pred: Predicted probabilities
            gamma: Focusing parameter
            alpha: Weighting factor for positive class
            
        Returns:
            Focal loss value
        """
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        pt = tf.where(y_true == 1, y_pred, 1 - y_pred)
        ce = -tf.math.log(pt)
        alpha_t = tf.where(y_true == 1, alpha, 1 - alpha)
        fl = alpha_t * tf.pow(1 - pt, gamma) * ce
        return tf.reduce_mean(fl)
    
    def _swish(self, x):
        """Swish activation function: x * sigmoid(x)"""
        return x * tf.nn.sigmoid(x)
    
    class DropConnectLayer(Layer):
        """
        Custom Keras layer for DropConnect regularization.
        """
        def __init__(self, rate=0.1, **kwargs):
            """
            Initialize the DropConnect layer.
            
            Args:
                rate: Drop probability
            """
            super().__init__(**kwargs)
            self.rate = rate

        def call(self, inputs, training=None):
            """
            Apply DropConnect to the input tensor.
            
            Args:
                inputs: Input tensor
                training: Boolean indicating whether in training mode
            
            Returns:
                Tensor with dropped connections
            """
            if training:
                keep_prob = 1.0 - self.rate
                batch_size = tf.shape(inputs)[0]
                random_tensor = keep_prob + tf.random.uniform([batch_size, 1], 0, 1, dtype=inputs.dtype)
                binary_tensor = tf.floor(random_tensor)
                output = inputs / keep_prob * binary_tensor
                return output
            return inputs

        def get_config(self):
            """
            Return the configuration of the layer for serialization.
            """
            config = super().get_config()
            config.update({'rate': self.rate})
            return config
    
    class FeatureInteractionLayer(Layer):
        """
        Custom Keras layer to capture pairwise feature interactions.
        """
        def __init__(self, units, **kwargs):
            """
            Initialize the feature interaction layer.
            
            Args:
                units: Number of units in the output dense layer
            """
            super().__init__(**kwargs)
            self.units = units
            self.dense = Dense(units, activation=lambda x: x * tf.nn.sigmoid(x), 
                             kernel_initializer='he_normal')

        def build(self, input_shape):
            """
            Build the layer by initializing the Dense layer with the correct input shape.
            
            Args:
                input_shape: Shape of the input tensor
            """
            input_dim = input_shape[-1]  # Number of input features
            pairwise_dim = input_dim * input_dim
            self.dense.build(input_shape=(None, pairwise_dim))
            super().build(input_shape)

        def call(self, inputs):
            """
            Compute pairwise feature interactions.
            
            Args:
                inputs: Input tensor with shape [batch_size, input_dim]
            
            Returns:
                Tensor with feature interactions, shape [batch_size, units]
            """
            # Debug: Print input shape
            tf.print("FeatureInteractionLayer input shape:", tf.shape(inputs))
            
            # Compute pairwise interactions using einsum
            pairwise = tf.einsum('bi,bj->bij', inputs, inputs)
            
            # Debug: Print pairwise shape
            tf.print("Pairwise interaction shape:", tf.shape(pairwise))
            
            # Get input_dim dynamically
            input_dim = tf.shape(inputs)[-1]
            pairwise_dim = input_dim * input_dim
            
            # Reshape to [batch_size, input_dim * input_dim]
            pairwise = tf.reshape(pairwise, [-1, pairwise_dim])
            
            # Debug: Print reshaped pairwise shape
            tf.print("Reshaped pairwise shape:", tf.shape(pairwise))
            
            # Apply dense layer with Swish activation
            interaction = self.dense(pairwise)
            return interaction

        def compute_output_shape(self, input_shape):
            """
            Compute the output shape of the layer.
            
            Args:
                input_shape: Shape of the input tensor
            
            Returns:
                Output shape as a tuple
            """
            return (input_shape[0], self.units)

        def get_config(self):
            """
            Return the configuration of the layer for serialization.
            """
            config = super().get_config()
            config.update({'units': self.units})
            return config
    
    def build_model(self, input_shape, learning_rate=0.001):
        """
        Build a deep neural network with residual connections and advanced regularization.
        
        Args:
            input_shape: Shape of input features (tuple, e.g., (30,))
            learning_rate: Learning rate for optimizer
            
        Returns:
            Compiled Keras model
        """
        # Input layer
        inputs = Input(shape=input_shape, name='input_layer')
        
        # Initial Gaussian Noise for robustness
        x = GaussianNoise(0.1)(inputs)
        
        # Feature interaction branch
        interaction = self.FeatureInteractionLayer(units=64)(x)
        
        # First residual block
        x1 = Dense(512, activation=self._swish, kernel_initializer='he_normal')(x)
        x1 = BatchNormalization()(x1)
        x1 = self.DropConnectLayer(rate=0.2)(x1)
        x1 = Dense(256, activation=self._swish, kernel_initializer='he_normal')(x1)
        x1 = BatchNormalization()(x1)
        residual = Dense(256, kernel_initializer='he_normal')(x)  # Match dimensions
        x1 = Add()([x1, residual])
        x1 = LayerNormalization()(x1)
        
        # Second residual block
        x2 = Dense(256, activation=self._swish, kernel_initializer='he_normal')(x1)
        x2 = BatchNormalization()(x2)
        x2 = self.DropConnectLayer(rate=0.2)(x2)
        x2 = Dense(128, activation=self._swish, kernel_initializer='he_normal')(x2)
        x2 = BatchNormalization()(x2)
        residual = Dense(128, kernel_initializer='he_normal')(x1)  # Match dimensions
        x2 = Add()([x2, residual])
        x2 = LayerNormalization()(x2)
        
        # Third residual block
        x3 = Dense(128, activation=self._swish, kernel_initializer='he_normal')(x2)
        x3 = BatchNormalization()(x3)
        x3 = self.DropConnectLayer(rate=0.2)(x3)
        x3 = Dense(64, activation=self._swish, kernel_initializer='he_normal')(x3)
        x3 = BatchNormalization()(x3)
        residual = Dense(64, kernel_initializer='he_normal')(x2)  # Match dimensions
        x3 = Add()([x3, residual])
        x3 = LayerNormalization()(x3)
        
        # Concatenate with interaction features
        x = concatenate([x3, interaction])
        
        # Final dense layers
        x = Dense(32, activation=self._swish, kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = self.DropConnectLayer(rate=0.1)(x)
        
        # Output layer
        outputs = Dense(1, activation='sigmoid', name='output_layer')(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs, name='advanced_fraud_detection_nn')
        
        # Compile model
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss=self.focal_loss,
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.AUC(name='pr_auc', curve='PR')
            ]
        )
        
        return model
    
    def train_model(self, X_train, y_train, epochs=100, batch_size=32, verbose=1):
        """
        Train the neural network model
        
        Args:
            X_train: Training features
            y_train: Training labels
            epochs: Number of training epochs
            batch_size: Batch size for training
            verbose: Verbosity mode
        """
        print("\nTraining Advanced Neural Network model...")
        start_time = time.time()
        
        # Build model
        self.model = self.build_model(input_shape=(X_train.shape[1],))
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_pr_auc', patience=10, mode='max', restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_pr_auc', factor=0.5, patience=5, min_lr=1e-6, mode='max'),
            ModelCheckpoint(
                filepath=os.path.join(self.output_dir, 'best_dnn_fraud_model.h5'),
                monitor='val_pr_auc',
                save_best_only=True,
                mode='max',
                verbose=1
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_split=self.validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose,
            shuffle=True
        )
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Save the final model
        self.save_model('final_dnn_fraud_model.h5')
        
        # Plot training history
        self.plot_training_history()
        
        return self.history
    
    def evaluate_model(self, X_test, y_test, fold=None):
        """Comprehensive model evaluation with threshold analysis"""
        print("\nEvaluating Advanced Neural Network model...")
        
        # Get predictions
        y_pred_proba = self.model.predict(X_test).flatten()
        
        # Find optimal threshold
        self.best_threshold = self._find_optimal_threshold(y_test, y_pred_proba)
        y_pred = (y_pred_proba > self.best_threshold).astype(int)
        
        # Generate evaluation metrics
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        # Calculate confusion matrix for business impact
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # Generate visualizations
        prefix = f"fold_{fold}_" if fold is not None else ""
        self._plot_roc_curve(fpr, tpr, roc_auc, prefix)
        self._plot_pr_curve(precision, recall, pr_auc, np.mean(y_test), prefix)
        self._plot_confusion_matrix(y_test, y_pred, prefix)
        self._plot_probability_distribution(y_test, y_pred_proba, prefix)
        
        # Print and return metrics
        print("\nOptimal Threshold:", self.best_threshold)
        print(classification_report(y_test, y_pred))
        print("\nBusiness Impact Analysis:")
        print(f"Frauds Caught (TP): {tp}")
        print(f"Frauds Missed (FN): {fn}")
        print(f"Legitimate Transactions Flagged (FP): {fp}")
        print(f"Legitimate Transactions Correct (TN): {tn}")
        print("\nRate Analysis:")
        print(f"True Positive Rate (TPR): {tpr:.4f}")
        print(f"False Positive Rate (FPR): {fpr:.4f}")
        print(f"True Negative Rate (TNR): {tnr:.4f}")
        print(f"False Negative Rate (FNR): {fnr:.4f}")
        
        return {
            'best_threshold': self.best_threshold,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'confusion_matrix': cm,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn,
            'tpr': tpr,
            'fpr': fpr,
            'tnr': tnr,
            'fnr': fnr
        }
    
    def _find_optimal_threshold(self, y_true, y_pred_proba, thresholds=None):
        """Find optimal threshold based on F1 score"""
        if thresholds is None:
            thresholds = np.linspace(0.1, 0.9, 50)
        
        best_threshold = 0.5
        best_f1 = 0
        
        for thresh in thresholds:
            y_pred = (y_pred_proba > thresh).astype(int)
            f1 = f1_score(y_true, y_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = thresh
                
        return best_threshold
    
    def plot_training_history(self):
        """Plot training metrics visualization"""
        if self.history is None:
            print("No training history available.")
            return
        
        # Plot training history
        plt.figure(figsize=(15, 10))
        
        # Accuracy plot
        plt.subplot(2, 3, 1)
        plt.plot(self.history.history['accuracy'], label='Train Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        
        # Loss plot
        plt.subplot(2, 3, 2)
        plt.plot(self.history.history['loss'], label='Train Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        
        # Precision plot
        plt.subplot(2, 3, 3)
        plt.plot(self.history.history['precision'], label='Train Precision')
        plt.plot(self.history.history['val_precision'], label='Validation Precision')
        plt.title('Model Precision')
        plt.ylabel('Precision')
        plt.xlabel('Epoch')
        plt.legend()
        
        # Recall plot
        plt.subplot(2, 3, 4)
        plt.plot(self.history.history['recall'], label='Train Recall')
        plt.plot(self.history.history['val_recall'], label='Validation Recall')
        plt.title('Model Recall')
        plt.ylabel('Recall')
        plt.xlabel('Epoch')
        plt.legend()
        
        # AUC plot
        plt.subplot(2, 3, 5)
        plt.plot(self.history.history['auc'], label='Train AUC')
        plt.plot(self.history.history['val_auc'], label='Validation AUC')
        plt.title('Model ROC AUC')
        plt.ylabel('AUC')
        plt.xlabel('Epoch')
        plt.legend()
        
        # PR AUC plot
        plt.subplot(2, 3, 6)
        plt.plot(self.history.history['pr_auc'], label='Train PR AUC')
        plt.plot(self.history.history['val_pr_auc'], label='Validation PR AUC')
        plt.title('Model PR AUC')
        plt.ylabel('PR AUC')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_history.png'), dpi=300)
        plt.close()
    
    def _plot_roc_curve(self, fpr, tpr, roc_auc, prefix=''):
        """Plot ROC curve"""
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.title('Receiver Operating Characteristic', fontsize=16)
        plt.legend(loc='lower right', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{prefix}roc_curve.png'), dpi=300)
        plt.close()
    
    def _plot_pr_curve(self, precision, recall, pr_auc, prevalence, prefix=''):
        """Plot Precision-Recall curve"""
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
        plt.plot([0, 1], [prevalence, prevalence], color='red', linestyle='--', 
                 label=f'Baseline ({prevalence:.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=14)
        plt.ylabel('Precision', fontsize=14)
        plt.title('Precision-Recall Curve', fontsize=16)
        plt.legend(loc='lower left', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{prefix}pr_curve.png'), dpi=300)
        plt.close()
    
    def _plot_confusion_matrix(self, y_true, y_pred, prefix=''):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Non-Fraud', 'Fraud'],
                    yticklabels=['Non-Fraud', 'Fraud'])
        plt.xlabel('Predicted', fontsize=14)
        plt.ylabel('Actual', fontsize=14)
        plt.title('Confusion Matrix', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{prefix}confusion_matrix.png'), dpi=300)
        plt.close()
    
    def _plot_probability_distribution(self, y_true, y_pred_proba, prefix=''):
        """Plot probability distribution by class"""
        plt.figure(figsize=(10, 6))
        for label in [0, 1]:
            sns.kdeplot(y_pred_proba[y_true == label], label=f'Class {label}')
        plt.axvline(x=self.best_threshold, color='k', linestyle='--', label=f'Threshold ({self.best_threshold:.2f})')
        plt.xlabel('Predicted Probability', fontsize=14)
        plt.ylabel('Density', fontsize=14)
        plt.title('Probability Distribution by Class', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{prefix}probability_distribution.png'), dpi=300)
        plt.close()
    
    def save_model(self, filename):
        """Save the trained model"""
        if self.model is not None:
            path = os.path.join(self.output_dir, filename)
            self.model.save(path)
            print(f"Model saved to {path}")
    
    def load_model(self, filename='best_dnn_fraud_model.h5'):
        """Load a trained model"""
        path = os.path.join(self.output_dir, filename)
        try:
            self.model = tf.keras.models.load_model(
                path, 
                custom_objects={
                    'focal_loss': self.focal_loss,
                    'FeatureInteractionLayer': self.FeatureInteractionLayer,
                    'DropConnectLayer': self.DropConnectLayer
                }
            )
            print(f"Model loaded from {path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict(self, X, threshold=None):
        """
        Make predictions with adjustable threshold
        
        Args:
            X: Input features
            threshold: Decision threshold (defaults to optimal threshold)
            
        Returns:
            predictions, probabilities
        """
        if self.model is None:
            print("Model not loaded or trained")
            return None, None
            
        if threshold is None:
            threshold = self.best_threshold
            
        proba = self.model.predict(X).flatten()
        return (proba > threshold).astype(int), proba
