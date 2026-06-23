import torch
import torch.nn as nn
import xgboost as xgb
import numpy as np

# ==========================================
# 1. SDNN (Standard Deep Neural Network)
# ==========================================
class SDNN(nn.Module):
    """
    Standard Deep Neural Network (Models 1 & 2 in the paper).
    Architecture determined via grid search: 3 hidden layers, 256 neurons each.
    Maps input (precipitation + actions) to scalar continuous CSO volume.
    """
    def __init__(self, input_dim):
        super(SDNN, self).__init__()
        
        # 3 hidden layers with 256 neurons each
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # Predicts continuous CSO volume
        )
        
    def forward(self, x):
        # We predict log(CSO) or scaled CSO, but linear output is standard
        return self.network(x)


# ==========================================
# 2. TWO-STEP SURROGATE MODEL
# ==========================================
class TwoStepSurrogateModel:
    """
    Implements the 2-step pipeline mentioned in the paper:
    1. XGBoost classifier: predicts if there is a zero-overflow state (success).
    2. DNN regressor (SDNN or BDNN): regresses continuous failure states (overflow volumes).
    """
    def __init__(self, input_dim, dnn_model=None):
        """
        Args:
            input_dim (int): Number of features (Rainfall timesteps + 10 actions).
            dnn_model (nn.Module): The DNN used for predicting CSO volume when overflow > 0.
                                   If None, creates a default SDNN.
        """
        self.classifier = xgb.XGBClassifier(
            n_estimators=100, 
            max_depth=4, 
            learning_rate=0.1, 
            use_label_encoder=False, 
            eval_metric='logloss'
        )
        
        # If no DNN is provided, we instantiate a standard SDNN
        self.regressor = dnn_model if dnn_model is not None else SDNN(input_dim)
        
    def train_classifier(self, X_train, y_train):
        """
        Trains the XGBoost classifier to identify zero-overflow vs failure.
        y_train is expected to be the actual continuous CSO volume.
        """
        # Convert continuous CSO to binary labels (0 = no overflow, 1 = overflow)
        y_binary = (y_train > 0.0).astype(int)
        self.classifier.fit(X_train, y_binary)
        
    def predict(self, X):
        """
        Predicts total CSO volume by chaining classifier and regressor.
        """
        # Step 1: Predict whether an overflow occurs
        overflow_preds = self.classifier.predict(X)
        
        # Initialize output array with zeros
        final_cso_preds = np.zeros(X.shape[0])
        
        # Step 2: For indices where overflow_preds == 1, use the DNN to regress volume
        overflow_indices = np.where(overflow_preds == 1)[0]
        
        if len(overflow_indices) > 0:
            X_tensor = torch.tensor(X[overflow_indices], dtype=torch.float32)
            
            self.regressor.eval()
            with torch.no_grad():
                # Get the continuous regression from the DNN
                dnn_preds = self.regressor(X_tensor).numpy().flatten()
            
            # Map back to the original array, bounding to avoid negative physical CSO
            final_cso_preds[overflow_indices] = np.maximum(dnn_preds, 0.0)
            
        return final_cso_preds
