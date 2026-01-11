
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv1D, Dropout, Activation, Add, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import os
import torch # Explicit import for loading .pt files

class TCN:
    """
    Implementação da Temporal Convolutional Network (TCN) em Keras (Manus AI Architecture).
    Adapted for Inside Bar Breakout Prediction.
    """

    def __init__(self, num_feat, num_classes, nb_filters=64, kernel_size=3, 
                 dilations=[1, 2, 4, 8, 16], nb_stacks=1, dropout_rate=0.2, use_skip_connections=True):
        self.num_feat = num_feat
        self.num_classes = num_classes
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.dilations = dilations
        self.nb_stacks = nb_stacks
        self.dropout_rate = dropout_rate
        self.use_skip_connections = use_skip_connections

    def _residual_block(self, x, dilation_rate):
        prev_x = x
        
        # 1st Conv with Dilation
        x = Conv1D(filters=self.nb_filters, kernel_size=self.kernel_size,
                   dilation_rate=dilation_rate, padding='causal')(x)
        x = Activation('relu')(x)
        x = Dropout(self.dropout_rate)(x)
        
        # 2nd Conv with Dilation
        x = Conv1D(filters=self.nb_filters, kernel_size=self.kernel_size,
                   dilation_rate=dilation_rate, padding='causal')(x)
        x = Activation('relu')(x)
        x = Dropout(self.dropout_rate)(x)
        
        # Skip Connection
        if self.use_skip_connections:
            # If channels mismatch, project input
            if prev_x.shape[-1] != x.shape[-1]:
                prev_x = Conv1D(filters=self.nb_filters, kernel_size=1, padding='same')(prev_x)
            return Add()([prev_x, x])
        else:
            return x

    def build_model(self, input_shape):
        inputs = Input(shape=input_shape)
        x = inputs
        
        # Stack of Residual Blocks
        for _ in range(self.nb_stacks):
            for d in self.dilations:
                x = self._residual_block(x, d)
                
        # Final Classification Head
        # Global Pooling to catch features from anywhere (or last step?)
        # TCN usually preserves sequence length.
        # For classification, we often take the last time step.
        # Lambda layer to slice last step: x[:, -1, :]
        x = tf.keras.layers.Lambda(lambda t: t[:, -1, :])(x)
        
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(self.num_classes)(x)
        outputs = Activation('softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model

def train_tcn_keras():
    print("--- Training TCN (Keras/TensorFlow) ---")
    
    # Load PyTorch Data and Convert to Numpy for Keras
    try:
        X_pt = torch.load('train_inputs_tcn.pt') # (N, Channels=25, L=120) for PyTorch
        y_pt = torch.load('train_labels_tcn.pt')
        
        X = X_pt.numpy()
        y = y_pt.numpy()
        
        # Keras expects (N, Length, Channels)
        # PyTorch has (N, Channels, Length)
        X = np.transpose(X, (0, 2, 1)) 
        
        # One-hot encode Y
        import tensorflow as tf
        y_onehot = tf.keras.utils.to_categorical(y, num_classes=3) # Use tf.keras.utils explicitly
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error loading TCN data: {e}")
        return

    print(f"Data Loaded: X={X.shape}, y={y_onehot.shape}")
    
    # Validation Split
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y_onehot[:split], y_onehot[split:]
    
    # Model Config
    N_TIMESTEPS = X.shape[1] # 120
    N_FEATURES = X.shape[2] # 25
    N_CLASSES = 3
    
    tcn_builder = TCN(num_feat=N_FEATURES, num_classes=N_CLASSES, nb_filters=64, kernel_size=3, dilations=[1,2,4,8,16])
    model = tcn_builder.build_model(input_shape=(N_TIMESTEPS, N_FEATURES))
    
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    model.summary()
    
    # Callbacks
    checkpoint = ModelCheckpoint('trade_tcn_keras.h5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    
    # Train
    history = model.fit(X_train, y_train, 
                        validation_data=(X_val, y_val),
                        epochs=50, 
                        batch_size=64,
                        callbacks=[checkpoint, reduce_lr],
                        verbose=1)
                        
    print("Training Complete. Model saved to trade_tcn_keras.h5")

if __name__ == '__main__':
    # Need torch to load the .pt file
    import torch 
    train_tcn_keras()
