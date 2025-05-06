import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def create_lstm_model(input_shape, num_classes):
    """Create and compile the LSTM model."""
    model = Sequential([
        # LSTM katmanları
        LSTM(64, input_shape=input_shape, return_sequences=True),
        Dropout(0.3),
        LSTM(32, return_sequences=False),
        Dropout(0.3),
        
        # Çıkış katmanları
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='sigmoid' if num_classes == 2 else 'softmax')
    ])

    # Model derleme
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy' if num_classes == 2 else 'categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(model, X_train, y_train_categorical, X_val, y_val_categorical, epochs=15, batch_size=32):
    """Train the LSTM model."""
    # Early stopping ekleyelim
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    # Learning rate azaltma
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=0.0001
    )
    
    # Model eğitimi
    history = model.fit(
        X_train, 
        y_train_categorical,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val_categorical),
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    return model, history