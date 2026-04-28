import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from models.metrics import compute_metrics

def _best_f1_threshold(y_true: np.ndarray, y_probs: np.ndarray) -> float:
    thresholds = np.arange(0.35, 0.66, 0.01)
    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in thresholds:
        preds = (y_probs >= threshold).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    return best_threshold


def run_lstm(X_train, y_train, X_val, y_val, X_recovery, y_recovery, args):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=args.max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_val_seq = tokenizer.texts_to_sequences(X_val)
    X_recovery_seq = tokenizer.texts_to_sequences(X_recovery)

    X_train_pad = tf.keras.preprocessing.sequence.pad_sequences(
        X_train_seq, maxlen=args.max_len, padding="post", truncating="post"
    )
    X_val_pad = tf.keras.preprocessing.sequence.pad_sequences(
        X_val_seq, maxlen=args.max_len, padding="post", truncating="post"
    )
    X_recovery_pad = tf.keras.preprocessing.sequence.pad_sequences(
        X_recovery_seq, maxlen=args.max_len, padding="post", truncating="post"
    )

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Embedding(input_dim=args.max_words, output_dim=96),
            tf.keras.layers.SpatialDropout1D(0.25),
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(48, dropout=0.3, recurrent_dropout=0.2)
            ),
            tf.keras.layers.Dense(
                32,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            ),
            tf.keras.layers.Dropout(0.35),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=8e-4, clipnorm=1.0),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True,
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=1,
        min_lr=1e-5,
    )

    y_train_np = np.asarray(y_train)
    class_weights_raw = compute_class_weight(
        class_weight="balanced", classes=np.array([0, 1]), y=y_train_np
    )
    class_weight = {
        0: float(class_weights_raw[0]),
        1: float(class_weights_raw[1]),
    }

    history = model.fit(
        X_train_pad,
        y_train_np,
        validation_data=(X_val_pad, np.asarray(y_val)),
        epochs=args.lstm_epochs,
        batch_size=args.lstm_batch_size,
        callbacks=[early_stop, reduce_lr],
        class_weight=class_weight,
        verbose=2,
    )

    val_probs = model.predict(X_val_pad, batch_size=256, verbose=0).ravel()
    recovery_probs = model.predict(X_recovery_pad, batch_size=256, verbose=0).ravel()
    decision_threshold = _best_f1_threshold(np.asarray(y_val), val_probs)
    val_pred = (val_probs >= decision_threshold).astype(int)
    recovery_pred = (recovery_probs >= decision_threshold).astype(int)

    return {
        "welfake_validation_20pct": compute_metrics(y_val, val_pred),
        "recovery_external_test": compute_metrics(y_recovery, recovery_pred),
        "config": {
            "tokenizer": {
                "max_words": args.max_words,
                "max_len": args.max_len,
                "oov_token": "<OOV>",
            },
            "model": {
                "type": "LSTM",
                "embedding_dim": 96,
                "lstm_units": 48,
                "dropout": 0.3,
                "epochs": args.lstm_epochs,
                "batch_size": args.lstm_batch_size,
                "decision_threshold": decision_threshold,
                "class_weight": class_weight,
                "optimizer": "Adam(learning_rate=8e-4, clipnorm=1.0)",
            },
            "history": {
                key: [float(value) for value in values]
                for key, values in history.history.items()
            },
        },
    }
