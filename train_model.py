print("TRAIN SCRIPT STARTED")
import numpy as np
import json
import pickle
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

os.makedirs('model', exist_ok=True)
print("=" * 60)
print("  CareerDNA — ANN Training")
print("=" * 60)

# ── Skill Feature Space (30 skills = input neurons) ───────────
SKILLS = [
    "python", "java", "javascript", "c++", "sql", "git",
    "html", "css", "machine learning", "deep learning",
    "tensorflow", "pytorch", "nlp", "react", "node",
    "docker", "kubernetes", "aws", "linux", "bash",
    "statistics", "mathematics", "data analysis",
    "networking", "security", "agile", "rest api",
    "flutter", "kotlin", "microservices"
]
N_FEATURES = len(SKILLS)
SKILL_IDX  = {s: i for i, s in enumerate(SKILLS)}

# ── Career Definitions ────────────────────────────────────────
CAREERS = {
    "AI/ML Engineer":        ["python","machine learning","deep learning","tensorflow","pytorch","statistics","mathematics","data analysis"],
    "Data Scientist":        ["python","sql","statistics","mathematics","data analysis","machine learning","rest api"],
    "Full Stack Developer":  ["javascript","html","css","react","node","sql","git","rest api","docker"],
    "Backend Engineer":      ["python","java","sql","git","linux","rest api","docker","microservices"],
    "Frontend Developer":    ["javascript","html","css","react","git"],
    "DevOps Engineer":       ["linux","docker","kubernetes","aws","bash","git","python","microservices"],
    "Cloud Architect":       ["aws","linux","networking","docker","kubernetes","python","bash"],
    "Cybersecurity Analyst": ["networking","linux","security","python","sql","bash"],
    "Mobile Developer":      ["kotlin","flutter","javascript","git","rest api"],
    "MLOps Engineer":        ["python","machine learning","docker","kubernetes","linux","aws","git"],
    "Data Engineer":         ["python","sql","linux","docker","aws","bash","statistics"],
    "NLP Engineer":          ["python","machine learning","deep learning","pytorch","nlp","statistics"],
    "Research Scientist":    ["python","deep learning","mathematics","statistics","machine learning","pytorch"],
    "Product Manager":       ["sql","agile","rest api","javascript","data analysis"],
}
CAREER_NAMES = list(CAREERS.keys())
N_CLASSES    = len(CAREER_NAMES)

print(f"  Careers:   {N_CLASSES}")
print(f"  Features:  {N_FEATURES} skills")

# ── Synthetic Dataset Generation ─────────────────────────────
np.random.seed(42)
N_SAMPLES = 2000
X_list, y_list = [], []

for career_idx, (career, core_skills) in enumerate(CAREERS.items()):
    n = N_SAMPLES // N_CLASSES
    for _ in range(n):
        vec = np.zeros(N_FEATURES, dtype=np.float32)
        # Core skills → high probability
        for s in core_skills:
            if s in SKILL_IDX:
                vec[SKILL_IDX[s]] = np.random.uniform(0.6, 1.0)
        # Random extra skills → noise
        for _ in range(np.random.randint(0, 5)):
            vec[np.random.randint(0, N_FEATURES)] = max(
                vec[np.random.randint(0, N_FEATURES)],
                np.random.uniform(0.1, 0.4)
            )
        X_list.append(vec)
        y_list.append(career_idx)

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.int32)
print(f"  Dataset:   {len(X)} samples")

# ── Feature Normalisation — MinMaxScaler ─────────────────────

scaler   = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
print(f"  Normalised: MinMaxScaler → [0, 1]")

# ── Train / Val / Test Split (70 / 15 / 15) ──────────────────
X_tr, X_tmp, y_tr, y_tmp = train_test_split(X_scaled, y, test_size=0.30, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp)
print(f"  Split:     Train={len(X_tr)} | Val={len(X_val)} | Test={len(X_test)}")

# ── ANN Model Definition ──────────────────────────────────────

tf.random.set_seed(42)

inp = keras.Input(shape=(N_FEATURES,), name='skill_input')

# Hidden Layer 1: Dense(128) + BatchNorm + ReLU + Dropout(0.3)
x = layers.Dense(128, kernel_regularizer=regularizers.l2(0.001))(inp)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.Dropout(0.30)(x)

# Hidden Layer 2: Dense(64) + BatchNorm + ReLU + Dropout(0.2)
x = layers.Dense(64, kernel_regularizer=regularizers.l2(0.001))(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.Dropout(0.20)(x)

# Hidden Layer 3: Dense(32) + ReLU
x = layers.Dense(32)(x)
x = layers.Activation('relu')(x)

# Output Layer: Softmax → probability over N_CLASSES careers
out = layers.Dense(N_CLASSES, activation='softmax', name='career_output')(x)

model = keras.Model(inputs=inp, outputs=out)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print(f"\n  Architecture:")
print(f"    Input({N_FEATURES}) → Dense(128)+BN+ReLU+Drop(0.3)")
print(f"                       → Dense(64)+BN+ReLU+Drop(0.2)")
print(f"                       → Dense(32)+ReLU")
print(f"                       → Output({N_CLASSES}, Softmax)")
print(f"    Parameters: {model.count_params():,}")

# ── Callbacks ─────────────────────────────────────────────────
callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-5, verbose=0),
]

# ── Train ─────────────────────────────────────────────────────
print(f"\n  Training (max 150 epochs, early stopping)...")
history = model.fit(
    X_tr, y_tr,
    validation_data=(X_val, y_val),
    epochs=150,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# ── Evaluate ─────────────────────────────────────────────────
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n  Test Accuracy: {test_acc*100:.1f}%")
print(f"  Test Loss:     {test_loss:.4f}")
print(f"  Epochs run:    {len(history.history['loss'])}")

# ── Save ─────────────────────────────────────────────────────
model.save('model/career_model')
with open('model/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

hist = history.history
with open('model/training_history.json', 'w') as f:
    json.dump({
        'epochs':       list(range(1, len(hist['loss'])+1)),
        'loss':         [float(v) for v in hist['loss']],
        'val_loss':     [float(v) for v in hist['val_loss']],
        'accuracy':     [float(v) for v in hist['accuracy']],
        'val_accuracy': [float(v) for v in hist['val_accuracy']],
    }, f)

with open('model/metadata.json', 'w') as f:
    json.dump({
        'n_features':    N_FEATURES,
        'n_classes':     N_CLASSES,
        'career_names':  CAREER_NAMES,
        'skill_index':   SKILL_IDX,
        'test_accuracy': round(float(test_acc) * 100, 1),
        'test_loss':     round(float(test_loss), 4),
        'total_params':  model.count_params(),
        'architecture':  [
            {'layer': 'Input',   'neurons': N_FEATURES, 'activation': '-'},
            {'layer': 'Dense 1', 'neurons': 128, 'activation': 'ReLU + BatchNorm + Dropout(0.3)'},
            {'layer': 'Dense 2', 'neurons': 64,  'activation': 'ReLU + BatchNorm + Dropout(0.2)'},
            {'layer': 'Dense 3', 'neurons': 32,  'activation': 'ReLU'},
            {'layer': 'Output',  'neurons': N_CLASSES, 'activation': 'Softmax'},
        ]
    }, f)

print(f"\n  Saved: model/career_model.keras")
print(f"  Saved: model/scaler.pkl")
print(f"  Saved: model/training_history.json")
print(f"  Saved: model/metadata.json")
print(f"\n{'='*60}")
print(f"  Done! Now run: python app.py")
print(f"  Open: http://localhost:5000")
print(f"{'='*60}")
