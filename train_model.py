import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────
IMG_SIZE   = 224
BATCH_SIZE = 32
TRAIN_DIR  = 'dataset/train'
TEST_DIR   = 'dataset/test'
MODEL_PATH = 'model/cancer_model.h5'

os.makedirs('model', exist_ok=True)

# ─────────────────────────────────────────────────────────────
#  STEP 1 — DATA LOADING & AUGMENTATION
#  - Mild augmentation to avoid overfitting
#  - 10% validation split from training data
#  - Separate clean test generator (no augmentation)
# ─────────────────────────────────────────────────────────────
print("=" * 50)
print("  Cancer Detection Model — Training Script")
print("=" * 50)
print("\nStep 1: Loading and augmenting data...")

train_datagen = ImageDataGenerator(
    rescale            = 1./255,
    rotation_range     = 15,        # mild rotation
    width_shift_range  = 0.1,       # slight horizontal shift
    height_shift_range = 0.1,       # slight vertical shift
    shear_range        = 0.1,       # mild shear
    zoom_range         = 0.15,      # slight zoom
    horizontal_flip    = True,      # flip X-rays horizontally
    brightness_range   = [0.85, 1.15],  # slight brightness change
    fill_mode          = 'nearest',
    validation_split   = 0.1        # 10% for validation
)

# Clean test generator — NO augmentation (real world condition)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size = (IMG_SIZE, IMG_SIZE),
    batch_size  = BATCH_SIZE,
    class_mode  = 'binary',
    subset      = 'training',
    shuffle     = True
)

val_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size = (IMG_SIZE, IMG_SIZE),
    batch_size  = BATCH_SIZE,
    class_mode  = 'binary',
    subset      = 'validation',
    shuffle     = False
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size = (IMG_SIZE, IMG_SIZE),
    batch_size  = BATCH_SIZE,
    class_mode  = 'binary',
    shuffle     = False
)

print(f"\nClasses found  : {train_generator.class_indices}")
print(f"Train samples  : {train_generator.samples}")
print(f"Val samples    : {val_generator.samples}")
print(f"Test samples   : {test_generator.samples}")

# ─────────────────────────────────────────────────────────────
#  STEP 2 — BUILD MODEL
#  - ResNet50 pretrained on ImageNet (transfer learning)
#  - Custom classification head
#  - Dropout + BatchNorm to prevent overfitting
#  - Single sigmoid output (binary classification)
# ─────────────────────────────────────────────────────────────
print("\nStep 2: Building ResNet50 transfer learning model...")

base_model = ResNet50(
    weights     = 'imagenet',
    include_top = False,
    input_shape = (IMG_SIZE, IMG_SIZE, 3)
)

# Freeze all base model layers in Phase 1
base_model.trainable = False

# Custom classification head
x      = base_model.output
x      = GlobalAveragePooling2D()(x)          # reduce spatial dims

x      = Dense(512, activation='relu')(x)     # fully connected layer 1
x      = BatchNormalization()(x)              # normalize activations
x      = Dropout(0.4)(x)                      # drop 40% to avoid overfit

x      = Dense(256, activation='relu')(x)     # fully connected layer 2
x      = BatchNormalization()(x)
x      = Dropout(0.3)(x)                      # drop 30%

x      = Dense(128, activation='relu')(x)     # fully connected layer 3
x      = Dropout(0.2)(x)                      # drop 20%

output = Dense(1, activation='sigmoid')(x)    # binary output (0=cancer, 1=normal)

model  = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer = Adam(learning_rate=0.0001),
    loss      = 'binary_crossentropy',
    metrics   = ['accuracy']
)

print(f"Total parameters   : {model.count_params():,}")
print(f"Trainable params   : {sum([np.prod(v.shape) for v in model.trainable_weights]):,}")

# ─────────────────────────────────────────────────────────────
#  STEP 3 — CALLBACKS
#  - ModelCheckpoint  : saves best model automatically
#  - EarlyStopping    : stops if no improvement (prevents overfit)
#  - ReduceLROnPlateau: reduces LR when stuck (prevents underfit)
# ─────────────────────────────────────────────────────────────
print("\nStep 3: Setting up callbacks...")

checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor        = 'val_accuracy',
    save_best_only = True,
    verbose        = 1
)

early_stop = EarlyStopping(
    monitor              = 'val_accuracy',
    patience             = 6,               # wait 6 epochs before stopping
    restore_best_weights = True,
    verbose              = 1
)

reduce_lr = ReduceLROnPlateau(
    monitor  = 'val_loss',
    factor   = 0.3,                         # reduce LR by 70%
    patience = 3,                           # wait 3 epochs
    min_lr   = 1e-7,
    verbose  = 1
)

# ─────────────────────────────────────────────────────────────
#  STEP 4 — PHASE 1 TRAINING (frozen base)
#  Goal: train classification head while base is frozen
#  Why: learn new features without destroying pretrained weights
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("  Phase 1: Training with frozen base (max 15 epochs)")
print("  Goal: Train classification head only")
print("=" * 50)

h1 = model.fit(
    train_generator,
    epochs          = 15,
    validation_data = val_generator,
    callbacks       = [checkpoint, early_stop, reduce_lr],
    verbose         = 1
)

best_p1 = max(h1.history['val_accuracy'])
print(f"\nPhase 1 completed!")
print(f"Best val_accuracy  : {best_p1 * 100:.2f}%")
print(f"Epochs trained     : {len(h1.history['accuracy'])}")

# ─────────────────────────────────────────────────────────────
#  STEP 5 — PHASE 2 FINE TUNING (unfreeze last 40 layers)
#  Goal: fine-tune deeper ResNet50 layers for better accuracy
#  Why: very low LR (0.00001) prevents destroying pretrained weights
#  Protection: EarlyStopping + ReduceLROnPlateau still active
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("  Phase 2: Fine-tuning last 40 layers (max 10 epochs)")
print("  Goal: Push accuracy higher without overfitting")
print("=" * 50)

# Freeze all first, then unfreeze last 40
for layer in base_model.layers:
    layer.trainable = False
for layer in base_model.layers[-40:]:
    layer.trainable = True

# Recompile with very low learning rate
model.compile(
    optimizer = Adam(learning_rate=0.00001),  # 10x lower than Phase 1
    loss      = 'binary_crossentropy',
    metrics   = ['accuracy']
)

# Fresh callbacks for Phase 2
early_stop2 = EarlyStopping(
    monitor              = 'val_accuracy',
    patience             = 5,
    restore_best_weights = True,
    verbose              = 1
)

reduce_lr2 = ReduceLROnPlateau(
    monitor  = 'val_loss',
    factor   = 0.3,
    patience = 2,
    min_lr   = 1e-8,
    verbose  = 1
)

h2 = model.fit(
    train_generator,
    epochs          = 10,
    validation_data = val_generator,
    callbacks       = [checkpoint, early_stop2, reduce_lr2],
    verbose         = 1
)

best_p2 = max(h2.history['val_accuracy'])
print(f"\nPhase 2 completed!")
print(f"Best val_accuracy  : {best_p2 * 100:.2f}%")
print(f"Epochs trained     : {len(h2.history['accuracy'])}")

# ─────────────────────────────────────────────────────────────
#  STEP 6 — FINAL EVALUATION ON TEST SET
#  Uses clean test data (no augmentation) — real world condition
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("  Step 6: Final evaluation on unseen test data")
print("=" * 50)

loss, accuracy = model.evaluate(test_generator, verbose=1)

print(f"\n{'=' * 50}")
print(f"  FINAL RESULTS")
print(f"{'=' * 50}")
print(f"  Test Accuracy : {accuracy * 100:.2f}%")
print(f"  Test Loss     : {loss:.4f}")
print(f"  Phase 1 Best  : {best_p1 * 100:.2f}%")
print(f"  Phase 2 Best  : {best_p2 * 100:.2f}%")
print(f"{'=' * 50}")

# Overfitting check
train_acc = h2.history['accuracy'][-1]
val_acc   = h2.history['val_accuracy'][-1]
gap       = train_acc - val_acc
print(f"\nOverfit check:")
print(f"  Train accuracy : {train_acc * 100:.2f}%")
print(f"  Val accuracy   : {val_acc * 100:.2f}%")
print(f"  Gap            : {gap * 100:.2f}%")
if gap < 0.05:
    print("  Status         : ✅ No overfitting detected")
elif gap < 0.10:
    print("  Status         : ⚠️  Mild overfitting — acceptable")
else:
    print("  Status         : ❌ Overfitting detected — consider more dropout")

# ─────────────────────────────────────────────────────────────
#  STEP 7 — PLOT TRAINING GRAPHS
#  Shows accuracy and loss for both phases
# ─────────────────────────────────────────────────────────────
print("\nStep 7: Saving training graphs...")

acc  = h1.history['accuracy']     + h2.history['accuracy']
val  = h1.history['val_accuracy'] + h2.history['val_accuracy']
ls   = h1.history['loss']         + h2.history['loss']
vls  = h1.history['val_loss']     + h2.history['val_loss']
p1_end = len(h1.history['accuracy']) - 1

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(acc, label='Train Accuracy', color='#6c63ff', linewidth=2)
plt.plot(val, label='Val Accuracy',   color='#3ecfcf', linewidth=2)
plt.axvline(x=p1_end, color='gray', linestyle='--', alpha=0.7, label='Fine-tune start')
plt.title('Model Accuracy', fontsize=14, fontweight='bold')
plt.xlabel('Epoch'); plt.ylabel('Accuracy')
plt.legend(); plt.grid(alpha=0.2)
plt.ylim([0.5, 1.0])

plt.subplot(1, 2, 2)
plt.plot(ls,  label='Train Loss', color='#6c63ff', linewidth=2)
plt.plot(vls, label='Val Loss',   color='#3ecfcf', linewidth=2)
plt.axvline(x=p1_end, color='gray', linestyle='--', alpha=0.7, label='Fine-tune start')
plt.title('Model Loss', fontsize=14, fontweight='bold')
plt.xlabel('Epoch'); plt.ylabel('Loss')
plt.legend(); plt.grid(alpha=0.2)

plt.suptitle(f'Cancer Detection Model — Final Accuracy: {accuracy*100:.2f}%',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('model/training_graph.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nModel saved at     : {MODEL_PATH}")
print(f"Graph saved at     : model/training_graph.png")
print("\n✅ Training complete! Run: python app.py")