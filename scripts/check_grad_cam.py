import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("TensorFlow version:", tf.__version__)

# ============================================================================
# 1. CREATE DUMMY METEOROLOGICAL DATA
# ============================================================================
# Simulating 4 meteorological variables (e.g., temperature, pressure, humidity, wind)
# Each with 6 past time frames, 256x256 spatial resolution

def create_dummy_data(batch_size=2):
    """
    Create dummy spatio-temporal meteorological data
    Returns 4 separate inputs, each of shape (batch_size, 6, 256, 256, 1)
    """
    inputs = []
    for i in range(4):
        # Create data with some spatial patterns
        data = np.random.randn(batch_size, 6, 256, 256, 1).astype(np.float32)
        # Add some spatial structure (meteorological fields often have smooth patterns)
        for b in range(batch_size):
            for t in range(6):
                x, y = np.meshgrid(np.linspace(-1, 1, 256), np.linspace(-1, 1, 256))
                pattern = np.sin(3*x + i) * np.cos(3*y + i) + 0.5 * np.random.randn(256, 256)
                data[b, t, :, :, 0] += pattern * 0.3
        inputs.append(data)
    
    # Create dummy binary labels (e.g., precipitation/no precipitation)
    labels = np.random.randint(0, 2, size=(batch_size,))
    
    return inputs, labels

# Create training and test data
train_inputs, train_labels = create_dummy_data(batch_size=10)
test_inputs, test_labels = create_dummy_data(batch_size=2)

print("Input shapes:")
for i, inp in enumerate(train_inputs):
    print(f"  Input {i}: {inp.shape}")
print(f"Labels shape: {train_labels.shape}")

# ============================================================================
# 2. BUILD MODEL WITH CONCATENATION APPROACH
# ============================================================================

def build_concatenated_model(input_shape=(6, 256, 256, 1)):
    """
    Model that concatenates 4 inputs along the channel dimension
    then applies convolution
    """
    # Define 4 separate inputs
    inputs = [layers.Input(shape=input_shape, name=f'input_{i}') for i in range(4)]
    
    # Concatenate along the channel dimension (axis=-1)
    # Shape: (batch, 6, 256, 256, 4)
    concatenated = layers.Concatenate(axis=-1, name='concatenate')(inputs)
    
    # Apply 3D convolution on the concatenated spatio-temporal data
    # This processes all 4 variables jointly
    conv1 = layers.Conv3D(32, (3, 7, 7), activation='relu', 
                         padding='same', name='conv3d_concat')(concatenated)
    pool1 = layers.MaxPooling3D((2, 4, 4))(conv1)
    
    conv2 = layers.Conv3D(64, (2, 5, 5), activation='relu', 
                         padding='same', name='conv3d_target')(pool1)
    pool2 = layers.MaxPooling3D((1, 4, 4))(conv2)
    
    # Flatten and classify
    flatten = layers.Flatten()(pool2)
    dense1 = layers.Dense(128, activation='relu')(flatten)
    dropout = layers.Dropout(0.5)(dense1)
    output = layers.Dense(2, activation='softmax', name='output')(dropout)
    
    model = Model(inputs=inputs, outputs=output)
    return model

concat_model = build_concatenated_model()
concat_model.compile(optimizer='adam', 
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

print("\n" + "="*70)
print("CONCATENATED MODEL ARCHITECTURE")
print("="*70)
concat_model.summary()

# ============================================================================
# 3. BUILD MODEL WITH TIMEDISTRIBUTED APPROACH (for comparison)
# ============================================================================

def build_timedistributed_model(input_shape=(6, 256, 256, 1)):
    """
    Model that processes each of 4 inputs with TimeDistributed Conv2D
    """
    inputs = [layers.Input(shape=input_shape, name=f'input_{i}') for i in range(4)]
    
    # Process each input separately with TimeDistributed Conv2D
    processed = []
    for i, inp in enumerate(inputs):
        # TimeDistributed applies Conv2D to each time frame independently
        x = layers.TimeDistributed(
            layers.Conv2D(8, (7, 7), activation='relu', padding='same'),
            name=f'timedist_conv_{i}'
        )(inp)
        x = layers.TimeDistributed(layers.MaxPooling2D((4, 4)))(x)
        processed.append(x)
    
    # Concatenate processed features
    concatenated = layers.Concatenate(axis=-1)(processed)
    
    # Further processing
    conv2 = layers.Conv3D(64, (2, 5, 5), activation='relu', 
                         padding='same', name='conv3d_timedist_target')(concatenated)
    pool2 = layers.MaxPooling3D((1, 4, 4))(conv2)
    
    flatten = layers.Flatten()(pool2)
    dense1 = layers.Dense(128, activation='relu')(flatten)
    dropout = layers.Dropout(0.5)(dense1)
    output = layers.Dense(2, activation='softmax', name='output')(dropout)
    
    model = Model(inputs=inputs, outputs=output)
    return model

timedist_model = build_timedistributed_model()
timedist_model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

print("\n" + "="*70)
print("TIMEDISTRIBUTED MODEL ARCHITECTURE")
print("="*70)
timedist_model.summary()

# ============================================================================
# 4. TRAIN BOTH MODELS (briefly, just for demonstration)
# ============================================================================

print("\n" + "="*70)
print("TRAINING MODELS")
print("="*70)

print("\nTraining concatenated model...")
concat_model.fit(train_inputs, train_labels, epochs=3, batch_size=2, verbose=1)

print("\nTraining TimeDistributed model...")
timedist_model.fit(train_inputs, train_labels, epochs=3, batch_size=2, verbose=1)

# ============================================================================
# 5. GRAD-CAM IMPLEMENTATION FOR CONCATENATED MODEL
# ============================================================================

def get_gradcam_per_input_concatenated(model, inputs, target_layer_name, 
                                      pred_index=None, epsilon=1e-10):
    """
    Extract Grad-CAM for each of the 4 concatenated inputs separately
    
    Args:
        model: Trained model
        inputs: List of 4 inputs, each shape (1, 6, 256, 256, 1)
        target_layer_name: Name of the conv layer to extract activations from
        pred_index: Target class (None = use predicted class)
        epsilon: Small value to avoid division by zero
    
    Returns:
        List of 4 Grad-CAM heatmaps, one per input
    """
    # Create a model that outputs both conv activations and predictions
    grad_model = Model(
        inputs=model.input,
        outputs=[model.get_layer(target_layer_name).output, model.output]
    )
    
    # Forward pass with gradient tape
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(inputs)
        
        # Use predicted class if not specified
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        
        # Get the score for the target class
        class_channel = predictions[:, pred_index]
    
    # Compute gradients of class score with respect to conv layer output
    grads = tape.gradient(class_channel, conv_outputs)
    
    # Get the shape of conv outputs
    # Expected shape: (batch, time, height, width, channels)
    num_total_channels = conv_outputs.shape[-1]
    num_channels_per_input = num_total_channels // 4
    
    print(f"\nConv layer output shape: {conv_outputs.shape}")
    print(f"Channels per input: {num_channels_per_input}")
    
    gradcams = []
    
    for i in range(4):
        # Extract channels corresponding to input i
        start_ch = i * num_channels_per_input
        end_ch = (i + 1) * num_channels_per_input
        
        print(f"\nProcessing input {i} (channels {start_ch}:{end_ch})...")
        
        # Get gradients and activations for this input's channels
        input_grads = grads[:, :, :, :, start_ch:end_ch]
        input_conv = conv_outputs[:, :, :, :, start_ch:end_ch]
        
        # Global average pooling of gradients (importance weights)
        # Average over batch, time, height, width dimensions
        pooled_grads = tf.reduce_mean(input_grads, axis=[0, 1, 2, 3])
        
        # Average activations across time dimension (6 frames -> 1)
        # Shape: (batch, height, width, channels)
        pooled_conv = tf.reduce_mean(input_conv, axis=1)
        
        # Weight each channel by its importance
        # Reshape pooled_grads to enable broadcasting: (channels,) -> (1, 1, 1, channels)
        weights = tf.reshape(pooled_grads, (1, 1, 1, num_channels_per_input))
        pooled_conv = pooled_conv * weights
        
        # Create heatmap by averaging across channels
        heatmap = tf.reduce_mean(pooled_conv, axis=-1)[0]
        
        # Apply ReLU to focus on positive influences
        heatmap = tf.maximum(heatmap, 0)
        
        # Normalize to [0, 1]
        heatmap_max = tf.reduce_max(heatmap)
        if heatmap_max > epsilon:
            heatmap = heatmap / heatmap_max
        
        gradcams.append(heatmap.numpy())
    
    return gradcams, pred_index.numpy()

# ============================================================================
# 6. GRAD-CAM IMPLEMENTATION FOR TIMEDISTRIBUTED MODEL
# ============================================================================

def get_gradcam_timedistributed(model, inputs, target_layer_name, 
                               pred_index=None, epsilon=1e-10):
    """
    Extract Grad-CAM for TimeDistributed model
    Computes separate Grad-CAMs for each input's TimeDistributed layer
    """
    grad_model = Model(
        inputs=model.input,
        outputs=[model.get_layer(target_layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(inputs)
        
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        
        class_channel = predictions[:, pred_index]
    
    grads = tape.gradient(class_channel, conv_outputs)
    
    num_total_channels = conv_outputs.shape[-1]
    num_channels_per_input = num_total_channels // 4
    
    gradcams = []
    
    for i in range(4):
        start_ch = i * num_channels_per_input
        end_ch = (i + 1) * num_channels_per_input
        
        input_grads = grads[:, :, :, :, start_ch:end_ch]
        input_conv = conv_outputs[:, :, :, :, start_ch:end_ch]
        
        pooled_grads = tf.reduce_mean(input_grads, axis=[0, 1, 2, 3])
        pooled_conv = tf.reduce_mean(input_conv, axis=1)
        
        # Weight each channel by its importance
        weights = tf.reshape(pooled_grads, (1, 1, 1, num_channels_per_input))
        pooled_conv = pooled_conv * weights
        
        heatmap = tf.reduce_mean(pooled_conv, axis=-1)[0]
        heatmap = tf.maximum(heatmap, 0)
        
        heatmap_max = tf.reduce_max(heatmap)
        if heatmap_max > epsilon:
            heatmap = heatmap / heatmap_max
        
        gradcams.append(heatmap.numpy())
    
    return gradcams, pred_index.numpy()

# ============================================================================
# 7. COMPUTE GRAD-CAM FOR BOTH MODELS
# ============================================================================

print("\n" + "="*70)
print("COMPUTING GRAD-CAM")
print("="*70)

# Use first test sample
test_sample = [inp[0:1] for inp in test_inputs]

print("\nComputing Grad-CAM for concatenated model...")
gradcams_concat, pred_concat = get_gradcam_per_input_concatenated(
    concat_model, 
    test_sample, 
    target_layer_name='conv3d_target'
)

print("\nComputing Grad-CAM for TimeDistributed model...")
gradcams_timedist, pred_timedist = get_gradcam_timedistributed(
    timedist_model,
    test_sample,
    target_layer_name='conv3d_timedist_target'
)

# ============================================================================
# 8. VISUALIZE RESULTS
# ============================================================================

print("\n" + "="*70)
print("VISUALIZATION")
print("="*70)

variable_names = ['Temperature', 'Pressure', 'Humidity', 'Wind Speed']

fig, axes = plt.subplots(3, 4, figsize=(16, 12))
fig.suptitle('Grad-CAM Comparison: Concatenated vs TimeDistributed Models', 
             fontsize=16, fontweight='bold')

for i in range(4):
    # Original input (average across time)
    original = np.mean(test_sample[i][0], axis=0)[:, :, 0]
    
    # Plot original
    im0 = axes[0, i].imshow(original, cmap='viridis')
    axes[0, i].set_title(f'{variable_names[i]}\n(Averaged Input)', fontsize=10)
    axes[0, i].axis('off')
    plt.colorbar(im0, ax=axes[0, i], fraction=0.046)
    
    # Plot Grad-CAM from concatenated model
    im1 = axes[1, i].imshow(gradcams_concat[i], cmap='jet', alpha=0.8)
    axes[1, i].set_title(f'Grad-CAM (Concat)\nPred: {pred_concat}', fontsize=10)
    axes[1, i].axis('off')
    plt.colorbar(im1, ax=axes[1, i], fraction=0.046)
    
    # Plot Grad-CAM from TimeDistributed model
    im2 = axes[2, i].imshow(gradcams_timedist[i], cmap='jet', alpha=0.8)
    axes[2, i].set_title(f'Grad-CAM (TimeDist)\nPred: {pred_timedist}', fontsize=10)
    axes[2, i].axis('off')
    plt.colorbar(im2, ax=axes[2, i], fraction=0.046)

plt.tight_layout()
plt.savefig('gradcam_comparison.png', dpi=150, bbox_inches='tight')
print("\nVisualization saved as 'gradcam_comparison.png'")
plt.show()

# ============================================================================
# 9. OVERLAY VISUALIZATION
# ============================================================================

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle('Grad-CAM Overlays on Original Inputs', fontsize=16, fontweight='bold')

for i in range(4):
    original = np.mean(test_sample[i][0], axis=0)[:, :, 0]
    
    # Concatenated model overlay
    axes[0, i].imshow(original, cmap='gray')
    axes[0, i].imshow(gradcams_concat[i], cmap='jet', alpha=0.4)
    axes[0, i].set_title(f'{variable_names[i]}\n(Concatenated)', fontsize=10)
    axes[0, i].axis('off')
    
    # TimeDistributed model overlay
    axes[1, i].imshow(original, cmap='gray')
    axes[1, i].imshow(gradcams_timedist[i], cmap='jet', alpha=0.4)
    axes[1, i].set_title(f'{variable_names[i]}\n(TimeDistributed)', fontsize=10)
    axes[1, i].axis('off')

plt.tight_layout()
plt.savefig('gradcam_overlays.png', dpi=150, bbox_inches='tight')
print("Overlay visualization saved as 'gradcam_overlays.png'")
plt.show()

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
print(f"\nConcatenated model prediction: {pred_concat}")
print(f"TimeDistributed model prediction: {pred_timedist}")
print(f"True label: {test_labels[0]}")
print("\nThe Grad-CAM heatmaps show which spatial regions each model")
print("focuses on for each meteorological variable when making predictions.")