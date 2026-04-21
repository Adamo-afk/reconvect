"""
Grad-CAM Implementation in TensorFlow: Understanding Feature Maps vs Gradient Maps
==================================================================================
This implementation demonstrates the key differences between:
1. Feature Maps: What patterns the network detects (forward pass)
2. Gradient Maps: What influences the decision (backward pass)  
3. Grad-CAM: How to combine both for interpretability

TensorFlow/Keras implementation with detailed visualizations.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ============================================================================
# STEP 1: Define a Simple CNN for MNIST using Keras
# ============================================================================

class SimpleMNISTCNN(keras.Model):
    """
    A simple CNN for MNIST digit classification using TensorFlow/Keras.
    Designed to be interpretable with clear layer structure.
    """
    def __init__(self):
        super(SimpleMNISTCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = layers.Conv2D(16, kernel_size=5, padding='same', activation='relu', name='conv1')
        self.pool1 = layers.MaxPooling2D(2, 2)
        
        self.conv2 = layers.Conv2D(32, kernel_size=5, padding='same', activation='relu', name='conv2')
        self.pool2 = layers.MaxPooling2D(2, 2)
        
        self.conv3 = layers.Conv2D(64, kernel_size=5, padding='same', activation='relu', name='conv3')
        self.pool3 = layers.MaxPooling2D(2, 2)
        
        # Fully connected layers
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(128, activation='relu')
        self.dropout = layers.Dropout(0.2)
        self.fc2 = layers.Dense(10)  # 10 classes for digits 0-9
        
    def call(self, inputs, training=False):
        # Forward pass through the network
        x = self.conv1(inputs)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.pool3(x)
        
        x = self.flatten(x)
        x = self.fc1(x)
        if training:
            x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def model(self):
        x = tf.keras.layers.Input(shape=(28, 28, 1))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

# ============================================================================
# STEP 2: Implement Grad-CAM using TensorFlow's GradientTape
# ============================================================================

class GradCAM:
    """
    Grad-CAM implementation for TensorFlow/Keras models.
    Shows how feature maps and gradients are combined for interpretability.
    """
    def __init__(self, model, layer_name='conv3'):
        self.model = model
        self.layer_name = layer_name
        
        # Create a model that outputs both predictions and feature maps
        self.grad_model = self._build_grad_model()
        
    def _build_grad_model(self):
        """Build a model that outputs both predictions and conv layer outputs"""
        # Get the layer object
        target_layer = None
        for layer in self.model.layers:
            if layer.name == self.layer_name:
                target_layer = layer
                break
        
        if target_layer is None:
            raise ValueError(f"Layer {self.layer_name} not found")
            
        # Create a model that maps input -> [predictions, last_conv_outputs]
        grad_model = keras.Model(
            inputs=self.model.input,
            outputs=[self.model.output, target_layer.output]
        )
        return grad_model
    
    def generate(self, input_image, target_class=None):
        """
        Generate Grad-CAM heatmap showing which regions are important.
        
        Args:
            input_image: Input image tensor (1, 28, 28, 1)
            target_class: Target class for explanation (if None, uses predicted class)
        
        Returns:
            heatmap: Grad-CAM heatmap
            feature_maps: Raw feature maps from target layer
            gradients: Raw gradients from backpropagation
            weights: Channel-wise importance weights
        """
        # Ensure input has batch dimension
        if len(input_image.shape) == 3:
            input_image = tf.expand_dims(input_image, 0)
        
        # Use GradientTape to compute gradients
        with tf.GradientTape() as tape:
            # Forward pass
            predictions, conv_outputs = self.grad_model(input_image)
            
            # Get target class if not specified
            if target_class is None:
                target_class = tf.argmax(predictions[0])
            
            # Get the score for target class
            class_score = predictions[:, target_class]
        
        # Compute gradients of class score with respect to feature maps
        gradients = tape.gradient(class_score, conv_outputs)
        
        # Remove batch dimension for processing
        conv_outputs = conv_outputs[0]
        gradients = gradients[0]
        
        # Compute channel-wise weights (Global Average Pooling of gradients)
        weights = tf.reduce_mean(gradients, axis=(0, 1))
        
        # Compute Grad-CAM by weighting feature maps
        grad_cam = tf.zeros(conv_outputs.shape[:2])
        for i in range(len(weights)):
            grad_cam += weights[i] * conv_outputs[:, :, i]
        
        # Apply ReLU (only keep positive influences)
        grad_cam = tf.nn.relu(grad_cam)
        
        # Normalize to 0-1 range
        if tf.reduce_max(grad_cam) > 0:
            grad_cam = grad_cam / tf.reduce_max(grad_cam)
        
        return grad_cam.numpy(), conv_outputs.numpy(), gradients.numpy(), weights.numpy()

# ============================================================================
# STEP 3: Feature Map Extraction Functions
# ============================================================================

def get_feature_maps(model, input_image, layer_names=['conv1', 'conv2', 'conv3']):
    """
    Extract feature maps from multiple layers.
    These show WHAT patterns the network detects at each layer.
    """
    # Create intermediate models for each layer
    feature_maps = {}
    
    for layer_name in layer_names:
        # Get index of layer name
        layer_index = layer_names.index(layer_name)

        # Create model that outputs this layer
        intermediate_model = keras.Model(
            inputs=model.inputs,
            outputs=model.layers[layer_index].output
        )
        # Get feature maps
        if len(input_image.shape) == 3:
            input_image_batch = tf.expand_dims(input_image, 0)
        else:
            input_image_batch = input_image
            
        features = intermediate_model(input_image_batch)
        feature_maps[layer_name] = features[0].numpy()  # Remove batch dimension
    
    return feature_maps

# ============================================================================
# STEP 4: Visualization Functions
# ============================================================================

def visualize_feature_maps(feature_maps, layer_name):
    """
    Visualize feature maps from a specific layer.
    Shows what patterns the network learned to detect.
    """
    features = feature_maps[layer_name]
    n_features = min(16, features.shape[-1])  # Show max 16 feature maps
    
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(f'Feature Maps from {layer_name} (Forward Pass - What Network Sees)', 
                 fontsize=16, fontweight='bold')
    
    for idx in range(n_features):
        plt.subplot(4, 4, idx + 1)
        plt.imshow(features[:, :, idx], cmap='viridis')
        plt.title(f'Filter {idx}')
        plt.axis('off')
        plt.colorbar(fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    return fig

def visualize_gradients(gradients, weights):
    """
    Visualize gradients from backpropagation.
    Shows HOW MUCH each region influences the decision.
    """
    n_channels = min(16, gradients.shape[-1])
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle('Gradients (Backward Pass - Importance for Decision)', 
                 fontsize=16, fontweight='bold')
    
    for idx in range(n_channels):
        plt.subplot(4, 4, idx + 1)
        grad_map = gradients[:, :, idx]
        plt.imshow(grad_map, cmap='RdBu_r', 
                  vmin=-np.abs(grad_map).max(), 
                  vmax=np.abs(grad_map).max())
        plt.title(f'Ch {idx} (w={weights[idx]:.2f})')
        plt.axis('off')
        plt.colorbar(fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    return fig

def visualize_gradcam_components(input_image, grad_cam, feature_maps, gradients, 
                                weights, predicted_class, confidence):
    """
    Comprehensive visualization showing all components of Grad-CAM.
    Clearly demonstrates how feature maps and gradients combine.
    """
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    # Original image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(input_image.squeeze(), cmap='gray')
    ax1.set_title(f'Input Image\nPredicted: {predicted_class}\nConfidence: {confidence:.1f}%')
    ax1.axis('off')
    
    # Average feature maps
    ax2 = fig.add_subplot(gs[0, 1])
    avg_features = np.mean(feature_maps, axis=-1)
    im2 = ax2.imshow(avg_features, cmap='viridis')
    ax2.set_title('Avg Feature Maps\n(What network detects)')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046)
    
    # Average absolute gradients
    ax3 = fig.add_subplot(gs[0, 2])
    avg_gradients = np.mean(np.abs(gradients), axis=-1)
    im3 = ax3.imshow(avg_gradients, cmap='hot')
    ax3.set_title('Avg |Gradients|\n(Importance)')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046)
    
    # Grad-CAM result
    ax4 = fig.add_subplot(gs[0, 3])
    im4 = ax4.imshow(grad_cam, cmap='jet')
    ax4.set_title('Grad-CAM\n(Features × Gradients)')
    ax4.axis('off')
    plt.colorbar(im4, ax=ax4, fraction=0.046)
    
    # Show top contributing feature maps
    top_indices = np.argsort(weights)[::-1][:4]
    for i, idx in enumerate(top_indices):
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(feature_maps[:, :, idx], cmap='viridis')
        ax.set_title(f'Top Feature {i+1}\nChannel {idx}\nWeight: {weights[idx]:.3f}')
        ax.axis('off')
    
    # Mathematical formula
    ax_math = fig.add_subplot(gs[2, :2])
    ax_math.text(0.5, 0.5, 
                'Grad-CAM Formula:\n\n'
                'Grad-CAM = ReLU(Σᵢ αᵢ · Aᵢ)\n\n'
                'where:\n'
                '• Aᵢ = Feature map from channel i\n'
                '• αᵢ = Global avg pooling of gradients\n'
                '       for channel i (importance weight)\n'
                '• ReLU = Keep only positive values',
                fontsize=12, ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax_math.axis('off')
    
    # Overlay on original
    ax_overlay = fig.add_subplot(gs[2, 2])
    ax_overlay.imshow(input_image.squeeze(), cmap='gray')
    # Resize Grad-CAM to match input size
    grad_cam_resized = tf.image.resize(
        tf.expand_dims(tf.expand_dims(grad_cam, -1), 0),
        (28, 28),
        method='bilinear'
    )[0, :, :, 0].numpy()
    ax_overlay.imshow(grad_cam_resized, cmap='jet', alpha=0.5)
    ax_overlay.set_title('Grad-CAM Overlay\n(Important regions)')
    ax_overlay.axis('off')
    
    # Channel weights distribution  
    ax_weights = fig.add_subplot(gs[2, 3])
    ax_weights.bar(range(len(weights)), sorted(weights, reverse=True))
    ax_weights.set_title('Channel Importance Weights\n(from gradient pooling)')
    ax_weights.set_xlabel('Channel (sorted)')
    ax_weights.set_ylabel('Weight')
    ax_weights.grid(True, alpha=0.3)
    
    fig.suptitle('Grad-CAM Components: Feature Maps × Gradient Weights = Attribution', 
                fontsize=16, fontweight='bold')
    
    return fig

# ============================================================================
# STEP 5: Training Function
# ============================================================================

def create_and_train_model(x_train, y_train, x_test, y_test, epochs=3):
    """
    Create and train the CNN model on MNIST.
    """
    # Build the model
    model = SimpleMNISTCNN().model()
    
    # Compile with optimizer and loss
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    # # Build the model by calling it once
    # model.build(input_shape=(None, 28, 28, 1))
    
    # # Display model summary
    # model.summary()
    
    # Train the model
    print("\n" + "="*50)
    print("Training CNN on MNIST...")
    print("="*50)
    
    history = model.fit(
        x_train, y_train,
        batch_size=64,
        epochs=epochs,
        validation_split=0.1,
        verbose=1
    )
    
    # Model summary
    model.summary()

    # Evaluate on test set
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest accuracy: {test_accuracy:.4f}")
    
    return model, history

# ============================================================================
# STEP 6: Main Execution
# ============================================================================

def main():
    print("TensorFlow Grad-CAM Implementation")
    print("="*50)
    
    # Load and preprocess MNIST dataset
    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Normalize and reshape
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Add channel dimension (28, 28) -> (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    
    print(f"Training data shape: {x_train.shape}")
    print(f"Test data shape: {x_test.shape}")
    
    # Create and train the model
    model, _ = create_and_train_model(x_train, y_train, x_test, y_test, epochs=3)
    
    # ============================================================
    # ANALYSIS: Select a test image for detailed visualization
    # ============================================================
    test_idx = 0
    test_image = x_test[test_idx]
    test_label = y_test[test_idx]
    
    # Make prediction
    test_image_batch = np.expand_dims(test_image, 0)
    predictions = model.predict(test_image_batch, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = tf.nn.softmax(predictions[0]).numpy().max() * 100
    
    print("\n" + "="*50)
    print(f"Test Image Analysis")
    print(f"True Label: {test_label}")
    print(f"Predicted: {predicted_class}")
    print(f"Confidence: {confidence:.1f}%")
    print("="*50)
    
    # ============================================================
    # VISUALIZATION 1: Feature Maps (Forward Pass)
    # ============================================================
    print("\n1. Generating Feature Maps (What the network sees)...")
    
    # Extract feature maps from all conv layers
    feature_maps = get_feature_maps(model, test_image, ['conv1', 'conv2', 'conv3'])
    
    # Visualize feature maps for each layer
    for layer_name in ['conv1', 'conv2', 'conv3']:
        fig = visualize_feature_maps(feature_maps, layer_name)
        plt.show()
    
    # ============================================================
    # VISUALIZATION 2: Grad-CAM Analysis (Backward Pass)
    # ============================================================
    print("\n2. Generating Grad-CAM Analysis (What matters for decision)...")
    
    # Create Grad-CAM analyzer
    grad_cam = GradCAM(model, layer_name='conv3')
    
    # Generate Grad-CAM heatmap and components
    heatmap, conv3_features, gradients, weights = grad_cam.generate(test_image, predicted_class)
    
    # Visualize gradients
    fig = visualize_gradients(gradients, weights)
    plt.show()
    
    # ============================================================
    # VISUALIZATION 3: Complete Component Analysis
    # ============================================================
    print("\n3. Generating Complete Grad-CAM Component Breakdown...")
    
    fig = visualize_gradcam_components(
        test_image, heatmap, conv3_features, gradients,
        weights, predicted_class, confidence
    )
    plt.show()
    
    # ============================================================
    # VISUALIZATION 4: Comparison Across Multiple Digits
    # ============================================================
    print("\n4. Comparing Grad-CAM Across Different Digits...")
    
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    fig.suptitle('Grad-CAM Analysis Across Different Digits', fontsize=16, fontweight='bold')
    
    # Analyze 5 different test images
    for i in range(5):
        test_idx = i * 100  # Sample different images
        test_image = x_test[test_idx]
        test_label = y_test[test_idx]
        
        # Predict
        predictions = model.predict(np.expand_dims(test_image, 0), verbose=0)
        predicted = np.argmax(predictions[0])
        
        # Generate Grad-CAM
        heatmap, _, _, _ = grad_cam.generate(test_image, predicted)
        
        # Original image
        axes[0, i].imshow(test_image.squeeze(), cmap='gray')
        axes[0, i].set_title(f'True: {test_label}\nPred: {predicted}')
        axes[0, i].axis('off')
        
        # Grad-CAM heatmap
        axes[1, i].imshow(heatmap, cmap='jet')
        axes[1, i].set_title('Grad-CAM')
        axes[1, i].axis('off')
        
        # Overlay
        axes[2, i].imshow(test_image.squeeze(), cmap='gray')
        heatmap_resized = tf.image.resize(
            tf.expand_dims(tf.expand_dims(heatmap, -1), 0),
            (28, 28)
        )[0, :, :, 0].numpy()
        axes[2, i].imshow(heatmap_resized, cmap='jet', alpha=0.5)
        axes[2, i].set_title('Overlay')
        axes[2, i].axis('off')
    
    axes[0, 0].set_ylabel('Input', fontsize=12)
    axes[1, 0].set_ylabel('Grad-CAM', fontsize=12)
    axes[2, 0].set_ylabel('Overlay', fontsize=12)
    
    plt.tight_layout()
    plt.show()
    
    # ============================================================
    # VISUALIZATION 5: Feature Evolution Through Layers
    # ============================================================
    print("\n5. Showing Feature Evolution Through Network Layers...")
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle('Feature Evolution: From Input to Deep Features', fontsize=16, fontweight='bold')
    
    # Input image
    axes[0].imshow(test_image.squeeze(), cmap='gray')
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    # Feature maps at different depths
    layer_names = ['conv1', 'conv2', 'conv3']
    for idx, layer_name in enumerate(layer_names):
        # Get average activation across all filters
        avg_activation = np.mean(feature_maps[layer_name], axis=-1)
        axes[idx + 1].imshow(avg_activation, cmap='viridis')
        axes[idx + 1].set_title(f'{layer_name}\nSize: {avg_activation.shape}')
        axes[idx + 1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # ============================================================
    # Summary and Key Insights
    # ============================================================
    print("\n" + "="*50)
    print("ANALYSIS COMPLETE!")
    print("="*50)
    print("\nKey Differences Demonstrated:")
    print("-" * 40)
    print("1. FEATURE MAPS (Forward Pass):")
    print("   - Show what patterns/features the network detects")
    print("   - Different filters detect different patterns")
    print("   - Get progressively more complex in deeper layers")
    print("   - Always positive after ReLU activation")
    print()
    print("2. GRADIENT MAPS (Backward Pass):")
    print("   - Show how sensitive the output is to each location")
    print("   - Can be positive or negative")
    print("   - Indicate importance for the final decision")
    print("   - Same resolution as the layer they're computed for")
    print()
    print("3. GRAD-CAM (Feature Maps × Gradient Weights):")
    print("   - Combines 'what network sees' with 'what matters'")
    print("   - Shows which detected features are important")
    print("   - Always positive (after ReLU)")
    print("   - Provides interpretable visualization of decisions")
    print()
    print("The key insight: Grad-CAM = Features × Importance")
    print("="*50)

if __name__ == "__main__":
    main()
