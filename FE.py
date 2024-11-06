import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Generate dummy dataset
def generate_dummy_faces(n_individuals=5, n_images_per_person=3, img_size=128):
    """
    Generate dummy face dataset
    Returns: array of shape (n_individuals * n_images_per_person, img_size, img_size)
    """
    n_total = n_individuals * n_images_per_person
    dummy_faces = []
    
    for i in range(n_individuals):
        # Create base pattern for each individual
        base_pattern = np.random.randn(img_size, img_size)
        
        # Generate slightly different versions for each person
        for j in range(n_images_per_person):
            # Add random noise to create variation
            variation = base_pattern + 0.1 * np.random.randn(img_size, img_size)
            dummy_faces.append(variation)
    
    return np.array(dummy_faces)

# Simulate Contourlet Transform (since we don't have actual Contourlet Transform)
def simulate_contourlet_transform(image, levels=2):
    """
    Simulate Contourlet Transform by creating sub-bands
    Returns: List of coefficients matrices
    """
    h, w = image.shape
    coeffs = []
    
    # Simulate different frequency bands
    for level in range(levels):
        size = h // (2 ** (level + 1))
        c1 = np.resize(image, (size, size))
        c2 = np.resize(image, (size, size)) + 0.1 * np.random.randn(size, size)
        c3 = np.resize(image, (size, size)) + 0.1 * np.random.randn(size, size)
        
        coeffs.extend([c1, c2, c3])
    
    return coeffs

# Feature extraction
def extract_features(image):
    """
    Extract features using simulated Contourlet Transform
    """
    coeffs = simulate_contourlet_transform(image)
    # Flatten and concatenate all coefficients
    feature_vector = np.concatenate([c.flatten() for c in coeffs])
    return feature_vector

# Main execution
if __name__ == "__main__":
    # Parameters
    N_INDIVIDUALS = 5
    N_IMAGES_PER_PERSON = 3
    IMAGE_SIZE = 128
    
    # Generate dummy dataset
    print("Generating dummy dataset...")
    faces = generate_dummy_faces(N_INDIVIDUALS, N_IMAGES_PER_PERSON, IMAGE_SIZE)
    print(f"Dataset shape: {faces.shape}")
    
    # Extract features for all images
    print("\nExtracting features...")
    features = np.array([extract_features(face) for face in faces])
    print(f"Feature matrix shape: {features.shape}")
    
    # Apply PCA
    print("\nApplying PCA...")
    pca = PCA(n_components=0.95)  # Keep 95% of variance
    reduced_features = pca.fit_transform(features)
    print(f"Reduced feature matrix shape: {reduced_features.shape}")
    
    # Visualize some results
    plt.figure(figsize=(15, 5))
    
    # Plot original image
    plt.subplot(131)
    plt.imshow(faces[0], cmap='gray')
    plt.title('Example Generated Face')
    
    # Plot feature vector
    plt.subplot(132)
    plt.plot(features[0])
    plt.title('Feature Vector')
    
    # Plot reduced features
    plt.subplot(133)
    plt.plot(reduced_features[0])
    plt.title('PCA Reduced Features')
    
    plt.tight_layout()
    plt.show()
    
    # Print some statistics
    print("\nDataset Statistics:")
    print(f"Number of individuals: {N_INDIVIDUALS}")
    print(f"Images per person: {N_IMAGES_PER_PERSON}")
    print(f"Original feature dimension: {features.shape[1]}")
    print(f"Reduced feature dimension: {reduced_features.shape[1]}")
    print(f"Dimension reduction: {features.shape[1]/reduced_features.shape[1]:.2f}x")

    # Generate a test image
    print("\nGenerating test image...")
    test_image = generate_dummy_faces(1, 1, IMAGE_SIZE)[0]
    test_features = extract_features(test_image)
    test_reduced = pca.transform([test_features])
    
    print(f"Test feature vector shape: {test_features.shape}")
    print(f"Test reduced feature vector shape: {test_reduced.shape}")