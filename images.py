import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hessenberg
from PIL import Image
import os
from pathlib import Path


class HessenbergLorenzEncryption:
    """
    Image encryption using Hessenberg decomposition and Lorenz chaotic system
    """
    
    def __init__(self, sigma=10, rho=28, beta=8/3, x0=0.1, y0=0.2, z0=0.3, dt=0.01):
        """Initialize Lorenz system parameters"""
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0
        self.dt = dt
    
    def generate_lorenz_sequences(self, num_pixels):
        """Generate Lorenz chaotic sequences"""
        x = np.zeros(num_pixels)
        y = np.zeros(num_pixels)
        z = np.zeros(num_pixels)
        
        x[0] = self.x0
        y[0] = self.y0
        z[0] = self.z0
        
        for i in range(num_pixels - 1):
            x[i+1] = x[i] + self.dt * (self.sigma * (y[i] - x[i]))
            y[i+1] = y[i] + self.dt * (x[i] * (self.rho - z[i]) - y[i])
            z[i+1] = z[i] + self.dt * (x[i] * y[i] - self.beta * z[i])
        
        return x, y, z
    
    def load_image_from_array(self, img_array):
        """Preprocess image array"""
        # Handle different array shapes
        if img_array.ndim == 3:  # (height, width, channels)
            # Convert RGB to grayscale if needed
            if img_array.shape[2] == 3:
                # RGB to grayscale conversion
                img_array = 0.299 * img_array[:,:,0] + 0.587 * img_array[:,:,1] + 0.114 * img_array[:,:,2]
            elif img_array.shape[2] == 1:
                img_array = img_array[:,:,0]
        
        # Normalize to [0, 1] if not already
        if img_array.max() > 1.0:
            img_array = img_array / 255.0
        
        return img_array.astype(np.float64)
    
    def encrypt(self, image_array):
        """Encrypt image using Hessenberg decomposition and Lorenz chaos"""
        rows, cols = image_array.shape
        num_pixels = rows * cols
        
        # Hessenberg decomposition
        H, Q = hessenberg(image_array, calc_q=True)
        
        # Generate Lorenz chaotic sequences
        x, y, z = self.generate_lorenz_sequences(num_pixels)
        
        # Generate chaotic keys
        X_key = np.mod(x, 1).reshape(rows, cols)
        Y_key = np.mod(y, 1).reshape(rows, cols)
        Z_key = np.mod(z, 1).reshape(rows, cols)
        
        # Encrypt Q and H matrices
        Q_encrypted = Q + X_key + Z_key
        H_encrypted = H + Y_key + Z_key
        
        # Convert to uint8 for visualization
        Q_vis = np.mod(np.round(Q_encrypted * 1e5), 256)
        H_vis = np.mod(np.round(H_encrypted * 1e5), 256)
        encrypted_image = np.mod(Q_vis + H_vis, 256).astype(np.uint8)
        
        # Store encryption data for decryption
        encryption_data = {
            'Q_encrypted': Q_encrypted,
            'H_encrypted': H_encrypted,
            'rows': rows,
            'cols': cols,
            'encrypted_image': encrypted_image
        }
        
        return encryption_data
    
    def decrypt(self, encryption_data):
        """Decrypt image by reversing the encryption process"""
        Q_encrypted = encryption_data['Q_encrypted']
        H_encrypted = encryption_data['H_encrypted']
        rows = encryption_data['rows']
        cols = encryption_data['cols']
        num_pixels = rows * cols
        
        # Regenerate the same chaotic keys
        x, y, z = self.generate_lorenz_sequences(num_pixels)
        
        X_key = np.mod(x, 1).reshape(rows, cols)
        Y_key = np.mod(y, 1).reshape(rows, cols)
        Z_key = np.mod(z, 1).reshape(rows, cols)
        
        # Reverse the encryption
        Q_decrypted = Q_encrypted - X_key - Z_key
        H_decrypted = H_encrypted - Y_key - Z_key
        
        # Reconstruct: A = Q * H * Q'
        decrypted_image = Q_decrypted @ H_decrypted @ Q_decrypted.T
        
        # Convert back to valid image format
        decrypted_image = np.real(decrypted_image)  # Take real part
        decrypted_image = np.clip(decrypted_image, 0, 1)  # Clip to [0, 1]
        
        return decrypted_image
    
    def calculate_metrics(self, original, decrypted):
        """Calculate MSE and PSNR"""
        mse = np.mean((original - decrypted) ** 2)
        
        if mse == 0:
            psnr = float('inf')
        else:
            psnr = 10 * np.log10(1 / mse)
        
        return mse, psnr
    
    def process_npz_archive(self, npz_path, output_dir='output', max_images=None, visualize_samples=3):
        """
        Process all images from a .npz archive file
        
        Parameters:
        -----------
        npz_path : str
            Path to the .npz file containing image dataset
        output_dir : str
            Directory to save encrypted and decrypted images
        max_images : int or None
            Maximum number of images to process (None = process all)
        visualize_samples : int
            Number of samples to visualize (first N images)
        """
        print(f"\n{'='*70}")
        print(f"Loading NPZ Archive: {npz_path}")
        print(f"{'='*70}\n")
        
        # Load the NPZ file
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"NPZ file not found: {npz_path}")
        
        # Load with allow_pickle=True to handle object arrays
        npz_data = np.load(npz_path, allow_pickle=True)
        
        # Display available arrays in the NPZ file
        print(f"Available arrays in NPZ file:")
        for key in npz_data.files:
            print(f"  - {key}: shape {npz_data[key].shape}, dtype {npz_data[key].dtype}")
        print()
        
        # Determine which array to use
        # Priority: 'images' > 'data' > 'X_train' > 'x_train' > first array
        data_key = None
        for key in ['images', 'data', 'X_train', 'x_train', 'X', 'x']:
            if key in npz_data.files:
                data_key = key
                break
        
        if data_key is None:
            data_key = npz_data.files[0]
        
        print(f"Using array: '{data_key}'")
        images = npz_data[data_key]
        
        # Handle object arrays
        if images.dtype == object:
            print(f"Dataset contains object arrays. Converting to numeric format...")
            # Try to convert object array to numeric
            try:
                # If it's an array of arrays, try to stack them
                images = np.array([np.array(img) for img in images])
            except Exception as e:
                print(f"Warning: Could not convert object array: {e}")
                print("Attempting to process individual objects...")
        
        print(f"Dataset shape: {images.shape}")
        print(f"Dataset dtype: {images.dtype}")
        
        # Determine number of images to process
        if images.ndim == 4:  # (N, H, W, C) or (N, C, H, W)
            num_images = images.shape[0]
        elif images.ndim == 3:  # (N, H, W) - grayscale
            num_images = images.shape[0]
        elif images.ndim == 1:  # Array of image objects
            num_images = images.shape[0]
            print(f"Detected 1D array of {num_images} image objects")
        else:
            raise ValueError(f"Unexpected data shape: {images.shape}")
        
        if max_images is not None:
            num_images = min(num_images, max_images)
        
        print(f"Processing {num_images} images...\n")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create subdirectories
        encrypted_dir = os.path.join(output_dir, 'encrypted')
        decrypted_dir = os.path.join(output_dir, 'decrypted')
        comparison_dir = os.path.join(output_dir, 'comparison')
        os.makedirs(encrypted_dir, exist_ok=True)
        os.makedirs(decrypted_dir, exist_ok=True)
        os.makedirs(comparison_dir, exist_ok=True)
        
        results = []
        
        for idx in range(num_images):
            print(f"{'='*70}")
            print(f"Processing Image {idx + 1}/{num_images}")
            print(f"{'='*70}")
            
            try:
                # Extract single image
                if images.ndim == 4:
                    img_array = images[idx]
                elif images.ndim == 3:
                    img_array = images[idx]
                elif images.ndim == 1:  # Array of objects
                    img_array = images[idx]
                    # Handle object dtype
                    if isinstance(img_array, (list, tuple)) or (hasattr(img_array, 'dtype') and img_array.dtype == object):
                        img_array = np.array(img_array)
                    # Ensure it's a numpy array
                    img_array = np.asarray(img_array)
                else:
                    raise ValueError(f"Unexpected array dimension: {images.ndim}")
                
                # Check if the image is valid
                if img_array.size == 0:
                    print(f"⚠️  Image {idx} is empty, skipping...")
                    continue
                
                # Preprocess image
                original_image = self.load_image_from_array(img_array)
                
                # Ensure image is 2D (required for Hessenberg decomposition)
                if original_image.ndim != 2:
                    print(f"⚠️  Image {idx} is not 2D after preprocessing (shape: {original_image.shape}), skipping...")
                    continue
                
                print(f"Image shape: {original_image.shape}")
                
                # Encrypt
                print("Encrypting...")
                encryption_data = self.encrypt(original_image)
                
                # Decrypt
                print("Decrypting...")
                decrypted_image = self.decrypt(encryption_data)
                
                # Calculate metrics
                mse, psnr = self.calculate_metrics(original_image, decrypted_image)
                
                print(f"\nMetrics:")
                print(f"  MSE:  {mse:.6e}")
                if np.isinf(psnr):
                    print(f"  PSNR: Inf dB (Perfect)")
                else:
                    print(f"  PSNR: {psnr:.4f} dB")
                
                # Save images
                # Save encrypted image
                encrypted_path = os.path.join(encrypted_dir, f'image_{idx:04d}_encrypted.png')
                Image.fromarray(encryption_data['encrypted_image']).save(encrypted_path)
                
                # Save decrypted image
                decrypted_uint8 = (decrypted_image * 255).astype(np.uint8)
                decrypted_path = os.path.join(decrypted_dir, f'image_{idx:04d}_decrypted.png')
                Image.fromarray(decrypted_uint8).save(decrypted_path)
                
                # Save original for comparison
                original_uint8 = (original_image * 255).astype(np.uint8)
                original_path = os.path.join(comparison_dir, f'image_{idx:04d}_original.png')
                Image.fromarray(original_uint8).save(original_path)
                
                # Visualize first few samples
                if idx < visualize_samples:
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    
                    axes[0].imshow(original_image, cmap='gray', vmin=0, vmax=1)
                    axes[0].set_title(f'Original Image {idx}', fontsize=12, fontweight='bold')
                    axes[0].axis('off')
                    
                    axes[1].imshow(encryption_data['encrypted_image'], cmap='gray')
                    axes[1].set_title(f'Encrypted Image {idx}', fontsize=12, fontweight='bold')
                    axes[1].axis('off')
                    
                    axes[2].imshow(decrypted_image, cmap='gray', vmin=0, vmax=1)
                    axes[2].set_title(f'Decrypted Image {idx}\nMSE: {mse:.6e}, PSNR: {psnr:.4f} dB', 
                                     fontsize=12, fontweight='bold')
                    axes[2].axis('off')
                    
                    plt.tight_layout()
                    
                    # Save comparison figure
                    comparison_fig_path = os.path.join(comparison_dir, f'comparison_{idx:04d}.png')
                    plt.savefig(comparison_fig_path, dpi=150, bbox_inches='tight')
                    plt.show()
                    plt.close()
                
                results.append({
                    'image_idx': idx,
                    'mse': mse,
                    'psnr': psnr,
                    'shape': original_image.shape
                })
                
                print(f"✅ Saved to {output_dir}\n")
                
            except Exception as e:
                print(f"❌ Error processing image {idx}: {e}\n")
                continue
        
        npz_data.close()
        
        # Print summary statistics
        if results:
            print(f"\n{'='*70}")
            print(f"PROCESSING COMPLETE - SUMMARY")
            print(f"{'='*70}")
            print(f"Total images processed: {len(results)}")
            print(f"Output directory: {output_dir}")
            
            mse_values = [r['mse'] for r in results]
            psnr_values = [r['psnr'] for r in results if not np.isinf(r['psnr'])]
            
            print(f"\nMSE Statistics:")
            print(f"  Mean: {np.mean(mse_values):.6e}")
            print(f"  Std:  {np.std(mse_values):.6e}")
            print(f"  Min:  {np.min(mse_values):.6e}")
            print(f"  Max:  {np.max(mse_values):.6e}")
            
            if psnr_values:
                print(f"\nPSNR Statistics:")
                print(f"  Mean: {np.mean(psnr_values):.4f} dB")
                print(f"  Std:  {np.std(psnr_values):.4f} dB")
                print(f"  Min:  {np.min(psnr_values):.4f} dB")
                print(f"  Max:  {np.max(psnr_values):.4f} dB")
            
            print(f"\nFiles saved:")
            print(f"  - Encrypted images: {encrypted_dir}")
            print(f"  - Decrypted images: {decrypted_dir}")
            print(f"  - Comparison images: {comparison_dir}")
            print(f"{'='*70}\n")
            
            # Save metrics to CSV
            import csv
            metrics_path = os.path.join(output_dir, 'metrics.csv')
            with open(metrics_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['image_idx', 'mse', 'psnr', 'shape'])
                writer.writeheader()
                writer.writerows(results)
            print(f"Metrics saved to: {metrics_path}\n")
        
        return results


def main():
    """Main function to process the NPZ archive"""
    
    # Configuration
    NPZ_PATH = 'dataset/full_archive.npz'
    OUTPUT_DIR = 'output_encrypted'
    MAX_IMAGES = None  # Set to a number to limit processing, None = process all
    VISUALIZE_SAMPLES = 3  # Number of samples to display
    
    # Initialize the encryption system
    print("Initializing Hessenberg-Lorenz Encryption System...")
    print(f"Lorenz Parameters: σ=10, ρ=28, β=8/3")
    print(f"Initial conditions: x0=0.1, y0=0.2, z0=0.3")
    print(f"Time step: dt=0.01\n")
    
    encryptor = HessenbergLorenzEncryption(
        sigma=10,
        rho=28,
        beta=8/3,
        x0=0.1,
        y0=0.2,
        z0=0.3,
        dt=0.01
    )
    
    # Process the NPZ archive
    try:
        results = encryptor.process_npz_archive(
            npz_path=NPZ_PATH,
            output_dir=OUTPUT_DIR,
            max_images=MAX_IMAGES,
            visualize_samples=VISUALIZE_SAMPLES
        )
        
        print("✅ All processing complete!")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print(f"\nPlease ensure the file exists at: {NPZ_PATH}")
        print("If your NPZ file has a different name or location, update NPZ_PATH in the code.")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()