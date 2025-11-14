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
    
    def load_image(self, image_path):
        """Load and preprocess image"""
        img = Image.open(image_path)
        
        # Convert to grayscale if RGB
        if img.mode == 'RGB':
            img = img.convert('L')
        
        # Convert to numpy array and normalize to [0, 1]
        img_array = np.array(img, dtype=np.float64) / 255.0
        
        return img_array
    
    def encrypt(self, image_array):
        """Encrypt image using Hessenberg decomposition and Lorenz chaos"""
        rows, cols = image_array.shape
        num_pixels = rows * cols
        
        print(f"Image dimensions: {rows} x {cols}")
        print("Performing Hessenberg decomposition...")
        
        # Hessenberg decomposition
        H, Q = hessenberg(image_array, calc_q=True)
        
        print("Generating Lorenz chaotic sequences...")
        x, y, z = self.generate_lorenz_sequences(num_pixels)
        
        # Generate chaotic keys
        X_key = np.mod(x, 1).reshape(rows, cols)
        Y_key = np.mod(y, 1).reshape(rows, cols)
        Z_key = np.mod(z, 1).reshape(rows, cols)
        
        print("Performing encryption...")
        # Encrypt Q and H matrices
        Q_encrypted = Q + X_key + Z_key
        H_encrypted = H + Y_key + Z_key
        
        # Convert to uint8 for visualization
        Q_vis = np.mod(np.round(Q_encrypted * 1e5), 256)
        H_vis = np.mod(np.round(H_encrypted * 1e5), 256)
        encrypted_image = np.mod(Q_vis + H_vis, 256).astype(np.uint8)
        
        print("✅ Encryption complete.\n")
        
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
        
        print("Regenerating Lorenz chaotic sequences for decryption...")
        x, y, z = self.generate_lorenz_sequences(num_pixels)
        
        # Regenerate the same chaotic keys
        X_key = np.mod(x, 1).reshape(rows, cols)
        Y_key = np.mod(y, 1).reshape(rows, cols)
        Z_key = np.mod(z, 1).reshape(rows, cols)
        
        print("Performing decryption...")
        # Reverse the encryption
        Q_decrypted = Q_encrypted - X_key - Z_key
        H_decrypted = H_encrypted - Y_key - Z_key
        
        print("Reconstructing original image from Hessenberg decomposition...")
        # Reconstruct: A = Q * H * Q'
        decrypted_image = Q_decrypted @ H_decrypted @ Q_decrypted.T
        
        # Convert back to valid image format
        decrypted_image = np.real(decrypted_image)  # Take real part
        decrypted_image = np.clip(decrypted_image, 0, 1)  # Clip to [0, 1]
        
        print("✅ Decryption complete.\n")
        
        return decrypted_image
    
    def calculate_metrics(self, original, decrypted):
        """Calculate MSE and PSNR"""
        mse = np.mean((original - decrypted) ** 2)
        
        if mse == 0:
            psnr = float('inf')
        else:
            psnr = 10 * np.log10(1 / mse)
        
        return mse, psnr
    
    def visualize_results(self, original, encrypted_vis, decrypted, mse, psnr):
        """Visualize original, encrypted, and decrypted images"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(original, cmap='gray')
        axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(encrypted_vis, cmap='gray')
        axes[1].set_title('Encrypted Image', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        axes[2].imshow(decrypted, cmap='gray')
        axes[2].set_title(f'Decrypted Image\nMSE: {mse:.6e}, PSNR: {psnr:.4f} dB', 
                         fontsize=12, fontweight='bold')
        axes[2].axis('off')
        
        plt.tight_layout()
        return fig
    
    def process_single_image(self, image_path, output_dir=None, visualize=True):
        """Process a single image: encrypt and decrypt"""
        print(f"\n{'='*60}")
        print(f"Processing: {image_path}")
        print(f"{'='*60}\n")
        
        # Load image
        original_image = self.load_image(image_path)
        
        # Encrypt
        encryption_data = self.encrypt(original_image)
        
        # Decrypt
        decrypted_image = self.decrypt(encryption_data)
        
        # Calculate metrics
        mse, psnr = self.calculate_metrics(original_image, decrypted_image)
        
        print(f"{'='*50}")
        print(f"Decryption Quality Metrics")
        print(f"{'='*50}")
        print(f"MSE: {mse:.6e}")
        if np.isinf(psnr):
            print(f"PSNR: Inf dB (Perfect)")
        else:
            print(f"PSNR: {psnr:.4f} dB")
        print(f"{'='*50}\n")
        
        # Visualize
        if visualize:
            fig = self.visualize_results(
                original_image, 
                encryption_data['encrypted_image'], 
                decrypted_image, 
                mse, 
                psnr
            )
            plt.show()
        
        # Save results if output directory specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            base_name = Path(image_path).stem
            
            # Save encrypted image
            encrypted_path = os.path.join(output_dir, f"{base_name}_encrypted.png")
            Image.fromarray(encryption_data['encrypted_image']).save(encrypted_path)
            
            # Save decrypted image
            decrypted_uint8 = (decrypted_image * 255).astype(np.uint8)
            decrypted_path = os.path.join(output_dir, f"{base_name}_decrypted.png")
            Image.fromarray(decrypted_uint8).save(decrypted_path)
            
            print(f"Results saved to: {output_dir}\n")
        
        return {
            'original': original_image,
            'encrypted': encryption_data['encrypted_image'],
            'decrypted': decrypted_image,
            'mse': mse,
            'psnr': psnr
        }
    
    def process_dataset(self, image_folder, output_dir=None, visualize_samples=3):
        """Process multiple images from a dataset folder"""
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(Path(image_folder).glob(f'*{ext}'))
            image_files.extend(Path(image_folder).glob(f'*{ext.upper()}'))
        
        if not image_files:
            print(f"No images found in {image_folder}")
            return
        
        print(f"\nFound {len(image_files)} images in dataset")
        print(f"{'='*60}\n")
        
        results = []
        
        for idx, image_path in enumerate(image_files):
            try:
                # Visualize only first few samples
                visualize = (idx < visualize_samples)
                
                result = self.process_single_image(
                    str(image_path), 
                    output_dir=output_dir,
                    visualize=visualize
                )
                results.append({
                    'filename': image_path.name,
                    'mse': result['mse'],
                    'psnr': result['psnr']
                })
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}\n")
        
        # Print summary statistics
        if results:
            print(f"\n{'='*60}")
            print(f"DATASET SUMMARY")
            print(f"{'='*60}")
            print(f"Total images processed: {len(results)}")
            
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
            
            print(f"{'='*60}\n")
        
        return results


# Example usage
if __name__ == "__main__":
    # Initialize the encryption system
    encryptor = HessenbergLorenzEncryption()
    
    # ============================================================
    # OPTION 1: Process a Single Image
    # ============================================================
    # Uncomment and modify the path below:
    
    # encryptor.process_single_image(
    #     image_path='cameraman.jpg',           # ← PUT YOUR IMAGE PATH HERE
    #     output_dir='output',                  # ← OUTPUT FOLDER
    #     visualize=True
    # )
    
    
    # ============================================================
    # OPTION 2: Process an Entire Image Dataset (Folder)
    # ============================================================
    # Uncomment and modify the path below:
    
    # results = encryptor.process_dataset(
    #     image_folder='my_images/',            # ← PUT YOUR DATASET FOLDER PATH HERE
    #     output_dir='results/',                # ← OUTPUT FOLDER
    #     visualize_samples=3                   # Show first 3 images
    # )
    
    
    # ============================================================
    # EXAMPLES OF VALID PATHS:
    # ============================================================
    # Windows:
    #   'C:/Users/YourName/Pictures/dataset/'
    #   'D:/datasets/images/'
    #   'dataset/'  (if folder is in same directory as script)
    #
    # Linux/Mac:
    #   '/home/username/datasets/images/'
    #   '~/datasets/images/'
    #   'dataset/'  (if folder is in same directory as script)
    # ============================================================
    
    print("\n" + "="*70)
    print("HOW TO USE THIS CODE:")
    print("="*70)
    print("\n1️⃣  FOR SINGLE IMAGE:")
    print("    Uncomment lines 278-282 and change the path:")
    print("    encryptor.process_single_image('path/to/image.jpg', output_dir='output')")
    
    print("\n2️⃣  FOR IMAGE DATASET (FOLDER):")
    print("    Uncomment lines 291-295 and change the path:")
    print("    encryptor.process_dataset('path/to/folder/', output_dir='results/')")
    
    print("\n3️⃣  YOUR FOLDER STRUCTURE SHOULD LOOK LIKE:")
    print("    my_images/")
    print("    ├── image1.jpg")
    print("    ├── image2.png")
    print("    ├── image3.jpg")
    print("    └── ...")
    
    print("\n4️⃣  SUPPORTED FORMATS:")
    print("    .jpg, .jpeg, .png, .bmp, .tif, .tiff")
    
    print("\n" + "="*70 + "\n")