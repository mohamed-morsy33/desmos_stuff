import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import json


def histogram_equalization(image: np.ndarray) -> np.ndarray:
    """
    Apply histogram equalization to enhance contrast.
    """
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    return cdf[image]


def prewitt_edge_detection(image: np.ndarray, 
                           sigma: float = 1.2,
                           threshold: float = 0.08,
                           use_histogram_eq: bool = True) -> tuple:
    """
    Prewitt edge detection optimized for faces.
    
    Args:
        image: Grayscale image
        sigma: Gaussian smoothing (1.0-2.0 for faces, lower = more detail)
        threshold: Edge threshold as fraction of max gradient (0.05-0.15, lower = more edges)
        use_histogram_eq: Enhance contrast for better feature detection
    
    Returns:
        tuple: (edges, gradient_magnitude) - binary edges and raw gradient
    """
    # Enhance contrast
    if use_histogram_eq:
        image = histogram_equalization(image)
    
    # Apply Gaussian blur
    blurred = ndimage.gaussian_filter(image.astype(float), sigma=sigma)
    
    # Prewitt kernels - equal weighting across rows/columns
    prewitt_x = np.array([
        [-1, 0, 1], 
        [-1, 0, 1], 
        [-1, 0, 1]
    ])
    
    prewitt_y = np.array([
        [-1, -1, -1], 
        [0, 0, 0], 
        [1, 1, 1]
    ])
    
    # Apply Prewitt operators
    gx = ndimage.convolve(blurred, prewitt_x)
    gy = ndimage.convolve(blurred, prewitt_y)
    
    # Compute gradient magnitude
    gradient = np.sqrt(gx**2 + gy**2)
    
    # Apply threshold
    threshold_val = gradient.max() * threshold
    edges = (gradient > threshold_val).astype(np.uint8) * 255
    
    return edges, gradient


def extract_edge_coordinates(edge_image: np.ndarray) -> list:
    """
    Extract (x, y) coordinates of all edge pixels.
    """
    edge_pixels = np.argwhere(edge_image > 0)
    coordinates = [(int(col), int(row)) for row, col in edge_pixels]
    return coordinates


def save_coordinates(coordinates: list, filename: str, format: str = 'json') -> None:
    """
    Save edge coordinates to a file.
    """
    if format == 'json':
        with open(filename, 'w') as f:
            json.dump(coordinates, f, indent=2)
    elif format == 'csv':
        np.savetxt(filename, coordinates, delimiter=',', fmt='%d', header='x,y', comments='')
    elif format == 'npy':
        np.save(filename, np.array(coordinates))
    
    print(f"✓ Saved {len(coordinates)} coordinates to {filename}")


def prewitt_edge_detect(image_file: str,
                        sigma: float = 1.2,
                        threshold: float = 0.08,
                        use_histogram_eq: bool = True,
                        export_coords: bool = True,
                        coord_format: str = 'json',
                        show_plot: bool = True) -> tuple:
    """
    Prewitt edge detection with coordinate export.
    
    Args:
        image_file: Path to input image
        sigma: Gaussian blur (1.0-2.0, lower = sharper edges)
        threshold: Edge threshold (0.05-0.15, lower = more edges)
        use_histogram_eq: Enhance contrast (recommended for faces)
        export_coords: Export coordinates to file
        coord_format: 'json', 'csv', or 'npy'
        show_plot: Display results
    
    Returns:
        tuple: (edges, coordinates, gradient_magnitude)
    """
    print(f"Processing {image_file} with Prewitt edge detection...")
    
    # Read and convert to grayscale
    image = iio.imread(image_file)
    if len(image.shape) == 3:
        # Handle RGBA (4 channels) or RGB (3 channels)
        if image.shape[2] == 4:
            # Convert RGBA to RGB
            rgb = image[:, :, :3]
            alpha = image[:, :, 3:4] / 255.0
            rgb = (rgb * alpha + 255 * (1 - alpha)).astype(np.uint8)
            gray = (0.2989 * rgb[:, :, 0] + 
                    0.5870 * rgb[:, :, 1] + 
                    0.1140 * rgb[:, :, 2]).astype(np.uint8)
        else:
            gray = (0.2989 * image[:, :, 0] + 
                    0.5870 * image[:, :, 1] + 
                    0.1140 * image[:, :, 2]).astype(np.uint8)
    else:
        gray = image
    
    print(f"Image shape: {image.shape}")
    print(f"Settings: sigma={sigma}, threshold={threshold}, histogram_eq={use_histogram_eq}")
    
    # Apply Prewitt edge detection
    edges, gradient = prewitt_edge_detection(gray, sigma, threshold, use_histogram_eq)
    
    # Count edges
    edge_count = np.sum(edges > 0)
    edge_percentage = (edge_count / edges.size) * 100
    print(f"Detected {edge_count:,} edge pixels ({edge_percentage:.2f}% of image)")
    
    # Extract coordinates
    coordinates = extract_edge_coordinates(edges)
    print(f"Extracted {len(coordinates):,} coordinate points")
    
    # Export coordinates
    if export_coords and len(coordinates) > 0:
        base_name = image_file.rsplit('.', 1)[0]
        extension = coord_format if coord_format != 'npy' else 'npy'
        save_coordinates(coordinates, f"{base_name}_prewitt_edges.{extension}", coord_format)
    
    # Visualize
    if show_plot:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Original image
        axes[0, 0].imshow(image if len(image.shape) == 3 else gray, cmap='gray' if len(image.shape) == 2 else None)
        axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Grayscale
        axes[0, 1].imshow(gray, cmap='gray')
        axes[0, 1].set_title('Grayscale', fontsize=14)
        axes[0, 1].axis('off')
        
        # Histogram equalized (if used)
        if use_histogram_eq:
            enhanced = histogram_equalization(gray)
            axes[0, 2].imshow(enhanced, cmap='gray')
            axes[0, 2].set_title('Histogram Equalized\n(Enhanced Contrast)', fontsize=14)
        else:
            axes[0, 2].imshow(gray, cmap='gray')
            axes[0, 2].set_title('No Enhancement', fontsize=14)
        axes[0, 2].axis('off')
        
        # Gradient magnitude
        gradient_normalized = (gradient / gradient.max() * 255).astype(np.uint8)
        axes[1, 0].imshow(gradient_normalized, cmap='hot')
        axes[1, 0].set_title('Gradient Magnitude\n(Prewitt)', fontsize=14)
        axes[1, 0].axis('off')
        
        # Detected edges
        axes[1, 1].imshow(edges, cmap='gray')
        axes[1, 1].set_title(f'Detected Edges\n{edge_count:,} pixels ({edge_percentage:.1f}%)', 
                            fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')
        
        # Overlay edges on original
        if len(image.shape) == 3:
            overlay = image.copy()
            edge_mask = edges > 0
            if overlay.shape[2] == 4:  # RGBA
                overlay[edge_mask] = [255, 0, 0, 255]
            else:  # RGB
                overlay[edge_mask] = [255, 0, 0]
        else:
            overlay = np.stack([gray]*3, axis=-1)
            edge_mask = edges > 0
            overlay[edge_mask] = [255, 0, 0]
        
        axes[1, 2].imshow(overlay)
        axes[1, 2].set_title('Edges Overlay (Red)\nOn Original Image', fontsize=14, fontweight='bold')
        axes[1, 2].axis('off')
        
        plt.suptitle(f'Prewitt Edge Detection Results\nσ={sigma}, threshold={threshold}', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig("prewitt_edge_detection.png", dpi=150, bbox_inches='tight')
        print("✓ Saved visualization to prewitt_edge_detection.png")
        plt.show()
    
    return edges, coordinates, gradient


def plot_coordinates(coordinates: list, 
                     image_shape: tuple = None, 
                     save_path: str = "edge_coordinates.png",
                     point_size: float = 0.5) -> None:
    """
    Plot edge coordinates as a scatter plot.
    
    Args:
        coordinates: List of (x, y) tuples
        image_shape: Original image shape for axis limits
        save_path: Output file path
        point_size: Size of scatter points (smaller = denser appearance)
    """
    if not coordinates:
        print("No coordinates to plot")
        return
    
    x_coords, y_coords = zip(*coordinates)
    
    plt.figure(figsize=(12, 12))
    plt.scatter(x_coords, y_coords, s=point_size, c='black', marker='.', alpha=0.8)
    plt.gca().invert_yaxis()
    plt.title(f'Edge Coordinates\n{len(coordinates):,} points', fontsize=16, fontweight='bold')
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.axis('equal')
    plt.grid(True, alpha=0.2, linestyle='--')
    
    if image_shape:
        plt.xlim(0, image_shape[1])
        plt.ylim(image_shape[0], 0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved coordinate plot to {save_path}")
    plt.show()


def load_coordinates(filename: str, format: str = None) -> list:
    """
    Load coordinates from a file.
    
    Args:
        filename: Input file path
        format: File format ('json', 'csv', 'npy') - auto-detected if None
    """
    if format is None:
        format = filename.rsplit('.', 1)[-1]
    
    if format == 'json':
        with open(filename, 'r') as f:
            coords = json.load(f)
    elif format == 'csv':
        data = np.loadtxt(filename, delimiter=',', skiprows=1)
        coords = [(int(x), int(y)) for x, y in data]
    elif format == 'npy':
        data = np.load(filename)
        coords = [(int(x), int(y)) for x, y in data]
    else:
        raise ValueError(f"Unknown format: {format}")
    
    print(f"✓ Loaded {len(coords)} coordinates from {filename}")
    return coords


# Main execution
if __name__ == "__main__":
    print("=" * 70)
    print("EDGE DETECTION SCRIPT")
    print("=" * 70)
    
    # Run Prewitt edge detection
    edges, coords, gradient = prewitt_edge_detect(
        "input_img.png",
        sigma=1.2,                 # Smoothing (1.0-2.0, lower = more detail)
        threshold=0.08,            # Edge threshold (0.05-0.15, lower = more edges)
        use_histogram_eq=True,     # Enhance contrast
        export_coords=True,        # Save coordinates
        coord_format='json',       # Options: 'json', 'csv', 'npy'
        show_plot=True             # Display results
    )
    
    # Plot just the coordinates
    if len(coords) > 0:
        plot_coordinates(coords, 
                        image_shape=edges.shape, 
                        save_path="prewitt_coordinates_plot.png",
                        point_size=0.5)
        
        print("\n" + "=" * 70)
        print("SUCCESS!")
        print("=" * 70)
        print(f"Total edge points: {len(coords):,}")
        print(f"\nFiles created:")
        print(f"  1. prewitt_edge_detection.png - Full visualization")
        print(f"  2. output_edges.json - Coordinate data")
        print(f"  3. prewitt_coordinates_plot.png - Scatter plot")
        print("\n" + "=" * 70)
        print("TUNING GUIDE:")
        print("=" * 70)
        print("For MORE edges:")
        print("  - Decrease threshold (try 0.06 or 0.05)")
        print("  - Decrease sigma (try 1.0)")
        print("\nFor FEWER/CLEANER edges:")
        print("  - Increase threshold (try 0.10 or 0.12)")
        print("  - Increase sigma (try 1.5 or 2.0)")
        print("\nFor BETTER facial features:")
        print("  - Keep use_histogram_eq=True")
        print("  - Try threshold between 0.06-0.10")
        print("=" * 70)
        
        # Show sample coordinates
        print(f"\nFirst 10 coordinates: {coords[:10]}")
        print(f"Last 10 coordinates: {coords[-10:]}")