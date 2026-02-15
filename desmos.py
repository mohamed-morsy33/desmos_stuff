import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import json
import sys
import base64
from io import BytesIO


def histogram_equalization(image: np.ndarray) -> np.ndarray:
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    return cdf[image]


def prewitt_edge_detection(image: np.ndarray, sigma: float = 1.2, threshold: float = 0.08, use_histogram_eq: bool = True) -> tuple:
    if use_histogram_eq:
        image = histogram_equalization(image)
    
    blurred = ndimage.gaussian_filter(image.astype(float), sigma=sigma)
    
    prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    
    gx = ndimage.convolve(blurred, prewitt_x)
    gy = ndimage.convolve(blurred, prewitt_y)
    
    gradient = np.sqrt(gx**2 + gy**2)
    threshold_val = gradient.max() * threshold
    edges = (gradient > threshold_val).astype(np.uint8) * 255
    
    return edges, gradient


def extract_edge_coordinates(edge_image: np.ndarray) -> list:
    edge_pixels = np.argwhere(edge_image > 0)
    coordinates = [(int(col), int(row)) for row, col in edge_pixels]
    return coordinates


def save_coordinates(coordinates: list, filename: str, format: str = 'json') -> None:
    if format == 'json':
        with open(filename, 'w') as f:
            json.dump(coordinates, f, indent=2)
    elif format == 'csv':
        np.savetxt(filename, coordinates, delimiter=',', fmt='%d', header='x,y', comments='')
    elif format == 'npy':
        np.save(filename, np.array(coordinates))
    
    print(f"Saved {len(coordinates)} coordinates to {filename}")


def export_desmos_json(coordinates: list, filename: str = "desmos_graph.json", image_shape: tuple = None) -> None:
    print(f"Exporting Desmos-compatible JSON...")
    
    x_coords = [coord[0] for coord in coordinates]
    y_coords = [-coord[1] for coord in coordinates]
    
    expressions = [{
        "type": "table",
        "id": "edge-points",
        "columns": [
            {"latex": "x_1", "values": x_coords, "hidden": False, "points": True, "lines": False, "dragMode": "NONE"},
            {"latex": "y_1", "values": y_coords, "hidden": False, "points": True, "lines": False, "dragMode": "NONE"}
        ]
    }]
    
    graph_settings = {"xAxisNumbers": True, "yAxisNumbers": True, "xAxisLabel": "X", "yAxisLabel": "Y (inverted)"}
    
    if image_shape:
        graph_settings["viewport"] = {
            "xmin": -50, "xmax": image_shape[1] + 50,
            "ymin": -(image_shape[0] + 50), "ymax": 50
        }
    
    desmos_state = {
        "version": 11,
        "randomSeed": "12345",
        "graph": {"viewport": graph_settings.get("viewport", {}), "squareAxes": False, "xAxisNumbers": True, "yAxisNumbers": True, "xAxisLabel": "X", "yAxisLabel": "Y"},
        "expressions": {"list": expressions}
    }
    
    with open(filename, 'w') as f:
        json.dump(desmos_state, f, indent=2)
    
    print(f"Saved Desmos JSON to {filename}")


def image_to_base64(image_array: np.ndarray) -> str:
    if len(image_array.shape) == 2:
        image_array = np.stack([image_array]*3, axis=-1)
    
    buffer = BytesIO()
    plt.imsave(buffer, image_array, format='png')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"


def generate_html_page(input_image_path: str, edges: np.ndarray, coordinates: list, original_image: np.ndarray, gradient: np.ndarray, sigma: float, threshold: float, desmos_api_key: str = None) -> None:
    print("Generating HTML page for GitHub Pages...")
    
    original_base64 = image_to_base64(original_image)
    edges_base64 = image_to_base64(edges)
    
    if len(original_image.shape) == 3:
        overlay = original_image.copy()
        edge_mask = edges > 0
        if overlay.shape[2] == 4:
            overlay[edge_mask] = [255, 0, 0, 255]
        else:
            overlay[edge_mask] = [255, 0, 0]
    else:
        overlay = np.stack([original_image]*3, axis=-1)
        edge_mask = edges > 0
        overlay[edge_mask] = [255, 0, 0]
    
    overlay_base64 = image_to_base64(overlay)
    
    edge_count = len(coordinates)
    edge_percentage = (edge_count / (edges.shape[0] * edges.shape[1])) * 100
    
    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Edge Detection Results</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Comic Sans MS', 'Comic Sans', cursive;
            background-color: #2d5016;
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: #f5f5f5;
            border-radius: 8px;
            padding: 40px;
            border: 3px solid #4a7c24;
        }}
        
        h1 {{
            text-align: center;
            color: #2d5016;
            font-size: 2.5em;
            margin-bottom: 30px;
        }}
        
        .stats {{
            display: flex;
            justify-content: space-around;
            margin: 30px 0;
            flex-wrap: wrap;
            gap: 15px;
        }}
        
        .stat-box {{
            background: #4a7c24;
            color: white;
            padding: 20px 30px;
            border-radius: 5px;
            text-align: center;
            flex: 1;
            min-width: 180px;
        }}
        
        .stat-number {{
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        
        .stat-text {{
            font-size: 1em;
        }}
        
        .image-section {{
            margin: 40px 0;
        }}
        
        .image-box {{
            background: white;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
            border: 2px solid #4a7c24;
        }}
        
        .image-box h3 {{
            color: #2d5016;
            margin-bottom: 15px;
            font-size: 1.3em;
        }}
        
        .image-box img {{
            width: 100%;
            border-radius: 3px;
            display: none;
        }}
        
        .show-btn {{
            background: #4a7c24;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-family: 'Comic Sans MS', cursive;
            font-size: 1em;
            margin-top: 10px;
        }}
        
        .show-btn:hover {{
            background: #5a8c34;
        }}
        
        .downloads {{
            background: white;
            padding: 30px;
            border-radius: 5px;
            margin: 30px 0;
            border: 2px solid #4a7c24;
        }}
        
        .downloads h2 {{
            color: #2d5016;
            margin-bottom: 20px;
            font-size: 1.8em;
        }}
        
        .download-links {{
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
        }}
        
        .download-link {{
            background: #4a7c24;
            color: white;
            padding: 12px 24px;
            border-radius: 5px;
            text-decoration: none;
            font-weight: bold;
        }}
        
        .download-link:hover {{
            background: #5a8c34;
        }}
        
        .params {{
            background: white;
            padding: 25px;
            border-radius: 5px;
            margin: 30px 0;
            border: 2px solid #4a7c24;
        }}
        
        .params h2 {{
            color: #2d5016;
            margin-bottom: 15px;
            font-size: 1.8em;
        }}
        
        .param-list {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        
        .param {{
            background: #f9f9f9;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #4a7c24;
        }}
        
        .param-label {{
            font-weight: bold;
            color: #2d5016;
            margin-bottom: 5px;
        }}
        
        .param-val {{
            color: #4a7c24;
            font-size: 1.2em;
            font-weight: bold;
        }}
        
        @media (max-width: 768px) {{
            .container {{
                padding: 20px;
            }}
            
            h1 {{
                font-size: 2em;
            }}
            
            .stat-box {{
                min-width: 100%;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Edge Detection Results</h1>
        
        <div class="stats">
            <div class="stat-box">
                <div class="stat-number">{edge_count:,}</div>
                <div class="stat-text">Edge Pixels</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">{edge_percentage:.1f}%</div>
                <div class="stat-text">Coverage</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">{edges.shape[1]}x{edges.shape[0]}</div>
                <div class="stat-text">Resolution</div>
            </div>
        </div>
        
        <div class="params">
            <h2>Settings</h2>
            <div class="param-list">
                <div class="param">
                    <div class="param-label">Sigma</div>
                    <div class="param-val">{sigma}</div>
                </div>
                <div class="param">
                    <div class="param-label">Threshold</div>
                    <div class="param-val">{threshold}</div>
                </div>
                <div class="param">
                    <div class="param-label">Algorithm</div>
                    <div class="param-val">Prewitt</div>
                </div>
            </div>
        </div>
        
        <div class="image-section">
            <div class="image-box">
                <h3>Original Image</h3>
                <button class="show-btn" onclick="showImage('img1')">Show Image</button>
                <img id="img1" src="{original_base64}" alt="Original">
            </div>
            
            <div class="image-box">
                <h3>Detected Edges</h3>
                <button class="show-btn" onclick="showImage('img2')">Show Image</button>
                <img id="img2" src="{edges_base64}" alt="Edges">
            </div>
            
            <div class="image-box">
                <h3>Edge Overlay</h3>
                <button class="show-btn" onclick="showImage('img3')">Show Image</button>
                <img id="img3" src="{overlay_base64}" alt="Overlay">
            </div>
        </div>
        
        <div class="downloads">
            <h2>Download Results</h2>
            <div class="download-links">
                <a href="output_edges.json" class="download-link" download>Coordinates (JSON)</a>
                <a href="desmos_graph.json" class="download-link" download>Desmos Graph</a>
                <a href="prewitt_edge_detection.png" class="download-link" download>Visualization</a>
            </div>
        </div>
    </div>
    
    <script>
        function showImage(id) {{
            var img = document.getElementById(id);
            if (img.style.display === 'none' || img.style.display === '') {{
                img.style.display = 'block';
            }} else {{
                img.style.display = 'none';
            }}
        }}
    </script>
</body>
</html>'''
    
    with open('index.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print("Generated index.html for GitHub Pages")


def prewitt_edge_detect(image_file: str, sigma: float = 1.2, threshold: float = 0.08, use_histogram_eq: bool = True, export_coords: bool = True, coord_format: str = 'json', show_plot: bool = True, generate_html: bool = True, desmos_api_key: str = None) -> tuple:
    print(f"Processing {image_file} with Prewitt edge detection...")
    
    image = iio.imread(image_file)
    if len(image.shape) == 3:
        if image.shape[2] == 4:
            rgb = image[:, :, :3]
            alpha = image[:, :, 3:4] / 255.0
            rgb = (rgb * alpha + 255 * (1 - alpha)).astype(np.uint8)
            gray = (0.2989 * rgb[:, :, 0] + 0.5870 * rgb[:, :, 1] + 0.1140 * rgb[:, :, 2]).astype(np.uint8)
        else:
            gray = (0.2989 * image[:, :, 0] + 0.5870 * image[:, :, 1] + 0.1140 * image[:, :, 2]).astype(np.uint8)
    else:
        gray = image
    
    print(f"Image shape: {image.shape}")
    print(f"Settings: sigma={sigma}, threshold={threshold}, histogram_eq={use_histogram_eq}")
    
    edges, gradient = prewitt_edge_detection(gray, sigma, threshold, use_histogram_eq)
    
    edge_count = np.sum(edges > 0)
    edge_percentage = (edge_count / edges.size) * 100
    print(f"Detected {edge_count:,} edge pixels ({edge_percentage:.2f}% of image)")
    
    coordinates = extract_edge_coordinates(edges)
    print(f"Extracted {len(coordinates):,} coordinate points")
    
    if export_coords and len(coordinates) > 0:
        save_coordinates(coordinates, "output_edges.json", coord_format)
        export_desmos_json(coordinates, "desmos_graph.json", edges.shape)
    
    if show_plot:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        axes[0, 0].imshow(image if len(image.shape) == 3 else gray, cmap='gray' if len(image.shape) == 2 else None)
        axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(gray, cmap='gray')
        axes[0, 1].set_title('Grayscale', fontsize=14)
        axes[0, 1].axis('off')
        
        if use_histogram_eq:
            enhanced = histogram_equalization(gray)
            axes[0, 2].imshow(enhanced, cmap='gray')
            axes[0, 2].set_title('Histogram Equalized', fontsize=14)
        else:
            axes[0, 2].imshow(gray, cmap='gray')
            axes[0, 2].set_title('No Enhancement', fontsize=14)
        axes[0, 2].axis('off')
        
        gradient_normalized = (gradient / gradient.max() * 255).astype(np.uint8)
        axes[1, 0].imshow(gradient_normalized, cmap='hot')
        axes[1, 0].set_title('Gradient Magnitude', fontsize=14)
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(edges, cmap='gray')
        axes[1, 1].set_title(f'Detected Edges\n{edge_count:,} pixels ({edge_percentage:.1f}%)', fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')
        
        if len(image.shape) == 3:
            overlay = image.copy()
            edge_mask = edges > 0
            if overlay.shape[2] == 4:
                overlay[edge_mask] = [255, 0, 0, 255]
            else:
                overlay[edge_mask] = [255, 0, 0]
        else:
            overlay = np.stack([gray]*3, axis=-1)
            edge_mask = edges > 0
            overlay[edge_mask] = [255, 0, 0]
        
        axes[1, 2].imshow(overlay)
        axes[1, 2].set_title('Edges Overlay', fontsize=14, fontweight='bold')
        axes[1, 2].axis('off')
        
        plt.suptitle(f'Prewitt Edge Detection Results\nσ={sigma}, threshold={threshold}', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig("prewitt_edge_detection.png", dpi=150, bbox_inches='tight')
        print("Saved visualization to prewitt_edge_detection.png")
        plt.show()
    
    if generate_html and len(coordinates) > 0:
        if len(image.shape) == 2:
            display_image = np.stack([gray]*3, axis=-1)
        else:
            display_image = image
        generate_html_page(image_file, edges, coordinates, display_image, gradient, sigma, threshold, desmos_api_key)
    
    return edges, coordinates, gradient


if __name__ == "__main__":
    print("=" * 70)
    print("EDGE DETECTION SCRIPT")
    print("=" * 70)
    
    if len(sys.argv) < 2:
        print("Usage: python script.py <input_image> [desmos_api_key]")
        sys.exit(1)
    
    input_img = sys.argv[1]
    desmos_api_key = sys.argv[2] if len(sys.argv) > 2 else None
    
    if desmos_api_key:
        print(f"Using Desmos API key: {desmos_api_key[:10]}...")
    
    edges, coords, gradient = prewitt_edge_detect(
        input_img,
        sigma=1.2,
        threshold=0.08,
        use_histogram_eq=True,
        export_coords=True,
        coord_format='json',
        show_plot=True,
        generate_html=True,
        desmos_api_key=desmos_api_key
    )