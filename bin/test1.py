import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
import implicit2d

def extract_features(image_path):
    # Read image
    img = cv2.imread(image_path)
    
    # Convert to HSV for better color separation
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Extract features by color
    black_mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 50))  # fault
    red_mask = cv2.inRange(hsv, (160, 100, 100), (180, 255, 255))  # red horizon
    green_mask = cv2.inRange(hsv, (35, 100, 100), (85, 255, 255))  # green horizon
    
    # Extract points and normalize coordinates
    h, w = img.shape[:2]
    
    def get_points(mask, skip=5):
        points = np.where(mask > 0)
        coords = np.column_stack((points[1], points[0]))  # x,y coordinates
        coords = coords[::skip]  # reduce density
        return coords.astype(float) / [w, h]  # normalize to [0,1]
    
    # Compute normals for a set of points
    def compute_normals(points):
        # Smooth points if needed
        x = gaussian_filter1d(points[:, 0], sigma=1)
        y = gaussian_filter1d(points[:, 1], sigma=1)
        
        # Compute tangents using finite differences
        dx = np.gradient(x)
        dy = np.gradient(y)
        
        # Rotate 90 degrees to get normals and normalize
        normals = np.column_stack((-dy, dx))
        norms = np.sqrt((normals ** 2).sum(axis=1))
        normals /= norms[:, np.newaxis]
        
        return normals

    # Get features
    fault_points = get_points(black_mask)
    red_points = get_points(red_mask)
    green_points = get_points(green_mask)
    
    red_normals = compute_normals(red_points)
    green_normals = compute_normals(green_points)
    
    return {
        'fault': fault_points,
        'red_horizon': (red_points, red_normals),
        'green_horizon': (green_points, green_normals)
    }

def build_implicit_function(points, normals):
    # Create builder
    builder = implicit2d.ImplicitBuilder()
    
    # Begin description
    builder.beginDescription()
    
    # Create regular grid of vertices (10x10 for example)
    nx, ny = 10, 10
    vertices = []
    for j in range(ny):
        for i in range(nx):
            x = i / (nx - 1)
            y = j / (ny - 1)
            vid = j * nx + i
            builder.addVertex(x, y)
            vertices.append((x, y))
    
    # Create triangles
    for j in range(ny - 1):
        for i in range(nx - 1):
            v1 = j * nx + i
            v2 = v1 + 1
            v3 = v1 + nx
            v4 = v2 + nx
            
            # Two triangles per grid cell
            builder.addTriangle(v1, v2, v3)
            builder.addTriangle(v2, v4, v3)
    
    # Add edges (no faults for simplicity)
    eid = 0
    for j in range(ny):
        for i in range(nx - 1):
            v1 = j * nx + i
            v2 = v1 + 1
            builder.addEdge(v1, v2)
            eid += 1
    
    for j in range(ny - 1):
        for i in range(nx):
            v1 = j * nx + i
            v2 = v1 + nx
            builder.addEdge(v1, v2)
            eid += 1
    
    # Add data points
    for point, normal in zip(points, normals):
        builder.addDataPoint(point[0], point[1], normal)
    
    # End description
    builder.endDescription()
    
    return builder

# Main execution
if __name__ == "__main__":
    # Extract features from image
    features = extract_features("../models/model1.png")
    
    # Build red horizon model
    red_points, red_normals = features['red_horizon']
    red_builder = build_implicit_function(red_points, red_normals)
    
    # Build green horizon model
    green_points, green_normals = features['green_horizon']
    green_builder = build_implicit_function(green_points, green_normals)