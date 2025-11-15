#!/usr/bin/env python3
"""Create PNG visualization files using OpenCV."""

import os
import cv2
import numpy as np

def create_rpm_chart_png():
    """Create RPM chart as PNG."""
    width, height = 800, 533
    img = np.ones((height, width, 3), dtype=np.uint8) * 248  # #f8f9fa background
    
    # Title
    cv2.putText(img, "RPM Over Time", (width//2 - 120, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (51, 51, 51), 2)
    
    # Axes
    cv2.line(img, (80, height-80), (width-50, height-80), (51, 51, 51), 2)  # X-axis
    cv2.line(img, (80, 70), (80, height-80), (51, 51, 51), 2)  # Y-axis
    
    # Y-axis labels
    y_values = [5000, 4000, 3000, 2000, 1000]
    for i, val in enumerate(y_values):
        y = 70 + i * (height - 150) // 4
        cv2.putText(img, str(val), (20, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (102, 102, 102), 1)
        cv2.line(img, (75, y), (85, y), (102, 102, 102), 1)
    
    # X-axis labels
    x_labels = ['0s', '5s', '10s', '15s', '20s']
    for i, label in enumerate(x_labels):
        x = 80 + i * (width - 130) // 4
        cv2.putText(img, label, (x-10, height-60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (102, 102, 102), 1)
        cv2.line(img, (x, height-85), (x, height-75), (102, 102, 102), 1)
    
    # Sample data points (normalized to chart area)
    rpm_data = [3500, 3300, 3100, 2900, 2700, 2800, 3000, 3200, 3100, 2900, 
                2700, 2500, 2600, 2800, 3000, 3200, 3400, 3600, 3500, 3300]
    
    points = []
    for i, rpm in enumerate(rpm_data):
        x = 80 + int((i / (len(rpm_data) - 1)) * (width - 130))
        y = height - 80 - int(((rpm - 1000) / 4000) * (height - 150))
        points.append((x, y))
    
    # Draw line chart
    for i in range(len(points) - 1):
        cv2.line(img, points[i], points[i+1], (107, 91, 255), 3)  # #FF6B6B in BGR
    
    # Draw points
    for point in points:
        cv2.circle(img, point, 5, (107, 91, 255), -1)
    
    # Axis labels
    cv2.putText(img, "Time (seconds)", (width//2 - 80, height-20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (51, 51, 51), 1)
    cv2.putText(img, "RPM", (10, height//2), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (51, 51, 51), 1)
    
    cv2.imwrite('static/visualizations/rpm-chart.png', img)
    print("✓ Created rpm-chart.png")

def create_speed_graph_png():
    """Create speed graph as PNG."""
    width, height = 800, 533
    img = np.ones((height, width, 3), dtype=np.uint8) * 248  # #f8f9fa background
    
    # Title
    cv2.putText(img, "Predicted Speed", (width//2 - 120, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (51, 51, 51), 2)
    
    # Axes
    cv2.line(img, (80, height-80), (width-50, height-80), (51, 51, 51), 2)  # X-axis
    cv2.line(img, (80, 70), (80, height-80), (51, 51, 51), 2)  # Y-axis
    
    # Y-axis labels
    y_values = [50, 40, 30, 20, 10]
    for i, val in enumerate(y_values):
        y = 70 + i * (height - 150) // 4
        cv2.putText(img, str(val), (40, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (102, 102, 102), 1)
        cv2.line(img, (75, y), (85, y), (102, 102, 102), 1)
    
    # X-axis labels
    x_labels = ['0s', '5s', '10s', '15s', '20s']
    for i, label in enumerate(x_labels):
        x = 80 + i * (width - 130) // 4
        cv2.putText(img, label, (x-10, height-60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (102, 102, 102), 1)
        cv2.line(img, (x, height-85), (x, height-75), (102, 102, 102), 1)
    
    # Sample data points
    speed_data = [15, 18, 21, 24, 23, 20, 18, 16, 18, 21, 
                  24, 27, 30, 32, 35, 37, 36, 34, 32, 30]
    
    points = []
    for i, speed in enumerate(speed_data):
        x = 80 + int((i / (len(speed_data) - 1)) * (width - 130))
        y = height - 80 - int(((speed - 10) / 40) * (height - 150))
        points.append((x, y))
    
    # Create filled area (polygon)
    polygon_points = [(80, height-80)] + points + [(width-50, height-80)]
    pts = np.array(polygon_points, np.int32)
    pts = pts.reshape((-1, 1, 2))
    
    # Draw filled area with transparency
    overlay = img.copy()
    cv2.fillPoly(overlay, [pts], (196, 205, 78))  # #4ECDC4 in BGR
    cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
    
    # Draw line on top
    for i in range(len(points) - 1):
        cv2.line(img, points[i], points[i+1], (196, 205, 78), 3)
    
    # Draw points
    for point in points:
        cv2.circle(img, point, 5, (196, 205, 78), -1)
    
    # Axis labels
    cv2.putText(img, "Time (seconds)", (width//2 - 80, height-20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (51, 51, 51), 1)
    cv2.putText(img, "Speed (m/s)", (10, height//2), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (51, 51, 51), 1)
    
    cv2.imwrite('static/visualizations/speed-graph.png', img)
    print("✓ Created speed-graph.png")

if __name__ == "__main__":
    print("Creating PNG visualizations...")
    print("=" * 60)
    
    # Ensure directory exists
    os.makedirs('static/visualizations', exist_ok=True)
    
    create_rpm_chart_png()
    create_speed_graph_png()
    
    print("=" * 60)
    print("✓ All PNG visualizations created successfully!")
