import tkinter as tk

# Set up the main application window
root = tk.Tk()
root.title("Random Walk Animation")

# Create a canvas widget
canvas_width = 800
canvas_height = 600
canvas = tk.Canvas(root, width=canvas_width, height=canvas_height, bg="white")
canvas.pack()

# Initial position and points list
x, y = canvas_width // 2, canvas_height // 2
points = [(x, y)]  # Starting point (optional to initialize with the first point)
# Example list of points generated previously
# Replace this with your actual list of points
generated_points = [(400, 300), (410, 300), (420, 290), (430, 290), (440, 300), (450, 310), (460, 320)]

# Append generated points to the points list
points.extend(generated_points)

def animate_points(index=0):
    if index < len(points):
        x, y = points[index]
        # Draw the point
        canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill="blue")
        # Schedule the next point to be drawn
        root.after(100, animate_points, index + 1)

# Start the animation
animate_points()

# Run the application
root.mainloop()
