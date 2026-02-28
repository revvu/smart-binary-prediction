import tkinter as tk
import random

# Set up the main application window
root = tk.Tk()
root.title("Random Walk Animation")

# Create a canvas widget
canvas_width = 800
canvas_height = 600
canvas = tk.Canvas(root, width=canvas_width, height=canvas_height, bg="white")
canvas.pack()

# Initial position
x, y = canvas_width // 2, canvas_height // 2
point = canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill="blue")

# Random walk parameters
step_size = 10

def random_walk():
    global x, y
    # Randomly choose a direction
    direction = random.choice(['left', 'right', 'up', 'down'])
    if direction == 'left':
        x -= step_size
    elif direction == 'right':
        x += step_size
    elif direction == 'up':
        y -= step_size
    elif direction == 'down':
        y += step_size

    # Ensure the point stays within the canvas bounds
    x = max(0, min(canvas_width, x))
    y = max(0, min(canvas_height, y))

    # Move the point
    canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill="blue")
    
    # Schedule the next step
    root.after(100, random_walk)

# Start the random walk animation
random_walk()

# Run the application
root.mainloop()
