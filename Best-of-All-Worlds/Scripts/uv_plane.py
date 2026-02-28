# Reevu Adakroy
# 7/26/2024

import tkinter as tk

def run_animation(title, points, speed=1):
    root = tk.Tk()
    root.title(title)

    canvas_width = 600
    canvas_height = 600
    margin = 50  # Define the margin size
    square_size = canvas_width - 2 * margin

    canvas = tk.Canvas(root, width=canvas_width, height=canvas_height, bg="white")
    canvas.pack()

    # Draw the 1x1 square with thicker lines and slightly curved corners
    canvas.create_rectangle(margin, margin, canvas_width - margin, canvas_height - margin,
                            outline="black", width=3)

    # Draw dotted diagonal lines
    for i in range(0, square_size, 10):
        canvas.create_line(margin + i, margin + i, margin + i + 5, margin + i + 5, fill="black")
        canvas.create_line(canvas_width - margin - i, margin + i, canvas_width - margin - i - 5, margin + i + 5, fill="black")

    arrow = None  # Variable to keep track of the current arrow

    def animate_points(index=0, previous_points=None):
        nonlocal arrow
        if previous_points is None:
            previous_points = []

        if index < len(points):
            # Redraw previous points in blue
            for px, py in previous_points:
                canvas.create_oval(px - 4, py - 4, px + 4, py + 4, fill="blue")

            # Get the new point
            x, y = points[index]
            x = margin + x * square_size
            y = canvas_height - (margin + y * square_size)

            # Draw the new point in red
            canvas.create_oval(x - 4, y - 4, x + 4, y + 4, fill="red")
            
            # Draw an arrow from the previous point to the new point
            if previous_points:
                prev_x, prev_y = previous_points[-1]
                if arrow:
                    canvas.delete(arrow)  # Delete the previous arrow
                arrow = canvas.create_line(prev_x, prev_y, x, y, arrow=tk.LAST, fill="black")
            
            # Add the new point to previous_points and schedule the next point to be drawn
            previous_points.append((x, y))
            root.after(1000//speed, animate_points, index + 1, previous_points)

    animate_points()
    root.mainloop()

def calc_uv(seq, actions):
    # assumes that the seq is 0,1
    n = len(seq)
    # hacky first case correct
    uv_list = [0]*(n-1)+[(0.5, 0)]
    for t in range(n):
        (last_u, last_v) = uv_list[t-1]
        u = (t*last_u + seq[t])/(t+1)
        v = (t*last_v + [1-actions[t], actions[t]][int(seq[t])])/(t+1)
        uv_list[t]=(u,v)
    return uv_list
