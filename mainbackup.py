import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# Function to compute the Mandelbrot set
def mandelbrot(c, max_iter):
    z = c
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z * z + c
    return max_iter


# Create a grid of complex numbers for the Mandelbrot set
def mandelbrot_set(xmin, xmax, ymin, ymax, width, height, max_iter):
    r1 = np.linspace(xmin, xmax, width)
    r2 = np.linspace(ymin, ymax, height)
    n3 = np.empty((width, height))

    for i in range(width):
        for j in range(height):
            n3[i, j] = mandelbrot(r1[i] + 1j * r2[j], max_iter)

    return n3.T


# Parameters for the initial view
xmin, xmax, ymin, ymax = -2.0, 1.0, -1.5, 1.5
width, height = 800, 800
max_iter = 256

# Create the figure and axis
fig, ax = plt.subplots()
ax.set_axis_off()

# Create an initial Mandelbrot image
mandelbrot_img = mandelbrot_set(xmin, xmax, ymin, ymax, width, height, max_iter)
img = ax.imshow(mandelbrot_img, cmap='hot', extent=(xmin, xmax, ymin, ymax))


# Animation function to zoom into the Mandelbrot set
def update(frame):
    global xmin, xmax, ymin, ymax
    zoom_factor = 0.9  # Adjust the zoom speed (smaller is slower)

    # Update the view limits for zooming in
    x_center, y_center = (xmin + xmax) / 2, (ymin + ymax) / 2
    width, height = (xmax - xmin) * zoom_factor, (ymax - ymin) * zoom_factor

    xmin, xmax = x_center - width / 2, x_center + width / 2
    ymin, ymax = y_center - height / 2, y_center + height / 2

    # Update the image with the new zoomed-in view
    mandelbrot_img = mandelbrot_set(xmin, xmax, ymin, ymax, width=800, height=800, max_iter=256)
    img.set_data(mandelbrot_img)
    img.set_extent((xmin, xmax, ymin, ymax))
    ax.set_title(f"Zoom Level: {frame}")
    return img,


# Create and display the animation
ani = FuncAnimation(fig, update, frames=100, interval=50, blit=True)

# To save the animation, use the following line:
# ani.save('mandelbrot_zoom.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

plt.show()
