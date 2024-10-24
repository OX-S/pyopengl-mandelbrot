import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpmath import mp, mpc, mpf

# Set the decimal precision (number of significant digits)
mp.dps = 50  # Increase this number for higher precision

def mandelbrot(c, max_iter):
    z = mpc(0, 0)
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z * z + c
    return max_iter

def compute_mandelbrot(xmin, xmax, ymin, ymax, width, height, max_iter):
    # Create a 2D array to hold the iteration counts
    pixels = np.zeros((height, width), dtype=int)

    # Generate a grid of complex numbers over the specified range
    xs = [mpf(x) for x in np.linspace(float(xmin), float(xmax), width)]
    ys = [mpf(y) for y in np.linspace(float(ymin), float(ymax), height)]

    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            c = mpc(x, y)
            pixels[i, j] = mandelbrot(c, max_iter)

    return pixels

def update(frame, images, ax, max_iter):
    # Get the bounds for the current frame
    xmin, xmax, ymin, ymax = images[frame]['bounds']

    # Convert bounds to float
    xmin_float = float(xmin)
    xmax_float = float(xmax)
    ymin_float = float(ymin)
    ymax_float = float(ymax)

    # Compute the Mandelbrot set for the current frame
    pixels = compute_mandelbrot(xmin, xmax, ymin, ymax, images[frame]['width'], images[frame]['height'], max_iter)

    # Clear the previous image
    ax.clear()
    ax.set_title(f'Zoom Level: {frame}, Max Iterations: {max_iter}')
    ax.imshow(pixels, extent=(xmin_float, xmax_float, ymin_float, ymax_float), cmap='hot', interpolation='bilinear')
    ax.axis('off')  # Hide the axis for better visualization

def main():
    # Initial parameters
    width, height = 800, 600
    max_iter = 100

    # Starting bounds of the Mandelbrot set
    xmin, xmax = mpf(-2.0), mpf(1.0)
    ymin, ymax = mpf(-1.5), mpf(1.5)

    # Zoom parameters
    zoom_center = mpc(-0.743643887037151, 0.131825904205330)  # Interesting point in the Mandelbrot set
    zoom_factor = 0.9  # Adjust to control zoom speed
    num_frames = 50  # Number of frames in the animation

    # Prepare the figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))

    # Precompute the bounds for each frame
    images = []
    for frame in range(num_frames):
        # Calculate the width and height of the current view
        xrange = (xmax - xmin) * (mpf(zoom_factor) ** frame)
        yrange = (ymax - ymin) * (mpf(zoom_factor) ** frame)

        # Update the bounds centered around the zoom center
        xmin_new = zoom_center.real - xrange / 2
        xmax_new = zoom_center.real + xrange / 2
        ymin_new = zoom_center.imag - yrange / 2
        ymax_new = zoom_center.imag + yrange / 2

        images.append({
            'bounds': (xmin_new, xmax_new, ymin_new, ymax_new),
            'width': width,
            'height': height
        })

    # Create the animation
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=num_frames,
        fargs=(images, ax, max_iter),
        interval=200,
        blit=False
    )

    # Display the animation
    plt.show()

if __name__ == '__main__':
    main()
