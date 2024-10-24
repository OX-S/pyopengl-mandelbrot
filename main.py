import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np

# Vertex shader code
vertex_shader = """
#version 400
in vec2 position;
out vec2 fragCoord;
void main()
{
    fragCoord = position;
    gl_Position = vec4(position, 0.0, 1.0);
}
"""

# Fragment shader code with double-precision arithmetic
fragment_shader = """
#version 400
#extension GL_ARB_gpu_shader_fp64 : enable

in vec2 fragCoord;
out vec4 outColor;

uniform double zoom;
uniform dvec2 center;
uniform int maxIter;

void main()
{
    // Map pixel coordinate to complex plane using double precision
    dvec2 c = center + dvec2(fragCoord) * zoom;
    dvec2 z = dvec2(0.0, 0.0);
    int i;
    for(i = 0; i < maxIter; i++)
    {
        if(length(z) > 2.0)
            break;
        // z = z^2 + c
        z = dvec2(
            z.x * z.x - z.y * z.y + c.x,
            2.0 * z.x * z.y + c.y
        );
    }
    
    // Smooth coloring
    double smooth_iter = i + 1.0 - log(float(log(length(z)))) / log(2.0);
    float color_value = float(0.5 + 0.5 * cos(3.0 + smooth_iter * 0.15));
    outColor = vec4(
        0.5 + 0.5 * cos(6.2831 * color_value + 0.0),
        0.5 + 0.5 * cos(6.2831 * color_value + 2.0),
        0.5 + 0.5 * cos(6.2831 * color_value + 4.0),
        1.0
    );
}

"""

def main():
    # Initialize Pygame and OpenGL context
    pygame.init()
    screen_size = (800, 600)

    # Set OpenGL attributes before creating the display
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 4)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 0)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)

    screen = pygame.display.set_mode(screen_size, DOUBLEBUF | OPENGL)
    pygame.display.set_caption("Real-Time Mandelbrot Set Zoom with Double Precision")

    # Check OpenGL version
    gl_version = glGetString(GL_VERSION).decode()
    print("OpenGL version:", gl_version)
    if int(gl_version.split('.')[0]) < 4:
        print("OpenGL 4.0 or higher is required.")
        pygame.quit()
        return

    # Compile shaders and program
    try:
        shader = compileProgram(
            compileShader(vertex_shader, GL_VERTEX_SHADER),
            compileShader(fragment_shader, GL_FRAGMENT_SHADER)
        )
    except RuntimeError as e:
        print("Shader compilation error:", e)
        pygame.quit()
        return

    # Define the quad covering the screen
    vertices = np.array([
        -1.0, -1.0,  # Bottom Left
         1.0, -1.0,  # Bottom Right
         1.0,  1.0,  # Top Right
        -1.0,  1.0   # Top Left
    ], dtype=np.float32)
    indices = np.array([
        0, 1, 2,
        2, 3, 0
    ], dtype=np.uint32)

    # Generate buffers and arrays
    VBO = glGenBuffers(1)
    VAO = glGenVertexArrays(1)
    EBO = glGenBuffers(1)

    # Bind VAO
    glBindVertexArray(VAO)

    # Vertex buffer
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    # Element buffer
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

    # Set attributes
    position = glGetAttribLocation(shader, 'position')
    glEnableVertexAttribArray(position)
    glVertexAttribPointer(position, 2, GL_FLOAT, GL_FALSE, 0, None)

    # Unbind VAO
    glBindVertexArray(0)

    # Uniform locations
    zoom_loc = glGetUniformLocation(shader, 'zoom')
    center_loc = glGetUniformLocation(shader, 'center')
    max_iter_loc = glGetUniformLocation(shader, 'maxIter')

    # Initial parameters
    zoom = 1.0
    center = [-0.74364388703, 0.13182590421]  # Example of an interesting point
    max_iter = 100
    zoom_speed = 0.99  # Adjust for zoom speed

    clock = pygame.time.Clock()
    running = True
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # Implement mouse interaction if desired

        # Update zoom
        zoom *= zoom_speed

        # Dynamically adjust max_iter based on zoom level
        if zoom <= 1e-12:
            max_iter = 1000  # Cap max_iter to prevent excessive computation
        else:
            max_iter = int(100 + abs(np.log10(zoom)) * 50)
            max_iter = min(max_iter, 1000)  # Optional: limit max_iter

        # Clear screen
        glClear(GL_COLOR_BUFFER_BIT)

        # Use shader program
        glUseProgram(shader)

        # Update uniforms
        glUniform1d(zoom_loc, zoom)
        glUniform2d(center_loc, center[0], center[1])
        glUniform1i(max_iter_loc, max_iter)

        # Bind VAO and draw
        glBindVertexArray(VAO)
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)

        # Swap buffers
        pygame.display.flip()

        # Cap the frame rate
        clock.tick(60)

    # Clean up
    glDeleteProgram(shader)
    glDeleteBuffers(1, [VBO])
    glDeleteBuffers(1, [EBO])
    glDeleteVertexArrays(1, [VAO])
    pygame.quit()

if __name__ == '__main__':
    main()
