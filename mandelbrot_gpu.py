import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import ctypes

# Vertex shader
vertex_shader = """
#version 400 core
in vec2 position;
void main()
{
    gl_Position = vec4(position, 0.0, 1.0);
}
"""

# Fragment shader with double-precision calculations
# Fragment shader with corrected function calls
fragment_shader = """
#version 400 core

out vec4 outColor;

uniform dvec2 center;
uniform double scale;
uniform int maxIter;
uniform dvec2 resolution;

// Gradient colors array
const vec3 gradient[8] = vec3[](
    vec3(0.3, 0.0, 0.2),  // Deep purple
    vec3(0.5, 0.0, 0.5),  // Purple
    vec3(0.8, 0.0, 0.5),  // Violet
    vec3(0.9, 0.5, 0.2),  // Orange
    vec3(0.9, 0.8, 0.2),  // Yellow
    vec3(0.2, 0.6, 0.4),  // Greenish
    vec3(0.2, 0.2, 0.8),  // Light Blue
    vec3(0.0, 0.1, 0.5)   // Deep Blue
);

void main()
{
    // Normalize pixel coordinates to [-1, 1]
    dvec2 uv = (gl_FragCoord.xy / resolution.xy) * 2.0 - dvec2(1.0);
    uv.x *= resolution.x / resolution.y; // Correct aspect ratio

    // Map to complex plane with high precision
    dvec2 c = center + uv * scale;

    // Initialize z to (0, 0)
    dvec2 z = dvec2(0.0, 0.0);
    int i;

    // Mandelbrot iteration with double precision
    for (i = 0; i < maxIter; i++)
    {
        if (dot(z, z) > 4.0) break;
        z = dvec2(z.x * z.x - z.y * z.y + c.x, 2.0 * z.x * z.y + c.y);
    }

    // Smooth coloring
    float iter = float(i);
    if (i < maxIter)
    {
        // Convert to float before passing to log
        float log_zn = log(float(dot(z, z))) / 2.0;
        float nu = log(log_zn / log(2.0)) / log(2.0);
        iter = float(i) + 1.0 - nu;
    }

    // Determine color from gradient based on smooth iteration count
    float t = fract(iter / float(maxIter));  // t is a value between 0 and 1
    int idx1 = int(t * 7.0);  // Index of the first color in the gradient
    int idx2 = (idx1 + 1) % 8; // Index of the next color in the gradient
    float blend = t * 7.0 - float(idx1); // How much to blend between colors

    vec3 color = mix(gradient[idx1], gradient[idx2], blend);  // Linear interpolation between gradient colors
    color = pow(color, vec3(0.8)); // Optional gamma correction for more vivid colors

    outColor = vec4(color, 1.0);
}

"""


# Screen quad vertex shader
screen_vertex_shader = """
#version 330 core
in vec2 position;
in vec2 texCoord;
out vec2 TexCoord;
void main()
{
    gl_Position = vec4(position, 0.0, 1.0);
    TexCoord = texCoord;
}
"""

# Screen quad fragment shader
screen_fragment_shader = """
#version 330 core
in vec2 TexCoord;
out vec4 outColor;
uniform sampler2D screenTexture;
void main()
{
    outColor = texture(screenTexture, TexCoord);
}
"""

class MandelbrotViewer:
    def __init__(self, width=800, height=600):
        pygame.init()
        # Request OpenGL 4.0 Core Profile
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 4)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 0)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
        pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Mandelbrot Set Viewer")
        self.width, self.height = width, height

        # Print OpenGL and GLSL versions
        version = glGetString(GL_VERSION).decode()
        print("OpenGL version:", version)
        glsl_version = glGetString(GL_SHADING_LANGUAGE_VERSION).decode()
        print("GLSL version:", glsl_version)

        self.program = self.init_shader()
        self.init_quad()
        self.init_fbo()
        self.init_screen_shader()
        self.init_screen_quad()

        self.maxIter = 2000
        self.scale = 2.5  # Initial scale (zoom level)
        self.center = np.array([-0.75, 0.0], dtype=np.float64)  # Use float64 for double precision
        self.dragging = False
        self.last_mouse_pos = None

        self.main_loop()

    def init_shader(self):
        # Compile shaders and create shader program
        try:
            program = compileProgram(
                compileShader(vertex_shader, GL_VERTEX_SHADER),
                compileShader(fragment_shader, GL_FRAGMENT_SHADER)
            )
        except RuntimeError as e:
            print("Shader compilation or linking failed.")
            print(e)
            pygame.quit()
            quit()
        return program

    def init_quad(self):
        # Define a quad that covers the viewport
        vertices = np.array([
            -1.0, -1.0,  # Bottom-left
             1.0, -1.0,  # Bottom-right
            -1.0,  1.0,  # Top-left
             1.0,  1.0   # Top-right
        ], dtype=np.float32)

        # Generate and bind VAO and VBO
        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)

        self.VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

        # Specify vertex attributes
        position = glGetAttribLocation(self.program, 'position')
        glEnableVertexAttribArray(position)
        glVertexAttribPointer(position, 2, GL_FLOAT, GL_FALSE, 0, None)

        # Unbind for cleanup
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

    def init_fbo(self):
        # High-resolution dimensions
        self.hi_res_width = self.width * 2  # Increase resolution
        self.hi_res_height = self.height * 2

        # Create texture
        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, self.hi_res_width, self.hi_res_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        # Create framebuffer
        self.fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.texture, 0)

        # Check if framebuffer is complete
        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            print("Error: Framebuffer is not complete")
            pygame.quit()
            quit()

        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def init_screen_shader(self):
        # Compile shaders and create shader program
        try:
            self.screen_program = compileProgram(
                compileShader(screen_vertex_shader, GL_VERTEX_SHADER),
                compileShader(screen_fragment_shader, GL_FRAGMENT_SHADER)
            )
        except RuntimeError as e:
            print("Screen shader compilation or linking failed.")
            print(e)
            pygame.quit()
            quit()

    def init_screen_quad(self):
        vertices = np.array([
            # Positions    # Texture Coords
            -1.0, -1.0,    0.0, 0.0,
             1.0, -1.0,    1.0, 0.0,
            -1.0,  1.0,    0.0, 1.0,
             1.0,  1.0,    1.0, 1.0,
        ], dtype=np.float32)

        self.screen_VAO = glGenVertexArrays(1)
        glBindVertexArray(self.screen_VAO)

        VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, VBO)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

        # Position attribute
        position = glGetAttribLocation(self.screen_program, 'position')
        glEnableVertexAttribArray(position)
        glVertexAttribPointer(position, 2, GL_FLOAT, GL_FALSE, 4 * ctypes.sizeof(ctypes.c_float), ctypes.c_void_p(0))

        # TexCoord attribute
        texCoord = glGetAttribLocation(self.screen_program, 'texCoord')
        glEnableVertexAttribArray(texCoord)
        glVertexAttribPointer(texCoord, 2, GL_FLOAT, GL_FALSE, 4 * ctypes.sizeof(ctypes.c_float), ctypes.c_void_p(2 * ctypes.sizeof(ctypes.c_float)))

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

    def main_loop(self):
        clock = pygame.time.Clock()
        running = True
        while running:
            clock.tick(60)  # Limit to 60 FPS
            for event in pygame.event.get():
                self.handle_event(event)
            self.render()
            pygame.display.flip()
        pygame.quit()

    def handle_event(self, event):
        if event.type == QUIT:
            pygame.quit()
            quit()
        elif event.type == MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click
                self.dragging = True
                self.last_mouse_pos = pygame.mouse.get_pos()
            elif event.button == 4:  # Scroll up
                self.zoom(0.9)
            elif event.button == 5:  # Scroll down
                self.zoom(1.1)
        elif event.type == MOUSEBUTTONUP:
            if event.button == 1:  # Left click release
                self.dragging = False
        elif event.type == MOUSEMOTION:
            if self.dragging:
                x, y = pygame.mouse.get_pos()
                dx = (x - self.last_mouse_pos[0]) / self.width * self.scale * 2.0
                dy = (y - self.last_mouse_pos[1]) / self.height * self.scale * 2.0
                dx *= self.width / self.height  # Correct aspect ratio
                self.center -= np.array([dx, -dy], dtype=np.float64)
                self.last_mouse_pos = (x, y)
        elif event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                pygame.quit()
                quit()

    def zoom(self, factor):
        # Get mouse position in screen coordinates
        mouse_pos = pygame.mouse.get_pos()
        # Convert to normalized device coordinates (-1 to 1)
        mouse_ndc = np.array([
            (mouse_pos[0] / self.width) * 2 - 1,
            (mouse_pos[1] / self.height) * -2 + 1
        ], dtype=np.float64)
        # Correct aspect ratio
        mouse_ndc[0] *= self.width / self.height
        # Convert to world coordinates
        mouse_world = self.center + mouse_ndc * self.scale

        # Apply zoom
        self.scale *= factor
        # Adjust center to zoom towards mouse position
        self.center = mouse_world - mouse_ndc * self.scale

        # Increase maxIter when zooming in, decrease when zooming out
        # if factor < 1.0:
        #     self.maxIter = int(self.maxIter * 1.1)
        # else:
        #     self.maxIter = int(self.maxIter / 1.1)
        # self.maxIter = max(100, min(self.maxIter, 1000000))  # Clamp values

    def render(self):
        # Render to the high-resolution framebuffer
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glViewport(0, 0, self.hi_res_width, self.hi_res_height)
        glClear(GL_COLOR_BUFFER_BIT)
        glUseProgram(self.program)

        # Set uniforms with high-resolution dimensions
        glUniform2d(glGetUniformLocation(self.program, 'resolution'), self.hi_res_width, self.hi_res_height)
        glUniform2d(glGetUniformLocation(self.program, 'center'), self.center[0], self.center[1])
        glUniform1d(glGetUniformLocation(self.program, 'scale'), self.scale)
        glUniform1i(glGetUniformLocation(self.program, 'maxIter'), self.maxIter)

        # Draw the quad
        glBindVertexArray(self.VAO)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glBindVertexArray(0)
        glUseProgram(0)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        # Now render the high-resolution texture to the screen
        glViewport(0, 0, self.width, self.height)
        glClear(GL_COLOR_BUFFER_BIT)
        glUseProgram(self.screen_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glUniform1i(glGetUniformLocation(self.screen_program, 'screenTexture'), 0)

        glBindVertexArray(self.screen_VAO)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glBindVertexArray(0)
        glUseProgram(0)


if __name__ == '__main__':
    MandelbrotViewer()
