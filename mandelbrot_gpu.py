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

uniform vec2 center;
uniform float scale;
uniform int maxIter;
uniform vec2 resolution;

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


const vec3 lightDir = normalize(vec3(0.5, 0.5, 1.0));
const vec3 viewDir = vec3(0.0, 0.0, 1.0);
const float shininess = 32.0;

vec2 complex_mul(vec2 a, vec2 b) {
    return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

void main()
{
    // normalize pixel coords to [-1, 1]
    vec2 uv = (gl_FragCoord.xy / resolution.xy) * 2.0 - vec2(1.0);
    uv.x *= resolution.x / resolution.y; // Correct aspect ratio

    // map to complex plane
    vec2 c = center + uv * scale;

    // init z and its derivative dz/dc
    vec2 z = vec2(0.0, 0.0);
    vec2 dz = vec2(1.0, 0.0); // For derivative dz/dc
    
    

    int i;
    float iter = 0.0;

    // stripe average 
    float stripeSum = 0.0;
    float k = 0.3; // stripe frequency

    // mandelbrot calc
    for (i = 0; i < maxIter; i++)
    {
        if (length(z) > 2.0) break;

      
        stripeSum += sin(k * atan(z.y, z.x));

        dz = 2.0 * complex_mul(z, dz) + vec2(1.0, 0.0);

        z = complex_mul(z, z) + c;
    }

    // smooth coloring
    if (i < maxIter)
    {
        float log_zn = log(length(z)) / 2.0;
        float nu = log(log_zn / log(2.0)) / log(2.0);
        iter = float(i) + 1.0 - nu;
    }
    else
    {
        iter = float(i);
    }

    // stripe average
    float stripeAverage = stripeSum / iter;
    float stripeColor = 0.5 + 0.5 * stripeAverage;

    // compute normal using screen space derivatives
    float dx = dFdx(iter);
    float dy = dFdy(iter);
    vec3 normal = normalize(vec3(-dx, -dy, 1.0));

    // lighting 
    float diffuse = max(dot(normal, lightDir), 0.0);
    vec3 halfDir = normalize(lightDir + viewDir);
    float specular = pow(max(dot(normal, halfDir), 0.0), shininess);

    // step shading 
    float steps = 500; // num of steps
    float iterStep = floor(iter / float(maxIter) * steps) / steps;

    // determine color from gradient based
    float t = fract(iter / float(maxIter));  // between 0 and 1
    int idx1 = int(t * 7.0);  // first color 
    int idx2 = (idx1 + 1) % 8; // next color 
    float blend = t * 7.0 - float(idx1); // blend 

    vec3 color = mix(gradient[idx1], gradient[idx2], blend);  // Linear interpolation 

    // stripe coloring
    color = mix(color, vec3(stripeColor), 0.5);

    // lighting
    color *= diffuse + 0.2; // Add ambient term
    color += vec3(specular);

    // step shading
    color = floor(color * steps) / steps;

    // gamma correction 
    color = pow(color, vec3(0.8));

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


def init_shader():
    # Compile shaders
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


class MandelbrotViewer:
    def __init__(self, width=800, height=600):
        self.screen_VAO = None
        self.texture = None
        self.fbo = None
        self.hi_res_height = None
        self.VBO = None
        self.hi_res_width = None
        self.VAO = None
        self.screen_program = None
        pygame.init()

        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 4)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 0)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
        pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Mandelbrot Set Viewer")
        self.width, self.height = width, height

        # Debugging
        version = glGetString(GL_VERSION).decode()
        print("OpenGL version:", version)
        glsl_version = glGetString(GL_SHADING_LANGUAGE_VERSION).decode()
        print("GLSL version:", glsl_version)

        self.program = init_shader()
        self.init_quad()
        self.init_fbo()
        self.init_screen_shader()
        self.init_screen_quad()

        self.maxIter = 25000
        self.scale = 2.5  # Initial Scale
        self.center = np.array([-0.75, 0.0], dtype=np.float64) # tried to add double precision (wip)
        self.dragging = False
        self.last_mouse_pos = None

        self.main_loop()

    def init_quad(self):

        vertices = np.array([
            -1.0, -1.0,
             1.0, -1.0,
            -1.0,  1.0,
             1.0,  1.0
        ], dtype=np.float32)


        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)

        self.VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)


        position = glGetAttribLocation(self.program, 'position')
        glEnableVertexAttribArray(position)
        glVertexAttribPointer(position, 2, GL_FLOAT, GL_FALSE, 0, None)


        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

    def init_fbo(self):

        # Increase resolution
        self.hi_res_width = self.width * 2
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
        # Compile shaders
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

            -1.0, -1.0,    0.0, 0.0,
             1.0, -1.0,    1.0, 0.0,
            -1.0,  1.0,    0.0, 1.0,
             1.0,  1.0,    1.0, 1.0,
        ], dtype=np.float32)

        self.screen_VAO = glGenVertexArrays(1)
        glBindVertexArray(self.screen_VAO)

        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

        # Position attribute
        position = glGetAttribLocation(self.screen_program, 'position')
        glEnableVertexAttribArray(position)
        glVertexAttribPointer(position, 2, GL_FLOAT, GL_FALSE, 4 * ctypes.sizeof(ctypes.c_float), ctypes.c_void_p(0))

        # TexCoord attribute
        tex_coord = glGetAttribLocation(self.screen_program, 'texCoord')
        glEnableVertexAttribArray(tex_coord)
        glVertexAttribPointer(tex_coord, 2, GL_FLOAT, GL_FALSE, 4 * ctypes.sizeof(ctypes.c_float), ctypes.c_void_p(2 * ctypes.sizeof(ctypes.c_float)))

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

    def main_loop(self):
        clock = pygame.time.Clock()
        running = True
        while running:
            clock.tick(60)  # 60 FPS
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
            elif event.button == 4:  # Scroll wheel up
                self.zoom(0.9)
            elif event.button == 5:  # Scroll wheel down
                self.zoom(1.1)
        elif event.type == MOUSEBUTTONUP:
            if event.button == 1:  # Left click release
                self.dragging = False
        elif event.type == MOUSEMOTION:
            if self.dragging:
                x, y = pygame.mouse.get_pos()
                dx = (x - self.last_mouse_pos[0]) / self.width * self.scale * 2.0
                dy = (y - self.last_mouse_pos[1]) / self.height * self.scale * 2.0
                dx *= self.width / self.height  # set aspect ratio
                self.center -= np.array([dx, -dy], dtype=np.float64)
                self.last_mouse_pos = (x, y)
        elif event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                pygame.quit()
                quit()

    def zoom(self, factor):

        # Get mouse position in screen coords
        mouse_pos = pygame.mouse.get_pos()
        # Normalize mouse coords
        mouse_ndc = np.array([
            (mouse_pos[0] / self.width) * 2 - 1,
            (mouse_pos[1] / self.height) * -2 + 1
        ], dtype=np.float64)
        # Correct for aspect ratio
        mouse_ndc[0] *= self.width / self.height
        # Convert to world coordinates
        mouse_world = self.center + mouse_ndc * self.scale

        # Apply zoom
        self.scale *= factor
        # Adjust zoom to center on mouse
        self.center = mouse_world - mouse_ndc * self.scale

        # Increase iter when zooming, currenty breaks everything
        # if factor < 1.0:
        #     self.maxIter = int(self.maxIter * 1.1)
        # else:
        #     self.maxIter = int(self.maxIter / 1.1)
        # self.maxIter = max(100, min(self.maxIter, 1000000))

    def render(self):
        # Render to the high-res framebuffer
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glViewport(0, 0, self.hi_res_width, self.hi_res_height)
        glClear(GL_COLOR_BUFFER_BIT)
        glUseProgram(self.program)

        # Set uniforms
        glUniform2f(glGetUniformLocation(self.program, 'resolution'), float(self.hi_res_width),
                    float(self.hi_res_height))
        glUniform2f(glGetUniformLocation(self.program, 'center'), float(self.center[0]), float(self.center[1]))
        glUniform1f(glGetUniformLocation(self.program, 'scale'), float(self.scale))
        glUniform1i(glGetUniformLocation(self.program, 'maxIter'), self.maxIter)

        # Draw quad
        glBindVertexArray(self.VAO)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glBindVertexArray(0)
        glUseProgram(0)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        # Render high-res texture to the screen
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
