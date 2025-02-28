import pygame
from pygame.locals import *
from OpenGL.GL import *
import numpy as np
import sys

vertex_shader_source = """
#version 330 core
layout (location = 0) in vec2 position;
void main(){
    gl_Position = vec4(position, 0.0, 1.0);
}
"""

fragment_shader_source = """
#version 330 core
out vec4 outColor;

uniform vec2 resolution;  
uniform vec2 pan;      
uniform float zoom;      
uniform int max_iter;     
uniform float ncycle;     

uniform vec3 lightDir;    // directional light (should be normalized)
uniform float ambient;
uniform float diffuse;
uniform float specular;
uniform float shininess;

float mandelbrot(vec2 c) {
    vec2 z = vec2(0.0);
    int i;
    float iter = 0.0;
    for (i = 0; i < max_iter; i++){
        if(dot(z, z) > 4.0)
            break;
        float x = z.x*z.x - z.y*z.y + c.x;
        float y = 2.0*z.x*z.y + c.y;
        z = vec2(x, y);
    }
    if(i < max_iter) {
        float log_zn = log(dot(z, z)) / 2.0;
        float nu = log(log_zn / log(2.0)) / log(2.0);
        iter = float(i) + 1.0 - nu;
    } else {
        iter = float(max_iter);
    }
    return iter;
}

void main(){
    vec2 uv = (gl_FragCoord.xy - 0.5 * resolution) / resolution.y;
    vec2 c = uv / zoom + pan;

    float iter = mandelbrot(c);
    float height = iter;

    vec3 n = normalize(vec3(dFdx(height), dFdy(height), 1.0));

    float t = mod(iter, ncycle) / ncycle;

    vec3 baseColor = vec3(0.5 + 0.5 * cos(6.28318 * t + vec3(0.0, 0.5, 1.0)));

    // Lighting: combine Lambert (diffuse) with Blinn–Phong (specular)
    vec3 L = normalize(lightDir);
    vec3 V = vec3(0.0, 0.0, 1.0);  // viewer looking along +Z
    vec3 H = normalize(L + V);
    float lambert = max(dot(n, L), 0.0);
    float spec = pow(max(dot(n, H), 0.0), shininess);

    vec3 color = ambient * baseColor + diffuse * lambert * baseColor + specular * spec * vec3(1.0);
    outColor = vec4(color, 1.0);
}
"""



def compile_shader(source, shader_type):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)

    result = glGetShaderiv(shader, GL_COMPILE_STATUS)
    if not result:
        error_msg = glGetShaderInfoLog(shader).decode()
        print("Shader compilation error:", error_msg)
        sys.exit(1)
    return shader


def create_shader_program(vertex_source, fragment_source):
    vertex_shader = compile_shader(vertex_source, GL_VERTEX_SHADER)
    fragment_shader = compile_shader(fragment_source, GL_FRAGMENT_SHADER)
    program = glCreateProgram()
    glAttachShader(program, vertex_shader)
    glAttachShader(program, fragment_shader)
    glLinkProgram(program)

    result = glGetProgramiv(program, GL_LINK_STATUS)
    if not result:
        error_msg = glGetProgramInfoLog(program).decode()
        print("Program linking error:", error_msg)
        sys.exit(1)
    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)
    return program


def main():
    pygame.init()
    width, height = 800, 600
    pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)

    shader_program = create_shader_program(vertex_shader_source, fragment_shader_source)
    glUseProgram(shader_program)

    vertices = np.array([
        -1.0, -1.0,
        1.0, -1.0,
        1.0, 1.0,
        -1.0, 1.0
    ], dtype=np.float32)
    indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)

    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    ebo = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

    pos_loc = glGetAttribLocation(shader_program, "position")
    glEnableVertexAttribArray(pos_loc)
    glVertexAttribPointer(pos_loc, 2, GL_FLOAT, GL_FALSE, 2 * vertices.itemsize, ctypes.c_void_p(0))

    resolution_loc = glGetUniformLocation(shader_program, "resolution")
    pan_loc = glGetUniformLocation(shader_program, "pan")
    zoom_loc = glGetUniformLocation(shader_program, "zoom")
    max_iter_loc = glGetUniformLocation(shader_program, "max_iter")
    ncycle_loc = glGetUniformLocation(shader_program, "ncycle")
    lightDir_loc = glGetUniformLocation(shader_program, "lightDir")
    ambient_loc = glGetUniformLocation(shader_program, "ambient")
    diffuse_loc = glGetUniformLocation(shader_program, "diffuse")
    specular_loc = glGetUniformLocation(shader_program, "specular")
    shininess_loc = glGetUniformLocation(shader_program, "shininess")

    glUniform2f(resolution_loc, width, height)
    glUniform1i(max_iter_loc, 300)
    glUniform1f(ncycle_loc, 50.0)
    glUniform3f(lightDir_loc, 0.5, 0.5, 1.0)
    glUniform1f(ambient_loc, 0.2)
    glUniform1f(diffuse_loc, 0.7)
    glUniform1f(specular_loc, 0.5)
    glUniform1f(shininess_loc, 32.0)

    pan = np.array([-0.5, 0.0], dtype=np.float32)
    zoom = 1.0

    clock = pygame.time.Clock()
    dragging = False
    last_mouse = None

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 1:
                    dragging = True
                    last_mouse = event.pos
                elif event.button == 4:
                    zoom *= 1.1
                elif event.button == 5:
                    zoom /= 1.1
            elif event.type == MOUSEBUTTONUP:
                if event.button == 1:
                    dragging = False
            elif event.type == MOUSEMOTION:
                if dragging:
                    dx = event.pos[0] - last_mouse[0]
                    dy = event.pos[1] - last_mouse[1]

                    pan[0] -= dx / width / zoom * 2.0
                    pan[1] += dy / height / zoom * 2.0
                    last_mouse = event.pos

        glClear(GL_COLOR_BUFFER_BIT)

        glUniform2f(pan_loc, pan[0], pan[1])
        glUniform1f(zoom_loc, zoom)

        glBindVertexArray(vao)
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == '__main__':
    main()
