import pygame
from pygame.locals import *
from OpenGL.GL import *
import numpy as np
import sys

vertex_shader_source = """
#version 400 core
layout (location = 0) in vec2 position;
void main(){
    gl_Position = vec4(position, 0.0, 1.0);
}
"""

fragment_shader_source = """
#version 400 core
#extension GL_ARB_gpu_shader_fp64 : enable
out vec4 outColor;

uniform vec2 resolution;  
uniform dvec2 pan;      
uniform double zoom;      
uniform int max_iter;     
uniform double ncycle;     

double dlog(double x) {
    return double(log(float(x)));
}

double mandelbrot(in dvec2 c)
{
    dvec2 z = dvec2(0.0);
    int i;
    double iter = 0.0;
    for(i = 0; i < max_iter; i++){
        if(dot(z, z) > 4.0)
            break;
        double x = z.x * z.x - z.y * z.y + c.x;
        double y = 2.0 * z.x * z.y + c.y;
        z = dvec2(x, y);
    }
    if(i < max_iter) {
        double log_zn = dlog(dot(z, z)) / 2.0;
        double nu = dlog(log_zn / dlog(2.0)) / dlog(2.0);
        iter = double(i) + 1.0 - nu;
    } else {
        iter = double(max_iter);
    }
    return iter;
}

void main(){
    dvec2 uv = (dvec2(gl_FragCoord.xy) - 0.5 * dvec2(resolution)) / dvec2(resolution).y;
    dvec2 c  = uv / zoom + pan;

    double iter = mandelbrot(c);
    
    float dither = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453);
    float t = float(mod(iter, ncycle) + 0.4 * dither) / float(ncycle);

    vec3 baseColor = vec3(0.5 + 0.5 * cos(6.28318 * t + vec3(0.0, 0.8, 1.5)));
    outColor = vec4(baseColor, 1.0);
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
    glUniform1d(ncycle_loc, 50.0)
    glUniform3d(lightDir_loc, 0.5, 0.5, 1.0)
    glUniform1d(ambient_loc, 0.2)
    glUniform1d(diffuse_loc, 0.7)
    glUniform1d(specular_loc, 0.5)
    glUniform1d(shininess_loc, 32.0)

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

        glUniform2d(pan_loc, pan[0], pan[1])
        glUniform1d(zoom_loc, zoom)

        glBindVertexArray(vao)
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == '__main__':
    main()
