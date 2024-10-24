from OpenGL.GL import shaders


vertex_shader = """
#version 400 core
in vec2 position;
flat out dvec2 fragCoord;
void main()
{
    fragCoord = dvec2(position);
    gl_Position = vec4(position, 0.0, 1.0);
}

"""

fragment_shader = """
#version 430 core

flat in vec2 fragCoord;  // Using vec2 for simplicity and compatibility

out vec4 outColor;

uniform vec2 center;  // Center of the view in the complex plane
uniform float scale;  // Zoom level
uniform int maxIter;  // Maximum number of iterations

void main()
{
    // Map the pixel coordinates to the complex plane
    vec2 c = center + fragCoord * scale;

    // Initialize the complex number z to (0, 0)
    vec2 z = vec2(0.0, 0.0);
    int i;
    float iter = 0.0;

    // Iterate the Mandelbrot equation
    for (i = 0; i < maxIter; i++)
    {
        if (dot(z, z) > 4.0) break;
        z = vec2(z.x * z.x - z.y * z.y + c.x, 2.0 * z.x * z.y + c.y);
        iter += 1.0;
    }

    // Smooth coloring calculation for better gradients
    if (i < maxIter)
    {
        float log_zn = log(dot(z, z)) / 2.0;
        float nu = log(log_zn / log(2.0)) / log(2.0);
        iter = iter + 1.0 - nu;
    }

    // Map the iteration count to a color
    float color = iter / float(maxIter);
    vec3 col = vec3(0.5 + 0.5 * cos(6.2831 * (color + vec3(0.0, 0.33, 0.67))));
    outColor = vec4(col, 1.0);
}

"""

try:
    # Example of compiling and attaching shaders
    vertex_shader = shaders.compileShader(vertex_shader, GL_VERTEX_SHADER)
    fragment_shader = shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER)
    shader_program = shaders.compileProgram(vertex_shader, fragment_shader)
except shaders.ShaderCompilationError as e:
    print("Shader Compilation Error:", e)
except shaders.ShaderValidationError as e:
    print("Shader Validation Error:", e)
