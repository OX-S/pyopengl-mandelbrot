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