#version 430

mat4 Mvp;
uniform mat4 prog;
uniform mat4 cam;

uniform mat4 lookat;


in vec3 translation;
in vec3 in_vert;
in vec3 point_color;
out vec3 pick_color;

void main() {
    Mvp=prog*cam*lookat;
    gl_Position = Mvp * vec4(in_vert, 1.0);
    pick_color=point_color;
}