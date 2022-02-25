#version 430

mat4 Mvp;
uniform mat4 prog;
uniform mat4 cam;
uniform mat4 trans;
uniform mat4 lookat;



in vec3 in_vert;
in vec3 point_color;
out vec3 pick_color;

void main() {


    gl_Position =  prog* cam* trans*lookat* vec4(in_vert, 1.0);
    pick_color=point_color;
}