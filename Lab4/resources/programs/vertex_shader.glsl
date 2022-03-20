#version 430


uniform mat4 prog;
uniform mat4 cam;
uniform mat4 trans;
uniform mat4 lookat;

in vec2 tex_coord;
in vec3 in_vert;
in vec3 point_color;
out vec3 pick_color;
out vec2 v_tex_coord;



void main() {


    gl_Position =  prog* cam*lookat* trans* vec4(in_vert, 1.0);
    pick_color=point_color;
    v_tex_coord = tex_coord;

}