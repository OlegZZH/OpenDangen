#version 430


uniform mat4 prog;
uniform mat4 cam;
uniform mat4 trans;
uniform mat4 lookat;



in vec3 normal ;
in vec3 in_vert;
in vec3 point_color;
out vec3 pick_color;
out vec3 Normal;
out vec3 FragPos;

void main() {


    gl_Position =  prog* cam*lookat* trans* vec4(in_vert, 1.0);
    pick_color=point_color;
    FragPos = vec3( trans * vec4(in_vert, 1.0f));
    Normal = mat3(transpose(inverse(trans))) * normal;
}