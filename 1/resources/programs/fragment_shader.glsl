#version 430
uniform vec3 switcher;
out vec4 color;
in vec3 pick_color;

void main() {
    if( switcher == vec3(1,2,3)){

        color = vec4(pick_color, 1.0);
    }
    else{color = vec4(switcher, 1.0);}

}