#version 430
uniform int switcher;
out vec4 f_color;
in vec3 pick_color;

void main() {
    switch (switcher)
    {
        case 0:

            f_color = vec4(0.1, 0.1, 0.1, 1.0);
            break;
        case 1:

            f_color = vec4(1, 0.0, 0.0, 1.0);
            break;
        case 2:
            f_color = vec4(0.0, 0.0, 1.0, 1.0);
            break;
        case 3:
            f_color = vec4(pick_color, 1.0);
            break;
    }

}