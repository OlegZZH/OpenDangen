#version 430
uniform int switcher;
uniform vec3 objectColor;
uniform sampler2D texture;
in vec2 v_tex_coord;
out vec4 f_color;
in vec3 pick_color;
in vec3 Normal;


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
        case 4:
            f_color=vec4(texture2D(texture, v_tex_coord).rgb, 1.0);

    }

}