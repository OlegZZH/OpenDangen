#version 430
uniform int switcher;
uniform vec3 objectColor;
uniform vec3 lightColor;
uniform vec3 lightPos;
uniform vec3 viewPos;
float specularStrength = 0.5f;
out vec4 f_color;
in vec3 pick_color;
in vec3 Normal;
in vec3 FragPos;

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
            f_color = vec4(1.0, 0.5, 0.31, 1.0);
            break;
        case 3:
            f_color = vec4(pick_color, 1.0);
            break;
        case 4:
            vec3 norm = normalize(Normal);
            vec3 lightDir = normalize(lightPos - FragPos);
            vec3 viewDir = normalize(viewPos - FragPos);
            vec3 reflectDir = reflect(-lightDir, norm);

            float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
            vec3 specular = specularStrength * spec * lightColor;

            float diff = max(dot(Normal, lightDir), 0.0);
            vec3 diffuse = diff * lightColor;

            float ambientStrength = 0.1f;
            vec3 ambient = ambientStrength * lightColor;

            vec3 result = (ambient + diffuse + specular) * objectColor;

            f_color = vec4(result, 1.0f);

            break;
    }

}