#version 430
uniform vec3 switcher;
uniform vec3 objectColor;
uniform vec3 lightColor;
uniform vec3 viewPos;
//float specularStrength = 0.3;
out vec4 color;
in vec3 pick_color;
in vec3 Normal;
in vec3 FragPos;

struct Material {
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    float shininess;
};
struct Light {
    vec3 position;

    vec3 ambient;
    vec3 diffuse;
    vec3 specular;

    float constant;
    float linear;
    float quadratic;


};

uniform Material material;
uniform Light light;
void main() {
    if (switcher == vec3(1, 2, 3)){

        color = vec4(pick_color, 1.0);
    }
    else if (switcher == vec3(6, 0, 1)){
        // ambient
        //float ambientStrength = 0.1f;
        vec3 ambient = material.ambient * light.ambient;

        // diffuse

        vec3 norm = normalize(Normal);
        vec3 lightDir = normalize(light.position - FragPos);
        float diff = max(dot(norm, lightDir), 0.0);
        vec3 diffuse = light.diffuse * (diff * material.diffuse);

        // specular
        vec3 viewDir = normalize(viewPos - FragPos);
        vec3 reflectDir = reflect(-lightDir, norm);
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), material.shininess);
        vec3 specular = light.specular * (spec * material.specular);


        float distance = length(light.position - FragPos);
        float attenuation = 1.0 / (light.constant + light.linear * distance +
    		    light.quadratic * (distance * distance));

        ambient  *= attenuation;
        diffuse  *= attenuation;
        specular *= attenuation;

        vec3 result = (ambient + diffuse + specular) * objectColor;

        color = vec4(result, 1.0f);


    }
    else {
        color = vec4(switcher, 1.0); }


}