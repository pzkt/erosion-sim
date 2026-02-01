#version 330 core

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProj;

out vec3 vWorldPos;
out vec3 vWorldNormal;
out vec3 vViewDir;

void main()
{
    vec4 worldPos = uModel * vec4(inPos, 1.0);
    vWorldPos = worldPos.xyz;

    mat3 normalMatrix = transpose(inverse(mat3(uModel)));
    vWorldNormal = normalize(normalMatrix * inNormal);

    vec3 camPos = vec3(inverse(uView)[3]); 
    vViewDir = normalize(camPos - vWorldPos);

    gl_Position = uProj * uView * worldPos;
}
