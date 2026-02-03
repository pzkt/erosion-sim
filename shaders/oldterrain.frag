#version 330 core

in vec3 vWorldNormal;
in vec3 vWorldPos;

out vec4 fragColor;

void main()
{
    vec3 wn = normalize(vWorldNormal) * 0.5 + 0.5;
    fragColor = vec4(wn, 1.0);
}
