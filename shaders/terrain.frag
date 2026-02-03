#version 330 core

in vec3 vWorldNormal;

out vec4 fragColor;

void main()
{
    vec3 wn = normalize(vWorldNormal) * 0.5 + 0.5; // Normalize and remap to 0-1 range
    float gray = dot(wn, vec3(0.299, 0.587, 0.114)); // Calculate the grayscale value using luminance
    gray *= 0.8;
    fragColor = vec4(gray, gray, gray, 1.0); // Set the RGBA color
}
