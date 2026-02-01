#include "Erosion.h"
#include <random>
#include <cmath>
#include <algorithm>

static inline float clampf(float v, float a, float b){ return v < a ? a : (v > b ? b : v); }

// Bilinear sample and gradient computation
static void sampleHeightAndGradient(const std::vector<float> &map, int size, float x, float y, float &height, float &gradX, float &gradY){
    int xi = (int)floor(x);
    int yi = (int)floor(y);
    float xf = x - xi;
    float yf = y - yi;

    int x0 = clampf(xi, 0, size-1);
    int x1 = clampf(xi+1, 0, size-1);
    int y0 = clampf(yi, 0, size-1);
    int y1 = clampf(yi+1, 0, size-1);

    float h00 = map[y0*size + x0];
    float h10 = map[y0*size + x1];
    float h01 = map[y1*size + x0];
    float h11 = map[y1*size + x1];

    // height
    float h0 = h00*(1 - xf) + h10*xf;
    float h1 = h01*(1 - xf) + h11*xf;
    height = h0*(1 - yf) + h1*yf;

    // gradients
    gradX = ( (h10 - h00)*(1 - yf) + (h11 - h01)*yf );
    gradY = ( (h01 - h00)*(1 - xf) + (h11 - h10)*xf );
}

void Erosion::applyHydraulicErosion(std::vector<float> &heightmap, const ErosionParams &p){
    int size = p.mapSize;
    std::mt19937 rng(1337);
    std::uniform_real_distribution<float> dist(0.0f, (float)size - 1.001f);

    for(int i=0;i<p.numDrops;i++){
        float x = dist(rng);
        float y = dist(rng);
        float dirX = 0.0f, dirY = 0.0f;
        float speed = 1.0f;
        float water = 1.0f;
        float sediment = 0.0f;

        for(int lifetime=0; lifetime < p.maxLifetime; ++lifetime){
            float height, gradX, gradY;
            sampleHeightAndGradient(heightmap, size, x, y, height, gradX, gradY);

            // update direction
            dirX = dirX * p.inertia - gradX * (1 - p.inertia);
            dirY = dirY * p.inertia - gradY * (1 - p.inertia);

            // normalize direction to avoid extremely small steps
            float len = std::sqrt(dirX*dirX + dirY*dirY);
            if(len == 0.0f){
                // random small perturbation
                dirX = (dist(rng)-size*0.5f)*1e-3f;
                dirY = (dist(rng)-size*0.5f)*1e-3f;
                len = std::sqrt(dirX*dirX + dirY*dirY);
            }
            dirX /= len; dirY /= len;

            x += dirX;
            y += dirY;

            if(x < 0 || x >= size-1 || y < 0 || y >= size-1) break;

            float newHeight, ngradX, ngradY;
            sampleHeightAndGradient(heightmap, size, x, y, newHeight, ngradX, ngradY);
            float deltaHeight = newHeight - height;

            float capacity = std::max(-deltaHeight * speed * water * p.sedimentCapacityFactor, p.minSedimentCapacity);

            if(sediment > capacity || deltaHeight > 0){
                float amountToDeposit = (deltaHeight > 0) ? std::min(deltaHeight, sediment) : (sediment - capacity) * p.depositSpeed;
                sediment -= amountToDeposit;
                // deposit on integer cell
                int cx = (int)clampf(x, 0, size-1);
                int cy = (int)clampf(y, 0, size-1);
                heightmap[cy*size + cx] += amountToDeposit;
            } else {
                float amountToErode = std::min((capacity - sediment) * p.erodeSpeed, -deltaHeight);
                if(amountToErode > 0){
                    int cx = (int)clampf(x, 0, size-1);
                    int cy = (int)clampf(y, 0, size-1);
                    float eroded = std::min(heightmap[cy*size + cx], amountToErode);
                    heightmap[cy*size + cx] -= eroded;
                    sediment += eroded;
                }
            }

            // update speed and water
            speed = std::sqrt(std::max(0.0f, speed*speed + deltaHeight * p.gravity));
            water *= (1 - p.evaporateSpeed);
            if(water <= 0) break;
        }
    }
}
