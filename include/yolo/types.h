#ifndef TYPES_H
#define TYPES_H

#include <string>
#include <vector>
#include "config.h"


struct Detection
{
    // x1, y1, x2, y2
    float bbox[4];
    float conf;
    int classId;
    float mask[kNumMasks];  // mask coefficient
    std::vector<float> maskMatrix;  // 2D mask after mask coefficient multiply proto, and scale to original image
};


#endif  // TYPES_H
