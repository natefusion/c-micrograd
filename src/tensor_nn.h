#include "tensor_engine.h"

#ifndef TENSOR_NN_H
#define TENSOR_NN_H

struct MLP {
    Array_Tensor layers;
};

struct MLP make_mlp(Array_int shape);

#endif
