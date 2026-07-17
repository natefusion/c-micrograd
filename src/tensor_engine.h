#ifdef SCALAR_ENGINE_H
#error "You can't include both the scalar engine and the tensor engine"
#endif

#ifndef TENSOR_ENGINE_H
#define TENSOR_ENGINE_H

#include <math.h>
#include <stdbool.h>
#include <stdio.h>

#include "arena.h"
#include "array.h"

typedef struct Tensor Tensor;

Array_define(int, Array_int);
Array_define(Tensor*, Array_Tensor);

typedef double (*Dyadic_Fn)(double, double);
typedef double (*Monadic_Fn)(double);

typedef struct {
    Tensor *lhs;
    Tensor *rhs;
} Partial_Derivative;

typedef struct {
    Tensor *lhs;
    Tensor *rhs;
} Children;

enum Flags : uint8_t {
    Flags_visited = 0b00000001,
    Flags_arity_1 = 0b00000010,
    Flags_arity_2 = 0b00000100,
};

struct Tensor_small {
    double *data;
    Array_int shape;
};

struct Tensor {
    double *data;
    double *grad;
    Array_int shape;
    Partial_Derivative (*pd_fn)(Tensor*);
    Children children;
    enum Flags flags;
    // debugging
    char const *op;
    char const *name;
};

void tensor_reset(ptrdiff_t mem);

bool tensor_is_scalar(Tensor *v);
bool tensor_is_shape_eq(Array_int lhs, Array_int rhs);
int tensor_order(Tensor* v);
int tensor_rank(Tensor* v, int dimension);
int tensor_numel(Array_int shape);
int tensor_at_lame(Tensor *v, int subscript, ...);
#define tensor_at(tensor, subscript, ...) tensor_at_lame(maybe_ref((tensor)), (subscript), __VA_ARGS__, -1)

Array_int tensor_shape_make_lame(int *array, int size);
#define tensor_shape_make(...) tensor_shape_make_lame((int[])__VA_ARGS__, sizeof((int[])__VA_ARGS__)/sizeof((int[])__VA_ARGS__[0]))
Tensor* tensor_make(double number, Array_int shape);
Array_Tensor tensor_make_array(int len);
Tensor* tensor_add(Tensor *a, Tensor *b);
Tensor* tensor_mul(Tensor *a, Tensor *b);
Tensor* tensor_sub(Tensor *a, Tensor *b);
Tensor* tensor_div(Tensor *a, Tensor *b);
Tensor *tensor_tanh(Tensor *a);
Tensor *tensor_relu(Tensor *a);
Tensor *tensor_expt(Tensor *base, Tensor* power);
Tensor* tensor_product(Tensor *a, Tensor *b);

Partial_Derivative tensor_add_pd(Tensor* v);
Partial_Derivative tensor_mul_pd(Tensor* v);
Partial_Derivative tensor_sub_pd(Tensor* v);
Partial_Derivative tensor_div_pd(Tensor* v);
Partial_Derivative tensor_tanh_pd(Tensor* v);
Partial_Derivative tensor_relu_pd(Tensor* v);
Partial_Derivative tensor_expt_pd(Tensor *v);

void tensor_topological_sort(Tensor **topo_array, int* offset_from_end, Tensor *value);
void tensor_backward(Tensor *value);
void draw_computational_graph(Tensor *v);

Arena *tensor_get_arena(void);

#endif // TENSOR_ENGINE_H
