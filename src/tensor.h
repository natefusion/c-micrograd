#ifndef TENSOR_H
#define TENSOR_H

#include "array.h"

Array_define(int, Array_int);

typedef struct Tensor {
    double *data;
    int *shape;
    int *stride;
    int dims;
    int numel;
} Tensor;

typedef double (*Dyadic_Fn)(double, double);
typedef double (*Monadic_Fn)(double);

int tensor_numel(Tensor *a);
bool tensor_shape_eq(Tensor *a, Tensor *b);
int tensor_dims(Tensor *a);
int tensor_dim_get(Tensor *a, int index);
double tensor_row_major_get(Tensor *a, int index);
double tensor_row_major_set(Tensor *a, int index, double value);
double tensor_row_major_mut(Tensor *a, int index, double value, Dyadic_Fn op);
void tensor_print(Tensor *a);

double tensor_scalar_add(double a, double b);
double tensor_scalar_sub(double a, double b);
double tensor_scalar_mul(double a, double b);
double tensor_scalar_div(double a, double b);
double tensor_scalar_inv(double a);
double tensor_scalar_neg(double a);
double tensor_scalar_relu(double a);

void tensor_out_apply_scalar_right(Tensor *dest, Dyadic_Fn op, Tensor *a, double scalar);
void tensor_out_apply_scalar_left(Tensor *dest, Dyadic_Fn op, double scalar, Tensor *a);
void tensor_out_dyadic_apply_element_wise(Tensor *z, Dyadic_Fn op, Tensor *a, Tensor *b);
void tensor_out_monadic_apply_element_wise(Tensor *z, Monadic_Fn op, Tensor *a);

double tensor_dot(Tensor *a, Tensor *b);
void tensor_out_matmul(Tensor *z, Tensor *a, Tensor *b);
void tensor_out_matvec(Tensor *z, Tensor *a, Tensor *b);

void tensor_calc_numel(Tensor *a);
void tensor_calc_stride(Tensor *a);

#endif // TENSOR_H
