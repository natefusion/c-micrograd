#include "tensor.h"

#include <stdio.h>

void tensor_calc_numel(Tensor *a) {
    // not quite right at the edges but good enough
    a->numel = 1;
    for (int i = 0; i < tensor_dims(a); ++i) {
        a->numel *= a->shape[i];
    }
}

void tensor_calc_stride(Tensor *a) {
    int dims = tensor_dims(a);
    a->stride[dims-1] = 1;

    int prev = a->shape[dims-1];
    for (int j = 1; j < dims; ++j) {
        int i = dims - 1 - j;
        a->stride[i] = prev;
        prev *= a->shape[i];
    }
}

void tensor_print(Tensor *a) {
    int dims = tensor_dims(a);
    if (dims == 0) {
        printf("%f\n", tensor_row_major_get(a, 0));
        return;
    }

    if (dims == 1) {
        printf("[");
        for (int i = 0; i < tensor_numel(a); ++i) {
            printf("%f ", tensor_row_major_get(a, i));
        }
        printf("]\n");
        return;
    }


    printf("[\n");
    for (int i = 0; i < tensor_numel(a); ++i) {
        printf("\t%f", tensor_row_major_get(a, i));

        if ((i+1) % a->stride[dims-2] != 0) printf(" ");

        if (i < tensor_numel(a) - 1) {
            for (int j = 1; j < dims; ++j) {
                int dim = a->stride[dims - 1 - j];
                if ((i+1) % dim == 0) {
                    printf("\n");
                }
            }
        }
    }
    printf("\n]\n");
}

double tensor_scalar_add(double a, double b) { return a + b; }
double tensor_scalar_sub(double a, double b) { return a - b; }
double tensor_scalar_mul(double a, double b) { return a * b; }
double tensor_scalar_div(double a, double b) { return a / b; }
double tensor_scalar_inv(double a)           { return 1.0 / a; }
double tensor_scalar_neg(double a)           { return -a; }
double tensor_scalar_relu(double a)          { return a > 0.0 ? a : 0.0; }

int tensor_numel(Tensor *a) {
    return a->numel;
}

bool tensor_shape_eq(Tensor *a, Tensor *b) {
    if (a->dims != b->dims) {
        return false;
    }
    
    for (int i = 0; i < a->dims; ++i) {
        if (a->shape[i] != b->shape[i]) {
            return false;
        }
    }
    
    return true;
}

int tensor_dims(Tensor *a) {
    return a->dims;
}

int tensor_dim_get(Tensor *a, int dim) {
    assert(dim < tensor_dims(a));
    return a->shape[dim];
}

double tensor_row_major_get(Tensor *a, int index) {
    assert(index < tensor_numel(a));
    return a->data[index];
}

double tensor_row_major_set(Tensor *a, int index, double value) {
    assert(index < tensor_numel(a));
    return a->data[index] = value;
}

double tensor_row_major_mut(Tensor *a, int index, double value, Dyadic_Fn op) {
    assert(index < tensor_numel(a));
    return a->data[index] = op(a->data[index], value);
}

void tensor_out_apply_scalar_right(Tensor *z, Dyadic_Fn op, Tensor *a, double scalar) {
    for (int i = 0; i < tensor_numel(a); ++i) {
        tensor_row_major_set(z, i, op(tensor_row_major_get(a, i), scalar));
    }
}

void tensor_out_apply_scalar_left(Tensor *z, Dyadic_Fn op, double scalar, Tensor *a) {
    for (int i = 0; i < tensor_numel(a); ++i) {
        tensor_row_major_set(z, i, op(scalar, tensor_row_major_get(a, i)));
    }
}

void tensor_out_dyadic_apply_element_wise(Tensor *z, Dyadic_Fn op, Tensor *a, Tensor *b) {
    assert(tensor_shape_eq(z, a));
    assert(tensor_shape_eq(a, b));
    
    for (int i = 0; i < tensor_numel(a); ++i) {
        tensor_row_major_set(z, i, op(tensor_row_major_get(a, i), tensor_row_major_get(b, i)));
    }
}

void tensor_out_monadic_apply_element_wise(Tensor *z, Monadic_Fn op, Tensor* a) {
    assert(tensor_shape_eq(z, a));
    
    for (int i = 0; i < tensor_numel(a); ++i) {
        tensor_row_major_set(z, i, op(tensor_row_major_get(a, i)));
    }
}

double tensor_dot(Tensor *a, Tensor *b) {
    assert(tensor_dims(a) == 1);
    assert(tensor_dims(b) == 1);
    assert(tensor_numel(a) == tensor_numel(b));

    double z = 0.0;
    for (int i = 0; i < tensor_numel(a); ++i) {
        z += tensor_row_major_get(a, i) * tensor_row_major_get(b, i);
    }

    return z;
}

void tensor_out_matmul(Tensor *z, Tensor *a, Tensor *b) {
    assert(tensor_dims(a) == 2);
    assert(tensor_dims(b) == 2);
    assert(tensor_dims(z) == 2);
    assert(tensor_dim_get(a, 1) == tensor_dim_get(b, 0));

    int m = tensor_dim_get(a, 0);
    int n = tensor_dim_get(a, 1);
    int p = tensor_dim_get(b, 1);

    assert(tensor_dim_get(z, 0) == m);
    assert(tensor_dim_get(z, 1) == p);

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < p; ++j) {
            for (int k = 0; k < n; ++k) {
                double a_val = tensor_row_major_get(a, i*n+k);
                double b_val = tensor_row_major_get(b, k*p+j);
                tensor_row_major_mut(z, i*p+j, a_val * b_val, tensor_scalar_add);
            }
        }
    }
}

void tensor_out_matvec(Tensor *z, Tensor *a, Tensor *b) {
    assert(tensor_dims(a) == 2);
    assert(tensor_dims(b) == 1);
    assert(tensor_dims(z) == 1);
    assert(tensor_dim_get(a, 1) == tensor_dim_get(b, 0));

    int m = tensor_dim_get(a, 0);
    int n = tensor_dim_get(a, 1);

    assert(tensor_dim_get(z, 0) == m);

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            double a_val = tensor_row_major_get(a, i*n+j);
            double b_val = tensor_row_major_get(b, j);
            tensor_row_major_mut(z, i, a_val * b_val, tensor_scalar_add);
        }

    }
}
