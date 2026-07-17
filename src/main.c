#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "tensor.h"

int main(void) {
    Tensor t;
    t.dims = 2;
    t.shape = malloc(t.dims * sizeof(int));
    t.stride = malloc(t.dims * sizeof(int));
    t.shape[0] = 2;
    t.shape[1] = 2;
    tensor_calc_numel(&t);
    tensor_calc_stride(&t);
    t.data = malloc(t.numel * sizeof(double));
    tensor_row_major_set(&t, 0, 1);
    tensor_row_major_set(&t, 1, 2);
    tensor_row_major_set(&t, 2, 3);
    tensor_row_major_set(&t, 3, 4);

    Tensor t1;
    t1.dims = 2;
    t1.shape = malloc(t1.dims * sizeof(int));
    t1.stride = malloc(t1.dims * sizeof(int));
    t1.shape[0] = 2;
    t1.shape[1] = 2;
    tensor_calc_numel(&t1);
    tensor_calc_stride(&t1);
    t1.data = malloc(t1.numel * sizeof(double));
    tensor_row_major_set(&t1, 0, 5);
    tensor_row_major_set(&t1, 1, 6);
    tensor_row_major_set(&t1, 2, 7);
    tensor_row_major_set(&t1, 3, 8);

    Tensor t2;
    t2.dims = 2;
    t2.shape = malloc(t2.dims * sizeof(int));
    t2.stride = malloc(t2.dims * sizeof(int));
    t2.shape[0] = 2;
    t2.shape[1] = 2;
    tensor_calc_numel(&t2);
    tensor_calc_stride(&t2);
    t2.data = calloc(t2.numel, sizeof(double));

    tensor_out_matmul(&t2, &t, &t1);

    tensor_print(&t);
    tensor_print(&t1);
    tensor_print(&t2);
}
