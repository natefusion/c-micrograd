#include <math.h>
#include <stdbool.h>
#include <stdio.h>

#include "scalar_engine.h"

int main(void) {
    Arena arena = Arena_init(50000);

    VALUE_(x1, value_make(&arena, 2));
    VALUE_(x2, value_make(&arena, 0));
    VALUE_(w1, value_make(&arena, -3));
    VALUE_(w2, value_make(&arena, 1));
    VALUE_(b, value_make(&arena, 6.8813735870195432));
    VALUE_(x1_x_w1, value_mul(&arena, x1, w1));

    VALUE_(x2_x_w2, value_mul(&arena, x2, w2));

    VALUE_(x1_x_w1_plus_x2_x_w2, value_mul(&arena, x1_x_w1, x2_x_w2));
    VALUE_(n, value_add(&arena, x1_x_w1_plus_x2_x_w2, b));
    VALUE_(o, value_tanh(&arena, n));

    value_backward(&arena, o);
    value_draw_computational_graph(o);
}
