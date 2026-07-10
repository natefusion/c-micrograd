#include <math.h>
#include <stdbool.h>
#include <stdio.h>

#include "scalar_engine.h"

int main(void) {
    value_reset(50000);

    VALUE_(x1, value_make(2));
    VALUE_(x2, value_make(0));
    VALUE_(w1, value_make(-3));
    VALUE_(w2, value_make(1));
    VALUE_(b, value_make(6.8813735870195432));
    VALUE_(x1_x_w1, value_mul(x1, w1));

    VALUE_(x2_x_w2, value_mul(x2, w2));

    VALUE_(x1_x_w1_plus_x2_x_w2, value_mul(x1_x_w1, x2_x_w2));
    VALUE_(n, value_add(x1_x_w1_plus_x2_x_w2, b));
    VALUE_(o, value_tanh(n));

    value_backward(o);
    value_draw_computational_graph(o);
}
