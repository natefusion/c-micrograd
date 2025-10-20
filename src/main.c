#include <math.h>
#include <stdbool.h>
#include <stdio.h>

#include "scalar_engine.h"

int main(void) {
    Arena arena = Arena_init(50000);

    Value *x1 = value_make(&arena, 2, "x1");
    Value *x2 = value_make(&arena, 0, "x2");
    Value *w1 = value_make(&arena, -3, "w1");
    Value *w2 = value_make(&arena, 1, "w2");
    Value *b = value_make(&arena, 6.8813735870195432, "b");

    Value *x1_x_w1 = value_mul(&arena, x1, w1);
    x1_x_w1->name = "x1_x_w1";
    
    Value *x2_x_w2 = value_mul(&arena, x2, w2);
    x2_x_w2->name = "x2_x_w2";

    Value *x1_x_w1_plus_x2_x_w2 = value_mul(&arena, x1_x_w1, x2_x_w2);
    x1_x_w1_plus_x2_x_w2->name = "x1_x_w1_plus_x2_x_w2";

    Value *n = value_add(&arena, x1_x_w1_plus_x2_x_w2, b);
    n->name = "n";

    Value *o = value_tanh(&arena, n);
    o->name = "o";

    value_backward(&arena, o);
    draw_computational_graph(o);

    /* Value *a = value_make(&arena, 2.0, "a"); */
    /* Value *b = value_make(&arena, 3.0, "b"); */
    /* Value *z = value_mul(&arena, a, b); */
    /* z->name = "z"; */

    /* Value *t = value_tanh(&arena, z); */
    /* t->name = "t"; */
    /* value_backward(&arena, t); */
    /* draw_computational_graph(t, t); */
    /* close_graphviz_graph(); */
    
}
