#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>

#define Array_define(T, name)                                                  \
  typedef struct {                                                             \
    T *data;                                                                   \
    int len;                                                                   \
    } name

typedef struct {
    char *beg;
    char *end;
} Arena;

Arena Arena_init(ptrdiff_t cap) {
    Arena a = {0};
    a.beg = malloc(cap);
    a.end = a.beg ? a.beg+cap : 0;
    return a;
}

void *alloc(Arena *a, ptrdiff_t size, ptrdiff_t align, ptrdiff_t count,
            int flags) {
    ptrdiff_t padding = -(uintptr_t)a->beg & (align - 1);
    ptrdiff_t available = a->end - a->beg - padding;
    if (available < 0 || count > available/size) {
        abort();  // one possible out-of-memory policy
    }
    void *p = a->beg + padding;
    a->beg += padding + count*size;
    return memset(p, 0, count*size);
}

#define new(...)            newx(__VA_ARGS__,new4,new3,new2)(__VA_ARGS__)
#define newx(a,b,c,d,e,...) e
#define new2(a, t)          (t *)alloc(a, sizeof(t), _Alignof(t), 1, 0)
#define new3(a, t, n)       (t *)alloc(a, sizeof(t), _Alignof(t), n, 0)
#define new4(a, t, n, f) (t *)alloc(a, sizeof(t), _Alignof(t), n, f)

typedef struct Value Value;

Array_define(Value, Vec_Value);

enum Flags : uint8_t {
    Flags_visited = 0b00000001,
    Flags_arity_1 = 0b00000010,
    Flags_arity_2 = 0b00000100,
};

struct Value {
    double data;
    double local_grads[2];
    double grad;
    Value *children[2];
    enum Flags flags;
    // debugging
    char const *op;
    char const *name;
};

static int Values = 0;

Value* value_make(Arena *arena, double number, char const* name) {
    Value *z = new(arena, Value, 1);
    *z = (Value){.data = number, .name = name};
    Values++;
    return z;
}

Value* value_add(Arena *arena, Value *a, Value *b) {
    Value *z = new(arena, Value, 1);
    *z = (Value) {
        .data = a->data + b->data,
        .children = {a, b},
        .local_grads = {1.0, 1.0},
        .flags = Flags_arity_2,
        .op = "+",
    };

    Values++;
    return z;
}

Value* value_mul(Arena *arena, Value *a, Value *b) {
    Value *z = new(arena, Value, 1);
    *z = (Value) {
        .data = a->data * b->data,
        .children = {a, b},
        .local_grads = {b->data, a->data},
        .flags = Flags_arity_2,
        .op = "*",
    };

    Values++;
    return z;
}

Value* value_sub(Arena *arena, Value *a, Value *b) {
    Value *z = new(arena, Value, 1);
    *z = (Value){
        .data = a->data - b->data,
        .children = {a, b},
        .local_grads = {1.0, -1.0},
        .flags = Flags_arity_2,
        .op = "-",
    };
    
    Values++;
    return z;
}

Value* value_div(Arena *arena, Value *a, Value *b) {
    Value *z = new(arena, Value, 1);
    *z = (Value) {
        .data = a->data / b->data,
        .children = {a, b},
        .local_grads = {1.0/b->data, (-a->data / (pow(b->data, 2.0)))},
        .flags = Flags_arity_2,
        .op = "/",
    };

    Values++;
    return z;
}

Value *value_tanh(Arena *arena, Value *a) {
    Value *z = new(arena, Value, 1);
    *z = (Value) {
        .data = tanh(a->data),
        .children = {a},
        .local_grads = {1.0 - pow(tanh(a->data), 2.0)},
        .flags = Flags_arity_1,
        .op = "tanh",
    };

    Values++;
    return z;
}

Value *value_relu(Arena *arena, Value *a) {
    Value *z = new(arena, Value, 1);
    *z = (Value) {
        .data = a->data > 0.0 ? a->data : 0.0,
        .children = {a},
        .local_grads = {a->data > 0.0 ? 1.0 : 0.0},
        .flags = Flags_arity_1,
        .op = "relu",
    };

    Values++;
    return z;
}

Value *value_expt(Arena *arena, Value *base, Value* power) {
    Value *z = new(arena, Value, 1);
    *z = (Value) {
        .data = pow(base->data, power->data),
        .children = {base, power},
        .local_grads = {power->data * pow(base->data, power->data - 1.0), log(base->data) * pow(base->data, power->data)},
        .flags = Flags_arity_2,
        .op = "expt",
    };

    Values++;
    return z;
}

void topological_sort(Value **topo_array, int* offset_from_end, Value *value) {
    if ((value->flags & Flags_visited) != 0) {
        return;
    }

    value->flags |= Flags_visited;
    
    if ((value->flags & Flags_arity_1) != 0) {
        topological_sort(topo_array, offset_from_end, value->children[0]);
    } else if ((value->flags & Flags_arity_2) != 0) {
        topological_sort(topo_array, offset_from_end, value->children[0]);
        topological_sort(topo_array, offset_from_end, value->children[1]);
    }

    topo_array[Values - 1 - *offset_from_end] = value;
    *offset_from_end += 1;
}

void value_backward(Arena *arena, Value *value) {
    Value **topo_array = new(arena, Value *, Values);
    int offset_from_end = 0;
    topological_sort(topo_array, &offset_from_end, value);
    value->grad = 1;

    for (int i = 0; i < Values; ++i) {
        Value *v = topo_array[i];
        if ((v->flags & Flags_arity_1) != 0) {
            v->children[0]->grad += v->local_grads[0] * v->grad;
        } else if ((v->flags & Flags_arity_2) != 0) {
            v->children[0]->grad += v->local_grads[0] * v->grad;
            v->children[1]->grad += v->local_grads[1] * v->grad;
        } 
    }
}

void draw_computational_graph(Value *parent, Value *v) {
    static bool graph_started = false;
    
    if (!graph_started) {
        printf("digraph computational_graph {\n");
        printf("  rankdir=TB;\n");
        printf("  node [shape=box, style=filled, fillcolor=lightblue];\n");
        graph_started = true;
    }
    
    // Print node information
    printf("  \"%s\" [label=\"%s\\nvalue: %.4f\\ngrad: %.4f\\nop: %s\"];\n", 
           v->name, v->name, v->data, v->grad, v->op ? v->op : "leaf");
    
    // Print edges from children to parent
    if ((v->flags & Flags_arity_1) != 0) {
        printf("  \"%s\" -> \"%s\";\n", v->children[0]->name, v->name);
        draw_computational_graph(v, v->children[0]);
    } else if ((v->flags & Flags_arity_2) != 0) {
        printf("  \"%s\" -> \"%s\";\n", v->children[0]->name, v->name);
        printf("  \"%s\" -> \"%s\";\n", v->children[1]->name, v->name);
        draw_computational_graph(v, v->children[0]);
        draw_computational_graph(v, v->children[1]);
    }
}

void close_graphviz_graph(void) {
    printf("}\n");
}

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
    draw_computational_graph(o, o);
    close_graphviz_graph();

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
