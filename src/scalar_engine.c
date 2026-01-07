#include <math.h>
#include <stdbool.h>
#include <stdio.h>

#include "scalar_engine.h"

static int Values = 0;

static Value** Topological_sorted_values = NULL;

void value_reset(void) {
    Values = 0;
    Topological_sorted_values = NULL;
}

Value* value_make(Arena *arena, double number, char const* name) {
    Value *z = new(arena, Value, 1);
    *z = (Value){.data = number, .name = name};
    Values++;
    return z;
}

Partial_Derivative value_add_pd(Value* v) {
    return (Partial_Derivative) {1.0, 1.0};
}

Value* value_add(Arena *arena, Value *a, Value *b) {
    Value *z = new(arena, Value, 1);
    *z = (Value) {
        .data = a->data + b->data,
        .children = {a, b},
        .pd_fn = value_add_pd,
        .flags = Flags_arity_2,
        .op = "+",
    };

    Values++;
    return z;
}

Partial_Derivative value_mul_pd(Value* v) {
    return (Partial_Derivative) {v->children.rhs->data, v->children.lhs->data};
}

Value* value_mul(Arena *arena, Value *a, Value *b) {
    Value *z = new(arena, Value, 1);
    *z = (Value) {
        .data = a->data * b->data,
        .children = {a, b},
        .pd_fn = value_mul_pd,
        .flags = Flags_arity_2,
        .op = "*",
    };

    Values++;
    return z;
}

Partial_Derivative value_sub_pd(Value* v) {
    return (Partial_Derivative) {1.0, -1.0};
}

Value* value_sub(Arena *arena, Value *a, Value *b) {
    Value *z = new(arena, Value, 1);
    *z = (Value){
        .data = a->data - b->data,
        .children = {a, b},
        .pd_fn = value_sub_pd,
        .flags = Flags_arity_2,
        .op = "-",
    };
    
    Values++;
    return z;
}

Partial_Derivative value_div_pd(Value* v) {
    return (Partial_Derivative) {1.0/v->children.rhs->data, -v->children.lhs->data / pow(v->children.rhs->data, 2.0)};
}

Value* value_div(Arena *arena, Value *a, Value *b) {
    Value *z = new(arena, Value, 1);
    *z = (Value) {
        .data = a->data / b->data,
        .children = {a, b},
        .pd_fn = value_div_pd,
        .flags = Flags_arity_2,
        .op = "/",
    };

    Values++;
    return z;
}

Partial_Derivative value_tanh_pd(Value* v) {
    return (Partial_Derivative) {1.0 - pow(tanh(v->children.lhs->data), 2.0)};
}

Value *value_tanh(Arena *arena, Value *a) {
    Value *z = new(arena, Value, 1);
    *z = (Value) {
        .data = tanh(a->data),
        .children = {a},
        .pd_fn = value_tanh_pd,
        .flags = Flags_arity_1,
        .op = "tanh",
    };

    Values++;
    return z;
}

Partial_Derivative value_relu_pd(Value* v) {
    return (Partial_Derivative) {v->children.rhs->data > 0.0 ? 1.0 : 0.0};
}

Value *value_relu(Arena *arena, Value *a) {
    Value *z = new(arena, Value, 1);
    *z = (Value) {
        .data = a->data > 0.0 ? a->data : 0.0,
        .children = {a},
        .pd_fn = value_relu_pd,
        .flags = Flags_arity_1,
        .op = "relu",
    };

    Values++;
    return z;
}

Partial_Derivative value_expt_pd(Value *v) {
    Value *base = v->children.lhs;
    Value *power = v->children.rhs;
    return (Partial_Derivative) {power->data * pow(base->data, power->data - 1.0), log(base->data) * pow(base->data, power->data)};
}

Value *value_expt(Arena *arena, Value *base, Value* power) {
    Value *z = new(arena, Value, 1);
    *z = (Value) {
        .data = pow(base->data, power->data),
        .children = {base, power},
        .pd_fn = value_expt_pd,
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
        topological_sort(topo_array, offset_from_end, value->children.lhs);
    } else if ((value->flags & Flags_arity_2) != 0) {
        topological_sort(topo_array, offset_from_end, value->children.lhs);
        topological_sort(topo_array, offset_from_end, value->children.rhs);
    }

    topo_array[Values - 1 - *offset_from_end] = value;
    *offset_from_end += 1;
}

void value_backward(Arena *arena, Value *value) {
    if (Topological_sorted_values == NULL) {
        Topological_sorted_values = new(arena, Value*, Values);
        int offset_from_end = 0;
        topological_sort(Topological_sorted_values, &offset_from_end, value);
    }
    
    value->grad = 1;

    for (int i = 0; i < Values; ++i) {
        Value *v = Topological_sorted_values[i];
        if ((v->flags & Flags_arity_1) != 0) {
            Partial_Derivative pd = v->pd_fn(v);
            v->children.lhs->grad += pd.lhs * v->grad;
        } else if ((v->flags & Flags_arity_2) != 0) {
            Partial_Derivative pd = v->pd_fn(v);
            v->children.lhs->grad += pd.lhs * v->grad;
            v->children.rhs->grad += pd.rhs * v->grad;
        } 
    }
}

void draw_computational_graph_recur(Value *parent, Value *v) {
    // Print node information
    printf("  \"%s\" [label=\"%s\\nvalue: %.4f\\ngrad: %.4f\\nop: %s\"];\n", 
           v->name, v->name, v->data, v->grad, v->op ? v->op : "leaf");
    
    // Print edges from children to parent
    if ((v->flags & Flags_arity_1) != 0) {
        printf("  \"%s\" -> \"%s\";\n", v->children.lhs->name, v->name);
        draw_computational_graph_recur(v, v->children.lhs);
    } else if ((v->flags & Flags_arity_2) != 0) {
        printf("  \"%s\" -> \"%s\";\n", v->children.lhs->name, v->name);
        printf("  \"%s\" -> \"%s\";\n", v->children.rhs->name, v->name);
        draw_computational_graph_recur(v, v->children.lhs);
        draw_computational_graph_recur(v, v->children.rhs);
    }
}

void draw_computational_graph(Value* v) {
    printf("digraph computational_graph {\n");
    printf("  rankdir=TB;\n");
    printf("  node [shape=box, style=filled, fillcolor=lightblue];\n");

    draw_computational_graph_recur(v, v);

    printf("}\n");
}
