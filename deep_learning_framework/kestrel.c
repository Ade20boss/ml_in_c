#include <stddef.h>
#include <sys/types.h>
#define KESTREL_CODE
#include "kestrel.h"

Chunk_memory arena = {.offset = 0, .data = NULL, .capacity = 100000000};

typedef struct
{
    matrix layer1_inputs;  // (1 × 2)  — one pair, 2 input numbers
    matrix layer1_weights; // (2 × 2)  — rows=inputs(2), cols=hidden neurons(2); does NOT depend on how many pairs
    matrix layer1_outputs; // (1 × 2)  — one pair's result, 2 hidden neuron outputs
    matrix layer1_biases;  // (2 × 1)  — one bias per hidden neuron (column vector)
    matrix layer2_weights; // (2 × 1)  — rows=hidden inputs(2), cols=output neuron(1)
    matrix layer2_biases;  // (1 × 1)  — one bias for the single output neuron
    matrix layer2_outputs; // (1 × 1)  — one pair's final output (1 number)
} Xor;

float cost(float output, float target)
{
    return;
}


float *forward_pass(Xor * xor, float x1, float x2)
{
    // set input values
    MAT_POS(xor->layer1_inputs, 0, 0) = x1;
    MAT_POS(xor->layer1_inputs, 0, 1) = x2;

    // layer1: (1×2) · (2×2) = (1×2), then add biases (2×1, broadcast over the row), then sigmoid
    matrix_dotproduct(xor->layer1_outputs, xor->layer1_inputs, xor->layer1_weights);
    matrix_sum_bias(xor->layer1_outputs, xor->layer1_biases);
    matrix_activate(xor->layer1_outputs);
    // layer2: (1×2) · (2×1) = (1×1), then add bias (1×1), then sigmoid
    matrix_dotproduct(xor->layer2_outputs, xor->layer1_outputs, xor->layer2_weights);
    matrix_sum_bias(xor->layer2_outputs, xor->layer2_biases);
    matrix_activate(xor->layer2_outputs);
    return xor->layer2_outputs.es;
}

int main(void)
{
    arena_init(&arena);
    srand(time(0));
    Xor xor ;
    xor.layer1_inputs = matrix_allocate(&arena, 1, 2);  // one pair × 2 inputs
    xor.layer1_weights = matrix_allocate(&arena, 2, 2); // 2 inputs × 2 hidden neurons (col = neuron)
    xor.layer1_outputs = matrix_allocate(&arena, 1, 2); // one pair × 2 hidden neuron outputs

    xor.layer1_biases = matrix_allocate(&arena, 1, 2); // row now, 1 bias per hidden neuron

    xor.layer2_weights = matrix_allocate(&arena, 2, 1); // 2 hidden inputs × 1 output neuron (col = neuron)

    xor.layer2_biases = matrix_allocate(&arena, 1, 1); // 1 output neuron × 1 (column vector)

    xor.layer2_outputs = matrix_allocate(&arena, 1, 1); // one pair × 1 output neuron
    matrix_randomize(xor.layer1_weights, 0, 1);
    matrix_randomize(xor.layer1_biases, 0, 1);
    matrix_randomize(xor.layer2_weights, 0, 1);
    matrix_randomize(xor.layer2_biases, 0, 1);
    MATRIX_PRINT(xor.layer1_weights);
    MATRIX_PRINT(xor.layer1_biases);
    MATRIX_PRINT(xor.layer2_weights);
    MATRIX_PRINT(xor.layer2_biases);

    // --- load ONE pair into the 1×2 input ---
    // a 1×2 input only holds a single pair. this is the exact spot where your
    // training loop will later swap in each of the 4 pairs, one per forward pass.
    xor.layer1_inputs.es[0] = 1; // first number of the pair
    xor.layer1_inputs.es[1] = 0; // second number of the pair
    // (network is untrained here, so the output is just whatever the random weights give)

    float *outputs = forward_pass(&xor);

    // one pair in -> one output out (layer2_outputs is 1×1)
    // once you loop over pairs, you'll print one of these per pass instead.
    printf("Output: %f\n", outputs[0]);

    custom_free(&arena);
    return 0;
}
