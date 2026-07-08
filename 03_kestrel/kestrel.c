#include <stddef.h>
#define KESTREL_CODE
#include "kestrel.h"

Chunk_memory arena = {.offset = 0, .data = NULL, .capacity = 100000000};

typedef struct
{
    matrix layer1_inputs;  // (1 × 2)  — one pair, 2 input numbers
    matrix layer1_weights; // (2 × 2)  — rows=inputs(2), cols=hidden neurons(2); does NOT depend on how many pairs
    matrix layer1_biases;  // (2 × 1)  — one bias per hidden neuron (column vector)
    matrix layer1_outputs; // (1 × 2)  — one pair's result, 2 hidden neuron outputs

    matrix layer2_weights; // (2 × 1)  — rows=hidden inputs(2), cols=output neuron(1)
    matrix layer2_biases;  // (1 × 1)  — one bias for the single output neuron
    matrix layer2_outputs; // (1 × 1)  — one pair's final output (1 number)
} Xor;

float *forward_pass(Xor * xor)
{
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

Xor xor_allocate(void)
{
    Xor xor ;
    xor.layer1_inputs = matrix_allocate(&arena, 1, 2);  // one pair × 2 inputs
    xor.layer1_weights = matrix_allocate(&arena, 2, 2); // 2 inputs × 2 hidden neurons (col = neuron)
    xor.layer1_outputs = matrix_allocate(&arena, 1, 2); // one pair × 2 hidden neuron outputs

    xor.layer1_biases = matrix_allocate(&arena, 1, 2); // row now, 1 bias per hidden neuron

    xor.layer2_weights = matrix_allocate(&arena, 2, 1); // 2 hidden inputs × 1 output neuron (col = neuron)

    xor.layer2_biases = matrix_allocate(&arena, 1, 1); // 1 output neuron × 1 (column vector)

    xor.layer2_outputs = matrix_allocate(&arena, 1, 1); // one pair × 1 output neuron

    return xor;
}

float cost(Xor model, matrix inputs, matrix outputs)
{
    KESTREL_ASSERT(outputs.rows == inputs.rows);
    KESTREL_ASSERT(model.layer2_outputs.rows == 1);
    KESTREL_ASSERT(model.layer2_outputs.cols == outputs.cols);

    float total_cost = 0.0f;
    for (size_t i = 0; i < inputs.rows; i++)
    {
        matrix input = row_matricize(inputs, i);
        matrix expected_output = row_matricize(outputs, i);
        matrix_copy(model.layer1_inputs, input);

        forward_pass(&model);

        for (size_t j = 0; j < outputs.cols; j++)
        {
            float difference = MAT_POS(model.layer2_outputs, 0, j) - MAT_POS(expected_output, 0, j);
            total_cost += difference * difference;
        }
    }
    total_cost /= inputs.rows;
    return total_cost;
}

void finite_difference(Xor xor, Xor gradient, float epsilon, matrix inputs, matrix outputs)
{
    float saved;

    float c = cost(xor, inputs, outputs);
    for (size_t i = 0; i < xor.layer1_weights.rows; i++)
    {
        for (size_t j = 0; j < xor.layer1_weights.cols; j++)
        {
            saved = MAT_POS(xor.layer1_weights, i, j);

            MAT_POS(xor.layer1_weights, i, j) += epsilon;

            MAT_POS(gradient.layer1_weights, i, j) = (cost(xor, inputs, outputs) - c) / epsilon;

            MAT_POS(xor.layer1_weights, i, j) = saved;
        }
    }

    for (size_t i = 0; i < xor.layer1_biases.rows; i++)
    {
        for (size_t j = 0; j < xor.layer1_biases.cols; j++)
        {
            saved = MAT_POS(xor.layer1_biases, i, j);

            MAT_POS(xor.layer1_biases, i, j) += epsilon;

            MAT_POS(gradient.layer1_biases, i, j) = (cost(xor, inputs, outputs) - c) / epsilon;

            MAT_POS(xor.layer1_biases, i, j) = saved;
        }
    }


    for (size_t i = 0; i < xor.layer2_weights.rows; i++)
    {
        for (size_t j = 0; j < xor.layer2_weights.cols; j++)
        {
            saved = MAT_POS(xor.layer2_weights, i, j);

            MAT_POS(xor.layer2_weights, i, j) += epsilon;

            MAT_POS(gradient.layer2_weights, i, j) = (cost(xor, inputs, outputs) - c) / epsilon;

            MAT_POS(xor.layer2_weights, i, j) = saved;
        }
    }

    for (size_t i = 0; i < xor.layer2_biases.rows; i++)
    {
        for (size_t j = 0; j < xor.layer2_biases.cols; j++)
        {
            saved = MAT_POS(xor.layer2_biases, i, j);

            MAT_POS(xor.layer2_biases, i, j) += epsilon;

            MAT_POS(gradient.layer2_biases, i, j) = (cost(xor, inputs, outputs) - c) / epsilon;

            MAT_POS(xor.layer2_biases, i, j) = saved;
        }
    }
}

void learn(Xor m, Xor g, float rate)
{
    for (size_t i = 0; i < m.layer1_weights.rows; i++)
    {
        for (size_t j = 0; j < m.layer1_weights.cols; j++)
        {
            MAT_POS(m.layer1_weights, i, j) -= rate * MAT_POS(g.layer1_weights, i, j);
        }
    }

    for (size_t i = 0; i < m.layer1_biases.rows; i++)
    {
        for (size_t j = 0; j < m.layer1_biases.cols; j++)
        {
            MAT_POS(m.layer1_biases, i, j) -= rate * MAT_POS(g.layer1_biases, i, j);
        }
    }

    for (size_t i = 0; i < m.layer2_weights.rows; i++)
    {
        for (size_t j = 0; j < m.layer2_weights.cols; j++)
        {
            MAT_POS(m.layer2_weights, i, j) -= rate * MAT_POS(g.layer2_weights, i, j);
        }
    }

    for (size_t i = 0; i < m.layer2_biases.rows; i++)
    {
        for (size_t j = 0; j < m.layer2_biases.cols; j++)
        {
            MAT_POS(m.layer2_biases, i, j) -= rate * MAT_POS(g.layer2_biases, i, j);
        }
    }
}

int classifier_xor(Xor * xor)
{
    for (size_t i = 0; i < xor->layer2_outputs.rows; i++)
    {
        for (size_t j = 0; j < xor->layer2_outputs.cols; j++)
        {
            if (MAT_POS(xor->layer2_outputs, i, j) >= 0.5)
            {
                return 1;
            }

            else
            {
                return 0;
            }
        }
    }

    return 0;
}


float data[] = {
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    0,
    1,
    1,
    1,
    0,
};


int main(void)
{


    arena_init(&arena);
    srand(time(0));

    size_t stride = 3;
    size_t n = sizeof(data) / sizeof(data[0]) / stride;

    matrix data_inputs = {.rows = n, .cols = 2, .stride = stride, .es = data};

    matrix data_outputs = {.rows = n, .cols = 1, .stride = stride, .es = &data[2]};

    MATRIX_PRINT(data_inputs);
    MATRIX_PRINT(data_outputs);

    size_t arch[] = {2, 2, 1};
    Neural_network test = nn_allocate(arch, ARRAY_LEN(arch), &arena);
    Nn_randomize(test, 0, 1);
    NN_PRINT(test);

    printf("%f", nn_cost(test, data_inputs, data_outputs));


    // matrix_copy(NN_INPUT(test), row_matricize(data_inputs, 1));
    // Nn_forward_pass(test);
    // MATRIX_PRINT(NN_OUTPUT(test));

    custom_free(&arena);
    return 0;
    custom_free(&arena);


    // Xor xor = xor_allocate();
    // Xor gradient = xor_allocate();

    // matrix_randomize(xor.layer1_weights, 0, 1);
    // matrix_randomize(xor.layer1_biases, 0, 1);
    // matrix_randomize(xor.layer2_weights, 0, 1);
    // matrix_randomize(xor.layer2_biases, 0, 1);

    // float eps = 1.0e-1;
    // float rate = 1.0e-1;

    // printf("Cost = %f\n", cost(xor, data_inputs, data_outputs));
    // for (size_t i = 0; i < 100000; i++)
    // {
    //     finite_difference(xor, gradient, eps, data_inputs, data_outputs);
    //     learn(xor, gradient, rate);
    //     printf("Cost = %f\n", cost(xor, data_inputs, data_outputs));
    // }

    // printf("-----------------------------------------------\n");

    // for (size_t i = 0; i < 4; i++)
    // {
    //     matrix input = row_matricize(data_inputs, i);
    //     matrix expected_output = row_matricize(data_outputs, i);
    //     matrix_copy(xor.layer1_inputs, input);

    //     forward_pass(&xor);

    //     int y = classifier_xor(&xor);


    //     MATRIX_PRINT(input);
    //     printf("   ");
    //     printf("%d", y);
    //     printf("\n\n");
    // }


    // return 0;
}
