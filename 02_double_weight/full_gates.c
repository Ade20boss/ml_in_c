#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define W1 0
#define W2 1
#define B1 2
#define W3 3
#define W4 4
#define B2 5
#define W5 6
#define W6 7
#define B3 8

//Function to classify output of the linear equation. Forces
//the output of the linear equation between 0.0 and 1.0
float sigmoid(float x)
{
    return 1.f/(1.f + expf(-x));
}


//Function to generate a random float for the weights and the bias
float random_float()
{
    //Dividing the generated number by RAND_MAX forces the production of a
    //floating point number between 0 and 1.
    return (float) rand() / (float)RAND_MAX;
    //RAND_MAX - Essentially the largest number that rand can produce though
}



//COST FUNCTION - calculates how wrong the model is
float cost(float *weights, float training_set[][3], size_t set_count)
{
    float error_result = 0.0f;

    //Calculate how wrong the model is for each input, adds the errors together
    for (size_t i = 0; i < set_count; i++)
    {
        float x1 = training_set[i][0];
        float x2 = training_set[i][1];
        float h1 = sigmoid((x1*weights[W1]) + (x2*weights[W2]) + weights[B1]);
        float h2 = sigmoid((x1*weights[W3]) + (x2*weights[W4]) + weights[B2]);
        float y = sigmoid((h1* weights[W5]) + (h2*weights[W6]) + weights[B3]);
        float d = y - training_set[i][2];

        error_result += d*d;

    }

    //This then averages it by dividing the accumulated error by the number of inputs
    error_result /= set_count;
    return error_result;
}


int classifier(float x1, float x2, float *weights)
{

    float h1 = sigmoid((x1*weights[W1]) + (x2*weights[W2]) + weights[B1]);
    float h2 = sigmoid((x1*weights[W3]) + (x2*weights[W4]) + weights[B2]);
    float classer = sigmoid((h1* weights[W5]) + (h2*weights[W6]) + weights[B3]);

    if (classer >= 0.5)
    {
        return 1;
    }

    else 
    {
        return 0;
    }
}




 

//Main function
int main()
{

    float training_set[][3] = { //Dataset(Description of how the model is supposed to behave)
        {0, 0, 0},
        {1, 0, 1},
        {0, 1, 1},
        {1, 1, 0},
    };

    size_t length = sizeof(training_set)/sizeof(training_set[0]);


    //seed rand to produce new numbers everytime the program runs
    srand(time(0));

    float epsilon = 1e-1;
    float rate = 1e-1;
    float c;
    float weights[9];

    for (size_t i = 0; i < 9; i++)
    {
        weights[i] = random_float();
    }

    #define WEIGHT_COUNT sizeof(weights)/sizeof(weights[0])


    //Gradient descent: Continuously nudge the weights and bias to get derivatives and then continuously
    //toggle the weights and bias until model conforms with data
    for (size_t i = 0; i < 320000; i++)
    {
        c = cost(weights, training_set, length);

        for (size_t j = 0; j < WEIGHT_COUNT; j++)
        {
            float original = weights[j];
            weights[j] += epsilon;
            float d =  (cost(weights, training_set, length) - c) / epsilon;
            weights[j] = original;
            weights[j] -= rate *d;
        }



        if (i % 10000 == 0)
        {
            printf("Training Weight 1: %f, Training Weight 2: %f, Training Bias 1: %f, Training Weight 3: %f, Training Weight 4: %f, Training Bias 2: %f,  Training Weight 5: %f, Training Weight 6: %f, Training Bias 3: %f, Error_cost: %f\n", weights[W1], weights[W2], weights[B1], weights[W3], weights[W4], weights[B2], weights[W5], weights[W6], weights[B3], c);
        }
        
    }

       printf("Trained Weight 1: %f, Trained Weight 2: %f, Trained Bias 1: %f, Trained Weight 3: %f, Trained Weight 4: %f, Trained Bias 2: %f,  Trained Weight 5: %f, Trained Weight 6: %f, Trained Bias 3: %f, Error_cost: %f\n", weights[W1], weights[W2], weights[B1], weights[W3], weights[W4], weights[B2], weights[W5], weights[W6], weights[B3], c);

    printf("\n");

    printf("OUTPUT NEURON\n");
    printf("\n");
    printf("__________________________\n");
    for (size_t i = 0; i < length; i++)
    {
        float x1 = training_set[i][0];
        float x2 = training_set[i][1];
        printf("%f | %f = %d\n", x1, x2, classifier(x1, x2, weights));
    }

    printf("\n");


    printf("HIDDEN NEURON 1\n");
    printf("\n");
    printf("__________________________\n");
    for (size_t i = 0; i < length; i++)
    {
        float x1 = training_set[i][0];
        float x2 = training_set[i][1];
        printf("%f | %f = %f\n", x1, x2, sigmoid((x1*weights[W1]) + (x2*weights[W2]) + weights[B1]));
    }



    printf("\n");

    printf("HIDDEN NEURON 2\n");
    printf("\n");
    printf("__________________________\n");
    for (size_t i = 0; i < length; i++)
    {
        float x1 = training_set[i][0];
        float x2 = training_set[i][1];
        printf("%f | %f = %f\n", x1, x2, sigmoid((x1*weights[W3]) + (x2*weights[W4]) + weights[B2]));
    }
    



    return 0;

}



