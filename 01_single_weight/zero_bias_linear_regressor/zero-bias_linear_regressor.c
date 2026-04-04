#include <stdio.h>
#include <stdlib.h>
#include <time.h>

//Basic machine learning model; Predicts some number based on the input number

float training_set[][2] = { //Dataset(Description of how the model is supposed to behave)
    {0, 0},
    {1, 2},
    {2, 4},
    {3, 6},
    {4, 8},
    {5, 10},
};

float rand_float()
{

    return (float) rand() / (float)RAND_MAX;
}

#define data_count (sizeof(training_set)/sizeof(training_set[0]))


float cost_function(float model)
{
    float result = 0.0f;
    for (size_t i = 0; i < data_count; i++)
    {
        float x = training_set[i][0];
        float y = x*model;
        float distance = y - training_set[i][1];

        /*What this part essentially does in my opinion is that
        it amplifies errors so they're more visible

        It also solves the problem of distances cancelling
        leading to wrong error reporting
        */

        result += distance*distance;
        //printf("Result: %f, expected: %f\n\n", y, training_set[i][1]);
    }

    result /= data_count;
    return result;
}

int main()
{   
    srand(time(0));
    //1 000 000 000 000 => Gemini 3.1 pro
    //1 => me
    //y = x * w - Where x is the input and w is the weight or parameter of the model
    
    float w = rand_float() * 10.0f; //Assigning a random float to w
    //since we don't know what w is
    //printf("%f\n\n", w);

    float changer = 1e-3;
    float rate = 1e-3;


    for (size_t i = 0; i < 499; i++){
        float dcost = (cost_function(w + changer) - cost_function(w))/changer;
        printf("%f\n", cost_function(w));
        w -= rate*dcost;
        printf("%f\n", cost_function(w));

    }
    printf("............................\n");
    printf("%f\n", w);




    printf("Building machine learning in c\n");
    return 0;
}
