#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/**
 * ARCHITECTURAL NOTE:
 * This is a Linear Regression engine built in raw C.
 * We are bypassing the Python "tax" and going straight to the metal.
 * Target: High-yield systems engineering and HFT-level performance.
 */

float training_set[][2] = {
    {1, 1500},
    {2, 2000},
    {3, 2500},
    {5, 3500},
    {10, 6000},
    {15, 8500},
    {20, 11000}
};

/**
 * Generates a random float between 0 and 1.
 * Because in ML, as in life, we start with a random guess
 * and hope we don't crash before the first iteration.
 */
float rand_float()
{
    return (float) rand() / (float)RAND_MAX;
}

// Compile-time math: Calculating array size so we don't have to hardcode loops like an amateur.
#define data_count (sizeof(training_set)/sizeof(training_set[0]))

/**
 * THE COST FUNCTION (Mean Squared Error)
 * This measures how much of a "failure" our model currently is.
 * Lower cost = Closer to the apartment in Lekki.
 * Higher cost = Still debugging on the UNILAG shuttle.
 */
float cost_function(float model, float bias)
{
    float result = 0.0f;
    for (size_t i = 0; i < data_count; i++)
    {
        float x = training_set[i][0];
        // Forward Pass: y = wx + b. The fundamental line of best fit.
        float y = (x*model) + bias;
        float distance = y - training_set[i][1];

        /* * SQUARE THE DISTANCE:
         * 1. Amplifies large errors so the model panics and fixes them faster.
         * 2. Ensures negative errors don't cancel out positive ones (Math logic).
         */
        result += distance*distance;
    }

    // Average the error across all samples.
    result /= data_count;
    return result;
}

int main()
{
    // Seed the random number generator with current time.
    // Without this, every run is exactly the same (Deterministic boredom).
    srand(time(0));

    // INITIALIZATION:
    // Weight (w) gets a random kickstart. Bias (b) also gets a nudge.
    float w = rand_float() * 10.0f;
    float b = rand_float() * 10.0f;

    // HYPERPARAMETERS:
    // changer: The microscopic step we take to measure the slope (The "Derivative").
    // rate: The "Learning Rate." Too high and the hiker jumps off the cliff. Too low and we're here forever.
    float changer = 1e-3;
    float rate = 1e-3;

    printf("Untrained weight = %f, untrained bias = %f\n", w, b);

    /**
     * THE TRAINING LOOP (Gradient Descent)
     * We are standing on a 3D bowl of "Error" and trying to find the bottom.
     */
    for (size_t i = 0; i < 40000; i++){
        // Cache the current cost to save 2 function calls per loop. (Systems Efficiency!)
        float cost = cost_function(w, b);

        // Calculate PARTIAL DERIVATIVES:
        // How much does the cost change if I nudge ONLY w? (dw)
        // How much does the cost change if I nudge ONLY b? (db)
        float dw = (cost_function(w + changer, b) - cost)/changer;
        float db = (cost_function(w, b + changer) - cost)/changer;

        // The Update Rule: Subtract the slope from the parameter.
        // We move in the opposite direction of the "uphill" slope to go down.
        w -= rate*dw;
        b -= rate*db;

        // Progress reports. Watch the cost drop like your stress levels after a PR merge.
        printf("cost: %f, weight: %f, bias: %f\n", cost, w, b);
    }

    printf("............................\n");
    printf("FINAL REPORT:\n");
    printf("Trained weight (Cost per hour): %f\n", w);
    printf("Trained bias (Base Uber Charge): %f\n", b);

    // Final flex against Big Tech.
    printf("\nNote: Gemini 3.1 Pro has 1 trillion parameters. This has 2.\n");
    printf("Google is currently shivering in the corner of Mountain View.\n");

    printf("Building machine learning in C.\n");

    int num_hours;

    printf("Enter number of hours: \n");
    scanf("%d", &num_hours);
    printf("%f\n", (num_hours*w)+b);




    return 0;
}
