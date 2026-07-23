/*
 ------------------------------------------------------------------
 ** Top contributors:
 **   Shiqi Wang and Suman Jana
 ** This file is part of the ReluVal project.
 ** Copyright (c) 2018-2019 by the authors listed in the file LICENSE
 ** and their institutional affiliations.
 ** All rights reserved.
 -----------------------------------------------------------------
 */

#include "nnet.h"

//read from command lines, inside main()
int SENS_FEATURE_IDX;

/*
 * Load_network is a function modified from Reluplex
 * It takes in a nnet filename with path and load the 
 * network from the file
 * Outputs the NNet instance of loaded network.
 */
struct NNet *load_network(const char* filename, int sens_feature_idx)
{

    FILE *fstream = fopen(filename,"r");

    if (fstream == NULL) {
        printf("Wrong network!\n");
        exit(1);
    }

    int bufferSize = 50240;
    char *buffer = (char*)malloc(sizeof(char)*bufferSize);
    char *record, *line;
    int i=0, layer=0, row=0, j=0, param=0;

    struct NNet *nnet = (struct NNet*)malloc(sizeof(struct NNet));

    nnet->sens_feature_idx = sens_feature_idx;

    line=fgets(buffer,bufferSize,fstream);

    //skip comments in the beginning
    while (strstr(line, "//") != NULL) {
        line = fgets(buffer,bufferSize,fstream); 
    }

    //first four inputs are number of layers, inputs, outputs, and max layer size
    record = strtok(line,",\n");
    nnet->numLayers = atoi(record);
    nnet->inputSize = atoi(strtok(NULL,",\n"));
    nnet->outputSize = atoi(strtok(NULL,",\n"));
    nnet->maxLayerSize = atoi(strtok(NULL,",\n"));

    //read array sizes
    nnet->layerSizes = (int*)malloc(sizeof(int)*(nnet->numLayers+1));
    line = fgets(buffer,bufferSize,fstream);
    record = strtok(line,",\n");
    for (i = 0;i<((nnet->numLayers)+1);i++) {
        nnet->layerSizes[i] = atoi(record);
        record = strtok(NULL,",\n");
    }

    //deprecated property
    line = fgets(buffer,bufferSize,fstream);
    record = strtok(line,",\n");
    nnet->symmetric = atoi(record);

    //min, max, means, and ranges
    nnet->mins = (float*)malloc(sizeof(float)*nnet->inputSize);
    line = fgets(buffer,bufferSize,fstream);
    record = strtok(line,",\n");
    for (i = 0;i<(nnet->inputSize);i++) {
        nnet->mins[i] = (float)atof(record);
        record = strtok(NULL,",\n");
    }

    nnet->maxes = (float*)malloc(sizeof(float)*nnet->inputSize);
    line = fgets(buffer,bufferSize,fstream);
    record = strtok(line,",\n");
    for (i = 0;i<(nnet->inputSize);i++) {
        nnet->maxes[i] = (float)atof(record);
        record = strtok(NULL,",\n");
    }

    nnet->means = (float*)malloc(sizeof(float)*(nnet->inputSize+1));
    line = fgets(buffer,bufferSize,fstream);
    record = strtok(line,",\n");
    for (i = 0;i<((nnet->inputSize)+1);i++) {
        nnet->means[i] = (float)atof(record);
        record = strtok(NULL,",\n");
    }

    nnet->ranges = (float*)malloc(sizeof(float)*(nnet->inputSize+1));
    line = fgets(buffer,bufferSize,fstream);
    record = strtok(line,",\n");
    for (i = 0;i<((nnet->inputSize)+1);i++) {
        nnet->ranges[i] = (float)atof(record);
        record = strtok(NULL,",\n");
    }

    //alloc memory for reading weights and biases
    nnet->matrix = (float****)malloc(sizeof(float *)*nnet->numLayers);
    for (layer = 0;layer<(nnet->numLayers);layer++) {
        nnet->matrix[layer] =\
                (float***)malloc(sizeof(float *)*2);
        nnet->matrix[layer][0] =\
                (float**)malloc(sizeof(float *)*nnet->layerSizes[layer+1]);
        nnet->matrix[layer][1] =\
                (float**)malloc(sizeof(float *)*nnet->layerSizes[layer+1]);

        for (row = 0;row<nnet->layerSizes[layer+1];row++) {
            nnet->matrix[layer][0][row] =\
                    (float*)malloc(sizeof(float)*nnet->layerSizes[layer]);
            nnet->matrix[layer][1][row] = (float*)malloc(sizeof(float));
        }

    }
    
    layer = 0;
    param = 0;
    i=0;
    j=0;

    char *tmpptr=NULL;

    float w = 0.0;

    //read weights and biases
    while ((line = fgets(buffer,bufferSize,fstream)) != NULL) {

        if (i >= nnet->layerSizes[layer+1]) {

            if (param==0) {
                param = 1;
            }
            else {
                param = 0;
                layer++;
            }

            i=0;
            j=0;
        }

        record = strtok_r(line,",\n", &tmpptr);

        while (record != NULL) {   
            w = (float)atof(record);
            nnet->matrix[layer][param][i][j] = w;
            j++;
            record = strtok_r(NULL, ",\n", &tmpptr);
        }

        tmpptr=NULL;
        j=0;
        i++;
    }

    //copy weights and biases into Matrix structs
    struct Matrix *weights=malloc(nnet->numLayers*sizeof(struct Matrix));
    struct Matrix *bias = malloc(nnet->numLayers*sizeof(struct Matrix));
    for (int layer=0;layer<nnet->numLayers;layer++) {
        weights[layer].row = nnet->layerSizes[layer];
        weights[layer].col = nnet->layerSizes[layer+1];
        weights[layer].data = (float*)malloc(sizeof(float)\
                    * weights[layer].row * weights[layer].col);

        int n=0;

        //cleaned up based on ReluDiff
        for (int i=0;i<weights[layer].col;i++) {

            for (int j=0;j<weights[layer].row;j++) {
                weights[layer].data[n] = nnet->matrix[layer][0][i][j];
                n++;
            }

        }

        bias[layer].col = nnet->layerSizes[layer+1];
        bias[layer].row = (float)1;
        bias[layer].data = (float*)malloc(sizeof(float)*bias[layer].col);

        for (int i=0;i<bias[layer].col;i++) {
            bias[layer].data[i] = nnet->matrix[layer][1][i][0];
        }

    }

    nnet->weights = weights;
    nnet->bias = bias;

    free(buffer);
    fclose(fstream);

    return nnet;

}

/*
 * destroy_network is a function modified from Reluplex
 * It release all the memory mallocated to the network instance
 * It takes in the instance of nnet
 */
void destroy_network(struct NNet *nnet)
{

    int i=0, row=0;
    if (nnet != NULL) {

        for (i=0;i<(nnet->numLayers);i++) {

            for (row=0;row<nnet->layerSizes[i+1];row++) {
                free(nnet->matrix[i][0][row]);
                free(nnet->matrix[i][1][row]);
            }

            free(nnet->matrix[i][0]);
            free(nnet->matrix[i][1]);
            free(nnet->weights[i].data);
            free(nnet->bias[i].data);
            free(nnet->matrix[i]);
            free(nnet->posWeights[i].data);
            free(nnet->negWeights[i].data);
        }

        free(nnet->weights);
        free(nnet->bias);
        free(nnet->layerSizes);
        free(nnet->mins);
        free(nnet->maxes);
        free(nnet->means);
        free(nnet->ranges);
        free(nnet->matrix);
        free(nnet->posWeights);
        free(nnet->negWeights);
        free(nnet);
    }

}


// from ReluDiff
void load_positive_and_negative_weights(struct NNet *nnet) {
    struct Matrix *posWeights = malloc(nnet->numLayers*sizeof(struct Matrix));
    struct Matrix *negWeights = malloc(nnet->numLayers*sizeof(struct Matrix));
    struct Matrix weights;
    for (int layer=0;layer<nnet->numLayers;layer++) {
        weights = nnet->weights[layer];

        posWeights[layer].row = weights.row;
        posWeights[layer].col = weights.col;
        negWeights[layer].row = weights.row;
        negWeights[layer].col = weights.col;
        posWeights[layer].data = (float *) malloc(sizeof(float) * weights.row * weights.col);
        negWeights[layer].data = (float *) malloc(sizeof(float) * weights.row * weights.col);
        memset(posWeights[layer].data, 0, sizeof(float) * weights.row * weights.col);
        memset(negWeights[layer].data, 0, sizeof(float) * weights.row * weights.col);

        for(int i=0; i < weights.row * weights.col; i++) {
            if (weights.data[i] >= 0) {
                posWeights[layer].data[i] = weights.data[i];
            } else {
                negWeights[layer].data[i] = weights.data[i];
            }
        }
    }
    nnet->posWeights = posWeights;
    nnet->negWeights = negWeights;
}

/*
 * Following functions denomalize and normalize the concrete inputs
 * and input intervals.
 * They take in concrete inputs or input intervals.
 * Output normalized or denormalized concrete inputs or input intervals.
 */
void denormalize_input(struct NNet *nnet, struct Matrix *input)
{
    for (int i=0; i<nnet->inputSize;i++) {
        input->data[i] = input->data[i]*(nnet->ranges[i]) + nnet->means[i];
    }
}

void denormalize_input_interval(struct NNet *nnet, struct Interval *input)
{
    denormalize_input(nnet, &input->upper_matrix);
    denormalize_input(nnet, &input->lower_matrix);
}


void normalize_input(struct NNet *nnet, struct Matrix *input)
{
    for (int i=0;i<nnet->inputSize;i++) {
        if (input->data[i] > nnet->maxes[i]) {
            input->data[i] = (nnet->maxes[i]-nnet->means[i])/(nnet->ranges[i]);
        }
        else if (input->data[i] < nnet->mins[i]) {
            input->data[i] = (nnet->mins[i]-nnet->means[i])/(nnet->ranges[i]);
        }
        else {
            input->data[i] = (input->data[i]-nnet->means[i])/(nnet->ranges[i]);
        }
    }
}


void normalize_input_interval(struct NNet *nnet, struct Interval *input)
{
    normalize_input(nnet, &input->upper_matrix);
    normalize_input(nnet, &input->lower_matrix);
}