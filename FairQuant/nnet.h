/*
 -----------------------------------------------------------------
 ** Top contributors:
 **   Shiqi Wang and Suman Jana
 ** This file is part of the ReluVal project.
 ** Copyright (c) 2018-2019 by the authors listed in the file LICENSE
 ** and their institutional affiliations.
 ** All rights reserved.
 -----------------------------------------------------------------
 */

#include "matrix.h"
#include "interval.h"
#include <string.h>

#ifndef NNET_H
#define NNET_H

/* values read from the command line */
extern int SENS_FEATURE_IDX;

/*
 * Network instance modified from Reluplex
 * malloc all the memory needed for network
 */
struct NNet 
{
    int symmetric;     
    int numLayers;     
    int inputSize;     
    int outputSize;    
    int maxLayerSize;  
    int *layerSizes;   

    float *mins;      
    float *maxes;     
    float *means; 
    float *ranges;

    /*
     * first dimension: the layer (k)
     * second dimension: is bias (0 = no, 1 = yes)
     * third dimension: neuron in layer (k)
     * fourth dimension: source neuron in layer (k - 1)
     */   
    float ****matrix;
                       
    struct Matrix* weights;
    struct Matrix* posWeights;
    struct Matrix* negWeights;
    struct Matrix* bias;

    int *feature_range;
    int feature_range_length;
    int split_feature;
    int sens_feature_idx;
    unsigned long long global_volume;
};


/* load the network from file */
struct NNet *load_network(const char *filename, int target);

/* free all the memory for the network */
void destroy_network(struct NNet *network);

/* allocates memory for and loads the positive/negative weights */
void load_positive_and_negative_weights(struct NNet *nnet);

/* denormalize input */
void denormalize_input(struct NNet *nnet, struct Matrix *input);

/* denormalize input range */
void denormalize_input_interval(struct NNet *nnet, struct Interval *input);

/* normalize input */
void normalize_input(struct NNet *nnet, struct Matrix *input);

/* normalize input range */
void normalize_input_interval(struct NNet *nnet, struct Interval *input);

#endif