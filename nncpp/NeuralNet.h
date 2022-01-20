#ifndef NEURALNET_H
#define NEURALNET_H

#include <vector>

namespace nn {
    struct Cell {
        float Val;
        float Bias;
        bool Activation;
    };
    
    struct Synapse {
        int Source;
        int Target;
        float Weight;
    };
    
    struct NeuralNet {
        std::vector<Cell> Cells;
        std::vector<Synapse> Synapses;
        int ins;
        int hid;
        int out;
        
        std::vector<int> Run(std::vector<int>);
    };
    
    std::vector<bool> RandomGenome(int, int, int, int);
    NeuralNet NewBrain(int, int, int, std::vector<bool>);
}

#endif
