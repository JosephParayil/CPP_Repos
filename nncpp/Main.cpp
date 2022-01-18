#include <iostream>

#include "NeuralNet.h"

int main() {
    std::vector<bool> genome = nn::RandomGenome(2,2,2,16);
    for (int i=0;i<genome.size();i++) {
        std::cout<<genome[i];
    }
    
    nn::NeuralNet brain = nn::NewBrain(2,2,2,genome);;
    std::cout<<'\n'<<genome.size()<<'\n';
    
    std::cout<<brain.ins<<'\n'<<brain.hid<<'\n'<<brain.out<<'\n';
    
    std::cout<<"CELLS:\n";
    for (int i=0;i<brain.Cells.size();i++) {
        std::cout<<' '<<brain.Cells[i].Val<<','<<brain.Cells[i].Bias<<','<<brain.Cells[i].Activation<<'\n';
    }
    std::cout<<"SYNAPSES:\n";
    for (int i=0;i<brain.Synapses.size();i++) {
        std::cout<<' '<<brain.Synapses[i].Source<<','<<brain.Synapses[i].Target<<','<<brain.Synapses[i].Weight<<'\n';
    }
    std::cout<<brain.Synapses.size();
    return 0;
}
