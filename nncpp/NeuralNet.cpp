#include <cmath>
#include <random>

#include "NeuralNet.h"

namespace  math {
    std::random_device dev;
	std::mt19937 rng(dev());
	std::uniform_int_distribution<std::mt19937::result_type> randbit(0,1);
	int BitsInNum(int num) {
	    if (num) return (int)ceil(log2(num));
	    return 0;
	}
}


std::vector<bool> nn::RandomGenome(int ins, int hid, int out, int synapses) {
	std::vector<bool> genome;
	for (int i=0;i<synapses*(math::BitsInNum(ins+hid+out)+math::BitsInNum(hid+out)+21);i++) {
	    genome.push_back(math::randbit(math::rng));
	}
	
	return genome;
}

nn::NeuralNet nn::NewBrain(int ins, int hid, int out, std::vector<bool> genome) {
    NeuralNet newNN;
    newNN.ins = ins;
    newNN.hid = hid;
    newNN.out = out;
    for (int i=0;i<ins+hid+out;i++) newNN.Cells.push_back({0,0,false});
    int len0 = math::BitsInNum(ins+hid+out);
    int len1 = math::BitsInNum(hid+out);
    int genelen = len0+len1+21;
    unsigned int i=0;
    for (int g=0;g<genome.size()/genelen;g++) {
        Synapse newSyn;
        //SOURCE ID TRANSLATION
        int num = 0;
        int p=1;
        int genepos = (g*genelen)+len0;
        while (i<genepos) {
            if (genome[i])
                num |= p;
            i++; p<<=1;
        }
        newSyn.Source = num%(ins+hid+out);
        
        //TARGET ID TRANSLATION
        num=0; p=1;
        genepos+=len1;
        while (i<genepos) {
            if (genome[i])
                num |= p;
            i++; p<<=1;
        }
        newSyn.Target = (num%(hid+out))+ins;
        
        
        //WEIGHT TRANSLATION
        num=0; p=1;
        genepos+=10;
        while (i<genepos) {
            if (genome[i])
                num |= p;
            i++; p<<=1;
        }
        if (genome[i])
            num-=1023;
        i++;
        
        newSyn.Weight=num/341.0f;
        
        //BIAS TRANSLATION
        num=0; p=1;
        genepos+=9;
        while (i<genepos) {
            if (genome[i])
                num |= p;
            i++; p<<=1;
        }
        if (genome[i])
            num-=255;
        i++;
        
        newNN.Cells[newSyn.Target].Bias=num/85.0f;
        
        //ACTIVATION FUNCTION TYPE TRANSLATION
        newNN.Cells[newSyn.Target].Activation=genome[i];
        i++;
        
        //ADDING THE NEW SYNAPSE TO THE SYNAPSES vector
        bool check = true;
        for (int i=0;i<newNN.Synapses.size();i++) {
            if ((newNN.Synapses[i].Source==newSyn.Source)&&(newNN.Synapses[i].Target==newSyn.Target)){
                check = false;
                newNN.Synapses[i].Weight+=newSyn.Weight;
            }
        }
        if (check) newNN.Synapses.push_back(newSyn);
    }
    
    return newNN;
}     
    
    
    
std::vector<float> nn::NeuralNet::Run(std::vector<float> input) {
    //APPLYING INPUT
    for (int i=0; i<input.size(); i++) this->Cells[i].Val=input[i];
    //INITIALIZING VALUE MAP TO APPLY TO CELLS
    std::vector<float> valMap; for (int i=0;i<(this->hid+this->out);i++) valMap.push_back(0.0f);
    
    for (int i=0; i<this->Synapses.size(); i++) {
        nn::Synapse* syn = &this->Synapses[i];
        //Adding Source cell value*synapse weight to the target cell value.
        valMap[syn->Target-this->ins]+= this->Cells[syn->Source].Val*syn->Weight;
    }
    
    for (int i=this->ins; i<this->Cells.size(); i++) {
        //APPLYING BIAS
        float val = valMap[i-this->ins]+this->Cells[i].Bias;
        //SIGMOID ACTIVATION FUNCTION, if the activation type is 1
        if (this->Cells[i].Activation) val = 1/(1+pow(2,-val));
        //STANDARD CROPPING to 1-0, if the ativation type is 0
        else if (val>1) val=1; else if (val<0) val=0;
        this->Cells[i].Val= val;
    }
    std::vector<float> output;
    for (int i=this->ins+this->hid; i<this->ins+this->hid+this->out; i++) {
        output.push_back(this->Cells[i].Val);
    }
    
    
    return output;
}

    
