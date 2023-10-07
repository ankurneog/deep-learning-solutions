//////////////////////////////////////////////
// Torch script inference with C++
// Author: Ankur Neog
//API details : 
//https://caffe2.ai/doxygen-c/html/structtorch_1_1jit_1_1script_1_1_module.html
//////////////////////////////////////////////
#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <iomanip>
#include <exception>
#include <fstream>
#include <string>
using namespace std;

int main(int argc,const char* argv[]){
    auto cuda_available = torch::cuda::is_available();
    torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
    std::cout << (cuda_available ? "GPU support available" : "Executing on CPU.") <<endl;
    if(argc!=2){
        std::cerr<<"usage : nliml <torchscript file>"<<endl;
        std::cerr<<"Model file should be stored in build directory"<<endl;
    }
    auto file = argv[1];
    ifstream filestream;
    filestream.open(file);
    if(!filestream.is_open()) {

        std::cout<<endl<<"Failed to open file !"<<endl<<endl;
        exit(0);
    }
    
    torch::jit::script::Module module;
    try {
        //deserialize
         module = torch::jit::load(filestream);
       
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model "<<endl<<e.what()<<endl;
        return -1;
    }
    cout<<"Model Successfully loaded "<<endl;
    module.eval(); //Evaluation mode
    cout<<"Is Training Mode : "<<module.is_training()<<endl;

    //TODO : extract data from CSV file and use normalizer to normalize
    //following is a normalized feed for illustration.
   //  avgPSD[dB_mW/GHz] PSDcorr [dB] SCF_CUT [n.u.] SCF_N [n.u.] Phi_CUT [n.u.]	Phi_N [n.u.] SymRate_CUT [GBaud] SymRate_N [GBaud]

    torch::Tensor tharray = torch::tensor({0.2,0.6,0.0,0.1,0.16667,0.0,0.01,0.596}, {torch::kFloat});
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(tharray);


    at::Tensor  output = module.forward(inputs).toTensor();
    cout<<"Computed NLI : "<<endl;
    cout<<output<<endl;
    return 0;

}


