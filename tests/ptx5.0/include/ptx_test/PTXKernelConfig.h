#ifndef PTX_KERNEL_CONFIG_H_INCLUDED
#define PTX_KERNEL_CONFIG_H_INCLUDED

#include <cuda.h>
#include <fstream>
#include <iostream>

#include <ocelot/ir/Dim3.h>
#include <ocelot/ir/PTXOperand.h>

#include <hydrazine/json.h>
#include <hydrazine/Exception.h>

class PTXKernelConfig {
	public:
		using ParamVector = std::vector<ir::PTXOperand::DataType>;
		using SizeVector = std::vector<ir::Dim3>;
		ir::Dim3 threads; // threads per block
		ir::Dim3 blocks;  // blocks per grid
		std::string kernelName;
		ParamVector paramVector;
		SizeVector sizeVector;
		int destinationIdx;
	public:
		PTXKernelConfig();
		PTXKernelConfig(ir::Dim3 _threads, ir::Dim3 _blocks, std::string _kernelName, const ParamVector& _paramVector, const SizeVector& _sizeVector, int _destinationIdx);
		PTXKernelConfig(std::string path);
		// void loadConfigFromFile(std::string filename);
		static ir::Dim3 initialize_dim3(hydrazine::json::Visitor vistor);
		static ir::PTXOperand::DataType stringToDataType(std::string str);
};




#endif