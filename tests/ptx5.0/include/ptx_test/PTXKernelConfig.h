#ifndef PTXKERNELCONFIG_H
#define PTXKERNELCONFIG_H

#include <cuda.h>
#include <cuda_fp16.h>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>

#include <ocelot/ir/Dim3.h>
#include <ocelot/ir/PTXOperand.h>

#include <hydrazine/json.h>
#include <hydrazine/Exception.h>


class PTXKernelConfig {
	public:
		union Value {
			int64_t s64;
			double f64;
			int32_t s32;
			uint32_t u32;
			uint64_t u64;
			float f32;
			half f16;
			int32_t f16x2; // we use int32_t to store f16x2 value

			// Value() : f16x2(0.0, 0.0){};
			
		};
		using ParamVector = std::vector<ir::PTXOperand::DataType>;
		using SizeVector = std::vector<ir::Dim3>;
		ir::Dim3 threads; // threads per block
		ir::Dim3 blocks;  // blocks per grid
		std::string kernelName;
		ParamVector paramVector;
		SizeVector sizeVector;
		int destinationIdx;
		size_t nParams;
		std::vector<std::vector<Value>> valuesVector;
		
	public:
		PTXKernelConfig();
		PTXKernelConfig(ir::Dim3 _threads, ir::Dim3 _blocks, std::string _kernelName, const ParamVector& _paramVector, const SizeVector& _sizeVector, int _destinationIdx);
		PTXKernelConfig(std::string path);
		// void loadConfigFromFile(std::string filename);
		static ir::Dim3 initialize_dim3(hydrazine::json::Visitor vistor);
		static ir::PTXOperand::DataType stringToDataType(std::string str);
};




#endif /* PTXKERNELCONFIG_H */
