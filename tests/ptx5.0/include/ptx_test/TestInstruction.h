#ifndef TESTINSTRUCTION_H
#define TESTINSTRUCTION_H

#include <cuda.h>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <any>
#include <type_traits>

#include <hydrazine/ArgumentParser.h>
#include <hydrazine/macros.h>
#include <hydrazine/debug.h>
#include <hydrazine/Exception.h>
#include <hydrazine/Test.h>

#include <ocelot/ir/Module.h>
#include <ocelot/executive/EmulatedKernel.h>
#include <ocelot/executive/RuntimeException.h>
#include <ocelot/executive/CooperativeThreadArray.h>
#include <ocelot/transforms/PassManager.h>
#include <ocelot/transforms/RemoveBarrierPass.h>
#include <ocelot/transforms/ConvertPredicationToSelectPass.h>
#include <ocelot/translator/PTXToLLVMTranslator.h>
#include <ocelot/ir/LLVMKernel.h>
#include <boost/random.hpp>
#include <ocelot/ir/Dim3.h>
#include <ocelot/ir/PTXOperand.h>
#include <ocelot/api/OcelotConfiguration.h>
#include <ocelot/executive/LLVMExecutableKernel.h>
#include <ptx_test/PTXKernelConfig.h>
#include <ptx_test/Instructions.h>

namespace test {
    class TestInstruction : public Test
	{	
		public: 
			struct ArrayWithSize {
				union Pointer
				{
					float* p_f32;
					double* p_f64;
					int32_t* p_i32;
					int64_t* p_i64;
					uint32_t* p_u32;
					uint64_t* p_u64;
				} array;
				size_t bytesize; // bytesize
				ir::Dim3 dim3;
				ir::PTXOperand::DataType type;
				bool returnAray;
				ArrayWithSize(float* p, size_t bsz, ir::Dim3 dim3);
				ArrayWithSize(double* p, size_t bsz, ir::Dim3 dim3);
				ArrayWithSize(int32_t* p, size_t bsz, ir::Dim3 dim3);
				ArrayWithSize(int64_t* p, size_t bsz, ir::Dim3 dim3);
				ArrayWithSize(uint32_t* p, size_t bsz, ir::Dim3 dim3);
				ArrayWithSize(uint64_t* p, size_t bsz, ir::Dim3 dim3);
				// ArrayWithSize();
				bool operator==(const ArrayWithSize& other) const;
				static void* getPointer(const ArrayWithSize& arr);
			};
			// template<typename... Args>
			// using ParamTuple = std::tuple<Args ...>;
		protected:

			PTXKernelConfig config;
			size_t nParams;
			void _loadConfig();
			bool _testTranslate();
			
			// template<typename T, typename... Args>
			// bool _runPTXKernel(ArrayWithSize<T> d, Args... args);
			// bool _runPTXKernel(ArrayWithSize d, std::vector<ArrayWithSize> args);
			bool _runPTXKernel(std::vector<ArrayWithSize> args);
			bool _runLLVMKernel(std::vector<ArrayWithSize> args);
			bool runPTXTest();

			template<typename T>
			T _handle_arg(T arg);
			template<typename T>
			CUdeviceptr _handle_arg(T* array, size_t size);

			ArrayWithSize _allocArray(ir::PTXOperand::DataType type, ir::Dim3 dim);
			ArrayWithSize _allocArray(ir::PTXOperand::DataType type, ir::Dim3 dim, bool random);
			ArrayWithSize _allocArray(const ArrayWithSize& array);

			bool _freeArray(ArrayWithSize array);

			template<typename T>
			void _randomArray(T* a, ir::Dim3 dim);
			template<typename T>
			void _randomArray(T* a, size_t size);

			template<typename T>
			T _random();
			
			

		public:
			TestInstruction();
			// TestInstruction(PTXKernelConfig config);
		
		public:
			std::string input;
			std::string configPath;
			/*! \brief Total amount of time to spend on tests in seconds */
			hydrazine::Timer::Second timeLimit;
			bool recursive;
			bool output;
			
			ir::Module module;
			ir::IRKernel* kernel;	
	};
}


#endif /* TESTINSTRUCTION_H */
