#include <ptx_test/TestInstruction.h>
#include <sys/types.h>
#include <cstdint>
#include <vector>


#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 2

#define CUDA_CHECK(f, msg) \
	if ((r = f) != CUDA_SUCCESS) { \
		status << msg << r;  \
		return false;            \
	}                            


namespace test {

	TestInstruction::ArrayWithSize::ArrayWithSize(float* p, size_t bsz, ir::Dim3 _dim3) {
		array.p_f32 = p;
		bytesize = bsz;
		dim3 = _dim3;
		returnAray = false;
		type = ir::PTXOperand::DataType::f32;
	}

	TestInstruction::ArrayWithSize::ArrayWithSize(double* p, size_t bsz, ir::Dim3 _dim3) {
		array.p_f64 = p;
		bytesize = bsz;
		dim3 = _dim3;
		returnAray = false;
		type = ir::PTXOperand::DataType::f64;
	}

	TestInstruction::ArrayWithSize::ArrayWithSize(int32_t* p, size_t bsz, ir::Dim3 _dim3) {
		array.p_i32 = p;
		bytesize = bsz;
		dim3 = _dim3;
		returnAray = false;
		type = ir::PTXOperand::DataType::s32;
	}

	TestInstruction::ArrayWithSize::ArrayWithSize(int64_t* p, size_t bsz, ir::Dim3 _dim3) {
		array.p_i64 = p;
		bytesize = bsz;
		dim3 = _dim3;
		returnAray = false;
		type = ir::PTXOperand::DataType::s64;
	}

	TestInstruction::ArrayWithSize::ArrayWithSize(uint32_t* p, size_t bsz, ir::Dim3 _dim3) {
		array.p_u32 = p;
		bytesize = bsz;
		dim3 = _dim3;
		returnAray = false;
		type = ir::PTXOperand::DataType::u32;
	}

	TestInstruction::ArrayWithSize::ArrayWithSize(uint64_t* p, size_t bsz, ir::Dim3 _dim3) {
		array.p_u64 = p;
		bytesize = bsz;
		dim3 = _dim3;
		returnAray = false;
		type = ir::PTXOperand::DataType::u64;
	}
	
	void* TestInstruction::ArrayWithSize::getPointer(const ArrayWithSize &arr) {
		switch (arr.type) {
            case ir::PTXOperand::DataType::f32:
                return static_cast<void*>(arr.array.p_f32);
            case ir::PTXOperand::DataType::f64:
                return static_cast<void*>(arr.array.p_f64);
            case ir::PTXOperand::DataType::s32:
                return static_cast<void*>(arr.array.p_i32);
            case ir::PTXOperand::DataType::s64:
                return static_cast<void*>(arr.array.p_i64);
			case ir::PTXOperand::DataType::b32: 
            case ir::PTXOperand::DataType::u32:
                return static_cast<void*>(arr.array.p_u32);
            case ir::PTXOperand::DataType::b64:
			case ir::PTXOperand::DataType::u64:
                return static_cast<void*>(arr.array.p_u64);
            default:
                return nullptr; 
        }
	}

	bool TestInstruction::ArrayWithSize::operator==(const ArrayWithSize& other) const {
		if (this->bytesize != other.bytesize || this->type != other.type) {
			return false; 
		}

		switch (type) {
			case ir::PTXOperand::DataType::f32: {
				return std::memcmp(this->array.p_f32, other.array.p_f32, bytesize) == 0;
			}
			case ir::PTXOperand::DataType::f64: {
				return std::memcmp(this->array.p_f64, other.array.p_f64, bytesize) == 0;
			}
			case ir::PTXOperand::DataType::s32: {
				return std::memcmp(this->array.p_i32, other.array.p_i32, bytesize) == 0;
			}
			case ir::PTXOperand::DataType::s64: {
				return std::memcmp(this->array.p_i64, other.array.p_i64, bytesize) == 0;
			}
			case ir::PTXOperand::DataType::u32: {
				return std::memcmp(this->array.p_u32, other.array.p_u32, bytesize) == 0;
			}
			case ir::PTXOperand::DataType::u64: {
				return std::memcmp(this->array.p_u64, other.array.p_u64, bytesize) == 0;
			}
			default: {
				throw std::invalid_argument("Unsupported data type for comparison.");
			}
		}
	}

	TestInstruction::TestInstruction() {
		name = "TestInstruction";

		description = "A unit test for the LLVM executive runtime.";
		description += " Test Points: 1) Execute a kernel with a loop. ";
		description += "2) Execute a matrix multiply kernel.";
	}

	void TestInstruction::_loadConfig() {
		config = PTXKernelConfig(configPath);
		nParams = config.paramVector.size();
	}

	template<typename T>
	T TestInstruction::_random() {
		if constexpr (std::is_same_v<T, int> || std::is_same_v<T, long>
				      || std::is_same_v<T, int32_t> || std::is_same_v<T, int64_t>) {
			boost::random::uniform_int_distribution<T> dist(std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
        	return dist(random);
		}
		else if constexpr (std::is_same_v<T, unsigned> || std::is_same_v<T, unsigned long>
				      || std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>) {
			boost::random::uniform_int_distribution<T> dist(std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
        	return dist(random);
		}
		else if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
			boost::random::uniform_real_distribution<T> dist(-100.0, 100.0); 
        	return dist(random);	
		}
		else {
			// Unsupported 
			static_assert(std::is_same_v<T, void>, "Unsupported type for _random");
		}
	}


	template<typename T>
	void TestInstruction::_randomArray(T* a, ir::Dim3 dim) {
		for (int i = 0 ; i < dim.size(); i++) {
			a[i] = _random<T>();
		}
	}

	template<typename T>
	void TestInstruction::_randomArray(T* a, size_t size) {
		for (int i = 0 ; i < size; i++) {
			a[i] = _random<T>();
		}
	}
	TestInstruction::ArrayWithSize TestInstruction::_allocArray(const ArrayWithSize& array) {
		auto newArray = _allocArray(array.type, array.dim3, false);
		newArray.returnAray = array.returnAray;
		size_t bytesize = newArray.bytesize;
		switch (array.type) {
			case ir::PTXOperand::DataType::f32: {
				newArray.array.p_f32 = new float[bytesize / sizeof(float)];
				std::memcpy(newArray.array.p_f32, array.array.p_f32, bytesize);
				break;
			}
			case ir::PTXOperand::DataType::f64: {
				newArray.array.p_f64 = new double[bytesize / sizeof(double)];
				std::memcpy(newArray.array.p_f64, array.array.p_f64, bytesize);
				break;
			}
			case ir::PTXOperand::DataType::s32: {
				newArray.array.p_i32 = new int32_t[bytesize / sizeof(int32_t)];
				std::memcpy(newArray.array.p_i32, array.array.p_i32, bytesize);
				break;
			}
			case ir::PTXOperand::DataType::s64: {
				newArray.array.p_i64 = new int64_t[bytesize / sizeof(int64_t)];
				std::memcpy(newArray.array.p_i64, array.array.p_i64, bytesize);
				break;
			}
			case ir::PTXOperand::DataType::u32: {
				newArray.array.p_u32 = new uint32_t[bytesize / sizeof(uint32_t)];
				std::memcpy(newArray.array.p_u32, array.array.p_u32, bytesize);
				break;
			}
			case ir::PTXOperand::DataType::u64: {
				newArray.array.p_u64 = new uint64_t[bytesize / sizeof(uint64_t)];
				std::memcpy(newArray.array.p_u64, array.array.p_u64, bytesize);
				break;
			}
			default: {
				throw std::invalid_argument("Unsupported data type for array allocation.");
			}
		}
		return newArray;
	}

	TestInstruction::ArrayWithSize TestInstruction::_allocArray(ir::PTXOperand::DataType type, ir::Dim3 dim) {
		return _allocArray(type, dim, false);
	}

	TestInstruction::ArrayWithSize TestInstruction::_allocArray(ir::PTXOperand::DataType type, ir::Dim3 dim, bool random) {
		size_t size = dim.size();
		size_t bytesize;
		switch (type) {
			case ir::PTXOperand::DataType::f64:
			{
				bytesize = size * sizeof(double);
				double* p = new double[size];
				if (random) _randomArray(p, size);
				else memset(p, 0, bytesize);
				return ArrayWithSize(p, bytesize, dim);
			}
			case ir::PTXOperand::DataType::f32:
			{
				bytesize = size * sizeof(float);
				float* p = new float[size];
				if (random) _randomArray(p, size);
				else memset(p, 0, bytesize);
				return ArrayWithSize(p, bytesize, dim);
			}
			case ir::PTXOperand::DataType::s64:
			{
				bytesize = size * sizeof(int64_t);
				int64_t* p = new int64_t[size];
				if (random) _randomArray(p, size);
				else memset(p, 0, bytesize);
				return ArrayWithSize(p, bytesize, dim);
			}
			case ir::PTXOperand::DataType::s32:
			{
				bytesize = size * sizeof(int32_t);
				int32_t* p = new int32_t[size];
				if (random) _randomArray(p, size);
				else memset(p, 0, bytesize);
				return ArrayWithSize(p, bytesize, dim);
			}
			case ir::PTXOperand::DataType::b64:
			case ir::PTXOperand::DataType::u64:
			{
				bytesize = size * sizeof(uint64_t);
				uint64_t* p = new uint64_t[size];
				if (random) _randomArray(p, size);
				else memset(p, 0, bytesize);
				return ArrayWithSize(p, bytesize, dim);
			}
			case ir::PTXOperand::DataType::b32:
			case ir::PTXOperand::DataType::u32:
			{
				bytesize = size * sizeof(uint32_t);
				uint32_t* p = new uint32_t[size];
				if (random) _randomArray(p, size);
				else memset(p, 0, bytesize);
				return ArrayWithSize(p, bytesize, dim);
			}
			default:
			{
				throw std::invalid_argument("Unsupported data type.");
			}
		}
	}

	bool TestInstruction::_freeArray(ArrayWithSize array) {
		switch (array.type) {
			case ir::PTXOperand::DataType::f64:
			{
				delete[] array.array.p_f64;
				break;
			}
			case ir::PTXOperand::DataType::f32:
			{
				delete[] array.array.p_f32;
				break;
			}
			case ir::PTXOperand::DataType::s64:
			{
				delete[] array.array.p_i64;
				break;
			}
			case ir::PTXOperand::DataType::s32:
			{
				delete[] array.array.p_i32;
				break;
			}
			case ir::PTXOperand::DataType::b64:
			case ir::PTXOperand::DataType::u64:
			{
				delete[] array.array.p_u64;
				break;
			}
			case ir::PTXOperand::DataType::b32:
			case ir::PTXOperand::DataType::u32:
			{
				delete[] array.array.p_u32;
				break;
			}
			default:
			{
				throw std::invalid_argument("Unsupported data type.");
			}
		}
		return true;
	}

	template<typename T>
	T TestInstruction::_handle_arg(T arg) {
		// unsupported
		static_assert(std::is_same_v<T, void>, "unsupported handle arg");
	}

	template<typename T>
	CUdeviceptr TestInstruction::_handle_arg(T* array, size_t size) {
		CUresult r;
		CUdeviceptr p = 0;
		// T* array = arg.first;
		size_t bytesize = size;
		
		CUDA_CHECK(cuMemAlloc(&p, bytesize), "cannot load allocate memory with error code: ");
		CUDA_CHECK(cuMemcpyHtoD(p, array, bytesize), "cannot copy memory from host to device with error code: ")

		return p;
	}

	template<>
	CUdeviceptr TestInstruction::_handle_arg(void* array, size_t size) {
		CUresult r;
		CUdeviceptr p = 0;
		// T* array = arg.first;
		size_t bytesize = size;
		CUDA_CHECK(cuMemAlloc(&p, bytesize), "cannot load allocate memory with error code: ");
		CUDA_CHECK(cuMemcpyHtoD(p, array, bytesize), "cannot copy memory from host to device with error code: ")

		return p;
	}

	
	bool TestInstruction::_runPTXKernel(std::vector<ArrayWithSize> args){
		// Get CUDA Device
		CUresult r;
		if ((r = cuInit(0)) != CUDA_SUCCESS) {
			status << "cannot init cuda driver with error code: " << r;
			return false;
		}

		CUdevice device;
		if ((r = cuDeviceGet(&device, 0)) != CUDA_SUCCESS) {
			status << "cannot get cuda device with error code: " << r;
			return false;
		}

		CUcontext context;
		if ((r = cuCtxCreate(&context, 0, device)) != CUDA_SUCCESS) {
			status << "cannot create cuda context with error code: " << r;
			return false;
		}

		// Load Module
		CUmodule module;
		CUfunction kernel;
		CUDA_CHECK(cuModuleLoad(&module, input.c_str()), "cannot load cuda module with error code: ");
		CUDA_CHECK(cuModuleGetFunction(&kernel, module, config.kernelName.c_str()), "cannot get cuda function with error code: ");
	
		// Unpack args
		void* d_host = nullptr;
		CUdeviceptr d_device = 0;
		size_t d_size;
		std::vector<CUdeviceptr> unpackedParams;
		for (size_t i = 0; i < args.size(); i++) {
			auto arg = args[i];
			if (arg.returnAray) {
				// Allocate dest array
				d_size = arg.bytesize;
				d_host = ArrayWithSize::getPointer(arg);
				CUDA_CHECK(cuMemAlloc(&d_device, d_size), "cannot load allocate memory with error code: ");
				unpackedParams.push_back(d_device);
			}
			else {
				unpackedParams.push_back(_handle_arg(ArrayWithSize::getPointer(arg), arg.bytesize));
			}
		}
		std::vector<void*> kernelParams;
		for (auto&& param : unpackedParams) {
			kernelParams.push_back(&param);
		}
		
		void** kernelParams_ptr = kernelParams.data();
		// void* kernelParams_ptr[] = {
		// 	&unpackedParams[0],
		// 	&unpackedParams[1],
		// 	&unpackedParams[2],
		// 	&unpackedParams[3],
		// };
		// for (int i = 0 ; i < 4; i++) {
		// 	std::cout << "upp: " << kernelParams_ptr[i] << ", kpp:" << kernelParams[i] << std::endl;
		// }
		// Launch kernel
		CUDA_CHECK(cuLaunchKernel(
					kernel,
					config.blocks.x, config.blocks.y, config.blocks.z,
					config.threads.x, config.threads.y, config.threads.z,
					0, 0,
					kernelParams_ptr,
					nullptr), "cannot launch cuda function with error code: ");

		CUDA_CHECK(cuCtxSynchronize(), "cannot synchronize cuda device with error code: ");
		// Copy result back
		CUDA_CHECK(cuMemcpyDtoH(d_host, d_device, d_size), "cannot copy memory from device to host back with error code:  ");

		// std::cout << *((uint32_t *)d_host) << std::endl;

		// Clean memory
		CUDA_CHECK(cuMemFree(d_device), "cannot free device dest memory with error code:  ");

		for (size_t i = 0; i < unpackedParams.size(); i++) {
			if (i == config.destinationIdx) continue; // we have free dest memory
			CUDA_CHECK(cuMemFree((CUdeviceptr)unpackedParams[i]), "cannot free device argument memory with error code:  ");
		}

		// Destroy context
		CUDA_CHECK(cuModuleUnload(module), "cannot unload cuda module with error code:  ");
    	CUDA_CHECK(cuCtxDestroy(context), "cannot destroy cuda context with error code:  ");

		return true;
	}

	bool TestInstruction::_runLLVMKernel(std::vector<ArrayWithSize> args) {
		bool result = true;
		bool loaded = false;
		// load kernel function
		try {
			loaded = module.load(input);
		}
		catch(const hydrazine::Exception& e) {
			status << " error - " << e.what() << "\n";
		}
		if(!loaded) {
			status << "failed to load module '" << input << "'\n";
			return (result = false);
		}
		kernel = module.getKernel(config.kernelName);
		if (!kernel) {
			status << "failed to get kernel\n";
			return (result = false);
		}

		// output translated llvm kernel
		if (output) {
			
			transforms::PassManager manager(&module);

			transforms::ConvertPredicationToSelectPass pass1;
			transforms::RemoveBarrierPass pass2;
			translator::PTXToLLVMTranslator translator;

			manager.addPass(&pass1);
			manager.addPass(&pass2);

			manager.runOnKernel(*kernel);
			manager.releasePasses();
			
			manager.addPass(&translator);
			manager.runOnKernel(*kernel);
			manager.releasePasses();

			ir::LLVMKernel* translatedKernel = dynamic_cast< ir::LLVMKernel* >( 
				translator.translatedKernel() );
			translatedKernel->assemble();
			
			std::string outputFile = input + "." + kernel->name + ".ll";
		
			if( output )
			{
				std::ofstream outFile( outputFile.c_str() );
				outFile << translatedKernel->code();
				outFile << "\n";
				outFile.close();
			}
			
			delete translatedKernel;
		}

		// std::cout << kernel->name << std::endl;
		// configure parameters
		std::vector<ir::Parameter*> params;
		std::stringstream ss;
		
		int level = api::OcelotConfiguration::get().executive.optimizationLevel;
		auto executableKerel = new executive::LLVMExecutableKernel(*kernel, 0, 
		( translator::Translator::OptimizationLevel ) level);
		for (size_t i = 0; i < nParams; i++) {
			ss << executableKerel->name << "_param_" << i;
			std::string paramName = ss.str();
			ss.str("");
			params.push_back(executableKerel->getParameter(paramName));
			
		}
		// set parameter values
		for (int i = 0; i < params.size(); i++) {
			auto p = params[i];
			auto arg = args[i];
			p->arrayValues.resize(1);
			p->arrayValues[0].val_u64 = (ir::PTXU64) ArrayWithSize::getPointer(arg);
			if (arg.returnAray) p->returnArgument = true;
		}
		executableKerel->updateArgumentMemory();

		executableKerel->setKernelShape( 1, 1, 1 );
		executableKerel->launchGrid( config.blocks.x, config.blocks.y, config.blocks.z );

		return result;
	}

	bool TestInstruction::runPTXTest() {
		bool result = true;
		std::vector<ArrayWithSize> args;
		ArrayWithSize d = _allocArray(config.paramVector[config.destinationIdx], config.sizeVector[config.destinationIdx]);
		d.returnAray = true;
		// build random test array
		for (int i = 0; i < config.paramVector.size(); i++) {
			auto type = config.paramVector[i];
			auto dim3 = config.sizeVector[i];
			if (i == config.destinationIdx)
				args.push_back(d);
			else
				args.push_back(_allocArray(type, dim3, true));
		}
		
		result = _runPTXKernel(args) && result;
		// copy dest array
		auto d_cuda = _allocArray(d);

		result = _runLLVMKernel(args) && result;

		if (!result) {
			// status.str();
			std::cout << "Status:" << status.str() << std::endl;
		} 
		bool equal = (d == d_cuda);
		if (!equal) {
			uint32_t* d_p = (uint32_t *) ArrayWithSize::getPointer(d);
			uint32_t* dcuda_p = (uint32_t *) ArrayWithSize::getPointer(d_cuda);
			std::cout << "Error when check correcty" << std::endl;
			std::cout << "PTX result: ";
			for (int i = 0; i < d_cuda.dim3.size(); i++)
			{
				std::cout << dcuda_p[i] <<  " " ;
			}
			std::cout << std::endl;

			std::cout << "IR result: ";
			for (int i = 0; i < d.dim3.size(); i++)
			{
				std::cout << d_p[i] <<  " " ;
			}
			std::cout << std::endl;

			std::cout << "Argument: " << std::endl;
			int argIdx = 0;
			for (int i = 0; i < args.size(); i++)
			{
				if (args[i].returnAray) continue;
				std::cout << "arg" << argIdx << ": ";
				argIdx++;
				uint32_t* p = (uint32_t *) ArrayWithSize::getPointer(args[i]);
				for (int j = 0; j < args[i].dim3.size(); j++)
				{
					std::cout << p[j] <<  " " ;
				}
				std::cout << std::endl;
			}
		}
		

		for (auto&& arg:args){
			_freeArray(arg);
		}

		return result && equal;
	}

}

// int main( int argc, char** argv )
// {
// 	hydrazine::ArgumentParser parser( argc, argv );
// 	test::TestInstruction test;
// 	parser.description( test.testDescription() );

// 	parser.parse("-c", test.configPath, "config.test", "Test configuration path");
// 	parser.parse( "-i", test.input, "../tests/ptx",
// 		"Input directory to search for ptx files." );
// 	parser.parse( "-r", test.recursive, true, 
// 		"Recursively search directories.");
// 	parser.parse( "-o", test.output, false,
// 		"Print out the internal representation of each parsed file." );
// 	parser.parse("-l", "--time-limit", test.timeLimit, 60, 
// 		"How many seconds to run tests.");
// 	parser.parse( "-s", test.seed, 0,
// 		"Set the random seed, 0 implies seed with time." );
// 	parser.parse( "-v", test.verbose, false, "Print out info after the test." );
// 	parser.parse();
	
// 	// test.test();
	
// 	// return test.passed();
// 	return 0;
// }
