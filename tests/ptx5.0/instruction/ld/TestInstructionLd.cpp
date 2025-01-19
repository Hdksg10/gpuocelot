#include <ptx_test/TestInstruction.h>



namespace test {
    class TestInstructionLd : public TestInstruction {
    public:

        TestInstructionLd() {
            name = "TestInstructionFma";

            description = "A unit test for the FMA instruction.";
        }
    protected:
        // Override runxxxKernel function so that we can load const memory
        bool _runPTXKernel(std::vector<ArrayWithSize> args){
            auto _config = *config;
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
            CUDA_CHECK(cuModuleGetFunction(&kernel, module, _config.kernelName.c_str()), "cannot get cuda function with error code: ");
        
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
            // Launch kernel
            CUDA_CHECK(cuLaunchKernel(
                        kernel,
                        _config.blocks.x, _config.blocks.y, _config.blocks.z,
                        _config.threads.x, _config.threads.y, _config.threads.z,
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
                if (i == _config.destinationIdx) continue; // we have free dest memory
                CUDA_CHECK(cuMemFree((CUdeviceptr)unpackedParams[i]), "cannot free device argument memory with error code:  ");
            }

            // Destroy context
            CUDA_CHECK(cuModuleUnload(module), "cannot unload cuda module with error code:  ");
            CUDA_CHECK(cuCtxDestroy(context), "cannot destroy cuda context with error code:  ");

            return true;
        }

        bool doTest() {
            bool result = true;
            _loadConfig();
            
            result = runPTXTest();

            return result;
        }
    };
} //namespace test


int main( int argc, char** argv )
{
	hydrazine::ArgumentParser parser( argc, argv );
	test::TestInstructionLd test;
	parser.description( test.testDescription() );

	parser.parse("-c", test.configPath, "../instruction/fma/config_f16.test", "Test configuration path.");
	parser.parse( "-i", test.input, "../instruction/fma/test_fma.ptx",
		"Test PTX path.");
	parser.parse( "-r", test.recursive, false, 
		"Recursively search directories.");
	parser.parse( "-o", test.output, false,
		"Print out the internal representation of each parsed file." );
	parser.parse("-l", "--time-limit", test.timeLimit, 60, 
		"How many seconds to run tests.");
	parser.parse( "-s", test.seed, 0,
		"Set the random seed, 0 implies seed with time." );
	parser.parse( "-v", test.verbose, false, "Print out info after the test." );
	parser.parse();
	
	test.test();
	
	return test.passed();

}