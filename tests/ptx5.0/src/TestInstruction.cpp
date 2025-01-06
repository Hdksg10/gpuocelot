#include <ptx_test/TestInstruction.h>
#include <sys/types.h>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <boost/filesystem.hpp>
#include <cmath>
#include "ocelot/ir/PTXOperand.h"

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
	const double THERESHOLD = 1e-2;
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

	TestInstruction::ArrayWithSize::ArrayWithSize(half* p, size_t bsz, ir::Dim3 _dim3) {
		array.p_f16 = p;
		bytesize = bsz;
		dim3 = _dim3;
		returnAray = false;
		type = ir::PTXOperand::DataType::f16;
	}
	
	TestInstruction::ArrayWithSize::ArrayWithSize(half2* p, size_t bsz, ir::Dim3 _dim3) {
		array.p_f16x2 = p;
		bytesize = bsz;
		dim3 = _dim3;
		returnAray = false;
		type = ir::PTXOperand::DataType::f16x2;
	}

	void* TestInstruction::ArrayWithSize::getPointer(const ArrayWithSize &arr) {
		switch (arr.type) {
			case ir::PTXOperand::DataType::f16:
				return static_cast<void*>(arr.array.p_f16);
			case ir::PTXOperand::DataType::f16x2:
				return static_cast<void*>(arr.array.p_f16x2);
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
			case ir::PTXOperand::DataType::f16: {
				bool equal = true;
				for (size_t i = 0; i < bytesize / sizeof(half); i++) {
					if (fabs(this->array.p_f16[i] - other.array.p_f16[i]) > THERESHOLD) {
						equal = false;
						break;
					}
				}
				return equal;
			}
			case ir::PTXOperand::DataType::f16x2: {
				bool equal = true;
				for (size_t i = 0; i < bytesize / sizeof(half2); i++) {
					bool result = (fabs(this->array.p_f16x2[i].x - other.array.p_f16x2[i].x) < THERESHOLD) && (fabs(this->array.p_f16x2[i].y - other.array.p_f16x2[i].y) < THERESHOLD);
					if (!result) {
						equal = false;
						break;
					}
				}
				return equal;
			}
			case ir::PTXOperand::DataType::f32: {
				bool equal = true;
				for (size_t i = 0; i < bytesize / sizeof(float); i++) {
					if (fabs(this->array.p_f32[i] - other.array.p_f32[i]) > THERESHOLD) {
						equal = false;
						break;
					}
				}
				return equal;
			}
			case ir::PTXOperand::DataType::f64: {
				bool equal = true;
				for (size_t i = 0; i < bytesize / sizeof(double); i++) {
					if (fabs(this->array.p_f64[i] - other.array.p_f64[i]) > THERESHOLD) {
						equal = false;
						break;
					}
				}
				return equal;
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
		if (recursive) {
			// load all configs in path with endname ".test"
			namespace fs = boost::filesystem;
			fs::path path = configPath;
			if (!fs::is_directory(path)) {
				if (fs::is_regular_file(path)) {
					// use relative path
					path = path.parent_path();
				}
			}
			fs::directory_iterator end;
			for (fs::directory_iterator file(path); file != end; ++file) {
				if (file->path().extension() == ".test") {
					configs.emplace_back(file->path().string());
				}
			}
		}
		else {
			configs.emplace_back(configPath);
		}

		if (verbose) {
			std::cout << "Found " << configs.size() << " configurations." << std::endl;
		}
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
		else if constexpr (std::is_same_v<T, half>) {
			boost::random::uniform_real_distribution<float> dist(-100.0f, 100.0f);
			return __half(dist(random));
		}
		else if constexpr (std::is_same_v<T, half2>) {
			boost::random::uniform_real_distribution<float> dist(-100.0f, 100.0f);
			return half2(__half(dist(random)), __half(dist(random)));
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
			case ir::PTXOperand::DataType::f16: {
				newArray.array.p_f16 = new half[bytesize / sizeof(half)];
				std::memcpy(newArray.array.p_f16, array.array.p_f16, bytesize);
				break;
			}
			case ir::PTXOperand::DataType::f16x2: {
				newArray.array.p_f16x2 = new half2[bytesize / sizeof(half2)];
				std::memcpy(newArray.array.p_f16x2, array.array.p_f16x2, bytesize);
				break;
			}
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
			case ir::PTXOperand::DataType::f16:
			{
				bytesize = size * sizeof(half);
				half* p = new half[size];
				if (random) _randomArray(p, size);
				else memset(p, 0, bytesize);
				return ArrayWithSize(p, bytesize, dim);
			}
			case ir::PTXOperand::DataType::f16x2:
			{
				bytesize = size * sizeof(half2);
				half2* p = new half2[size];
				if (random) _randomArray(p, size);
				else memset(p, 0, bytesize);
				return ArrayWithSize(p, bytesize, dim);
			}
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

	TestInstruction::ArrayWithSize TestInstruction::_allocArray(const std::vector<PTXKernelConfig::Value>& values, ir::PTXOperand::DataType type) {
		switch (type) {
			case ir::PTXOperand::DataType::s32: {
				auto array = _allocArray(ir::PTXOperand::DataType::s32, ir::Dim3(values.size()), false);
				for (size_t i = 0; i < values.size(); ++i) {
					array.array.p_i32[i] = values[i].s32;
				}
				return array;
			}
			case ir::PTXOperand::DataType::s64: {
				auto array = _allocArray(ir::PTXOperand::DataType::s64, ir::Dim3(values.size()), false);
				for (size_t i = 0; i < values.size(); ++i) {
					array.array.p_i64[i] = values[i].s64;
				}
				return array;
			}
			case ir::PTXOperand::DataType::u32: {
				auto array = _allocArray(ir::PTXOperand::DataType::u32, ir::Dim3(values.size()), false);
				for (size_t i = 0; i < values.size(); ++i) {
					array.array.p_u32[i] = values[i].u32;
				}
				return array;
			}
			case ir::PTXOperand::DataType::u64: {
				auto array = _allocArray(ir::PTXOperand::DataType::u64, ir::Dim3(values.size()), false);
				for (size_t i = 0; i < values.size(); ++i) {
					array.array.p_u64[i] = values[i].u64;
				}
				return array;
			}
			case ir::PTXOperand::DataType::f16: {
				auto array = _allocArray(ir::PTXOperand::DataType::f16, ir::Dim3(values.size()), false);
				for (size_t i = 0; i < values.size(); ++i) {
					array.array.p_f16[i] = values[i].f16;
				}
				return array;
			}
			case ir::PTXOperand::DataType::f16x2: {
				auto array = _allocArray(ir::PTXOperand::DataType::f16x2, ir::Dim3(values.size()), false);
				for (size_t i = 0; i < values.size(); ++i) {
					auto value = int32ToHalf2(values[i].f16x2);
					array.array.p_f16x2[i].x = value.x;
					array.array.p_f16x2[i].y = value.y;
				}
				return array;
			}
			case ir::PTXOperand::DataType::f32: {
				auto array = _allocArray(ir::PTXOperand::DataType::f32, ir::Dim3(values.size()), false);
				for (size_t i = 0; i < values.size(); ++i) {
					array.array.p_f32[i] = values[i].f32;
				}
				return array;
			}
			case ir::PTXOperand::DataType::f64: {
				auto array = _allocArray(ir::PTXOperand::DataType::f64, ir::Dim3(values.size()), false);
				for (size_t i = 0; i < values.size(); ++i) {
					array.array.p_f64[i] = values[i].f64;
				}
				return array;
			}
			default: {
				throw std::invalid_argument("Unsupported data type for array allocation.");
			}
		}
	}

	void TestInstruction::copyToArray(const ArrayWithSize& array, const std::vector<PTXKernelConfig::Value>& values) {
		switch (array.type) {
			case ir::PTXOperand::DataType::s32: {
				for (size_t i = 0; i < values.size(); ++i) {
					array.array.p_i32[i] = values[i].s32;
				}
				break;
			}
			case ir::PTXOperand::DataType::s64: {
				for (size_t i = 0; i < values.size(); ++i) {
					array.array.p_i64[i] = values[i].s64;
				}
				break;
			}
			case ir::PTXOperand::DataType::u32: {
				for (size_t i = 0; i < values.size(); ++i) {
					array.array.p_u32[i] = values[i].u32;
				}
				break;
			}
			case ir::PTXOperand::DataType::u64: {
				for (size_t i = 0; i < values.size(); ++i) {
					array.array.p_u64[i] = values[i].u64;
				}
				break;
			}
			case ir::PTXOperand::DataType::f16: {
				for (size_t i = 0; i < values.size(); ++i) {
					array.array.p_f16[i] = values[i].f16;
				}
				break;
			}
			case ir::PTXOperand::DataType::f16x2: {
				for (size_t i = 0; i < values.size(); ++i) {
					auto value = int32ToHalf2(values[i].f16x2);
					array.array.p_f16x2[i].x = value.x;
					array.array.p_f16x2[i].y = value.y;
				}
				break;
			}
			case ir::PTXOperand::DataType::f32: {
				for (size_t i = 0; i < values.size(); ++i) {
					array.array.p_f32[i] = values[i].f32;
				}
				break;
			}
			case ir::PTXOperand::DataType::f64: {
				for (size_t i = 0; i < values.size(); ++i) {
					array.array.p_f64[i] = values[i].f64;
				}
				break;
			}
			default: {
				throw std::invalid_argument("Unsupported data type for array allocation.");
			}
		}
	}

	void TestInstruction::printArray(const ArrayWithSize& array) {
		switch (array.type) {
			case ir::PTXOperand::DataType::s32: {
				for (size_t i = 0; i < array.dim3.size(); ++i) {
					std::cout << array.array.p_i32[i] << " ";
				}
				std::cout << std::endl;
				break;
			}
			case ir::PTXOperand::DataType::s64: {	
				for (size_t i = 0; i < array.dim3.size(); ++i) {
					std::cout << array.array.p_i64[i] << " ";
				}
				std::cout << std::endl;
				break;
			}
			case ir::PTXOperand::DataType::u32: {
				for (size_t i = 0; i < array.dim3.size(); ++i) {
					std::cout << array.array.p_u32[i] << " ";
				}
				std::cout << std::endl;
				break;
			}
			case ir::PTXOperand::DataType::u64: {
				for (size_t i = 0; i < array.dim3.size(); ++i) {
					std::cout << array.array.p_u64[i] << " ";
				}
				std::cout << std::endl;
				break;
			}
			case ir::PTXOperand::DataType::f16: {
				for (size_t i = 0; i < array.dim3.size(); ++i) {
					std::cout << array.array.p_f16[i] << " ";
				}
				std::cout << std::endl;
				break;
			}
			case ir::PTXOperand::DataType::f16x2: {
				for (size_t i = 0; i < array.dim3.size(); ++i) {
					std::cout << "[" << array.array.p_f16x2[i].x << " , " << array.array.p_f16x2[i].y << "] ";
				}
				std::cout << std::endl;
				break;
			}
			case ir::PTXOperand::DataType::f32: {
				for (size_t i = 0; i < array.dim3.size(); ++i) {
					std::cout << array.array.p_f32[i] << " ";
				}
				std::cout << std::endl;
				break;
			}
			case ir::PTXOperand::DataType::f64: {
				for (size_t i = 0; i < array.dim3.size(); ++i) {
					std::cout << array.array.p_f64[i] << " ";
				}
				std::cout << std::endl;
				break;
			}
			default: {
				throw std::invalid_argument("Unsupported data type for array printing.");
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
			case ir::PTXOperand::DataType::f16:
			{
				delete[] array.array.p_f16;
				break;
			}
			case ir::PTXOperand::DataType::f16x2:
			{
				delete[] array.array.p_f16x2;
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
	half2 TestInstruction::int32ToHalf2(int32_t a) {
		half2 h2;
		// store to half2, x = a[31,...,16], y = a[15,...,0]
		int16_t x = a >> 16;
		int16_t y = a & 0xffff;
		h2.x = *(reinterpret_cast<half*>(reinterpret_cast<void*>(&x)));
		h2.y = *(reinterpret_cast<half*>(reinterpret_cast<void*>(&y)));
		return h2;
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

	bool TestInstruction::_runLLVMKernel(std::vector<ArrayWithSize> args) {
		auto _config = *config;
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
		kernel = module.getKernel(_config.kernelName);
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
		for (size_t i = 0; i < _config.nParams; i++) {
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
		executableKerel->launchGrid( _config.blocks.x, _config.blocks.y, _config.blocks.z );

		return result;
	}

	bool TestInstruction::_runPTXTest() {
		auto _config = *config;
		bool result = true;
		std::vector<ArrayWithSize> args;
		ArrayWithSize d = _allocArray(_config.paramVector[_config.destinationIdx], _config.sizeVector[_config.destinationIdx]);
		d.returnAray = true;

		// build random test array
		for (size_t i = 0; i < _config.paramVector.size(); i++) {
			auto type = _config.paramVector[i];
			auto dim3 = _config.sizeVector[i];
			if (i == _config.destinationIdx)
				args.push_back(d);
			else
				args.push_back(_allocArray(type, dim3, true));
		}
		
		// if spicified input data
		for (size_t i = 0; i < _config.valuesVector.size(); i++) {
			// auto type = _config.paramVector[i];
			auto dim3 = _config.sizeVector[i];
			// if (i == _config.destinationIdx)
			// 	args.push_back(d);
			// else
			// 	args.push_back(_allocArray(type, dim3, true));
			const auto & values = _config.valuesVector[i];
			if (values.size() == args[i].dim3.size()) {
				copyToArray(args[i], values);
			}
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
		if (!equal || verbose) {
			if (!equal) {
				std::cout << "Error when check correcty" << std::endl;
			}
			std::cout << "PTX result: ";
			printArray(d_cuda);

			std::cout << "IR result: ";
			printArray(d);

			std::cout << "Argument: " << std::endl;
			int argIdx = 0;
			for (int i = 0; i < args.size(); i++)
			{
				if (args[i].returnAray) continue;
				std::cout << "arg" << argIdx << ": ";
				argIdx++;
				printArray(args[i]);
			}
		}
		_freeArray(d_cuda);
		for (auto&& arg:args){
			_freeArray(arg);
		}

		return result && equal;
	}

	bool TestInstruction::runPTXTest() {
		bool result = true;
		for (auto&& _config:configs){
			if (verbose) {
				std::cout << "Running test: " << _config.kernelName << std::endl;
			}
			config = &_config;
			bool r = _runPTXTest();
			if (verbose) {
				std::cout << "Test result: " << (r ? "PASS" : "FAIL") << std::endl;
			}
			result = r && result;
		}
		return result;
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
