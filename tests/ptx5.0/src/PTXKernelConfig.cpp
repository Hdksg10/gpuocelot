#ifndef PTXKERNELCONFIG_H
#define PTXKERNELCONFIG_H

#include <ptx_test/PTXKernelConfig.h>
#include "hydrazine/json.h"

PTXKernelConfig::PTXKernelConfig() {
    threads = ir::Dim3(1, 1, 1);
	blocks = ir::Dim3(1, 1, 1);
    destinationIdx = 0;
    kernelName = "invalid_kernel_name";
}

PTXKernelConfig::PTXKernelConfig(ir::Dim3 _threads, ir::Dim3 _blocks, std::string _kernelName, const ParamVector& _paramVector, const SizeVector& _sizeVector, int _destinationIdx) : threads(_threads), blocks(_blocks), kernelName(_kernelName), paramVector(_paramVector), sizeVector(_sizeVector), destinationIdx(_destinationIdx) {}

PTXKernelConfig::PTXKernelConfig(std::string path) {
	hydrazine::json::Parser parser;
	hydrazine::json::Object *config = 0;
	std::ifstream file(path.c_str());
	try {
		config = parser.parse_object(file);
        hydrazine::json::Visitor main(config);
        // if (main.find("threads")) {
        //     threads = initialize_dim3(main["trheads"]);
        // }
        // if (main.find("blocks")) {
        //     blocks = initialize_dim3(main["blocks"]);
        // }
        threads = initialize_dim3(main["trheads"]);
        blocks = initialize_dim3(main["blocks"]);
        kernelName = main.parse<std::string> ("kernel", "invalid_kernel_name");
        destinationIdx =  main.parse<int> ("dest", 0);
        // hydrazine::json::Array *params = main.parse<hydrazine::json::Array*>("param", nullptr);
        auto paramsVistor = main["params"];
        auto params = static_cast<hydrazine::json::Array *>(paramsVistor.value);
        for (auto&& value: *params) {
            paramVector.push_back(stringToDataType(value->as_string()));
        }
        auto sizeVistor = main["size"];
        auto size = static_cast<hydrazine::json::Array *>(sizeVistor.value);
        for (auto value: *size) {
            // paramVector.push_back(stringToDataType(value->as_string()));
            sizeVector.push_back(initialize_dim3(hydrazine::json::Visitor(value)));
        }
        
	}
	catch (hydrazine::Exception exp) {
		std::cerr << "==Ocelot== WARNING: Could not parse config file '" 
			<< path << "', loading defaults.\n" << std::endl;
			
		std::cerr << "exception:\n" << exp.what() << std::endl;
	}
	
}

ir::Dim3 PTXKernelConfig::initialize_dim3(hydrazine::json::Visitor vistor) {
    int x, y, z;
    x = vistor.parse<int>("x", 1);
    y = vistor.parse<int>("y", 1);
    z = vistor.parse<int>("z", 1);
    return ir::Dim3(x, y ,z);
}

ir::PTXOperand::DataType PTXKernelConfig::stringToDataType(std::string str) {
    using DataType = ir::PTXOperand::DataType;
    static const std::unordered_map<std::string, DataType> stringToEnumMap = {
        {"TypeSpecifier_invalid", DataType::TypeSpecifier_invalid},
        {"s8", DataType::s8},
        {"s16", DataType::s16},
        {"s32", DataType::s32},
        {"s64", DataType::s64},
        {"u8", DataType::u8},
        {"u16", DataType::u16},
        {"u32", DataType::u32},
        {"u64", DataType::u64},
        {"f16", DataType::f16},
        {"f32", DataType::f32},
        {"f64", DataType::f64},
        {"b8", DataType::b8},
        {"b16", DataType::b16},
        {"b32", DataType::b32},
        {"b64", DataType::b64},
        {"pred", DataType::pred}
    };
    auto it = stringToEnumMap.find(str);
    if (it != stringToEnumMap.end()) {
        return it->second;
    } else {
        return DataType::TypeSpecifier_invalid;
    }
}
#endif /* PTXKERNELCONFIG_H */

