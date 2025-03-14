
#include <ocelot/executive/LLVMModuleManager.h>
#include <ocelot/executive/ExecutableKernel.h>
#include <ocelot/executive/LLVMExecutionManager.h>
#include <ocelot/ir/ControlFlowGraph.h>
#include <ocelot/ir/Module.h>
#include <ocelot/ir/LLVMKernel.h>
#include <ocelot/api/OcelotConfiguration.h>
#include <ocelot/translator/PTXToLLVMTranslator.h>
#include <ocelot/transforms/PassManager.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Linker/Linker.h>
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include <llvm/Transforms/Scalar.h>
#include <llvm/IR/LegacyPassManager.h>
#include "ocelot/ir/PTXKernel.h"

#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/AsmParser/Parser.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/Scalar/GVN.h>
#include <llvm/Transforms/Scalar/SimpleLoopUnswitch.h>
#include <llvm/Transforms/InstCombine/InstCombine.h>
// Hydrazine Includes
#include <hydrazine/Thread.h>

// Standard Library Includes
#include <vector>
#include <iostream>
using namespace executive;
using namespace ir;

namespace llvm {

    LLVMContext &getGlobalContext();
    
} // namespace llvm

static void translate(llvm::Module*& module, ir::PTXKernel& kernel,
	translator::Translator::OptimizationLevel optimization,
	const ir::ExternalFunctionSet& externals)
{
	assert(module == 0);

	report(" Translating kernel.");
	
	report("  Converting from PTX IR to LLVM IR.");
	translator::PTXToLLVMTranslator translator(optimization, &externals);

	transforms::PassManager manager(const_cast<ir::Module*>(kernel.module));
	
	manager.addPass(&translator);
	manager.runOnKernel(kernel);
	manager.releasePasses();

	ir::LLVMKernel* llvmKernel = static_cast<ir::LLVMKernel*>(
		translator.translatedKernel());
	
	report("  Assembling LLVM kernel.");
	llvmKernel->assemble();
	llvm::SMDiagnostic error;

	report("  Parsing LLVM assembly.");
	module = llvm::parseAssemblyString(llvmKernel->code().c_str(), 
		error, llvm::getGlobalContext()).release();

	if(module == 0)
	{
		report("   Parsing kernel failed, dumping code:\n" 
			<< llvmKernel->numberedCode());
		std::string m;
		llvm::raw_string_ostream message(m);
		message << "LLVM Parser failed: ";
		error.print(kernel.name.c_str(), message);

		throw hydrazine::Exception(message.str());
	}

	report("  Checking llvm module for errors.");
	std::string verifyError;
	llvm::raw_string_ostream verifyOutput(verifyError);	
	if(llvm::verifyModule(*module, &verifyOutput))
	{
		report("   Checking kernel failed, dumping code:\n" 
			<< llvmKernel->numberedCode());
		delete llvmKernel;
		delete module;
		module = 0;

		throw hydrazine::Exception("LLVM Verifier failed for kernel: " 
			+ kernel.name + " : \"" + verifyError + "\"");
	}

	module->setModuleIdentifier(kernel.name.c_str());
	std::error_code EC;
	llvm::raw_fd_ostream OS(kernel.name + ".ll", EC);
	// module->print(OS, nullptr);
	// llvm::raw_fd_ostream OS2("module.bc", EC);
	// llvm::WriteBitcodeToFile(*module, OS2);
	delete llvmKernel;
}

void translateModule(const std::string& inputPath, const std::string& outputPath) {
    ir::Module module;
	ir::PTXKernel* kernel;
    module.load(inputPath);
    auto optLevel = translator::Translator::OptimizationLevel::NoOptimization; // or other optimization levels
    LLVMState::moduleManager()->loadModule(&module, optLevel, 0);

    // translate all kernels
    auto kernels = module.kernels();
    llvm::Module* llvmModule = 0;
    ir::ExternalFunctionSet* externel = 0;
    for (auto it = kernels.begin(); it != kernels.end(); ++it) {
        kernel = it->second;
		llvm::Module* transltaedKernel = 0;
        translate(transltaedKernel, *kernel, optLevel, *externel);
		if (transltaedKernel) {
			if (!llvmModule) {
				llvmModule = transltaedKernel;
				continue;
			}
			// link the translated LLVM module with the main module
			bool err = llvm::Linker::linkModules(*llvmModule, std::unique_ptr<llvm::Module>(transltaedKernel));
			if (err) {
				std::cerr << "Failed to link LLVM modules" << std::endl;
				return;
			}
		}
    }
	// std::error_code EC;
	// llvm::raw_fd_ostream OS("test.ll", EC);
	// std::cout << "Writing LLVM IR to test.ll" << std::endl;
	// for (llvm::Function &F : *llvmModule) {
	// 	llvm::outs() << F.getName() << "\n"; // Print each function to the output stream
	// }
	// llvmModule->print(OS, nullptr);

}

void executeLLVMModuleKernel(const std::string& kernelName) {

}

int main() {
	translateModule("./test_const_memory.ptx", "output.ll");
	return 0;
}