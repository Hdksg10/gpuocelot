/*! \file LLVMState.cpp
	\date Friday September 24, 2010
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief The header file for the LLVMState class.
*/
#include <llvm/Support/Error.h>
#include <llvm/Support/Host.h>
#include <llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
// Ocelot Includes
#include <ocelot/executive/LLVMState.h>
#include <ocelot/executive/LLVMModuleManager.h>

// Hydrazine Includes
#include <hydrazine/debug.h>

#include <llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include "llvm/ExecutionEngine/MCJIT.h"


#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>

// Preprocessor Defines
#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 0

namespace llvm {

// https://reviews.llvm.org/D19094
LLVMContext& getGlobalContext()
{
	return *executive::LLVMState::context();
}

} // namespace llvm

namespace executive
{

LLVMState& LLVMState::get()
{
	static LLVMState s;
	return s;
}

llvm::ExecutionEngine* LLVMState::jit()
{
	return LLVMState::get()._jit;
}

llvm::LLVMContext* LLVMState::context()
{
	return LLVMState::get()._context;
}

llvm::orc::LLJIT* LLVMState::orcjit()
{
	return LLVMState::get()._orcjit;
}

LLVMModuleManager* LLVMState::moduleManager()
{
	return LLVMState::get()._manager;
}

llvm::orc::ThreadSafeContext* LLVMState::threadSafeContext()
{
	return LLVMState::get()._tsc;	
}



LLVMState::LLVMState() : _jit(0), _context(0), _module(0), _tsc(0), _orcjit(0)
{
	_context = new llvm::LLVMContext();

	_tsc = new llvm::orc::ThreadSafeContext(std::make_unique<llvm::LLVMContext>());	

	_context = _tsc->getContext();

	report("Bringing the LLVM JIT-Compiler online.");

	// https://stackoverflow.com/a/38801376/4063520	
	llvm::InitializeNativeTarget();
	llvm::InitializeNativeTargetAsmParser();
	llvm::InitializeNativeTargetAsmPrinter();

	auto m = std::make_unique<llvm::Module>("Ocelot-LLVM-JIT-Blank Module", *_context);
	_module = m.get();
	assertM(_module != 0, "Creating global module failed.");

	llvm::TargetOptions Opts;
	llvm::RTDyldMemoryManager* MemMgr = new llvm::SectionMemoryManager();
	llvm::EngineBuilder factory(std::move(m));
	factory.setEngineKind(llvm::EngineKind::JIT);
	factory.setTargetOptions(Opts);
	factory.setMCJITMemoryManager(std::unique_ptr<llvm::RTDyldMemoryManager>(MemMgr));
	_jit = factory.create();
	_jit->DisableLazyCompilation(true);

	std::string TargetTriple = llvm::sys::getProcessTriple();
	std::string CPU = llvm::sys::getHostCPUName().str();
	#if defined(__riscv)
		// manually set RISC-V features
		report(" Set RISC-V Target Machine");
		auto jtmb = llvm::orc::JITTargetMachineBuilder(llvm::Triple(TargetTriple));
		jtmb.setCPU(CPU);
		jtmb.addFeatures({"+m", "+a", "+f", "+d", "+c"});
	#else
		auto jtmb = llvm::cantFail(llvm::orc::JITTargetMachineBuilder::detectHost());
	#endif
	for (const auto &Feature : jtmb.getFeatures().getFeatures()) {
        std::cout << Feature << " ";
    }
	auto orcjitExpected = llvm::orc::LLJITBuilder().setJITTargetMachineBuilder(jtmb).create();
    assertM(orcjitExpected, "Creating the OrcJIT failed.");
	_orcjit = orcjitExpected->release();
	

	// assertM(_targetMachine != 0, "Creating target machine failed.");
	assertM(_jit != 0, "Creating the JIT failed.");
	report(" The JIT is alive.");

	_manager = new LLVMModuleManager();
}

LLVMState::~LLVMState()
{
	delete _manager;
	delete _context;
}

} // namespace executive

