/*! \file LLVMState.cpp
	\date Friday September 24, 2010
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief The header file for the LLVMState class.
*/

#ifndef LLVM_STATE_CPP_INCLUDED
#define LLVM_STATE_CPP_INCLUDED

// Ocelot Includes
#include <ocelot/executive/LLVMState.h>

// Hydrazine Includes
#include <hydrazine/debug.h>

#include <llvm/Support/TargetSelect.h>
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/ExecutionEngine/JIT.h> 
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>

// Preprocessor Defines
#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 0

namespace executive
{

llvm::ExecutionEngine* LLVMState::jit()
{
	return _wrapper.jit();
}

LLVMState::StateWrapper::StateWrapper() : _jit(0), _module(0)
{

}

llvm::ExecutionEngine* LLVMState::StateWrapper::jit()
{
	if(_jit == 0)
	{
		report("Bringing the LLVM JIT-Compiler online.");

		_module = new llvm::Module("Ocelot-LLVM-JIT-Blank Module", 
			llvm::getGlobalContext());
		assertM(_module != 0, "Creating global module failed.");
	
		llvm::InitializeNativeTarget();
		
		_jit = llvm::EngineBuilder(_module).create();
		_jit->DisableLazyCompilation(true);
	
		assertM(_jit != 0, "Creating the JIT failed.");
		report(" The JIT is alive.");
	}
	return _jit;
}

LLVMState::StateWrapper::~StateWrapper()
{
	// no need to delete anything, llvm will handle it for us
}

LLVMState::StateWrapper LLVMState::_wrapper;

}

#endif
