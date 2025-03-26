/*! \file LLVMState.h
	\date Friday September 24, 2010
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief The header file for the LLVMState class.
*/

#ifndef LLVM_STATE_H_INCLUDED
#define LLVM_STATE_H_INCLUDED

// #include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"

namespace llvm
{
	
	class ExecutionEngine;
	class LLVMContext;
	class Module;
	class TargetMachine;
	namespace orc{
		class JITTargetMachineBuilder;
		class LLJIT;
	}
}

namespace executive
{

class LLVMModuleManager;

/*! \brief A class for managing global llvm state */
class LLVMState
{
public:
	/*! \brief Get a reference to the LLVM context */
	static llvm::LLVMContext* context();

	/*! \brief Get a reference to the jit */
	static llvm::ExecutionEngine* jit();

	/*! \brief Get a reference to the module manager */
	static LLVMModuleManager* moduleManager();

	/*! \brief OrcJIT engine */
	static llvm::orc::LLJIT* orcjit(); 

	/*! \brief OrcJIT target machine */
	static llvm::TargetMachine* targetMachine(); 

	LLVMState(LLVMState const&) = delete;
	LLVMState& operator=(LLVMState const&) = delete;

	~LLVMState();

private:

	static LLVMState& get();

	/*! \brief Build the jit */
	LLVMState();

	/*! \brief LLVM context */
	llvm::LLVMContext* _context;

	/*! \brief LLVM JIT Engine */
	llvm::ExecutionEngine* _jit;
	
	/*! \brief LLVM fake mofule */
	llvm::Module* _module;

	/*! \brief OrcJIT engine */
	llvm::orc::LLJIT* _orcjit;

	/*! \brief OrcJIT target machine */
	llvm::TargetMachine* _targetMachine;

	LLVMModuleManager* _manager;
};

}

#endif

