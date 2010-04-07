/*! \file Device.cpp
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date Jan 16, 2009
	\brief The source file for the Device class
*/

#include <ocelot/executive/interface/Device.h>

executive::Device::Properties::Properties(int id) {
	ISA = ir::Instruction::Emulated;
	name = "PTX Emulator";
	guid = id;
	totalMemory = (1 << 22);
	multiprocessorCount = 4;
	maxThreadsPerBlock = 768;
	maxThreadsDim[0] = maxThreadsDim[1] = maxThreadsDim[2] = maxThreadsPerBlock;
	maxGridSize[0] = maxGridSize[1] = maxGridSize[2] = 65536;
	sharedMemPerBlock = 16384;
	totalConstantMemory = 8192;
	SIMDWidth = 32;
	memPitch = (4<<10);
	regsPerBlock = 8192;
	clockRate = 2400000;
	textureAlign = 16;
	addressSpace = 0;
}

std::ostream& executive::Device::Properties::write(std::ostream &out) const {
	out << name << "( " << guid << " ):\n";
	out << "  " << "total memory: " << (totalMemory >> 10) << " kB\n";
	out << "  " << "ISA: " << ir::Instruction::toString(ISA) << "\n";
	out << "  " << "multiprocessors: " << multiprocessorCount << "\n";
	out << "  " << "max threads: " << maxThreadsPerBlock << "\n";
	out << "  " << "shared memory: " << (sharedMemPerBlock >> 10) << " kB\n";
	out << "  " << "const memory: " << (totalConstantMemory >> 10) << " kB\n";
	out << "  " << "SIMD width: " << SIMDWidth << "\n";
	out << "  " << "regs per block: " << regsPerBlock << "\n";
	out << "  " << "clock rate: " << clockRate << " Hz\n";
	return out;
}

executive::Device::Device(int guid) : _properties(guid), 
	_driverVersion(0), _runtimeVersion(0) {
}

executive::Device::~Device() {
}

std::string executive::Device::nearbyAllocationsToString(void* pointer) const {
	std::stringstream result;
	MemoryAllocationVector allocations = getNearbyAllocations(pointer);
	
	for(MemoryAllocationVector::iterator allocation = allocations.begin(); 
		allocation != allocations.end(); ++allocation)
	{
		result << "[" << (*allocation)->pointer() << "] - [" 
			<< ((char*)(*allocation)->pointer() + (*allocation)->size()) 
			<< "]\n";
	}
	
	return result.str();
}

const executive::Device::Properties& executive::Device::properties() const {
	return _properties;
}

int executive::Device::driverVersion() const {
	return _driverVersion;
}

int executive::Device::runtimeVersion() const {
	return _runtimeVersion;
}
