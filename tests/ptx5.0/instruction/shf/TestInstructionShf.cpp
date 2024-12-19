#include <ptx_test/TestInstruction.h>
#include <cstdint>
#include <iostream>
#include <vector>




namespace test {
    class TestInstructionShf : public TestInstruction {
    public:

        TestInstructionShf() {
            name = "TestInstructionShf";

            description = "A unit test for the SHF instruction.";
            description += " Test Points: 1) Execute a kernel with a loop. ";
            description += "2) Execute a matrix multiply kernel.";
        }
    protected:


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
	test::TestInstructionShf test;
	parser.description( test.testDescription() );

	parser.parse("-c", test.configPath, "../instruction/shf/config_shr_l_clamp.test", "Test configuration path.");
	parser.parse( "-i", test.input, "../instruction/shf/test_shf.ptx",
		"Test PTX path.");
	parser.parse( "-r", test.recursive, true, 
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