#include <ptx_test/TestInstruction.h>



namespace test {
    class TestInstructionSubC : public TestInstruction {
    public:

        TestInstructionSubC() {
            name = "TestInstructionSub";

            description = "A unit test for the SUB instruction.";
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
	test::TestInstructionSubC test;
	parser.description( test.testDescription() );

	parser.parse("-c", test.configPath, "../instruction/sub/config.test", "Test configuration path.");
	parser.parse( "-i", test.input, "../instruction/sub/test_sub.ptx",
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