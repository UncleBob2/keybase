import unittest
import json
import os
import sys
import argparse
from unittest.mock import patch, mock_open, MagicMock

# It's good practice to ensure that the module we are testing is imported 
# *after* potential mocks are set up if the module-level code performs actions 
# like API configuration immediately upon import. However, for this specific structure,
# gemini_processor.py's API setup is guarded by try-except and prints, which we'll intercept.
# We will import specific functions or classes as needed, or the whole module and reload it.
import gemini_processor 

# Helper to load sample file content
def load_sample_text(filename):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Assuming input_files is in the same directory as the project root,
    # and tests are run from the project root.
    # If input_files is relative to the test script, adjust path.
    # For the sandbox, assuming it's relative to the /app root.
    file_path = os.path.join(os.getcwd(), "input_files", filename)
    if not os.path.exists(file_path): # Fallback if tests are in a subdirectory
        file_path = os.path.join(base_dir, "..", "input_files", filename)

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Test setup error: Could not find {filename} at {file_path}")
        return None

class TestAPIKeyLoading(unittest.TestCase):
    # To test module-level code, we might need to reload the module under different mock conditions.
    # This is complex. A simpler way for this script is to put the API init logic into a function.
    # Since it's not, we'll patch os.getenv and genai.configure and then call a function that uses them,
    # or re-evaluate the module level code by reloading.
    # For simplicity, we assume the current structure where genai.configure is called at module level.
    # We will patch 'os.getenv' and 'genai.configure' and then 'importlib.reload(gemini_processor)'

    @patch('gemini_processor.genai.configure')
    @patch('gemini_processor.os.getenv')
    @patch('builtins.print')
    @patch('sys.exit')
    def test_api_key_loaded_successfully(self, mock_exit, mock_print, mock_getenv, mock_genai_configure):
        mock_getenv.return_value = "TEST_API_KEY"
        # Reloading the module to trigger the API key loading logic
        import importlib
        importlib.reload(gemini_processor)
        
        mock_getenv.assert_called_with("GOOGLE_API_KEY")
        mock_genai_configure.assert_called_with(api_key="TEST_API_KEY")
        mock_exit.assert_not_called()

    @patch('gemini_processor.os.getenv')
    @patch('gemini_processor.genai.configure')
    @patch('builtins.print')
    @patch('sys.exit')
    def test_api_key_missing(self, mock_exit, mock_print, mock_getenv, mock_genai_configure):
        mock_getenv.return_value = None # Simulate API key not found
        
        import importlib
        importlib.reload(gemini_processor)

        mock_getenv.assert_called_with("GOOGLE_API_KEY")
        mock_genai_configure.assert_not_called() # Should not be called if key is missing
        mock_print.assert_any_call("Error: GOOGLE_API_KEY not found in .env file or environment variables, or it's empty.")
        mock_exit.assert_called_with(1)

    @patch('gemini_processor.os.getenv')
    @patch('gemini_processor.genai.configure')
    @patch('builtins.print')
    @patch('sys.exit')
    def test_api_key_empty(self, mock_exit, mock_print, mock_getenv, mock_genai_configure):
        mock_getenv.return_value = "" # Simulate API key is empty
        
        import importlib
        importlib.reload(gemini_processor)

        mock_getenv.assert_called_with("GOOGLE_API_KEY")
        mock_genai_configure.assert_not_called()
        mock_print.assert_any_call("Error: GOOGLE_API_KEY not found in .env file or environment variables, or it's empty.")
        mock_exit.assert_called_with(1)


class TestArgParsing(unittest.TestCase):

    @patch('gemini_processor.os.path.isdir') 
    @patch('gemini_processor.os.makedirs')
    @patch('gemini_processor.glob.glob') # To prevent file operations
    @patch('builtins.print') # To suppress prints from main
    def run_main_with_args(self, args_list, mock_print, mock_glob, mock_makedirs, mock_isdir):
        """Helper to run main with mocked args and essential file ops."""
        mock_isdir.return_value = True # Assume input dir exists for parsing tests
        mock_glob.return_value = []    # No files found, to stop early
        with patch.object(sys, 'argv', ['gemini_processor.py'] + args_list):
            gemini_processor.main()

    @patch('gemini_processor.argparse.ArgumentParser.parse_args')
    def test_argparse_defaults(self, mock_parse_args):
        # Simulate no command-line arguments (default behavior)
        mock_parse_args.return_value = argparse.Namespace(input_dir="input_files/", output_dir="output_json/")
        
        # We need to check that these defaults are used.
        # This requires deeper mocking of what main() does with these paths.
        # For now, we check parse_args is called and what it would return.
        # A more complete test would be in TestMainFunction.
        
        # This test is more about ensuring our setup of ArgumentParser is correct.
        parser = gemini_processor.argparse.ArgumentParser() # Get a new parser instance like in the script
        parser.add_argument("-i", "--input_dir", default="input_files/", help="Input directory")
        parser.add_argument("-o", "--output_dir", default="output_json/", help="Output directory")
        args = parser.parse_args([]) # Parse empty list
        
        self.assertEqual(args.input_dir, "input_files/")
        self.assertEqual(args.output_dir, "output_json/")

    @patch('gemini_processor.argparse.ArgumentParser.parse_args')
    def test_argparse_custom_values(self, mock_parse_args):
        custom_input = "./custom_in"
        custom_output = "./custom_out"
        mock_parse_args.return_value = argparse.Namespace(input_dir=custom_input, output_dir=custom_output)

        parser = gemini_processor.argparse.ArgumentParser()
        parser.add_argument("-i", "--input_dir", default="input_files/", help="Input directory")
        parser.add_argument("-o", "--output_dir", default="output_json/", help="Output directory")
        args = parser.parse_args(['-i', custom_input, '-o', custom_output])

        self.assertEqual(args.input_dir, custom_input)
        self.assertEqual(args.output_dir, custom_output)

    # More integrated tests for argument parsing are in TestMainFunction


class TestProcessTextFile(unittest.TestCase):

    def setUp(self):
        # Mock the model object within gemini_processor
        self.mock_model = MagicMock()
        # This patch assumes 'model' is a global in gemini_processor module
        self.model_patcher = patch('gemini_processor.model', self.mock_model)
        self.model_patcher.start()

    def tearDown(self):
        self.model_patcher.stop()

    def test_successful_api_call(self):
        dummy_filepath = "dummy/path/to/file.txt"
        dummy_content = "This is some sample text."
        expected_json_string = '{"key": "value"}'

        # Mock the response from model.generate_content()
        mock_response = MagicMock()
        mock_response.text = expected_json_string
        # Simulate the structure for non-empty candidates/parts
        mock_candidate = MagicMock()
        mock_part = MagicMock()
        mock_part.text = expected_json_string 
        mock_candidate.content.parts = [mock_part]
        mock_response.candidates = [mock_candidate]
        
        self.mock_model.generate_content.return_value = mock_response

        result = gemini_processor.process_text_file(dummy_filepath, dummy_content)

        self.mock_model.generate_content.assert_called_once()
        # You could add more detailed assertions about the call arguments here if needed
        # e.g., checking parts of the SYSTEM_PROMPT or the generation_config
        self.assertEqual(result, expected_json_string)

    @patch('builtins.print')
    def test_api_call_empty_response(self, mock_print):
        dummy_filepath = "dummy/file.txt"
        dummy_content = "Content"
        
        mock_response = MagicMock()
        mock_response.candidates = [] # Simulate empty candidates
        self.mock_model.generate_content.return_value = mock_response

        result = gemini_processor.process_text_file(dummy_filepath, dummy_content)
        
        self.assertIsNone(result)
        mock_print.assert_any_call(f"Warning: Received an empty or unexpected response from the API for {dummy_filepath}.")

    @patch('builtins.print')
    def test_api_call_raises_exception(self, mock_print):
        dummy_filepath = "dummy/file.txt"
        dummy_content = "Content"
        
        # Simulate an API error
        # Using a generic Exception for simplicity, but could be specific google.api_core.exceptions error
        self.mock_model.generate_content.side_effect = Exception("API Communication Error")

        result = gemini_processor.process_text_file(dummy_filepath, dummy_content)

        self.assertIsNone(result)
        mock_print.assert_any_call(f"Error during API call for {dummy_filepath}: API Communication Error")


class TestValidateJsonOutput(unittest.TestCase): # Copied from previous content

    @classmethod
    def setUpClass(cls):
        cls.sample_case_01_text = load_sample_text("sample_case_01.txt")
        # cls.humphreys_text = load_sample_text("humphreys_v_hanne.txt") # Not used in these tests yet
        
        if not cls.sample_case_01_text:
            raise unittest.SkipTest("Skipping tests: sample_case_01.txt not found.")

    def get_base_valid_structure(self, original_text_content):
        """Helper to create a minimally valid JSON structure based on the new validator."""
        return {
            "structured_data": {
                "citation": "Test Citation", "date": "2024-07-31", "file_number": "TEST/2024/001",
                "registry": "Test Registry", "court": "Test Court", "document_type": "Test Judgment",
                "summary": "This is a test summary.", "outcome_summary": "Test outcome achieved.",
                "procedural_history": [{"court_name": "LC", "judge_or_judges": "LJ", "date_of_action": "2023-01-01", "description_of_action": "Filed", "prior_citation": None}],
                "legal_issues": ["Issue 1?"], "legal_concepts": ["Concept 1"],
                "holding_decision": [{"issue_addressed": "Issue 1?", "holding": "Yes", "disposition_on_issue": "Affirmed"}],
                "reasoning_for_holdings": [{"holding_addressed": "Yes", "reasoning_text": "Because.", "key_reasoning_paragraphs": ["1"]}],
                "key_findings_and_reasoning": [{"finding_phrase": "Finding 1", "supporting_paragraphs": ["1"], "relevant_concepts": ["Concept 1"]}],
                "parties": [{"role": "P", "name": "Plaintiff"}], "judges": ["Judge A"],
                "counsel": [{"name": "Counsel A", "represents": "Plaintiff"}],
                "logical_sections": {"introduction": ["1"], "facts": ["2"], "analysis": ["3"], "conclusion": ["4"]},
                "paragraphs": [{
                    "number": "1", "text": "Para 1 text.", "sentences": ["Sentence 1."],
                    "key_phrases_local": ["phrase 1"], 
                    "entities_local": {"people": [], "organizations": [], "dates": [], "locations": [], "legal_terms": []}
                }]
            },
            "original_text": original_text_content
        }

    def test_valid_json_perfect_match_new_validator(self):
        valid_data = self.get_base_valid_structure(self.sample_case_01_text)
        json_string = json.dumps(valid_data)
        is_valid, parsed_data = gemini_processor.validate_json_output(json_string, self.sample_case_01_text)
        self.assertTrue(is_valid, f"Validation failed for a valid JSON. Errors printed by validator might show why.")
        self.assertEqual(parsed_data, valid_data)
    
    # ... (Keep other TestValidateJsonOutput tests as they are, they should still be relevant) ...
    # Note: Some tests might need adjustment if the new validator is stricter/different for certain edge cases
    # compared to the old one. For example, the new validator is strict about list/dict types vs null.

    def test_malformed_json(self):
        is_valid, parsed_data = gemini_processor.validate_json_output("this is not json", self.sample_case_01_text)
        self.assertFalse(is_valid)
        self.assertIsNone(parsed_data)

    def test_missing_structured_data_key(self):
        invalid_data = {"original_text": self.sample_case_01_text}
        json_string = json.dumps(invalid_data)
        is_valid, parsed_data = gemini_processor.validate_json_output(json_string, self.sample_case_01_text)
        self.assertFalse(is_valid)

    def test_original_text_mismatch(self):
        valid_data = self.get_base_valid_structure(self.sample_case_01_text)
        json_string = json.dumps(valid_data)
        is_valid, _ = gemini_processor.validate_json_output(json_string, "Different text")
        self.assertFalse(is_valid)
        
    def test_field_should_be_empty_list_is_null(self):
        data = self.get_base_valid_structure(self.sample_case_01_text)
        data["structured_data"]["legal_issues"] = None 
        json_string = json.dumps(data)
        is_valid, _ = gemini_processor.validate_json_output(json_string, self.sample_case_01_text)
        self.assertFalse(is_valid, "Validator should require list, not null, for legal_issues.")

    def test_procedural_history_judge_is_list(self):
        data = self.get_base_valid_structure(self.sample_case_01_text)
        data["structured_data"]["procedural_history"][0]["judge_or_judges"] = ["Judge A", "Judge B"]
        json_string = json.dumps(data)
        is_valid, _ = gemini_processor.validate_json_output(json_string, self.sample_case_01_text)
        self.assertTrue(is_valid, "judge_or_judges being a list of strings should be valid.")

    def test_procedural_history_judge_is_null(self): # Allowed by schema
        data = self.get_base_valid_structure(self.sample_case_01_text)
        data["structured_data"]["procedural_history"][0]["judge_or_judges"] = None
        json_string = json.dumps(data)
        is_valid, _ = gemini_processor.validate_json_output(json_string, self.sample_case_01_text)
        self.assertTrue(is_valid, "judge_or_judges being null should be valid.")


class TestMainFunction(unittest.TestCase):

    @patch('gemini_processor.os.path.isdir')
    @patch('gemini_processor.os.makedirs')
    @patch('gemini_processor.glob.glob')
    @patch('builtins.open', new_callable=mock_open)
    @patch('gemini_processor.process_text_file')
    @patch('gemini_processor.validate_json_output')
    @patch('builtins.print') # Suppress print statements from main
    @patch('sys.exit') # To ensure exit is not called in success case
    def test_successful_end_to_end_one_file(
        self, mock_sys_exit, mock_print, mock_validate, mock_process, 
        mock_file_open, mock_glob, mock_makedirs, mock_isdir):

        mock_isdir.return_value = True  # Input directory exists
        dummy_filepath = "input_files/dummy.txt"
        mock_glob.return_value = [dummy_filepath] # Found one file
        
        dummy_content = "dummy file content"
        mock_file_open.return_value.read.return_value = dummy_content
        
        api_response_json_string = '{"structured_data": {...}, "original_text": "dummy file content"}' # Minimal
        mock_process.return_value = api_response_json_string
        
        parsed_valid_data = json.loads(api_response_json_string) # Assume it's valid
        parsed_valid_data["structured_data"] = TestValidateJsonOutput().get_base_valid_structure(dummy_content)["structured_data"] # ensure it has valid structure
        api_response_json_string = json.dumps(parsed_valid_data)
        mock_process.return_value = api_response_json_string


        mock_validate.return_value = (True, parsed_valid_data)

        with patch.object(sys, 'argv', ['gemini_processor.py', '-i', 'input_files', '-o', 'output_json']):
            gemini_processor.main()

        mock_isdir.assert_called_with("input_files/") # Check if input_dir from args is used
        mock_makedirs.assert_called_with("output_json/", exist_ok=True)
        mock_glob.assert_called_with(os.path.join("input_files/", "*.txt"))
        
        mock_file_open.assert_any_call(dummy_filepath, 'r', encoding='utf-8') # Reading input
        mock_process.assert_called_with(dummy_filepath, dummy_content)
        mock_validate.assert_called_with(api_response_json_string, dummy_content)
        
        expected_output_path = os.path.join("output_json/", "dummy.json")
        mock_file_open.assert_any_call(expected_output_path, 'w', encoding='utf-8') # Writing output
        
        # Check if json.dump was called (indirectly via handle().write())
        # The first call to mock_file_open is read, second is write
        # This is a bit fragile; depends on call order.
        # A better way is to separate read and write mocks if they have different behaviors.
        mock_file_open.return_value.write.assert_called_once_with(json.dumps(parsed_valid_data, indent=2, ensure_ascii=False))
        
        mock_sys_exit.assert_not_called()
        mock_print.assert_any_call(f"Successfully processed and saved '{expected_output_path}'")

    @patch('gemini_processor.os.path.isdir')
    @patch('gemini_processor.os.makedirs')
    @patch('gemini_processor.glob.glob')
    @patch('builtins.open', new_callable=mock_open, read_data="dummy content")
    @patch('gemini_processor.process_text_file')
    @patch('builtins.print')
    @patch('sys.exit')
    def test_main_api_error(self, mock_exit, mock_print, mock_process, mock_open_file, mock_glob, mock_makedirs, mock_isdir):
        mock_isdir.return_value = True
        mock_glob.return_value = ["input_files/error_file.txt"]
        mock_process.return_value = None # Simulate API error

        with patch.object(sys, 'argv', ['gemini_processor.py']): # Use default dirs
            gemini_processor.main()
        
        mock_print.assert_any_call("Failed to get API response for error_file.txt.")
        # Assert that no output file was attempted for writing (open not called with 'w')
        write_calls = [call for call in mock_open_file.call_args_list if call[0][1] == 'w']
        self.assertEqual(len(write_calls), 0)

    @patch('gemini_processor.os.path.isdir')
    @patch('gemini_processor.os.makedirs')
    @patch('gemini_processor.glob.glob')
    @patch('builtins.open', new_callable=mock_open, read_data="dummy content")
    @patch('gemini_processor.process_text_file')
    @patch('gemini_processor.validate_json_output')
    @patch('builtins.print')
    @patch('sys.exit')
    def test_main_validation_error(self, mock_exit, mock_print, mock_validate, mock_process, mock_open_file, mock_glob, mock_makedirs, mock_isdir):
        mock_isdir.return_value = True
        mock_glob.return_value = ["input_files/invalid_json_file.txt"]
        mock_process.return_value = '{"bad": "json"}' # Valid JSON string, but content is invalid
        mock_validate.return_value = (False, None)   # Simulate validation failure

        with patch.object(sys, 'argv', ['gemini_processor.py']):
            gemini_processor.main()

        mock_print.assert_any_call("Validation failed for invalid_json_file.txt. Output not saved.")
        write_calls = [call for call in mock_open_file.call_args_list if call[0][1] == 'w']
        self.assertEqual(len(write_calls), 0)

    @patch('gemini_processor.os.path.isdir')
    @patch('builtins.print')
    @patch('sys.exit') # Not strictly needed if main returns, but good for consistency
    def test_main_input_dir_not_found(self, mock_exit, mock_print, mock_isdir):
        mock_isdir.return_value = False # Simulate input directory does not exist
        
        with patch.object(sys, 'argv', ['gemini_processor.py', '-i', 'non_existent_dir']):
            gemini_processor.main()
            
        mock_print.assert_any_call("Error: Input directory 'non_existent_dir/' not found.")
        # mock_exit.assert_called_with(1) # Or check if main simply returns

    @patch('gemini_processor.os.path.isdir')
    @patch('gemini_processor.os.makedirs')
    @patch('gemini_processor.glob.glob')
    @patch('builtins.print')
    def test_main_no_txt_files_found(self, mock_print, mock_glob, mock_makedirs, mock_isdir):
        mock_isdir.return_value = True
        mock_glob.return_value = [] # Simulate no .txt files found
        
        with patch.object(sys, 'argv', ['gemini_processor.py']):
            gemini_processor.main()
            
        mock_print.assert_any_call("No .txt files found in 'input_files/'.")


if __name__ == '__main__':
    # unittest.main() # This will run all tests in the file
    # To run specific tests, you can use:
    # suite = unittest.TestSuite()
    # suite.addTest(TestAPIKeyLoading('test_api_key_loaded_successfully'))
    # runner = unittest.TextTestRunner()
    # runner.run(suite)
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
