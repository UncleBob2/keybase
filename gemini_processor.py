"""
Processes legal text files from an input directory using the Google Gemini API (gemini-1.5-flash-latest model)
to extract structured data based on a predefined system prompt. The output is saved as JSON files
in an output directory.

Prerequisites:
1. Install necessary packages: pip install google-generativeai python-dotenv
2. Create a .env file in the same directory as this script, containing your API key:
   GOOGLE_API_KEY="YOUR_ACTUAL_API_KEY"

Usage:
  python gemini_processor.py [--input_dir INPUT_PATH] [--output_dir OUTPUT_PATH]
  python gemini_processor.py -i ./my_texts -o ./my_json_outputs
  python gemini_processor.py --help (for more details)
"""
import google.generativeai as genai
import os
import json
from dotenv import load_dotenv # Import load_dotenv
import argparse # Add this

# Load environment variables from .env file
load_dotenv() 

# Configure the API key
# Users should create a .env file in the script's directory with:
# GOOGLE_API_KEY="YOUR_API_KEY"
try:
    api_key = os.getenv("GOOGLE_API_KEY") # Use os.getenv, which returns None if not found
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in .env file or environment variables, or it's empty.")
    genai.configure(api_key=api_key)
except ValueError as ve:
    print(f"Error: {ve}")
    print("Please ensure you have a .env file in the script's directory with the line:")
    print("GOOGLE_API_KEY='YOUR_API_KEY'")
    print("Or that the GOOGLE_API_KEY environment variable is set.")
    exit(1)
except Exception as e: # Catch any other unexpected errors during configuration
    print(f"An unexpected error occurred during API configuration: {e}")
    exit(1)

import glob # For finding files

# Initialize the GenerativeModel
# Using gemini-1.5-flash as gemini-2.5-flash is not a recognized model name via the API at this time.
# The model name might need adjustment if a new specific name for "Gemini 2.5 Flash" is released.
# For now, 'gemini-1.5-flash-latest' or 'gemini-1.5-flash-001' are typical. Let's use 'gemini-1.5-flash-latest'.
try:
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    print("Successfully initialized the Gemini 1.5 Flash model.")
except Exception as e:
    print(f"Error initializing the GenerativeModel: {e}")
    print("Please ensure your API key is correct and has access to the 'gemini-1.5-flash-latest' model.")
    exit(1)

# System prompt (will be defined in a later step)
SYSTEM_PROMPT = """
You are an expert legal document processor. Your task is to receive the full text of a legal case document (provided as plain text), process it, and generate a single, valid JSON object as output. This output JSON object must contain two top-level keys: structured_data and original_text.
Input: The complete content of a plain text (.txt) file representing a legal judgment.
Output: A single, valid JSON object. CRITICAL: Do NOT include any explanatory text, markdown formatting (like ```json), or any other content before or after the JSON object. Your entire response must be the JSON itself, structured as defined below.
Processing Steps & Requirements:
Capture Original Text: Store the entire input text exactly as received. This will eventually be placed under the original_text key in the final output JSON.
Process and Structure Data: Perform the following steps based on the input text to generate the data for the structured_data key:
a.  Parse Metadata:
*   Carefully read the beginning of the text document.
*   Identify and extract the following metadata fields: citation, date (standardize to YYYY-MM-DD if possible), file_number, registry, court, document_type, parties (array of objects with name and role), judges (array of strings), counsel (array of objects with name and represents).
b.  Analyze Content (LLM Tasks - Document Level):
*   Based on the entire text content (especially the extracted paragraphs and their sentences):
*   Generate Summaries: Create a concise overall summary and a specific outcome_summary.
*   *** NEW *** Extract Procedural History: Identify and list the key steps in the procedural history of this case. For each step, include the court_name, judge_or_judges involved, date_of_action (YYYY-MM-DD), a brief description_of_action (e.g., "Petition heard, orders granted"), and any prior_citation if available. Store as an array of objects under procedural_history.
*   Identify Key Information: List the main legal_issues (array of strings, phrased as questions the court addressed). Also, list key legal_concepts (array of strings, e.g., "procedural fairness", "constructive trust") discussed in the case.
*   *** NEW *** Determine Holdings for Each Issue: For each distinct legal_issue identified, determine the court's specific holding (the direct answer or ruling on that issue, e.g., "Yes, the judge erred," "No, the RTA did not apply"). Also, state the disposition_on_issue (e.g., "Error found," "Claim dismissed"). Store as an array of objects under holding_decision, where each object contains issue_addressed (string, matching or summarizing an item from legal_issues), holding (string), and disposition_on_issue (string).
*   *** NEW *** Summarize Reasoning for Holdings: For each significant holding in holding_decision, provide a concise summary of the court's primary reasoning_text (why the court reached that holding, referencing key legal principles or facts). Identify the key_reasoning_paragraphs (array of paragraph numbers) that most directly support this reasoning. Store as an array of objects under reasoning_for_holdings, where each object contains holding_addressed (string, referencing the holding), reasoning_text (string), and key_reasoning_paragraphs (array of strings/integers).
*   Identify Logical Sections: Determine paragraph numbers for introduction, facts, analysis, and conclusion. Store these as arrays of paragraph numbers under the logical_sections key.
*   Identify Key Findings and Reasoning (Overall Takeaways): Based on the overall analysis (especially the outcome, legal issues, and core reasoning paragraphs), identify 5-10 concise key findings, significant holdings, or important procedural points of the case (e.g., 'Appellant denied right to make submissions', 'Bona fide triable issues required trial'). For each finding:
*   Determine the specific paragraph numbers (supporting_paragraphs) that provide the primary evidence or reasoning for this finding.
*   Identify the main legal_concepts (from the previously generated list) relevant to this finding.
*   Store these as an array of objects under a new top-level key key_findings_and_reasoning, where each object contains finding_phrase (string), supporting_paragraphs (array of strings/integers matching paragraph numbers), and relevant_concepts (array of strings matching items in the legal_concepts list).
c.  Extract and Segment Paragraphs (Paragraph Level Granularity):
*   Process the body of the judgment.
*   Identify each numbered paragraph (e.g., [1], [2]).
*   For each paragraph:
*   Extract its number (integer or string, matching source).
*   Extract its full text content.
*   Perform sentence segmentation: Accurately split the full paragraph text into individual sentences using standard English sentence boundary detection rules. Store as an array of strings named sentences.
*   Extract Key Local Phrases: Identify and extract 3-7 salient multi-word key phrases directly from this paragraph's text that capture its core topic, action, or concept (e.g., 'significant procedural errors', 'refusing to allow submissions', 'constructive trust claim', 'successor in title'). Store these in an array named key_phrases_local. If no distinct phrases are identifiable, use an empty array [].
*   (Optional but Recommended) Extract Local Entities: Identify named entities (people, organizations, dates, locations) and key legal terms mentioned within this specific paragraph. Store these in an object named entities_local categorized by type (e.g., people: ["Satinder"], dates: ["2014-09-17"], legal_terms: ["procedural errors", "submissions"]). If none are found, use an empty object {}.
*   Store the results for the paragraph (number, text, sentences, key_phrases_local, entities_local) as an object within the paragraphs array.
Structure and Output Final JSON:
Assemble the results into a single JSON object with the following top-level structure. Note the reordered top-level keys (structured_data first, original_text last):
{
  "structured_data": {
    "citation": "string | null",
    "date": "string | null", // Ideally YYYY-MM-DD
    "file_number": "string | null",
    "registry": "string | null",
    "court": "string | null",
    "document_type": "string | null",
    "summary": "string | null", // LLM generated overall summary
    "outcome_summary": "string | null", // LLM generated summary of the outcome
    // *** NEW *** Procedural History Section
    "procedural_history": [
      {
        "court_name": "string | null",
        "judge_or_judges": "string | array | null", // Can be a single judge or list of judges
        "date_of_action": "string | null", // YYYY-MM-DD
        "description_of_action": "string | null",
        "prior_citation": "string | null"
      }
      // ... more procedural steps
    ],
    "legal_issues": [ // Phrased as questions
      "string"
      // ... more issues
    ],
    "legal_concepts": [
      "string"
      // ... more concepts
    ],
    // *** NEW *** Holding/Decision Section
    "holding_decision": [
      {
        "issue_addressed": "string", // Matches or summarizes an item from legal_issues
        "holding": "string", // Direct answer to the issue
        "disposition_on_issue": "string | null" // e.g., "Error found", "Claim dismissed"
      }
      // ... more holdings
    ],
    // *** NEW *** Reasoning for Holdings Section
    "reasoning_for_holdings": [
      {
        "holding_addressed": "string", // References the holding being explained
        "reasoning_text": "string", // Summary of the court's rationale for this specific holding
        "key_reasoning_paragraphs": ["integer | string", ...] // Paragraphs supporting this reasoning
      }
      // ... more reasoning entries
    ],
    "key_findings_and_reasoning": [ // Overall takeaways
      {
        "finding_phrase": "string",
        "supporting_paragraphs": ["integer | string", ...],
        "relevant_concepts": ["string", ...]
      }
      // ... more findings
    ],
    "parties": [
      {
        "role": "string",
        "name": "string"
      }
      // ... more parties
    ],
    "judges": [
      "string"
      // ... more judges
    ],
    "counsel": [
      {
        "name": "string",
        "represents": "string"
      }
      // ... more counsel
    ],
    "logical_sections": {
      "introduction": ["integer | string", ...],
      "facts": ["integer | string", ...],
      "analysis": ["integer | string", ...],
      "conclusion": ["integer | string", ...]
    },
    "paragraphs": [
      {
        "number": "integer | string",
        "text": "string",
        "sentences": [
          "string"
        ],
        "key_phrases_local": [
            "string"
        ],
        "entities_local": { // Optional
            "people": ["string", ...],
            "organizations": ["string", ...],
            "dates": ["string", ...],
            "locations": ["string", ...],
            "legal_terms": ["string", ...]
        }
      }
      // ... more paragraphs
    ]
  },
  "original_text": "string"
}
Use null for fields within structured_data that cannot be found or generated, unless an empty array [] (e.g., for parties, judges, counsel, legal_issues, legal_concepts, sentences, key_phrases_local, supporting_paragraphs, relevant_concepts, key_reasoning_paragraphs, entity arrays, procedural_history, holding_decision, reasoning_for_holdings) or an empty object {} (e.g., for entities_local) is more appropriate.
Ensure the final output is only the valid JSON object conforming to this two-key structure (structured_data first, original_text last). Be meticulous about JSON syntax.
"""

generation_config = genai.types.GenerationConfig(
    temperature=0.5, # Controls randomness
    top_p=0.95,      # Nucleus sampling
    top_k=40,        # Top-k sampling
    max_output_tokens=8192,
    response_mime_type="application/json" # Request JSON output
)

def process_text_file(file_path: str, original_file_content: str): # Modified to accept original_file_content
    print(f"Processing file: {file_path}...")
    try:
        # The system prompt is implicitly used by the model configuration or can be part of the message
        # For explicit system prompt usage with generate_content, it's often passed in the 'system_instruction' field for specific models/versions
        # or included directly in the contents list for multi-turn conversation.
        # Here, we assume the model object `model` has been initialized with system instructions if necessary,
        # or we prepend system prompt to the user's text if that's the preferred method for this SDK version
        # For this task, the prompt asks the LLM to behave as an expert.
        
        messages = [
            {'role':'user', 'parts': [SYSTEM_PROMPT, "\n\nInput Text Document Below:\n", original_file_content]}
        ]
        
        response = model.generate_content(
            messages, # Corrected to pass the messages list
            generation_config=generation_config,
            # stream=False # Not streaming for this use case
        )
        
        # Debug: Print raw response parts if available
        # print(f"Raw response: {response}")
        # if response.parts:
        #     for part in response.parts:
        #         print(f"Response part: {part.text if hasattr(part, 'text') else part}")
        # else:
        #     print("No parts in response.")

        # Check for empty or problematic response before accessing .text
        if not response.candidates or not response.candidates[0].content.parts:
            print(f"Warning: Received an empty or unexpected response from the API for {file_path}.")
            # print(f"Prompt feedback: {response.prompt_feedback}") # If available
            return None

        return response.text # Assuming response.text gives the direct JSON string as requested

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error during API call for {file_path}: {e}")
        # Attempt to print more details from the exception if it's an API error
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            print(f"API Error Details: {e.response.text}")
        return None

def validate_json_output(json_string: str, original_text_content: str) -> tuple[bool, dict | None]:
    """
    Validates the JSON string from LLM against DETAILED requirements from the system prompt.
    Returns a tuple: (bool indicating validity, parsed JSON data if valid else None).
    """
    if not json_string:
        print("Validation Error: LLM response is empty.")
        return False, None
    try:
        data = json.loads(json_string)
    except json.JSONDecodeError as e:
        print(f"Validation Error: Not a valid JSON. {e}")
        print(f"Problematic string (first 500 chars): '{json_string[:500]}...'")
        return False, None

    # 1. Check top-level keys and type
    if not isinstance(data, dict):
        print("Validation Error: Root is not a dictionary.")
        return False, None
    expected_top_keys = {"structured_data", "original_text"}
    if not expected_top_keys == set(data.keys()):
        print(f"Validation Error: Top-level keys mismatch. Expected {expected_top_keys}, Got {set(data.keys())}")
        return False, None

    # 2. Check original_text (strict match)
    if not isinstance(data["original_text"], str): # Check type first
        print("Validation Error: 'original_text' is not a string.")
        return False, None
    if data["original_text"] != original_text_content:
        print("Validation Error: 'original_text' content does not match input.")
        return False, None

    # 3. Check 'structured_data'
    if "structured_data" not in data or not isinstance(data["structured_data"], dict):
        print("Validation Error: 'structured_data' key missing or not a dictionary.")
        return False, None

    sd = data["structured_data"] # Alias for brevity

    # 4. Check presence and basic types of all direct keys in 'structured_data'
    expected_sd_keys_types = {
        "citation": (str, type(None)), "date": (str, type(None)), "file_number": (str, type(None)),
        "registry": (str, type(None)), "court": (str, type(None)), "document_type": (str, type(None)),
        "summary": (str, type(None)), "outcome_summary": (str, type(None)),
        "procedural_history": list, "legal_issues": list, "legal_concepts": list,
        "holding_decision": list, "reasoning_for_holdings": list,
        "key_findings_and_reasoning": list, "parties": list, "judges": list,
        "counsel": list, "logical_sections": dict, "paragraphs": list
    }

    for key, expected_type in expected_sd_keys_types.items():
        if key not in sd:
            print(f"Validation Error: Missing key '{key}' in 'structured_data'.")
            return False, None
        
        value = sd[key]
        if isinstance(expected_type, tuple): 
            is_type_valid = False
            for et in expected_type:
                if et is type(None) and value is None:
                    is_type_valid = True
                    break
                if et is not type(None) and isinstance(value, et):
                    is_type_valid = True
                    break
            if not is_type_valid:
                print(f"Validation Error: Key '{key}' in 'structured_data' has incorrect type. Expected one of {expected_type}, Got {type(value)}.")
                return False, None
        elif not isinstance(value, expected_type):
            print(f"Validation Error: Key '{key}' in 'structured_data' has incorrect type. Expected {expected_type}, Got {type(value)}.")
            return False, None
        
        if expected_type is list and value is None:
             print(f"Validation Error: Key '{key}' in 'structured_data' is null, but should be an empty list [] if no content.")
             return False, None
        if expected_type is dict and value is None:
             print(f"Validation Error: Key '{key}' in 'structured_data' is null, but should be an empty object {{}} if no content.")
             return False, None

    # 5. DETAILED VALIDATION OF NESTED STRUCTURES
    def validate_list_of_objects(parent_dict_name, parent_dict_value, list_key, item_schema, allow_empty_list=True):
        target_list = parent_dict_value.get(list_key) 
        
        if not isinstance(target_list, list):
            print(f"Validation Error: '{list_key}' is not a list in '{parent_dict_name}'. Expected list, got {type(target_list)}")
            return False

        if not allow_empty_list and not target_list:
            print(f"Validation Error: '{list_key}' in '{parent_dict_name}' must not be empty.")
            return False

        for i, item in enumerate(target_list):
            if not isinstance(item, dict):
                print(f"Validation Error: Item {i} in '{list_key}' (within '{parent_dict_name}') is not a dictionary.")
                return False
            
            for schema_key in item_schema.keys():
                if schema_key not in item:
                    # Allow 'prior_citation' in 'procedural_history' to be missing as per prompt "prior_citation if available"
                    if list_key == "procedural_history" and schema_key == "prior_citation":
                        continue
                    print(f"Validation Error: Item {i} in '{list_key}' (within '{parent_dict_name}') missing key: '{schema_key}'.")
                    return False

            for key, type_info in item_schema.items():
                if key not in item and not (list_key == "procedural_history" and key == "prior_citation"): 
                    continue

                value = item.get(key) 
                
                expected_item_type = type_info.get('type')
                can_be_null = type_info.get('can_be_null', True) 
                is_list_of_type = type_info.get('is_list', False) # Renamed from 'is_list' for clarity
                list_item_type_def = type_info.get('list_item_type') # Renamed from 'list_item_type'
                allow_empty_nested_list = type_info.get('allow_empty_list', True)

                if value is None:
                    if not can_be_null:
                        print(f"Validation Error: Key '{key}' in item {i} of '{list_key}' (within '{parent_dict_name}') is null but cannot be.")
                        return False
                    continue 

                if is_list_of_type: # Check if the field itself is supposed to be a list
                    if not isinstance(value, list):
                        print(f"Validation Error: Key '{key}' in item {i} of '{list_key}' (within '{parent_dict_name}') is not a list. Got {type(value)}.")
                        return False
                    if not allow_empty_nested_list and not value:
                        print(f"Validation Error: List for key '{key}' in item {i} of '{list_key}' (within '{parent_dict_name}') cannot be empty.")
                        return False
                    if list_item_type_def: 
                        for nested_idx, nested_item in enumerate(value):
                            current_item_type_valid = False
                            if isinstance(list_item_type_def, tuple): 
                                for lit in list_item_type_def:
                                    if lit is type(None) and nested_item is None: 
                                        current_item_type_valid = True
                                        break
                                    if lit is not type(None) and isinstance(nested_item, lit):
                                        current_item_type_valid = True
                                        break
                            elif isinstance(nested_item, list_item_type_def): 
                                current_item_type_valid = True
                            
                            if not current_item_type_valid:
                                print(f"Validation Error: Item {nested_idx} in list '{key}' (item {i} of '{list_key}', in '{parent_dict_name}') is not type/one of types {list_item_type_def}. Got {type(nested_item)}.")
                                return False
                else: 
                    current_field_type_valid = False
                    if isinstance(expected_item_type, tuple): 
                        for et_option in expected_item_type: # Renamed 'et' to 'et_option'
                            if et_option is type(None) and value is None: 
                                current_field_type_valid = True
                                break
                            if et_option is not type(None) and isinstance(value, et_option):
                                current_field_type_valid = True
                                break
                    elif isinstance(value, expected_item_type): 
                        current_field_type_valid = True
                    
                    if not current_field_type_valid:
                        print(f"Validation Error: Key '{key}' in item {i} of '{list_key}' (within '{parent_dict_name}') is not type/one of types {expected_item_type}. Got {type(value)}.")
                        return False
        return True

    # Schemas for validate_list_of_objects
    procedural_history_schema = {
        "court_name": {'type': str, 'can_be_null': True},
        "judge_or_judges": {'type': (str, list), 'can_be_null': True, 'is_list': True, 'list_item_type': str, 'allow_empty_list': True}, 
        "date_of_action": {'type': str, 'can_be_null': True},
        "description_of_action": {'type': str, 'can_be_null': True},
        "prior_citation": {'type': str, 'can_be_null': True} 
    }
    holding_decision_schema = {
        "issue_addressed": {'type': str, 'can_be_null': False},
        "holding": {'type': str, 'can_be_null': False},
        "disposition_on_issue": {'type': str, 'can_be_null': True}
    }
    reasoning_for_holdings_schema = {
        "holding_addressed": {'type': str, 'can_be_null': False},
        "reasoning_text": {'type': str, 'can_be_null': False},
        "key_reasoning_paragraphs": {'type': list, 'can_be_null': False, 'is_list': True, 'list_item_type': (str, int), 'allow_empty_list': True}
    }
    key_findings_schema = {
        "finding_phrase": {'type': str, 'can_be_null': False},
        "supporting_paragraphs": {'type': list, 'can_be_null': False, 'is_list': True, 'list_item_type': (str, int), 'allow_empty_list': True},
        "relevant_concepts": {'type': list, 'can_be_null': False, 'is_list': True, 'list_item_type': str, 'allow_empty_list': True}
    }
    parties_schema = {"role": {'type': str, 'can_be_null': False}, "name": {'type': str, 'can_be_null': False}}
    counsel_schema = {"name": {'type': str, 'can_be_null': False}, "represents": {'type': str, 'can_be_null': False}}
    paragraph_schema = {
        "number": {'type': (str, int), 'can_be_null': False},
        "text": {'type': str, 'can_be_null': False},
        "sentences": {'type': list, 'can_be_null': False, 'is_list': True, 'list_item_type': str, 'allow_empty_list': True},
        "key_phrases_local": {'type': list, 'can_be_null': False, 'is_list': True, 'list_item_type': str, 'allow_empty_list': True},
        "entities_local": {'type': dict, 'can_be_null': False} 
    }

    # Actual validation calls using the helper
    if not validate_list_of_objects("structured_data", sd, "procedural_history", procedural_history_schema, allow_empty_list=True): return False, None
    if not all(isinstance(item, str) for item in sd.get("legal_issues", [])): # Should be an empty list if no issues, not null
        print(f"Validation Error: Not all items in 'legal_issues' are strings.")
        return False, None
    if not all(isinstance(item, str) for item in sd.get("legal_concepts", [])): # Should be an empty list if no concepts, not null
        print(f"Validation Error: Not all items in 'legal_concepts' are strings.")
        return False, None
    if not validate_list_of_objects("structured_data", sd, "holding_decision", holding_decision_schema, allow_empty_list=True): return False, None
    if not validate_list_of_objects("structured_data", sd, "reasoning_for_holdings", reasoning_for_holdings_schema, allow_empty_list=True): return False, None
    if not validate_list_of_objects("structured_data", sd, "key_findings_and_reasoning", key_findings_schema, allow_empty_list=True): return False, None
    if not validate_list_of_objects("structured_data", sd, "parties", parties_schema, allow_empty_list=True): return False, None
    if not all(isinstance(item, str) for item in sd.get("judges", [])): # Should be an empty list if no judges, not null
        print(f"Validation Error: Not all items in 'judges' are strings.")
        return False, None
    if not validate_list_of_objects("structured_data", sd, "counsel", counsel_schema, allow_empty_list=True): return False, None

    # 5.10 logical_sections
    ls_node = sd.get("logical_sections", {}) # Should be an empty dict if no sections, not null
    if not isinstance(ls_node, dict): 
         print(f"Validation Error: 'logical_sections' is not a dictionary.")
         return False, None
    expected_ls_keys = {"introduction", "facts", "analysis", "conclusion"}
    for ls_key in expected_ls_keys: # Keys should be present, mapping to empty lists if no content for that section
        if ls_key not in ls_node: 
            print(f"Validation Error: 'logical_sections' missing key: '{ls_key}'.")
            return False, None
        val_list = ls_node[ls_key]
        if not isinstance(val_list, list):
            print(f"Validation Error: Value for '{ls_key}' in 'logical_sections' is not a list. Got {type(val_list)}.")
            return False, None
        if not all(isinstance(item, (str, int)) for item in val_list): 
            print(f"Validation Error: Not all items in 'logical_sections.{ls_key}' are string or int.")
            return False, None
            
    # 5.11 paragraphs
    if not validate_list_of_objects("structured_data", sd, "paragraphs", paragraph_schema, allow_empty_list=True): return False, None # Paragraphs list can be empty
    
    for i, p_item in enumerate(sd.get("paragraphs", [])): 
        if not isinstance(p_item, dict): continue 

        el_node = p_item.get("entities_local", {}) 
        if not isinstance(el_node, dict): 
             print(f"Validation Error: 'entities_local' in paragraph item {i} is not a dictionary. Got {type(el_node)}.")
             return False, None
        
        allowed_entity_keys = {"people", "organizations", "dates", "locations", "legal_terms"}
        for entity_type, entity_list in el_node.items():
            if entity_type not in allowed_entity_keys:
                print(f"Validation Error: Unknown key '{entity_type}' in 'entities_local' for paragraph {i}.")
                return False, None 
            if not isinstance(entity_list, list):
                print(f"Validation Error: Entity type '{entity_type}' in 'entities_local' (paragraph {i}) is not a list. Got {type(entity_list)}.")
                return False, None
            if not all(isinstance(entity_val, str) for entity_val in entity_list):
                print(f"Validation Error: Not all entity values for type '{entity_type}' in 'entities_local' (paragraph {i}) are strings.")
                return False, None

    print("JSON validation successful.")
    return True, data

def main():
    parser = argparse.ArgumentParser(
        description="Processes legal text files using Google Gemini API and outputs structured JSON.",
        formatter_class=argparse.RawTextHelpFormatter, # For better help text formatting
        epilog="""\
Example usage:
  python gemini_processor.py
  python gemini_processor.py -i ./my_legal_texts -o ./processed_output

Ensure a .env file with your GOOGLE_API_KEY is present in the script's directory or that the environment variable is set.
GOOGLE_API_KEY="YOUR_API_KEY"
"""
    )
    parser.add_argument(
        "-i", "--input_dir",
        default="input_files/",
        help="Directory containing input .txt files. (Default: input_files/)"
    )
    parser.add_argument(
        "-o", "--output_dir",
        default="output_json/",
        help="Directory where output .json files will be saved. (Default: output_json/)"
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    if not os.path.isdir(input_dir):
        print(f"Error: Input directory '{input_dir}' not found.")
        print("Please create it and place your .txt files inside.")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"Input directory: {os.path.abspath(input_dir)}")
    print(f"Output directory: {os.path.abspath(output_dir)}")

    txt_files = glob.glob(os.path.join(input_dir, "*.txt"))

    if not txt_files:
        print(f"No .txt files found in '{input_dir}'.")
        return

    successful_processing_count = 0
    failed_processing_count = 0

    print(f"Found {len(txt_files)} .txt files to process.")

    for txt_file_path in txt_files:
        filename = os.path.basename(txt_file_path)
        print(f"--- Processing file: {filename} ---")

        try:
            with open(txt_file_path, 'r', encoding='utf-8') as f:
                original_file_content = f.read()
        except Exception as e:
            print(f"Error reading file {filename}: {e}")
            failed_processing_count += 1
            continue

        # Call LLM to process the text
        # Note: process_text_file now takes file_path and original_file_content
        # The file_path argument in process_text_file is mostly for logging context now
        raw_json_output = process_text_file(txt_file_path, original_file_content)

        if raw_json_output is None:
            print(f"Failed to get API response for {filename}.")
            failed_processing_count += 1
            continue

        # Validate the JSON output
        is_valid, parsed_json_data = validate_json_output(raw_json_output, original_file_content)

        if is_valid and parsed_json_data:
            output_filename = os.path.splitext(filename)[0] + ".json"
            output_filepath = os.path.join(output_dir, output_filename)
            try:
                with open(output_filepath, 'w', encoding='utf-8') as outfile:
                    json.dump(parsed_json_data, outfile, indent=2, ensure_ascii=False)
                print(f"Successfully processed and saved '{output_filepath}'")
                successful_processing_count += 1
            except Exception as e:
                print(f"Error saving JSON output for {filename} to {output_filepath}: {e}")
                failed_processing_count += 1
        else:
            print(f"Validation failed for {filename}. Output not saved.")
            # validate_json_output already prints detailed validation errors
            failed_processing_count += 1
        print("--- Finished processing file ---")


    print("\n----- Processing Summary -----")
    print(f"Total files found: {len(txt_files)}")
    print(f"Successfully processed: {successful_processing_count}")
    print(f"Failed to process: {failed_processing_count}")
    print("-----------------------------")

if __name__ == "__main__":
    # Potentially clear screen or add some initial welcome message
    print("Starting Legal Document Processor...")
    main()
    print("Processing complete.")
