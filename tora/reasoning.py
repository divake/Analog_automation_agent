"""
LLM reasoning component for TORA-AICircuit.
Uses Anthropic Claude API to provide explanations and refinement suggestions.
"""

import os
import json
import requests
from typing import Dict, Any, List, Tuple
from dotenv import load_dotenv
import logging

class ReasoningModule:
    """
    LLM reasoning component for TORA-AICircuit integration.
    Uses Anthropic's Claude API to provide explanations and refinement suggestions.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the reasoning module with configuration.
        
        Args:
            config: Configuration dictionary from config.yaml
        """
        # Load environment variables for API key
        load_dotenv()
        
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
        
        # Initialize from config
        self.api_url = config["reasoning"]["api_url"]
        self.model = config["reasoning"]["model"]
        self.temperature = config["reasoning"]["temperature"]
        self.max_tokens = config["reasoning"]["max_tokens"]
        
        # Load circuit knowledge
        self.knowledge_path = config["circuit"]["knowledge_path"]
        with open(self.knowledge_path, 'r') as f:
            self.circuit_knowledge = json.load(f)
        
        # Load prompt templates
        with open(config["reasoning"]["prompts"]["explanation_template"], 'r') as f:
            self.explanation_template = f.read()
            
        with open(config["reasoning"]["prompts"]["refinement_template"], 'r') as f:
            self.refinement_template = f.read()
        
        self.logger = logging.getLogger(__name__)
    
    def explain_parameters(self, circuit_type: str, 
                         specifications: Dict[str, float],
                         predicted_parameters: Dict[str, float]) -> Dict[str, Any]:
        """
        Generate explanations for the predicted parameters.
        
        Args:
            circuit_type: Type of circuit (e.g., "CSVA")
            specifications: Performance specifications
            predicted_parameters: ML-predicted parameters
            
        Returns:
            Dictionary with explanations and reasoning
        """
        # Format the inputs
        specs_formatted = json.dumps(specifications, indent=2)
        params_formatted = json.dumps(predicted_parameters, indent=2)
        knowledge_formatted = json.dumps(self.circuit_knowledge, indent=2)
        
        # Create prompt from template
        prompt = self.explanation_template.format(
            circuit_type=circuit_type,
            specifications=specs_formatted,
            parameters=params_formatted,
            circuit_knowledge=knowledge_formatted
        )
        
        # Call Claude API
        response = self._call_claude_api(prompt)
        
        # Process and structure the response
        try:
            # Extract JSON content from response - Claude often wraps JSON in markdown codeblocks
            json_content = self._extract_json_from_response(response)
            structured_explanation = json.loads(json_content)
            return structured_explanation
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse explanation response as JSON: {e}")
            return {
                "error": "Failed to parse response",
                "raw_response": response
            }
    
    def suggest_refinements(self, circuit_type: str,
                          specifications: Dict[str, float],
                          current_parameters: Dict[str, float],
                          simulation_results: Dict[str, float]) -> Dict[str, Any]:
        """
        Suggest parameter refinements based on simulation results.
        
        Args:
            circuit_type: Type of circuit
            specifications: Target specifications
            current_parameters: Current parameter values
            simulation_results: Results from circuit simulation
            
        Returns:
            Dictionary with parameter adjustment suggestions and reasoning
        """
        # Format the inputs
        specs_formatted = json.dumps(specifications, indent=2)
        params_formatted = json.dumps(current_parameters, indent=2)
        results_formatted = json.dumps(simulation_results, indent=2)
        knowledge_formatted = json.dumps(self.circuit_knowledge, indent=2)
        
        # Calculate percent differences for each specification
        diffs = {}
        for spec, target in specifications.items():
            if spec in simulation_results:
                achieved = simulation_results[spec]
                percent_diff = ((achieved - target) / target) * 100
                diffs[spec] = {
                    "target": target,
                    "achieved": achieved,
                    "percent_diff": percent_diff
                }
        
        diffs_formatted = json.dumps(diffs, indent=2)
        
        # Create prompt from template
        prompt = self.refinement_template.format(
            circuit_type=circuit_type,
            specifications=specs_formatted,
            parameters=params_formatted,
            simulation_results=results_formatted,
            specification_differences=diffs_formatted,
            circuit_knowledge=knowledge_formatted
        )
        
        # Call Claude API
        response = self._call_claude_api(prompt)
        
        # Process and structure the response
        try:
            # Extract JSON content from response
            json_content = self._extract_json_from_response(response)
            structured_refinements = json.loads(json_content)
            return structured_refinements
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse refinement response as JSON: {e}")
            return {
                "error": "Failed to parse response",
                "raw_response": response
            }
    
    def _call_claude_api(self, prompt: str) -> str:
        """
        Call Claude API and get response.
        
        Args:
            prompt: The prompt to send to Claude
            
        Returns:
            Claude's response as a string
        """
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        try:
            self.logger.debug("Calling Claude API")
            response = requests.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
            
            # Extract content from Claude's response
            content = response.json()["content"][0]["text"]
            return content
        except Exception as e:
            self.logger.error(f"Error calling Claude API: {e}")
            raise
    
    def _extract_json_from_response(self, response: str) -> str:
        """
        Extract JSON content from Claude's response, handling cases where
        JSON is wrapped in markdown code blocks.
        
        Args:
            response: Claude's full text response
            
        Returns:
            Extracted JSON string
        """
        # Check if the response is already valid JSON
        try:
            json.loads(response)
            return response
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from markdown code blocks
        if "```json" in response and "```" in response:
            # Extract content between ```json and ```
            start_idx = response.find("```json") + 7
            end_idx = response.find("```", start_idx)
            json_content = response[start_idx:end_idx].strip()
            return json_content
        
        # If still no valid JSON, look for JSON-like structure with braces
        if "{" in response and "}" in response:
            start_idx = response.find("{")
            # Find the matching closing brace (handling nested braces)
            brace_count = 1
            for i in range(start_idx + 1, len(response)):
                if response[i] == "{":
                    brace_count += 1
                elif response[i] == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i + 1
                        break
            
            if brace_count == 0:  # Found matching closing brace
                return response[start_idx:end_idx]
        
        # If all extraction attempts fail, return the original response
        self.logger.warning("Could not extract JSON from response")
        return response 