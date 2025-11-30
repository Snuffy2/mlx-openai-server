"""
Parsers for MiniMax model response formats.

This module provides specialized parsers for MiniMax model's tool calls and
thinking traces, handling MiniMax-specific JSON parsing and message conversion.
"""

import ast
import json
import re
from typing import Any

from loguru import logger

from .base import BaseMessageConverter, BaseThinkingParser, BaseToolParser

TOOL_OPEN = "<minimax:tool_call>"
TOOL_CLOSE = "</minimax:tool_call>"
THINKING_OPEN = "<think>"
THINKING_CLOSE = "</think>"


class MinimaxThinkingParser(BaseThinkingParser):
    """Parser for MiniMax model's thinking response format."""

    def __init__(self) -> None:
        super().__init__(thinking_open=THINKING_OPEN, thinking_close=THINKING_CLOSE)


class MinimaxToolParser(BaseToolParser):
    """Parser for MiniMax model's tool response format with XML-style arguments."""

    def __init__(self) -> None:
        """
        Initialize the parser with MiniMax tool markers and compile regex patterns used to extract tool-call details.
        
        Sets:
        - `func_detail_regex`: matches an `<invoke name="...">` block and captures the function name and the rest of the content.
        - `func_arg_regex`: matches `<parameter name="...">value</parameter>` entries and captures parameter names and their string values.
        """
        super().__init__(tool_open=TOOL_OPEN, tool_close=TOOL_CLOSE)
        # Regex patterns for parsing MiniMax tool calls
        self.func_detail_regex = re.compile(r'<invoke name="([^"]+)"\s*>(.*)', re.DOTALL)
        self.func_arg_regex = re.compile(
            r'<parameter name="([^"]+)"\s*>([^<]*)</parameter>',
            re.DOTALL,
        )

    def _deserialize_value(self, value: str) -> Any:
        """
        Convert a string representation into the corresponding Python value.
        
        Attempts to parse the stripped input as JSON, then as a Python literal; if both parse attempts fail, returns the stripped original string.
        
        Parameters:
            value (str): The input string to deserialize; leading and trailing whitespace will be removed.
        
        Returns:
            Any: The deserialized Python object (e.g., dict, list, int, float, bool), or the stripped original string if parsing fails.
        """
        value = value.strip()

        # Try JSON parsing first
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            pass

        # Try literal eval for Python literals
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass

        # Return as string if all else fails
        return value

    def _parse_tool_content(self, tool_content: str) -> dict[str, Any] | None:
        """
        Parse a MiniMax-formatted tool call string into a dictionary with the function name and its arguments.
        
        Parameters:
            tool_content (str): The raw MiniMax tool call content to parse.
        
        Returns:
            dict[str, Any] | None: A dictionary with keys:
                - "name": the parsed function name (str).
                - "arguments": a dict mapping argument names (str) to their deserialized values.
            Returns `None` if the content cannot be parsed.
        """
        try:
            # Extract function name and arguments section
            detail_match = self.func_detail_regex.search(tool_content)
            if not detail_match:
                return None

            func_name = detail_match.group(1).strip()
            args_section = detail_match.group(2)

            # Extract all key-value pairs
            arg_pairs = self.func_arg_regex.findall(args_section)

            arguments = {}
            for key, value in arg_pairs:
                arg_key = key.strip()
                arg_value = self._deserialize_value(value)
                arguments[arg_key] = arg_value

            # Build tool call object

        except (AttributeError, KeyError, TypeError, ValueError, re.error) as e:
            logger.warning(
                f"Error parsing MiniMax tool call content: {tool_content}. {type(e).__name__}: {e}",
            )
            return None
        else:
            return {"name": func_name, "arguments": arguments}


class MiniMaxMessageConverter(BaseMessageConverter):
    """MiniMax-specific message format converter."""

    def _parse_arguments_string(self, arguments_str: str) -> Any:
        """Parse MiniMax-specific argument string format."""
        try:
            return json.loads(arguments_str)
        except json.JSONDecodeError:
            return arguments_str