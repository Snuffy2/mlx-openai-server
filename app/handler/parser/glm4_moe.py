"""
Parsers for GLM4 MoE model response formats.

This module provides specialized parsers for GLM4 MoE model's tool calls and
thinking traces, handling GLM4-specific JSON parsing and message conversion.
"""

import ast
import json
import re
from typing import Any

from loguru import logger

from .base import BaseMessageConverter, BaseThinkingParser, BaseToolParser

TOOL_OPEN = "<tool_call>"
TOOL_CLOSE = "</tool_call>"
THINKING_OPEN = "<think>"
THINKING_CLOSE = "</think>"


class Glm4MoEThinkingParser(BaseThinkingParser):
    """Parser for GLM4 model's thinking response format."""

    def __init__(self) -> None:
        super().__init__(thinking_open=THINKING_OPEN, thinking_close=THINKING_CLOSE)


class Glm4MoEToolParser(BaseToolParser):
    """Parser for GLM4 model's tool response format with XML-style arguments."""

    def __init__(self) -> None:
        """
        Initialize the GLM4 MoE tool parser with GLM4-specific tool markers and compiled regex patterns.

        Sets the tool open/close markers to GLM4 values and prepares two regex patterns:
        - `func_detail_regex`: matches a function detail block (function name on the first line and the remaining block of arguments).
        - `func_arg_regex`: matches individual argument entries expressed as `<arg_key>...</arg_key><arg_value>...</arg_value>` pairs.
        """
        super().__init__(tool_open=TOOL_OPEN, tool_close=TOOL_CLOSE)
        # Regex patterns for parsing GLM4 XML-style tool calls
        self.func_detail_regex = re.compile(r"([^\n]*)\n(.*)", re.DOTALL)
        self.func_arg_regex = re.compile(
            r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>",
            re.DOTALL,
        )

    def _deserialize_value(self, value: str) -> Any:
        """
        Deserialize a string into the most appropriate Python value.

        Parameters:
            value (str): A string containing a serialized value (JSON, Python literal, or plain text).

        Returns:
            The deserialized Python object parsed from `value` (e.g., dict, list, int, float, bool), or the original stripped string if parsing fails.
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
        Parse a GLM4-style tool call string into a tool descriptor with name and arguments.

        Parses `tool_content` for a function name and a sequence of key/value argument pairs in the GLM4 MoE tool-call format. On successful parse returns a dictionary with the function `name` and an `arguments` mapping; if the content does not match the expected format or a parsing error occurs, returns `None`.

        Parameters:
            tool_content (str): Raw tool-call text produced by the GLM4 model.

        Returns:
            dict: A dictionary with keys:
                - "name" (str): The parsed function name.
                - "arguments" (dict[str, Any]): Mapping of argument names to deserialized values.
            None: If the input does not match the expected format or parsing fails.
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

        except (
            re.error,
            ValueError,
            SyntaxError,
            KeyError,
            IndexError,
            AttributeError,
            TypeError,
        ) as e:
            logger.warning(
                f"Error parsing GLM4 tool call content: {tool_content}. {type(e).__name__}: {e}",
            )
            return None
        else:
            return {"name": func_name, "arguments": arguments}


class Glm4MoEMessageConverter(BaseMessageConverter):
    """GLM4 MoE-specific message format converter."""

    def _parse_arguments_string(self, arguments_str: str) -> Any:
        """Parse GLM4 MoE-specific argument string format."""
        try:
            return json.loads(arguments_str)
        except json.JSONDecodeError:
            return arguments_str
