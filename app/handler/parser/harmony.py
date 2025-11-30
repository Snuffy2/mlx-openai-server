"""
Parsers for Harmony model response formats.

This module provides specialized parsers for Harmony model's structured outputs,
using the openai-harmony library for encoding and parsing.
"""

from enum import Enum
from typing import Any

from loguru import logger
from openai_harmony import HarmonyEncodingName, Role, StreamableParser, load_harmony_encoding


class ChannelType(Enum):
    """Enumeration of harmony channel types."""

    ANALYSIS = "analysis"
    COMMENTARY = "commentary"
    FINAL = "final"


class ParsingState(Enum):
    """Enumeration of parsing states."""

    IDLE = "idle"
    PROCESSING_TOKENS = "processing_tokens"
    TOOL_PARSING = "tool_parsing"
    STREAM_ENDED = "stream_ended"


# Harmony Parsing Helper Functions
class HarmonyParser:
    """
    Enhanced helper class for parsing GPT-OSS model responses using harmony encoding.

    This parser handles streaming and non-streaming responses with proper state management,
    error handling, and support for different harmony channels (analysis, commentary, final).
    """

    tool_state: bool
    end_stream: bool
    parsing_state: ParsingState
    _accumulated_content: dict[str, list[str]]
    _current_function_name: str | None
    _function_arguments: list[str]

    def __init__(self) -> None:
        """Initialize the harmony parser with encoding and state management."""
        try:
            self.enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
            self.parser = StreamableParser(self.enc, role=Role.ASSISTANT)
        except Exception as e:
            logger.error(f"Failed to initialize harmony encoding: {type(e).__name__}: {e}")
            raise

        # Configuration
        self.end_tool_chunk = "<|call|>"

        # State management
        self._reset_state()

    def _reset_state(self) -> None:
        """Reset the parser state to initial values."""
        self.tool_state = False
        self.end_stream = False
        self.parsing_state = ParsingState.IDLE
        self._accumulated_content = {
            ChannelType.ANALYSIS.value: [],
            ChannelType.COMMENTARY.value: [],
            ChannelType.FINAL.value: [],
        }
        self._current_function_name = None
        self._function_arguments = []

    def parse_stream(self, text: str | None = None) -> tuple[Any | None, bool]:
        """
        Parse a streaming chunk of assistant output and extract channeled content or tool call information.

        Processes a single text chunk (which may be a token sequence or an end-of-stream marker), updates internal parser state and accumulators, and returns the parsed payload for the current channel along with whether the overall stream has ended.

        Parameters:
            text (str | None): A chunk of streamed text to parse. If equal to the parser's end-tool marker, the stream is marked ended; if None or empty, no progress is made.

        Returns:
            tuple[Any | None, bool]:
                - The parsed content for the current channel: a dict for analysis/tool commentary, a string for final content, or None if nothing was produced.
                - A boolean that is True when the stream has been marked ended, False otherwise.
        """
        # Handle end of stream marker
        if text == self.end_tool_chunk:
            logger.debug("End tool chunk detected, marking stream as ended")
            self.end_stream = True
            self.parsing_state = ParsingState.STREAM_ENDED
            return None, True

        # Handle empty or None text
        if not text:
            return None, self.end_stream

        try:
            self.parsing_state = ParsingState.PROCESSING_TOKENS
            text_tokens = self.enc.encode(text, allowed_special="all")

            # Initialize local variables for this chunk
            contents: list[str] = []
            function_name: str | None = None
            function_arguments: list[str] = []
            reasoning_content: list[str] = []
            current_channel: str | None = None

            # Process each token
            for text_token in text_tokens:
                try:
                    stream_text = self.parser.process(text_token)
                    current_channel = stream_text.current_channel
                    content = stream_text.last_content_delta

                    if not content:
                        continue

                    # Handle different channels
                    if current_channel == ChannelType.ANALYSIS.value:
                        reasoning_content.append(content)
                        self._accumulated_content[ChannelType.ANALYSIS.value].append(content)

                    elif current_channel == ChannelType.COMMENTARY.value:
                        self.parsing_state = ParsingState.TOOL_PARSING

                        if self.tool_state:
                            # Already parsing function arguments
                            function_arguments.append(content)
                            self._function_arguments.append(content)
                        else:
                            # Start of new function call
                            self.tool_state = True
                            if (
                                hasattr(stream_text, "current_recipient")
                                and stream_text.current_recipient
                            ):
                                function_name = stream_text.current_recipient.replace(
                                    "functions.",
                                    "",
                                )
                                self._current_function_name = function_name
                            function_arguments = [content]
                            self._function_arguments = [content]

                    elif current_channel == ChannelType.FINAL.value:
                        contents.append(content)
                        self._accumulated_content[ChannelType.FINAL.value].append(content)

                except (
                    AttributeError,
                    KeyError,
                    TypeError,
                    UnicodeDecodeError,
                    ValueError,
                ) as e:
                    logger.warning(f"Error processing token {text_token}. {type(e).__name__}: {e}")
                    continue

            # Return appropriate response based on current channel
            return self._build_response(
                current_channel,
                {
                    "reasoning_content": reasoning_content,
                    "function_name": function_name,
                    "function_arguments": function_arguments,
                    "contents": contents,
                },
            )

        except (AttributeError, KeyError, TypeError, ValueError) as e:
            logger.exception(
                f"Error in parse_stream with known exception type: {type(e).__name__}: {e}",
            )
            return None, self.end_stream

    def _build_response(
        self,
        current_channel: str | None,
        content_data: dict[str, Any],
    ) -> tuple[dict[str, Any] | str | None, bool]:
        """
        Constructs a parsed response object from accumulated content for the given channel.

        Parameters:
            current_channel (str | None): The harmony channel currently being processed (e.g., "analysis", "commentary", "final").
            content_data (dict[str, Any]): Extracted fragments for the channel; expected keys vary by channel:
                - "reasoning_content": list[str] for analysis channel
                - "function_name" and "function_arguments": for commentary channel
                - "contents": list[str] for final channel

        Returns:
            tuple[dict[str, Any] | str | None, bool]:
                parsed_content: For the analysis channel, a dict {"reasoning_content": <str>}; for the commentary channel, a dict with optional
                    "name" and "arguments"; for the final channel, a concatenated string of content. Returns `None` if there is no response to produce.
                is_complete: `True` if the stream has been marked ended, `False` otherwise.
        """
        if not current_channel:
            return None, self.end_stream

        try:
            if current_channel == ChannelType.ANALYSIS.value:
                reasoning_content = content_data.get("reasoning_content", [])
                if reasoning_content:
                    return {"reasoning_content": "".join(reasoning_content)}, self.end_stream

            elif current_channel == ChannelType.COMMENTARY.value:
                function_name = content_data.get("function_name")
                function_arguments = content_data.get("function_arguments", [])

                response = {}
                if function_name:
                    response["name"] = function_name
                if function_arguments:
                    response["arguments"] = "".join(function_arguments)

                if response:
                    return response, self.end_stream

            elif current_channel == ChannelType.FINAL.value:
                contents = content_data.get("contents", [])
                if contents:
                    return "".join(contents), self.end_stream
        except (KeyError, TypeError, ValueError) as e:
            logger.error(
                f"Error building response for channel {current_channel}: {type(e).__name__}: {e}",
            )

        return None, self.end_stream

    def reset(self) -> None:
        """Reset the parser to initial state for reuse."""
        logger.debug("Resetting harmony parser state")
        self._reset_state()

    def get_accumulated_content(self, channel: str | None = None) -> dict[str, str]:
        """
        Get accumulated content for all channels or a specific channel.

        Args:
            channel: Optional specific channel to retrieve content for

        Returns
        -------
            Dictionary of channel content
        """
        if channel and channel in self._accumulated_content:
            return {channel: "".join(self._accumulated_content[channel])}

        return {
            ch: "".join(content) for ch, content in self._accumulated_content.items() if content
        }

    def parse(self, text: str) -> dict[str, Any]:
        """
        Parse a complete Harmony-encoded assistant response and extract structured channels.

        Parses the provided full response text (non-streaming) into three possible outputs:
        reasoning content from the analysis channel, tool call(s) from the commentary channel,
        and final assistant content from the final channel. Partial results are returned when
        individual message parsing errors occur; unexpected top-level errors are logged and
        partial results are preserved.

        Parameters:
            text (str): The complete response text to parse.

        Returns:
            dict[str, Any]: A dictionary with keys:
                - "reasoning_content": extracted analysis text or None,
                - "tool_calls": list of tool call objects (each with "name" and "arguments") or None,
                - "content": extracted final response text or None.
        """
        # Initialize result structure
        result: dict[str, Any] = {"reasoning_content": None, "tool_calls": None, "content": None}

        if not text:
            logger.warning("Empty text provided to parse method")
            return result

        try:
            # Remove end tool chunk if present
            clean_text = text
            if self.end_tool_chunk in text:
                clean_text = text.split(self.end_tool_chunk)[0]
                logger.debug(f"Removed end tool chunk, processing {len(clean_text)} characters")

            # Encode and parse messages
            tokens = self.enc.encode(clean_text, allowed_special="all")
            parsed_messages = self.enc.parse_messages_from_completion_tokens(
                tokens,
                role=Role.ASSISTANT,
            )

            # Process each parsed message
            for message in parsed_messages:
                try:
                    if not hasattr(message, "channel") or not hasattr(message, "content"):
                        logger.warning(f"Invalid message structure: {message}")
                        continue

                    if message.channel == ChannelType.ANALYSIS.value:
                        if message.content and len(message.content) > 0:
                            result["reasoning_content"] = message.content[0].text
                            logger.debug("Extracted reasoning content")

                    elif message.channel == ChannelType.COMMENTARY.value:
                        if (
                            hasattr(message, "recipient")
                            and message.recipient
                            and message.content
                            and len(message.content) > 0
                        ):
                            tool_call = {
                                "name": message.recipient.replace("functions.", ""),
                                "arguments": message.content[0].text,
                            }
                            result["tool_calls"] = [tool_call]
                            logger.debug(f"Extracted tool call: {tool_call['name']}")

                    elif message.channel == ChannelType.FINAL.value:
                        if message.content and len(message.content) > 0:
                            result["content"] = message.content[0].text
                            logger.debug("Extracted final content")

                except (AttributeError, IndexError, KeyError, TypeError, ValueError) as e:
                    logger.warning(f"Error processing message: {type(e).__name__}: {e}")
                    continue

        except Exception as e:
            # Safety net for unexpected errors during encoding/parsing
            logger.error(f"Error in parse method: {type(e).__name__}: {e}")
            # Return partial results if available, don't raise

        return result

    def is_stream_ended(self) -> bool:
        """Check if the stream has ended."""
        return self.end_stream

    def get_parsing_state(self) -> ParsingState:
        """Get the current parsing state."""
        return self.parsing_state

    def is_tool_parsing_active(self) -> bool:
        """Check if currently parsing tool calls."""
        return self.tool_state

    def get_current_function_info(self) -> dict[str, str | None]:
        """
        Get information about the currently parsed function.

        Returns
        -------
            Dictionary with function name and accumulated arguments
        """
        return {
            "name": self._current_function_name,
            "arguments": "".join(self._function_arguments) if self._function_arguments else None,
        }

    def __repr__(self) -> str:
        """Return a string representation of the parser state."""
        return (
            f"HarmonyParser(state={self.parsing_state.value}, "
            f"tool_state={self.tool_state}, "
            f"stream_ended={self.end_stream})"
        )
