import unittest

from mlx_openai_server.handler.parser.base import BaseThinkingParser, BaseToolParser


class TestBaseToolParser(unittest.TestCase):
    def setUp(self):
        self.test_cases = [
            {
                "name": "simple function call",
                "chunks": """#<tool_call>#
#{"#name#":# "#get#_weather#",# "#arguments#":# {"#city#":# "#H#ue#"}}
#</tool_call>#
#<tool_call>#
#{"#name#":# "#get#_weather#",# "#arguments#":# {"#city#":# "#Sy#dney#"}}
#</tool_call>##""".split("#"),
                "expected_outputs": [
                    {"name": "get_weather", "arguments": ""},
                    {"name": None, "arguments": ' {"'},
                    {"name": None, "arguments": "city"},
                    {"name": None, "arguments": '":'},
                    {"name": None, "arguments": ' "'},
                    {"name": None, "arguments": "H"},
                    {"name": None, "arguments": "ue"},
                    {"name": None, "arguments": '"}'},
                    "\n",
                    {"name": "get_weather", "arguments": ""},
                    {"name": None, "arguments": ' {"'},
                    {"name": None, "arguments": "city"},
                    {"name": None, "arguments": '":'},
                    {"name": None, "arguments": ' "'},
                    {"name": None, "arguments": "Sy"},
                    {"name": None, "arguments": "dney"},
                    {"name": None, "arguments": '"}'},
                ],
            },
            {
                "name": "code function call",
                "chunks": r"""<tool_call>@@
@@{"@@name@@":@@ "@@python@@",@@ "@@arguments@@":@@ {"@@code@@":@@ "@@def@@ calculator@@(a@@,@@ b@@,@@ operation@@):\@@n@@   @@ if@@ operation@@ ==@@ '@@add@@'\@@n@@       @@ return@@ a@@ +@@ b@@\n@@   @@ if@@ operation@@ ==@@ '@@subtract@@'\@@n@@       @@ return@@ a@@ -@@ b@@\n@@   @@ if@@ operation@@ ==@@ '@@multiply@@'\@@n@@       @@ return@@ a@@ *@@ b@@\n@@   @@ if@@ operation@@ ==@@ '@@divide@@'\@@n@@       @@ return@@ a@@ /@@ b@@\n@@   @@ return@@ '@@Invalid@@ operation@@'@@"}}
@@</tool_call>@@@@""".split("@@"),
                "expected_outputs": [
                    {"name": "python", "arguments": ""},
                    {"name": None, "arguments": ' {"'},
                    {"name": None, "arguments": "code"},
                    {"name": None, "arguments": '":'},
                    {"name": None, "arguments": ' "'},
                    {"name": None, "arguments": "def"},
                    {"name": None, "arguments": " calculator"},
                    {"name": None, "arguments": "(a"},
                    {"name": None, "arguments": ","},
                    {"name": None, "arguments": " b"},
                    {"name": None, "arguments": ","},
                    {"name": None, "arguments": " operation"},
                    {"name": None, "arguments": "):\\"},
                    {"name": None, "arguments": "n"},
                    {"name": None, "arguments": "   "},
                    {"name": None, "arguments": " if"},
                    {"name": None, "arguments": " operation"},
                    {"name": None, "arguments": " =="},
                    {"name": None, "arguments": " '"},
                    {"name": None, "arguments": "add"},
                    {"name": None, "arguments": "'\\"},
                    {"name": None, "arguments": "n"},
                    {"name": None, "arguments": "       "},
                    {"name": None, "arguments": " return"},
                    {"name": None, "arguments": " a"},
                    {"name": None, "arguments": " +"},
                    {"name": None, "arguments": " b"},
                    {"name": None, "arguments": "\\n"},
                    {"name": None, "arguments": "   "},
                    {"name": None, "arguments": " if"},
                    {"name": None, "arguments": " operation"},
                    {"name": None, "arguments": " =="},
                    {"name": None, "arguments": " '"},
                    {"name": None, "arguments": "subtract"},
                    {"name": None, "arguments": "'\\"},
                    {"name": None, "arguments": "n"},
                    {"name": None, "arguments": "       "},
                    {"name": None, "arguments": " return"},
                    {"name": None, "arguments": " a"},
                    {"name": None, "arguments": " -"},
                    {"name": None, "arguments": " b"},
                    {"name": None, "arguments": "\\n"},
                    {"name": None, "arguments": "   "},
                    {"name": None, "arguments": " if"},
                    {"name": None, "arguments": " operation"},
                    {"name": None, "arguments": " =="},
                    {"name": None, "arguments": " '"},
                    {"name": None, "arguments": "multiply"},
                    {"name": None, "arguments": "'\\"},
                    {"name": None, "arguments": "n"},
                    {"name": None, "arguments": "       "},
                    {"name": None, "arguments": " return"},
                    {"name": None, "arguments": " a"},
                    {"name": None, "arguments": " *"},
                    {"name": None, "arguments": " b"},
                    {"name": None, "arguments": "\\n"},
                    {"name": None, "arguments": "   "},
                    {"name": None, "arguments": " if"},
                    {"name": None, "arguments": " operation"},
                    {"name": None, "arguments": " =="},
                    {"name": None, "arguments": " '"},
                    {"name": None, "arguments": "divide"},
                    {"name": None, "arguments": "'\\"},
                    {"name": None, "arguments": "n"},
                    {"name": None, "arguments": "       "},
                    {"name": None, "arguments": " return"},
                    {"name": None, "arguments": " a"},
                    {"name": None, "arguments": " /"},
                    {"name": None, "arguments": " b"},
                    {"name": None, "arguments": "\\n"},
                    {"name": None, "arguments": "   "},
                    {"name": None, "arguments": " return"},
                    {"name": None, "arguments": " '"},
                    {"name": None, "arguments": "Invalid"},
                    {"name": None, "arguments": " operation"},
                    {"name": None, "arguments": "'"},
                    {"name": None, "arguments": '"}'},
                ],
            },
        ]

    def test_parse_stream(self):
        for test_case in self.test_cases:
            with self.subTest(msg=test_case["name"]):
                parser = BaseToolParser("<tool_call>", "</tool_call>")
                outputs = []

                for chunk in test_case["chunks"]:
                    result = parser.parse_stream(chunk)
                    if result:
                        outputs.append(result)

                self.assertEqual(
                    len(outputs),
                    len(test_case["expected_outputs"]),
                    f"Expected {len(test_case['expected_outputs'])} outputs, got {len(outputs)}",
                )

                for i, (output, expected) in enumerate(zip(outputs, test_case["expected_outputs"])):
                    self.assertEqual(
                        output, expected, f"Chunk {i}: Expected {expected}, got {output}"
                    )


class TestBaseThinkingParser(unittest.TestCase):
    def setUp(self):
        self.parser = BaseThinkingParser("<think>", "</think>")

    def test_parse(self):
        test_cases = [
            {
                "name": "normal case",
                "content": "prefix<think>thinking content</think>suffix",
                "expected": ("thinking content", "prefixsuffix"),
            },
            {
                "name": "closing only",
                "content": "thinking content</think>suffix",
                "expected": ("thinking content", "suffix"),
            },
            {"name": "no tags", "content": "normal content", "expected": (None, "normal content")},
            {
                "name": "opening no closing",
                "content": "prefix<think>thinking content",
                "expected": (None, "prefix<think>thinking content"),
            },
            {
                "name": "empty thinking",
                "content": "prefix<think></think>suffix",
                "expected": ("", "prefixsuffix"),
            },
        ]

        for case in test_cases:
            with self.subTest(msg=case["name"]):
                thinking, remaining = self.parser.parse(case["content"])
                self.assertEqual(thinking, case["expected"][0])
                self.assertEqual(remaining, case["expected"][1])

    def test_parse_stream(self):
        test_cases = [
            {
                "name": "normal case",
                "chunk": "prefix<think>thinking</think>suffix",
                "expected": ("prefixsuffix", True),
            },
            {
                "name": "opening only",
                "chunk": "prefix<think>thinking",
                "expected": ({"reasoning_content": "thinking"}, False),
            },
            {
                "name": "closing in thinking",
                "chunk": "more thinking</think>suffix",
                "setup": lambda p: setattr(p, "is_thinking", True),
                "expected": ({"reasoning_content": "more thinking", "content": "suffix"}, True),
            },
            {
                "name": "closing only",
                "chunk": "thinking content</think>suffix",
                "expected": ({"reasoning_content": "thinking content", "content": "suffix"}, True),
            },
            {"name": "no tags", "chunk": "normal content", "expected": (None, False)},
        ]

        for case in test_cases:
            with self.subTest(msg=case["name"]):
                if "setup" in case:
                    case["setup"](self.parser)
                result, complete = self.parser.parse_stream(case["chunk"])
                self.assertEqual(result, case["expected"][0])
                self.assertEqual(complete, case["expected"][1])
                # Reset parser state
                self.parser.is_thinking = False


if __name__ == "__main__":
    unittest.main()
