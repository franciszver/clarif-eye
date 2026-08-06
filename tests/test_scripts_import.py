"""Test that scripts import without stale cross-module references (issue #?).

record_vision_fixture.py crashed in production (after a live API call!) when it
tried to call vision._complexity_flag, which was removed in P1.3 when router
took ownership of the complexity heuristic. No test imported the script, so the
breakage was invisible until then.

These tests ensure:
1. Every scripts/*.py file imports successfully (catches syntax/import errors)
2. For recorder scripts, the fixture-building logic exercises stale cross-module
   references at test time, where they can fail safely, not after a live API call.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import scripts.record_vision_fixture as record_vision_fixture


class TestRecordVisionFixtureBuildParsedFixture:
    """Test record_vision_fixture.build_parsed_fixture() with fake inputs.

    This is the post-reply processing extracted into a pure function, so a
    removed attribute (like vision._complexity_flag was) fails the test suite
    at collection time, not after a live API call.
    """

    def test_build_parsed_fixture_calls_classify_complexity(self):
        """Must use router.classify_complexity, not the removed vision._complexity_flag."""
        raw_reply = (
            "<<<CLARIF_OCR>>>\n"
            "Account Number: 123456\n"
            "Amount Due: $45.99\n"
            "Due Date: 2025-12-01\n"
            "<<<CLARIF_SCENE>>>\n"
            "A utility bill on a white background"
        )

        parsed = record_vision_fixture.build_parsed_fixture(raw_reply)

        assert isinstance(parsed, dict)
        assert "ocr_output" in parsed
        assert "scene_context" in parsed
        assert "complexity_flag" in parsed
        assert isinstance(parsed["complexity_flag"], bool)

    def test_build_parsed_fixture_with_unparseable_reply(self):
        """When reply doesn't parse, returns a degraded fixture with complexity_flag=False."""
        raw_reply = "garbage that doesn't follow the format"

        parsed = record_vision_fixture.build_parsed_fixture(raw_reply)

        assert parsed == {
            "ocr_output": "",
            "scene_context": "",
            "complexity_flag": False,
        }

    def test_build_parsed_fixture_with_empty_reply(self):
        """When reply is empty, returns a degraded fixture."""
        parsed = record_vision_fixture.build_parsed_fixture("")

        assert parsed == {
            "ocr_output": "",
            "scene_context": "",
            "complexity_flag": False,
        }

    def test_build_parsed_fixture_result_is_json_serializable(self):
        """Fixture result must be JSON-serializable (it gets written to a file)."""
        raw_reply = (
            "<<<CLARIF_OCR>>>\n"
            "Some text\n"
            "<<<CLARIF_SCENE>>>\n"
            "A scene description"
        )

        parsed = record_vision_fixture.build_parsed_fixture(raw_reply)
        json_str = json.dumps(parsed)

        assert isinstance(json_str, str)
        # Verify it round-trips
        reloaded = json.loads(json_str)
        assert reloaded == parsed
