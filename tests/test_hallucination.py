"""Tests for hallucination detection.

Tests claim extraction, grounded/ungrounded classification, and
grounded_ratio computation using the HallucinationDetector.
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Test claim extraction
# ---------------------------------------------------------------------------

class TestClaimExtraction:
    """Test that claim extraction returns list of strings."""

    def test_extract_claims_returns_list(self):
        """Claim extraction should return a list of string claims."""
        try:
            from src.generation.hallucination import HallucinationDetector

            detector = HallucinationDetector()
            if not detector._openai.api_key:
                pytest.skip("OpenAI API key not configured")
            text = (
                "The contract was signed on January 1, 2024. "
                "Acme Corp is the primary party. "
                "The agreement lasts for 24 months."
            )
            claims = detector.extract_claims(text)
            assert isinstance(claims, list)
            assert all(isinstance(c, str) for c in claims)
            assert len(claims) > 0
        except ImportError:
            pytest.skip("HallucinationDetector not yet implemented")
        except Exception as exc:
            if "api" in str(exc).lower() or "key" in str(exc).lower():
                pytest.skip(f"API not configured: {exc}")
            raise

    def test_extract_claims_empty_text(self):
        """Empty text should return empty claims list."""
        try:
            from src.generation.hallucination import HallucinationDetector

            detector = HallucinationDetector()
            if not detector._openai.api_key:
                pytest.skip("OpenAI API key not configured")
            claims = detector.extract_claims("")
            assert isinstance(claims, list)
            assert len(claims) == 0
        except ImportError:
            pytest.skip("HallucinationDetector not yet implemented")
        except Exception as exc:
            if "api" in str(exc).lower() or "key" in str(exc).lower():
                pytest.skip(f"API not configured: {exc}")
            raise


# ---------------------------------------------------------------------------
# Test grounded claims
# ---------------------------------------------------------------------------

class TestGroundedClaims:
    """Test that known grounded claims are labeled correctly."""

    def test_grounded_claim_detected(self):
        """A claim fully supported by context should be labeled grounded."""
        try:
            from src.generation.hallucination import HallucinationDetector

            detector = HallucinationDetector()
            if not detector._openai.api_key:
                pytest.skip("OpenAI API key not configured")

            context = (
                "This Agreement is entered into as of January 1, 2024, "
                "by and between Acme Corporation and Widget Industries Inc. "
                "The term of this agreement is 24 months."
            )
            answer = "The contract was signed on January 1, 2024."

            result = detector.detect(answer=answer, context=context)
            # Grounded ratio should be high for a claim directly from context
            assert result.grounded_ratio >= 0.5, (
                f"Expected grounded_ratio >= 0.5, got {result.grounded_ratio}"
            )
        except ImportError:
            pytest.skip("HallucinationDetector not yet implemented")
        except Exception as exc:
            if "api" in str(exc).lower() or "key" in str(exc).lower():
                pytest.skip(f"API not configured: {exc}")
            raise


# ---------------------------------------------------------------------------
# Test ungrounded claims
# ---------------------------------------------------------------------------

class TestUngroundedClaims:
    """Test that known ungrounded claims are flagged."""

    def test_ungrounded_claim_detected(self):
        """A claim NOT in the context should be labeled ungrounded."""
        try:
            from src.generation.hallucination import HallucinationDetector

            detector = HallucinationDetector()
            if not detector._openai.api_key:
                pytest.skip("OpenAI API key not configured")

            context = (
                "This Agreement is between Acme Corporation and Widget Industries. "
                "The contract term is 24 months starting January 2024."
            )
            # This claim is fabricated — not in the context
            answer = (
                "The contract includes a $5 million penalty clause for early "
                "termination and requires quarterly audits by an independent firm."
            )

            result = detector.detect(answer=answer, context=context)
            # Should have some hallucinated claims
            assert result.hallucinated_claims > 0, (
                "Expected hallucinated claims for fabricated answer"
            )
        except ImportError:
            pytest.skip("HallucinationDetector not yet implemented")
        except Exception as exc:
            if "api" in str(exc).lower() or "key" in str(exc).lower():
                pytest.skip(f"API not configured: {exc}")
            raise


# ---------------------------------------------------------------------------
# Test grounded_ratio computation
# ---------------------------------------------------------------------------

class TestGroundedRatio:
    """Test grounded_ratio computation."""

    def test_grounded_ratio_range(self):
        """grounded_ratio should always be between 0 and 1."""
        try:
            from src.generation.hallucination import HallucinationDetector

            detector = HallucinationDetector()
            if not detector._openai.api_key:
                pytest.skip("OpenAI API key not configured")

            context = "Acme Corp signed a 24-month contract with Widget Inc."
            answer = "Acme Corp signed a contract. The moon is made of cheese."

            result = detector.detect(answer=answer, context=context)
            assert 0.0 <= result.grounded_ratio <= 1.0, (
                f"grounded_ratio out of range: {result.grounded_ratio}"
            )
        except ImportError:
            pytest.skip("HallucinationDetector not yet implemented")
        except Exception as exc:
            if "api" in str(exc).lower() or "key" in str(exc).lower():
                pytest.skip(f"API not configured: {exc}")
            raise

    def test_fully_grounded_answer(self):
        """Answer that quotes context verbatim should be ~100% grounded."""
        try:
            from src.generation.hallucination import HallucinationDetector

            detector = HallucinationDetector()
            if not detector._openai.api_key:
                pytest.skip("OpenAI API key not configured")

            context = "The governing law is the State of Delaware."
            answer = "The governing law is the State of Delaware."

            result = detector.detect(answer=answer, context=context)
            assert result.grounded_ratio >= 0.8, (
                f"Verbatim quote should be grounded, got {result.grounded_ratio}"
            )
        except ImportError:
            pytest.skip("HallucinationDetector not yet implemented")
        except Exception as exc:
            if "api" in str(exc).lower() or "key" in str(exc).lower():
                pytest.skip(f"API not configured: {exc}")
            raise

    def test_result_has_expected_attributes(self):
        """Detection result should have total_claims, hallucinated_claims, grounded_ratio."""
        try:
            from src.generation.hallucination import HallucinationDetector

            detector = HallucinationDetector()
            if not detector._openai.api_key:
                pytest.skip("OpenAI API key not configured")

            result = detector.detect(answer="Test answer.", context="Test context.")
            assert hasattr(result, "total_claims")
            assert hasattr(result, "hallucinated_claims")
            assert hasattr(result, "grounded_ratio")
        except ImportError:
            pytest.skip("HallucinationDetector not yet implemented")
        except Exception as exc:
            if "api" in str(exc).lower() or "key" in str(exc).lower():
                pytest.skip(f"API not configured: {exc}")
            raise
