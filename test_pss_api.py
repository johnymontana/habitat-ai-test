# /// script
# requires-python = ">=3.10"
# dependencies = ["requests"]
# ///
"""
Test script for PSS (Persistent Session Store) API v1.
Base URL: https://pss.versino.de/api/v1

Tests:
  1. Health check
  2. Create a new session via /run (first turn, no session_id)
  3. Continue session via /run (subsequent turn with session_id)
  4. Store a response via /store
  5. Resume session and verify context includes stored memories
  6. Inline pattern (response passed back in next /run call)
  7. Error handling (missing key, bad session, missing fields)
"""

import requests
import sys
import os
import json
import time

BASE_URL = os.environ.get("PSS_BASE_URL", "https://pss.versino.de/api/v1")
API_KEY = os.environ.get("PSS_API_KEY", "")

# Allow overriding via CLI arg
if len(sys.argv) > 1:
    API_KEY = sys.argv[1]

if not API_KEY:
    print("Error: PSS_API_KEY environment variable is required.")
    print("Usage: PSS_API_KEY=pss_... uv run test_pss_api.py")
    sys.exit(1)


def headers(api_key=API_KEY):
    return {
        "X-API-Key": api_key,
        "Content-Type": "application/json",
    }


def print_result(name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}")
    if detail:
        for line in detail.strip().split("\n"):
            print(f"         {line}")


def test_health():
    """Test 1: Health endpoint (no auth required)."""
    print("\n--- Test 1: Health Check ---")
    resp = requests.get(f"{BASE_URL}/health")
    print_result(
        "GET /health returns 200",
        resp.status_code == 200,
        f"Status: {resp.status_code}",
    )
    data = resp.json()
    print_result(
        "Response has 'status' field",
        "status" in data,
        json.dumps(data, indent=2),
    )
    return data


def test_run_new_session():
    """Test 2: Create new session via /run (no session_id)."""
    print("\n--- Test 2: /run — New Session (first turn) ---")
    payload = {"message": "Was kostet das Hosting?"}
    resp = requests.post(f"{BASE_URL}/run", headers=headers(), json=payload)
    print_result(
        "POST /run returns 200",
        resp.status_code == 200,
        f"Status: {resp.status_code} | Body: {resp.text[:300]}",
    )
    if resp.status_code != 200:
        return None

    data = resp.json()
    print_result(
        "Response contains 'session_id'",
        "session_id" in data,
        f"session_id: {data.get('session_id', 'MISSING')}",
    )
    print_result(
        "Response contains 'context'",
        "context" in data,
        f"context (first 200 chars): {str(data.get('context', ''))[:200]}",
    )
    return data


def test_run_continue_session(session_id):
    """Test 3: Continue an existing session via /run with session_id."""
    print("\n--- Test 3: /run — Continue Session ---")
    payload = {
        "message": "Welche OCR-Technologie wird verwendet?",
        "session_id": session_id,
    }
    resp = requests.post(f"{BASE_URL}/run", headers=headers(), json=payload)
    print_result(
        "POST /run with session_id returns 200",
        resp.status_code == 200,
        f"Status: {resp.status_code}",
    )
    if resp.status_code != 200:
        return None

    data = resp.json()
    print_result(
        "Returned session_id matches",
        data.get("session_id") == session_id,
        f"Expected: {session_id}, Got: {data.get('session_id')}",
    )
    print_result(
        "Context returned",
        bool(data.get("context")),
        f"context (first 200 chars): {str(data.get('context', ''))[:200]}",
    )
    return data


def test_store(session_id):
    """Test 4: Store an LLM response via /store."""
    print("\n--- Test 4: /store — Save Response ---")
    payload = {
        "session_id": session_id,
        "response": "Das Hosting kostet 600 EUR pro Monat auf ECS Fargate.",
    }
    resp = requests.post(f"{BASE_URL}/store", headers=headers(), json=payload)
    print_result(
        "POST /store returns 200",
        resp.status_code == 200,
        f"Status: {resp.status_code} | Body: {resp.text[:300]}",
    )
    return resp.status_code == 200


def test_inline_pattern():
    """Test 5: Inline pattern — pass previous response in next /run call."""
    print("\n--- Test 5: Inline Pattern (response passed to next /run) ---")

    # Turn 1: new session
    resp1 = requests.post(
        f"{BASE_URL}/run",
        headers=headers(),
        json={"message": "Wann ist Go-Live?"},
    )
    if resp1.status_code != 200:
        print_result("Turn 1 /run", False, f"Status: {resp1.status_code}")
        return None

    data1 = resp1.json()
    session_id = data1["session_id"]
    simulated_llm_response = "Go-Live ist am 15. März 2026."

    # Turn 2: pass previous response inline
    resp2 = requests.post(
        f"{BASE_URL}/run",
        headers=headers(),
        json={
            "message": "Gibt es Risiken für den Go-Live Termin?",
            "session_id": session_id,
            "response": simulated_llm_response,
        },
    )
    print_result(
        "Turn 2 /run with response returns 200",
        resp2.status_code == 200,
        f"Status: {resp2.status_code}",
    )
    if resp2.status_code != 200:
        return None

    data2 = resp2.json()
    print_result(
        "Session persisted across turns",
        data2.get("session_id") == session_id,
    )
    print_result(
        "Context returned for turn 2",
        bool(data2.get("context")),
        f"context (first 200 chars): {str(data2.get('context', ''))[:200]}",
    )
    return data2


def test_resume_session_with_memory(session_id):
    """Test 6: Resume a session and check that stored memories appear in context."""
    print("\n--- Test 6: Resume Session — Verify Memory ---")
    time.sleep(1)  # brief pause to let store propagate
    payload = {
        "message": "Was haben wir über Hosting besprochen?",
        "session_id": session_id,
    }
    resp = requests.post(f"{BASE_URL}/run", headers=headers(), json=payload)
    print_result(
        "POST /run (resume) returns 200",
        resp.status_code == 200,
        f"Status: {resp.status_code}",
    )
    if resp.status_code != 200:
        return None

    data = resp.json()
    context = str(data.get("context", ""))
    has_memories = "memories" in context.lower() or "relevant" in context.lower()
    print_result(
        "Context contains memory/relevant indicators",
        has_memories,
        f"context (first 300 chars): {context[:300]}",
    )
    return data


def test_error_missing_api_key():
    """Test 7a: Request without API key should return 401."""
    print("\n--- Test 7a: Error — Missing API Key ---")
    resp = requests.post(
        f"{BASE_URL}/run",
        headers={"Content-Type": "application/json"},
        json={"message": "test"},
    )
    print_result(
        "Missing key returns 401",
        resp.status_code == 401,
        f"Status: {resp.status_code} | Body: {resp.text[:200]}",
    )


def test_error_invalid_api_key():
    """Test 7b: Request with invalid API key should return 401."""
    print("\n--- Test 7b: Error — Invalid API Key ---")
    resp = requests.post(
        f"{BASE_URL}/run",
        headers=headers(api_key="pss_invalid_key_12345"),
        json={"message": "test"},
    )
    print_result(
        "Invalid key returns 401",
        resp.status_code == 401,
        f"Status: {resp.status_code} | Body: {resp.text[:200]}",
    )


def test_error_missing_message():
    """Test 7c: /run without message field."""
    print("\n--- Test 7c: Error — Missing Message ---")
    resp = requests.post(f"{BASE_URL}/run", headers=headers(), json={})
    print_result(
        "Missing message returns 4xx",
        resp.status_code in (400, 422),
        f"Status: {resp.status_code} | Body: {resp.text[:200]}",
    )


def test_error_store_missing_fields():
    """Test 7d: /store without required fields."""
    print("\n--- Test 7d: Error — /store Missing Fields ---")
    resp = requests.post(f"{BASE_URL}/store", headers=headers(), json={})
    print_result(
        "/store with empty body returns 4xx",
        resp.status_code in (400, 422),
        f"Status: {resp.status_code} | Body: {resp.text[:200]}",
    )


def main():
    print("=" * 60)
    print("PSS API v1 Test Suite")
    print(f"Base URL: {BASE_URL}")
    print(f"API Key:  {API_KEY[:12]}...{API_KEY[-4:]}")
    print("=" * 60)

    # 1. Health check (no auth needed)
    test_health()

    # 2. New session
    run_result = test_run_new_session()
    if not run_result:
        print("\n*** /run failed — cannot continue session tests. Check API key. ***")
        # Still run error tests
        test_error_missing_api_key()
        test_error_invalid_api_key()
        test_error_missing_message()
        test_error_store_missing_fields()
        print("\n" + "=" * 60)
        print("Done (partial — auth issue).")
        return

    session_id = run_result["session_id"]

    # 3. Continue session
    test_run_continue_session(session_id)

    # 4. Store response
    test_store(session_id)

    # 5. Inline pattern
    test_inline_pattern()

    # 6. Resume and verify memory
    test_resume_session_with_memory(session_id)

    # 7. Error handling
    test_error_missing_api_key()
    test_error_invalid_api_key()
    test_error_missing_message()
    test_error_store_missing_fields()

    print("\n" + "=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
