# Fix Plan: Token Vocabulary Mismatch

## Problem Statement
The smoke test generates input token IDs that exceed the model's vocabulary size, causing metrics computation to fail with:
```
RuntimeError: index 50256 is out of bounds for dimension 2 with size 768
```

## Root Cause
The smoke test API (`/api/v1/smoke/run`) generates random token IDs using a hardcoded vocabulary size that doesn't match the actual model vocabulary. Token ID 50256 (GPT-2 EOS token) exceeds the model's vocab size of 768.

## Solution Strategy
Add input validation and sanitization to ensure all token IDs are within the model's vocabulary range.

## Tasks

### Task 1: Add Vocabulary Size to Smoke Test Response
**File:** `backend/app/api/v1/smoke.py`
**Goal:** Include model vocabulary size in smoke test response

**Steps:**
1. Read the smoke test endpoint implementation
2. Extract vocabulary size from the compiled model
3. Add `vocab_size` field to response schema
4. Return vocab_size in response payload

**Verification:**
```bash
# Run smoke test and check response includes vocab_size
curl -X POST http://localhost:8000/api/v1/smoke/run \
  -H "Content-Type: application/json" \
  -d '{"graph_id": "test", "max_seq": 10}' | jq '.vocab_size'
```

### Task 2: Add Token ID Validation in Execution Runner
**File:** `backend/app/services/execution_runner.py`
**Goal:** Validate and clamp token IDs to vocabulary range before execution

**Steps:**
1. After receiving input_data, extract input_ids
2. Get vocabulary size from compiled model
3. Add validation: check if any token_id >= vocab_size
4. If invalid tokens found:
   - Log warning with details
   - Clamp to valid range: `token_id % vocab_size`
5. Use sanitized token_ids for execution

**Code location:** Lines 116-121 (after preparing input tensor)

**Verification:**
- Unit test with out-of-range token IDs
- Verify clamping works correctly
- Verify metrics computation succeeds

### Task 3: Add Defensive Handling in Metrics Computation
**File:** `backend/app/services/execution_runner.py`
**Goal:** Add additional safety checks in metrics computation try block

**Steps:**
1. Before `gather()` operation (line 294), add assertion:
   - Check targets tensor max value < vocab_size
   - Log error if assertion fails
2. Add more specific exception types in except block:
   - Catch RuntimeError separately for gather errors
   - Catch IndexError for list access errors
   - Keep general Exception as fallback

**Code location:** Lines 284-349 (metrics computation block)

**Verification:**
- Smoke test should complete successfully
- Browser console should show non-null summary values
- Charts should render with data

### Task 4: Update Frontend Error Handling
**File:** `frontend/src/hooks/useSmokeTest.ts`
**Goal:** Add validation for received summary data

**Steps:**
1. In onComplete callback, check if summary has valid data
2. If all metric values are null, log warning
3. Consider adding user-facing error message

**Verification:**
- Run smoke test
- Check browser console for validation messages
- Verify charts show data when metrics are valid

### Task 5: Integration Test
**Goal:** Verify end-to-end fix works

**Steps:**
1. Start backend server
2. Start frontend dev server
3. Run smoke test from UI
4. Verify backend logs show no token validation warnings
5. Verify browser console shows:
   - Non-null summary values (ce, ppl, bits_per_token)
   - Per-token arrays have data
6. Verify UI displays:
   - All 6 metric tiles show numbers (not "—")
   - All 4 charts render with data (not placeholder)

**Success Criteria:**
- ✅ No RuntimeError in backend logs
- ✅ Summary values are numbers, not null
- ✅ Charts render with visualization data
- ✅ Metric tiles show computed values

## Notes
- This fix handles the immediate symptom (token ID mismatch)
- The root issue is that smoke test should use model-specific vocab size
- Consider updating smoke test generation in future to use actual model vocabulary
