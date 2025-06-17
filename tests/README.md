# Test Data Generation

This directory contains tools and utilities for generating test data for the log analyzer system.

## Data Generators

### `data_generators/generate_test_logs.py`

A sophisticated log generator that creates realistic test data with the following features:

- Multiple sessions with unique session IDs
- 4-level URC-UID hierarchy for request tracking
- Realistic error scenarios and propagation
- Distributed logs across multiple services
- Proper timestamp sequencing
- Realistic API endpoints and transaction types

#### Usage

```bash
python tests/data_generators/generate_test_logs.py
```

This will generate three log files in the `logs/` directory:
- `3scale_api_gateway.log`
- `payment_service.log`
- `tibco_businessworks.log`

#### Generated Data Structure

Each session follows this hierarchy:
1. Level 0: Root (login/session start)
2. Level 1: Main API requests (2-3 calls)
3. Level 2: Service calls (1-2 per level 1)
4. Level 3: Internal operations (0-2 per level 2)
5. Level 4: Deep operations (0-1 per level 3)

#### Example URC-UID Chain

```
Level 0: urc-root-abc123
Level 1: urc-payment-req-01 (UID=urc-root-abc123)
Level 2: urc-payment-validation-01 (UID=urc-payment-req-01)
Level 3: urc-payment-validation-db-01 (UID=urc-payment-validation-01)
Level 4: urc-payment-validation-db-retry-01 (UID=urc-payment-validation-db-01)
```

## Test Data Usage

The generated test data is used for:
- Testing the log analysis system
- Verifying URC-UID correlation
- Testing error detection and propagation
- Validating log parsing and processing
- Testing the analysis agent's capabilities 