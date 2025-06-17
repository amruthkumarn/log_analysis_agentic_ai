#!/usr/bin/env python3
"""
Generate test log data with multiple sessions and proper URC-UID hierarchies.
Each session will have 4 levels of API calls with realistic error scenarios.
"""

import random
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import uuid

class LogGenerator:
    def __init__(self):
        self.services = [
            "3scale API Gateway",
            "api_gateway", 
            "payment_service",
            "order_service",
            "user_profile_service",
            "inventory_service",
            "notification_service",
            "audit_service",
            "tibco_businessworks"
        ]
        
        self.endpoints = [
            "/api/v1/transfer",
            "/api/v1/payment",
            "/api/v1/orders",
            "/api/v1/profile",
            "/api/v1/inventory",
            "/api/v1/notifications",
            "/api/v1/audit",
            "/api/v2/orders",
            "/api/v2/payments"
        ]
        
        self.transaction_types = [
            "transfer", "payment", "debit", "credit", "refund",
            "create_order", "update_order", "cancel_order",
            "profile_check", "inventory_check", "notification",
            "audit_log", "validation", "authorization"
        ]
        
        self.error_scenarios = [
            ("timeout", "Connection timeout after 5000ms"),
            ("authentication", "Invalid credentials provided"),
            ("rate_limit", "Rate limit exceeded for API endpoint"),
            ("validation", "Invalid request parameters"),
            ("business", "Insufficient funds for transaction"),
            ("connection", "Database connection failed"),
            ("permission", "Access denied for requested resource")
        ]

    def generate_session_id(self) -> str:
        """Generate a session ID similar to abc123, def456, etc."""
        letters = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=3))
        numbers = ''.join(random.choices('0123456789', k=3))
        return f"{letters}{numbers}"

    def generate_urc_uid_hierarchy(self, session_id: str) -> List[Tuple[str, str, int, str]]:
        """Generate a 4-level URC-UID hierarchy with intuitive naming."""
        hierarchy = []
        
        # Level 0: Root (login/session start)
        root_urc = f"urc-root-{session_id}"
        hierarchy.append((root_urc, None, 0, "login"))
        
        # Level 1: Main API requests (2-3 calls) - Make session-specific
        level1_count = random.randint(2, 3)
        level1_urcs = []
        api_types = ["payment", "order", "profile"]
        
        for i in range(level1_count):
            api_type = api_types[i % len(api_types)]
            urc = f"urc-{api_type}-req-{session_id}-{i+1:02d}"
            uid = root_urc
            hierarchy.append((urc, uid, 1, f"{api_type}_request"))
            level1_urcs.append(urc)
        
        # Level 2: Service calls (1-2 per level 1) - Make session-specific
        level2_urcs = []
        service_types = ["validation", "processing", "auth"]
        
        for parent_urc in level1_urcs:
            level2_count = random.randint(1, 2)
            parent_type = parent_urc.split('-')[1]  # Extract payment/order/profile
            
            for i in range(level2_count):
                service_type = service_types[i % len(service_types)]
                urc = f"urc-{parent_type}-{service_type}-{session_id}-{len(level2_urcs)+1:02d}"
                uid = parent_urc
                hierarchy.append((urc, uid, 2, f"{service_type}_service"))
                level2_urcs.append(urc)
        
        # Level 3: Internal operations (0-2 per level 2) - Make session-specific
        level3_urcs = []
        internal_types = ["db", "cache", "queue"]
        
        for parent_urc in level2_urcs:
            level3_count = random.randint(0, 2)
            parent_parts = parent_urc.split('-')
            parent_type = parent_parts[1]  # payment/order/profile
            parent_service = parent_parts[2]  # validation/processing/auth
            
            for i in range(level3_count):
                internal_type = internal_types[i % len(internal_types)]
                urc = f"urc-{parent_type}-{parent_service}-{internal_type}-{session_id}-{len(level3_urcs)+1:02d}"
                uid = parent_urc
                hierarchy.append((urc, uid, 3, f"{internal_type}_operation"))
                level3_urcs.append(urc)
        
        # Level 4: Deep operations (0-1 per level 3) - Make session-specific
        deep_types = ["retry", "fallback", "cleanup"]
        
        for parent_urc in level3_urcs:
            if random.random() < 0.6:  # 60% chance of level 4
                parent_parts = parent_urc.split('-')
                parent_type = parent_parts[1]
                parent_service = parent_parts[2]
                parent_internal = parent_parts[3]
                deep_type = random.choice(deep_types)
                
                urc = f"urc-{parent_type}-{parent_service}-{parent_internal}-{deep_type}-{session_id}-01"
                uid = parent_urc
                hierarchy.append((urc, uid, 4, f"{deep_type}_operation"))
        
        return hierarchy

    def generate_log_entry(self, timestamp: datetime, level: str, service: str, 
                          message: str) -> str:
        """Generate a single log entry in the expected format."""
        ts_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        return f"[{ts_str}] [{level}] {service}: {message}"

    def generate_session_logs(self, session_id: str, start_time: datetime) -> List[str]:
        """Generate complete log sequence for a session with URC-UID hierarchy."""
        logs = []
        current_time = start_time
        
        # Generate hierarchy
        hierarchy = self.generate_urc_uid_hierarchy(session_id)
        
        # 1. Session start/login - ONLY place where session_id appears
        root_urc = hierarchy[0][0]
        user_id = f"user_{random.randint(1000, 9999)}"
        cif_id = f"{random.randint(100000, 999999)}"
        
        message = f"User '{user_id}' logged in session_id={session_id} cif_id={cif_id} URC={root_urc} response_time={random.randint(100, 300)}ms"
        logs.append(self.generate_log_entry(current_time, "INFO", "api_gateway", message))
        current_time += timedelta(seconds=random.randint(1, 3))
        
        # 2. Generate API calls following hierarchy - NO session_id in these
        error_introduced = False
        for urc, uid, level, operation_type in hierarchy[1:]:  # Skip root
            # Request
            endpoint = random.choice(self.endpoints)
            transaction_type = random.choice(self.transaction_types)
            service = random.choice(self.services)
            
            # Build message without session_id - ALWAYS include UID for correlation
            message = f"Request received for {endpoint}, URC={urc}, UID={uid}, transaction_type={transaction_type}"
            
            logs.append(self.generate_log_entry(current_time, "INFO", service, message))
            current_time += timedelta(milliseconds=random.randint(100, 500))
            
            # Introduce errors at deeper levels (30% chance)
            if level >= 2 and random.random() < 0.3 and not error_introduced:
                error_type, error_msg = random.choice(self.error_scenarios)
                error_level = "ERROR" if error_type in ["timeout", "connection", "authentication"] else "WARN"
                
                # Create error URC based on current URC
                error_urc = urc.replace("urc-", "urc-err-")
                
                error_message = f"{error_msg}, URC={error_urc}, UID={uid}, error_code={error_type.upper()}_{random.randint(1, 999):03d}"
                
                logs.append(self.generate_log_entry(current_time, error_level, service, error_message))
                current_time += timedelta(milliseconds=random.randint(50, 200))
                error_introduced = True
                
                # Propagate error up the chain
                if level > 1 and uid:
                    parent_service = random.choice(self.services)
                    # Create failure URC for parent
                    parent_fail_urc = uid.replace("urc-", "urc-fail-")
                    propagation_msg = f"Downstream service error detected, URC={parent_fail_urc}, UID={urc}, reason={error_msg}"
                    
                    logs.append(self.generate_log_entry(current_time, "ERROR", parent_service, propagation_msg))
                    current_time += timedelta(milliseconds=random.randint(50, 200))
            
            # Response (if no error or after error handling)
            response_time = random.randint(150, 800) if not error_introduced else random.randint(500, 2000)
            status_code = 200 if not error_introduced else random.choice([400, 401, 403, 500, 503])
            
            response_message = f"Response sent for {endpoint}, URC={urc}, UID={uid}, status_code={status_code}, response_time={response_time}ms"
            
            logs.append(self.generate_log_entry(current_time, "INFO", service, response_message))
            current_time += timedelta(seconds=random.randint(1, 2))
        
        return logs

    def generate_multiple_sessions(self, num_sessions: int = 5) -> Dict[str, List[str]]:
        """Generate logs for multiple sessions."""
        all_logs = {
            "3scale_api_gateway.log": [],
            "payment_service.log": [],
            "tibco_businessworks.log": []
        }
        
        base_time = datetime(2024, 3, 20, 10, 15, 0)
        
        for i in range(num_sessions):
            session_id = self.generate_session_id()
            session_start = base_time + timedelta(minutes=i*10)
            session_logs = self.generate_session_logs(session_id, session_start)
            
            # Distribute logs across files based on service
            for log in session_logs:
                if "api_gateway" in log or "3scale" in log:
                    all_logs["3scale_api_gateway.log"].append(log)
                elif "payment" in log or "order" in log:
                    all_logs["payment_service.log"].append(log)
                else:
                    all_logs["tibco_businessworks.log"].append(log)
        
        return all_logs

def main():
    generator = LogGenerator()
    
    # Generate logs for 5 different sessions
    print("Generating test logs with intuitive URC-UID hierarchies...")
    all_logs = generator.generate_multiple_sessions(5)
    
    # Write to log files
    for filename, logs in all_logs.items():
        filepath = f"logs/{filename}"
        with open(filepath, 'w') as f:
            for log in logs:
                f.write(log + '\n')
        print(f"Generated {len(logs)} log entries in {filepath}")
    
    print("\nGenerated sessions with intuitive URC-UID relationships:")
    print("- session_id only appears in 'logged in' messages")
    print("- URC naming: urc-{type}-{operation}-{sequence}")
    print("- Clear parent-child relationships via URC-UID")
    print("- 4 levels: root -> request -> service -> internal -> deep")
    print("- Example hierarchy:")
    print("  Level 0: urc-root-abc123")
    print("  Level 1: urc-payment-req-01 (UID=urc-root-abc123)")
    print("  Level 2: urc-payment-validation-01 (UID=urc-payment-req-01)")
    print("  Level 3: urc-payment-validation-db-01 (UID=urc-payment-validation-01)")
    print("  Level 4: urc-payment-validation-db-retry-01 (UID=urc-payment-validation-db-01)")

if __name__ == "__main__":
    main() 