import json
import random
from typing import List, Dict


class TrainingDataGenerator:
    """Generate generic synthetic training data that works with any API"""

    def __init__(self):
        # Generic flow patterns that apply to ANY REST API
        self.flow_patterns = {
            # Basic REST patterns
            "list_to_detail": ["GET /{resource}", "GET /{resource}/{id}"],
            "detail_to_update": [
                "GET /{resource}/{id}",
                "PUT /{resource}/{id}",
                "GET /{resource}/{id}",
            ],
            "create_workflow": [
                "GET /{resource}",
                "POST /{resource}",
                "GET /{resource}/{id}",
            ],
            "delete_workflow": [
                "GET /{resource}",
                "GET /{resource}/{id}",
                "DELETE /{resource}/{id}",
                "GET /{resource}",
            ],
            # Search and filter patterns
            "search_then_detail": [
                "GET /{resource}",
                "GET /{resource}?search={query}",
                "GET /{resource}/{id}",
            ],
            "filter_then_action": [
                "GET /{resource}?filter={filter}",
                "GET /{resource}/{id}",
                "PUT /{resource}/{id}",
            ],
            # Pagination patterns
            "pagination_browse": [
                "GET /{resource}",
                "GET /{resource}?page=2",
                "GET /{resource}?page=3",
                "GET /{resource}?page=4",
                "GET /{resource}/{id}",
            ],
            # Nested resource patterns
            "nested_list_detail": [
                "GET /{resource}",
                "GET /{resource}/{id}",
                "GET /{resource}/{id}/{subresource}",
                "GET /{resource}/{id}/{subresource}/{sub_id}",
            ],
            "nested_create": [
                "GET /{resource}/{id}",
                "GET /{resource}/{id}/{subresource}",
                "POST /{resource}/{id}/{subresource}",
                "GET /{resource}/{id}/{subresource}/{sub_id}",
            ],
            "nested_update": [
                "GET /{resource}/{id}/{subresource}/{sub_id}",
                "PUT /{resource}/{id}/{subresource}/{sub_id}",
                "GET /{resource}/{id}/{subresource}/{sub_id}",
            ],
            # Multi-resource workflows
            "cross_resource_lookup": [
                "GET /{resource1}",
                "GET /{resource1}/{id1}",
                "GET /{resource2}",
                "GET /{resource2}/{id2}",
                "POST /{resource3}",
            ],
            "reference_following": [
                "GET /{resource1}/{id1}",
                "GET /{resource2}/{ref_id}",
                "GET /{resource2}/{ref_id}/{subresource}",
            ],
            # Batch and bulk patterns
            "batch_operation": [
                "GET /{resource}",
                "POST /{resource}/batch",
                "GET /{resource}/batch/{batch_id}",
                "GET /{resource}/batch/{batch_id}/status",
            ],
            "bulk_update": [
                "GET /{resource}",
                "PUT /{resource}/bulk",
                "GET /{resource}",
            ],
            # Status and lifecycle patterns
            "status_workflow": [
                "GET /{resource}/{id}",
                "PUT /{resource}/{id}/status",
                "GET /{resource}/{id}",
                "GET /{resource}/{id}/history",
            ],
            "approval_workflow": [
                "POST /{resource}",
                "GET /{resource}/{id}",
                "PUT /{resource}/{id}/submit",
                "PUT /{resource}/{id}/approve",
                "GET /{resource}/{id}",
            ],
            # Error and retry patterns
            "validation_retry": [
                "POST /{resource}",
                "GET /{resource}/validate",
                "POST /{resource}",
                "GET /{resource}/{id}",
            ],
            "failed_then_recover": [
                "PUT /{resource}/{id}",
                "GET /{resource}/{id}/errors",
                "PUT /{resource}/{id}",
                "GET /{resource}/{id}",
            ],
            # Common exploration patterns
            "browse_compare": [
                "GET /{resource}",
                "GET /{resource}/{id1}",
                "GET /{resource}/{id2}",
                "GET /{resource}/{id1}",
            ],
            "deep_dive": [
                "GET /{resource}",
                "GET /{resource}/{id}",
                "GET /{resource}/{id}/{detail1}",
                "GET /{resource}/{id}/{detail2}",
                "GET /{resource}/{id}/{detail1}/{detail_id}",
            ],
        }

        # Generic prompt templates
        self.prompt_templates = {
            "read": [
                "view details",
                "show information",
                "get data",
                "check this",
                "see more",
            ],
            "create": ["create new", "add this", "make new", "set up", "initialize"],
            "update": ["update this", "edit", "modify", "change", "save changes"],
            "delete": ["remove this", "delete", "clear", "cancel", "deactivate"],
            "search": ["find", "search for", "look up", "filter", "browse"],
            "process": ["continue", "next step", "proceed", "finish this", "complete"],
            "validate": ["check this", "verify", "validate", "confirm", "review"],
            "compare": ["compare these", "check differences", "evaluate options"],
            "manage": ["handle this", "process", "deal with", "work on"],
        }

    def generate_training_samples(self, n_samples: int = 10000) -> List[Dict]:
        """Generate generic labeled training samples"""
        samples = []
        patterns = list(self.flow_patterns.keys())
        # Ensure we get at least 1 sample per pattern even for small requests
        samples_per_pattern = max(1, n_samples // (len(patterns) * 3))

        for pattern_name in patterns:
            for _ in range(samples_per_pattern):
                pattern = self.flow_patterns[pattern_name]
                flow_instance = self._generate_flow_instance(pattern)

                # Positive samples (correct next steps)
                for i in range(1, len(flow_instance)):
                    if i >= 1:  # Need at least 1 item in history
                        samples.append(
                            {
                                "history": [{"endpoint": e} for e in flow_instance[:i]],
                                "candidate": {"endpoint": flow_instance[i]},
                                "prompt": self._generate_prompt(
                                    flow_instance[i], pattern_name
                                ),
                                "label": 1,
                            }
                        )

                # Negative samples (wrong next steps)
                for i in range(1, len(flow_instance)):
                    if i >= 1:
                        wrong_endpoint = self._generate_wrong_endpoint(
                            flow_instance[:i]
                        )
                        samples.append(
                            {
                                "history": [{"endpoint": e} for e in flow_instance[:i]],
                                "candidate": {"endpoint": wrong_endpoint},
                                "prompt": self._generate_prompt(
                                    flow_instance[i], pattern_name
                                ),
                                "label": 0,
                            }
                        )

                # Neutral/alternative samples
                for i in range(1, min(3, len(flow_instance))):
                    if i >= 1:
                        neutral_endpoint = self._generate_neutral_endpoint()
                        samples.append(
                            {
                                "history": [{"endpoint": e} for e in flow_instance[:i]],
                                "candidate": {"endpoint": neutral_endpoint},
                                "prompt": "",
                                "label": random.choice([0, 0, 1]),  # Mostly negative
                            }
                        )

        random.shuffle(samples)
        return samples[:n_samples]

    def _generate_flow_instance(self, pattern: List[str]) -> List[str]:
        """Generate a specific instance of a generic pattern"""
        # Generic resource names that could apply to any API
        resources = [
            "items",
            "records",
            "entries",
            "objects",
            "resources",
            "entities",
            "data",
            "content",
            "documents",
            "files",
        ]
        subresources = [
            "details",
            "metadata",
            "properties",
            "attributes",
            "settings",
            "config",
            "status",
            "history",
            "logs",
            "comments",
        ]

        # Assign resources - fix the reference bug
        chosen_resources = random.sample(resources, min(3, len(resources)))
        chosen_subresources = random.sample(subresources, min(2, len(subresources)))

        resource_map = {
            "resource": chosen_resources[0],
            "resource1": chosen_resources[0],
            "resource2": chosen_resources[1] if len(chosen_resources) > 1 else chosen_resources[0],
            "resource3": chosen_resources[2] if len(chosen_resources) > 2 else chosen_resources[0],
            "subresource": chosen_subresources[0],
            "detail1": chosen_subresources[0],
            "detail2": chosen_subresources[1] if len(chosen_subresources) > 1 else chosen_subresources[0],
        }

        # Generate IDs
        id_map = {
            "id": str(random.randint(1, 10000)),
            "id1": str(random.randint(1, 1000)),
            "id2": str(random.randint(1001, 2000)),
            "sub_id": str(random.randint(1, 500)),
            "ref_id": str(random.randint(1, 1000)),
            "batch_id": str(random.randint(1, 100)),
            "detail_id": str(random.randint(1, 200)),
            "query": random.choice(["search", "filter", "find"]),
            "filter": random.choice(["active", "pending", "completed"]),
        }

        flow = []
        for endpoint_template in pattern:
            endpoint = endpoint_template

            # Replace resource placeholders
            for key, value in resource_map.items():
                endpoint = endpoint.replace(f"{{{key}}}", value)

            # Replace ID placeholders
            for key, value in id_map.items():
                endpoint = endpoint.replace(f"{{{key}}}", value)

            flow.append(endpoint)

        return flow

    def _generate_prompt(self, endpoint: str, pattern_name: str) -> str:
        """Generate contextual prompt based on endpoint and pattern"""
        if random.random() < 0.4:  # 40% no prompt
            return ""

        # Determine action from endpoint
        if endpoint.startswith("GET"):
            if "?" in endpoint:
                category = "search"
            elif endpoint.count("/") > 3:  # Deep nesting
                category = "read"
            else:
                category = "read"
        elif endpoint.startswith("POST"):
            if "batch" in endpoint:
                category = "process"
            else:
                category = "create"
        elif endpoint.startswith("PUT"):
            if "status" in endpoint:
                category = "process"
            else:
                category = "update"
        elif endpoint.startswith("DELETE"):
            category = "delete"
        else:
            category = "process"

        # Add pattern context
        if "compare" in pattern_name:
            category = "compare"
        elif "validate" in pattern_name or "retry" in pattern_name:
            category = "validate"
        elif "workflow" in pattern_name or "approval" in pattern_name:
            category = "process"
        elif "manage" in pattern_name or "bulk" in pattern_name:
            category = "manage"

        return random.choice(self.prompt_templates.get(category, ["continue"]))

    def _generate_wrong_endpoint(self, history: List[str]) -> str:
        """Generate plausible but incorrect endpoints"""
        if not history:
            return "GET /invalid"

        last = history[-1]

        wrong_strategies = [
            # Wrong HTTP method
            lambda: last.replace("GET", "DELETE")
            if "GET" in last
            else last.replace(last.split()[0], "GET"),
            # Wrong resource depth
            lambda: last + "/invalid/deep",
            # Completely different resource
            lambda: "GET /different/resource",
            # Invalid ID format
            lambda: last.replace(last.split("/")[-1], "invalid-id")
            if "/" in last
            else last,
            # Non-existent sub-resource
            lambda: last.rstrip("/") + "/nonexistent",
        ]

        return random.choice(wrong_strategies)()

    def _generate_neutral_endpoint(self) -> str:
        """Generate neutral endpoints that could be valid in any API"""
        neutral_options = [
            "GET /health",
            "GET /status",
            "GET /info",
            "GET /version",
            "GET /ping",
            "POST /auth/refresh",
            "GET /me",
            "GET /dashboard",
        ]

        return random.choice(neutral_options)
