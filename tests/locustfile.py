"""Locust load tests for noema API."""

from locust import HttpUser, between, tag, task


class NoemaUser(HttpUser):
    wait_time = between(1, 3)

    def on_start(self):
        """Verify API is reachable before running tests."""
        with self.client.get("/health", catch_response=True) as resp:
            if resp.status_code != 200:
                resp.failure(f"Health check failed: {resp.status_code}")

    @tag("health")
    @task(2)
    def health_check(self):
        self.client.get("/health")

    @tag("health")
    @task(1)
    def readiness(self):
        self.client.get("/ready")

    @tag("think")
    @task(3)
    def think_simple(self):
        payload = {
            "title": "Build a simple REST API",
            "description": "Create a REST API with FastAPI for user management",
            "complexity": "simple",
            "tags": ["api", "python"],
            "requirements": [
                {"category": "api", "description": "RESTful endpoints", "priority": 5}
            ],
        }
        with self.client.post(
            "/think",
            json=payload,
            catch_response=True,
            name="/think [simple]",
        ) as resp:
            if resp.status_code > 499:
                resp.failure(f"think failed: {resp.status_code}")

    @tag("think")
    @task(2)
    def think_moderate(self):
        payload = {
            "title": "Build a microservice with auth",
            "description": "Design a microservice with JWT auth, rate limiting, PostgreSQL",
            "complexity": "moderate",
            "tags": ["api", "python", "auth", "database"],
            "requirements": [
                {"category": "auth", "description": "JWT auth", "priority": 5},
                {
                    "category": "database",
                    "description": "PostgreSQL with SQLAlchemy",
                    "priority": 4,
                },
            ],
        }
        self.client.post("/think", json=payload, name="/think [moderate]")

    @tag("admin")
    @task(1)
    def admin_metrics(self):
        self.client.get("/admin/metrics")

    @tag("admin")
    @task(1)
    def admin_tasks_history(self):
        self.client.get("/admin/tasks/history?limit=10")

    @tag("admin")
    @task(1)
    def health_infra(self):
        self.client.get("/health/infra")

    @tag("knowledge")
    @task(1)
    def knowledge_stats(self):
        self.client.get("/knowledge/stats")

    @tag("ops")
    @task(1)
    def worker_stats(self):
        self.client.get("/workers/stats")

    @tag("think")
    @task(1)
    def think_complex(self):
        payload = {
            "title": "Design a high-load event processing system",
            "description": "Design a system that processes 1M events/sec with Kafka, Spark, Cassandra",
            "complexity": "complex",
            "tags": ["distributed", "streaming", "kafka", "spark"],
            "requirements": [
                {
                    "category": "performance",
                    "description": "1M events/sec throughput",
                    "priority": 5,
                },
                {"category": "scalability", "description": "Horizontal scaling", "priority": 4},
            ],
        }
        self.client.post("/think", json=payload, name="/think [complex]")


class ThinkStreamUser(HttpUser):
    """Heavier user that tests streaming endpoint."""

    wait_time = between(5, 15)

    @tag("stream")
    @task(1)
    def think_stream_simple(self):
        payload = {
            "title": "Build a CLI tool in Python",
            "description": "Create a CLI tool for file processing with click/typer",
            "complexity": "simple",
            "tags": ["cli", "python"],
        }
        with self.client.post(
            "/think/stream",
            json=payload,
            catch_response=True,
            name="/think/stream [simple]",
            stream=True,
        ) as resp:
            if resp.status_code > 499:
                resp.failure(f"stream failed: {resp.status_code}")
