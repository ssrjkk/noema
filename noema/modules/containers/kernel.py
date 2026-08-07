"""Container module — Dockerfile, docker-compose, Kubernetes manifests generation."""

from dataclasses import dataclass, field
from typing import Any, TypedDict


@dataclass
class ContainerConfig:
    port: int = 8000
    env_vars: dict[str, str] = field(default_factory=dict)
    volumes: list[str] = field(default_factory=list)
    health_check: str = ""
    user: str = "appuser"


class MultiStageBuild(TypedDict):
    builder: str
    runtime: str
    build_steps: list[str]


class StackConfig(TypedDict):
    base_image: str
    workdir: str
    copy_cmd: str
    install_cmd: str
    copy_app: str
    cmd: str
    port: int
    multi_stage_build: MultiStageBuild


STACK_CONFIGS: dict[str, StackConfig] = {
    "python": {
        "base_image": "python:3.12-slim",
        "workdir": "/app",
        "copy_cmd": "COPY requirements.txt .",
        "install_cmd": "RUN pip install --no-cache-dir -r requirements.txt",
        "copy_app": "COPY . .",
        "cmd": 'CMD ["python", "app.py"]',
        "port": 8000,
        "multi_stage_build": {
            "builder": "python:3.12-slim",
            "runtime": "python:3.12-slim",
            "build_steps": [
                "RUN pip install --no-cache-dir --prefix=/install -r requirements.txt",
            ],
        },
    },
    "node": {
        "base_image": "node:20-alpine",
        "workdir": "/app",
        "copy_cmd": "COPY package*.json ./",
        "install_cmd": "RUN npm ci --only=production",
        "copy_app": "COPY . .",
        "cmd": 'CMD ["node", "server.js"]',
        "port": 3000,
        "multi_stage_build": {
            "builder": "node:20-alpine",
            "runtime": "node:20-alpine",
            "build_steps": [
                "RUN npm ci",
                "RUN npm run build",
            ],
        },
    },
    "go": {
        "base_image": "golang:1.22-alpine",
        "workdir": "/app",
        "copy_cmd": "COPY go.mod go.sum ./",
        "install_cmd": "RUN go mod download",
        "copy_app": "COPY . .",
        "cmd": 'CMD ["./main"]',
        "port": 8080,
        "multi_stage_build": {
            "builder": "golang:1.22-alpine",
            "runtime": "alpine:3.19",
            "build_steps": [
                "RUN CGO_ENABLED=0 GOOS=linux go build -o /app/main .",
            ],
        },
    },
    "java": {
        "base_image": "eclipse-temurin:21-jdk",
        "workdir": "/app",
        "copy_cmd": "COPY pom.xml .",
        "install_cmd": "RUN mvn dependency:go-offline -B",
        "copy_app": "COPY . .",
        "cmd": 'CMD ["java", "-jar", "target/app.jar"]',
        "port": 8080,
        "multi_stage_build": {
            "builder": "eclipse-temurin:21-jdk",
            "runtime": "eclipse-temurin:21-jre",
            "build_steps": [
                "RUN mvn package -DskipTests -B",
            ],
        },
    },
}


class DockerfileGenerator:
    def generate(self, stack: str, options: dict[str, Any] | None = None) -> str:
        options = options or {}
        stack_lower = stack.lower()
        config = STACK_CONFIGS.get(stack_lower, STACK_CONFIGS["python"])

        port = options.get("port", config["port"])
        user = options.get("user", "appuser")
        env_vars = options.get("env_vars", {})
        health_check = options.get("health_check", "")
        expose_non_root = options.get("non_root", True)

        lines = []
        lines.append(f"FROM {config['base_image']}")
        lines.append("")

        if expose_non_root:
            lines.append(f"RUN groupadd -r {user} && useradd -r -g {user} {user}")
            lines.append("")

        lines.append(f"WORKDIR {config['workdir']}")
        lines.append("")

        lines.append(config["copy_cmd"])
        lines.append(config["install_cmd"])
        lines.append("")

        lines.append(config["copy_app"])
        lines.append("")

        if env_vars:
            for key, value in env_vars.items():
                lines.append(f'ENV {key}="{value}"')
            lines.append("")

        lines.append(f"EXPOSE {port}")
        lines.append("")

        if health_check:
            lines.append("HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \\")
            lines.append(f"  CMD {health_check}")
            lines.append("")

        if expose_non_root:
            lines.append(f"USER {user}")
            lines.append("")

        lines.append(config["cmd"])

        return "\n".join(lines)

    def optimize_multi_stage(self, stack: str) -> str:
        stack_lower = stack.lower()
        config = STACK_CONFIGS.get(stack_lower, STACK_CONFIGS["python"])
        ms = config["multi_stage_build"]

        lines = []
        lines.append("# Stage 1: Builder")
        lines.append(f"FROM {ms['builder']} AS builder")
        lines.append(f"WORKDIR {config['workdir']}")
        lines.append("")
        lines.append(config["copy_cmd"])
        for step in ms["build_steps"]:
            lines.append(step)
        lines.append("")
        lines.append(config["copy_app"])
        lines.append("")

        lines.append("# Stage 2: Runtime")
        lines.append(f"FROM {ms['runtime']} AS runtime")
        lines.append(f"WORKDIR {config['workdir']}")
        lines.append("")

        if stack_lower == "python":
            lines.append("COPY --from=builder /install /usr/local")
            lines.append("COPY --from=builder /app .")
        elif stack_lower == "go":
            lines.append("COPY --from=builder /app/main .")
        elif stack_lower == "node":
            lines.append("COPY --from=builder /app/node_modules ./node_modules")
            lines.append("COPY --from=builder /app/dist ./dist")
            lines.append("COPY --from=builder /app/package.json .")
        elif stack_lower == "java":
            lines.append("COPY --from=builder /app/target/*.jar app.jar")

        lines.append("")
        lines.append(f"EXPOSE {config['port']}")
        lines.append("")
        lines.append(config["cmd"])

        return "\n".join(lines)


class DockerComposeGenerator:
    def generate(self, services: list[dict[str, Any]]) -> str:
        lines = ['version: "3.8"', "services:"]

        for svc in services:
            name = svc.get("name", "app")
            image = svc.get("image", "alpine:latest")
            port = svc.get("port", 8000)
            env = svc.get("env", {})
            volumes = svc.get("volumes", [])
            depends_on = svc.get("depends_on", [])

            lines.append(f"  {name}:")
            if "build" in svc:
                lines.append(f"    build: {svc['build']}")
            else:
                lines.append(f"    image: {image}")

            if port:
                lines.append("    ports:")
                lines.append(f'      - "{port}:{port}"')

            if env:
                lines.append("    environment:")
                for k, v in env.items():
                    lines.append(f'      {k}: "{v}"')

            if volumes:
                lines.append("    volumes:")
                for v in volumes:
                    lines.append(f"      - {v}")

            if depends_on:
                lines.append("    depends_on:")
                for dep in depends_on:
                    lines.append(f"      - {dep}")

            lines.append("    restart: unless-stopped")
            lines.append("")

        return "\n".join(lines)

    def add_network(self, compose: str, name: str) -> str:
        network_block = f"""
networks:
  {name}:
    driver: bridge
    ipam:
      config:
        - subnet: 172.28.0.0/16
"""
        return compose.rstrip() + "\n" + network_block

    def add_volume(self, compose: str, name: str) -> str:
        volume_block = f"""
volumes:
  {name}:
    driver: local
"""
        return compose.rstrip() + "\n" + volume_block


class KubernetesGenerator:
    def generate_deployment(self, name: str, image: str, replicas: int = 1) -> str:
        return f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: {name}
  labels:
    app: {name}
spec:
  replicas: {replicas}
  selector:
    matchLabels:
      app: {name}
  template:
    metadata:
      labels:
        app: {name}
    spec:
      containers:
        - name: {name}
          image: {image}
          ports:
            - containerPort: 8000
          resources:
            requests:
              memory: "128Mi"
              cpu: "100m"
            limits:
              memory: "512Mi"
              cpu: "500m"
          readinessProbe:
            httpGet:
              path: /healthz
              port: 8000
            initialDelaySeconds: 5
            periodSeconds: 10
          livenessProbe:
            httpGet:
              path: /healthz
              port: 8000
            initialDelaySeconds: 15
            periodSeconds: 20"""

    def generate_service(self, name: str, port: int = 80) -> str:
        return f"""apiVersion: v1
kind: Service
metadata:
  name: {name}
  labels:
    app: {name}
spec:
  type: ClusterIP
  selector:
    app: {name}
  ports:
    - protocol: TCP
      port: {port}
      targetPort: 8000"""

    def generate_ingress(self, host: str, path: str = "/") -> str:
        service_name = host.split(".")[0].replace("-", "")
        return f"""apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: {service_name}-ingress
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  ingressClassName: nginx
  tls:
    - hosts:
        - {host}
      secretName: {service_name}-tls
  rules:
    - host: {host}
      http:
        paths:
          - path: {path}
            pathType: Prefix
            backend:
              service:
                name: {service_name}
                port:
                  number: 80"""


class ContainersModule:
    NAME = "containers"
    DESCRIPTION = (
        "Container configuration: Dockerfile, docker-compose, Kubernetes manifests generation"
    )

    def __init__(self) -> None:
        self.dockerfile_gen = DockerfileGenerator()
        self.compose_gen = DockerComposeGenerator()
        self.k8s_gen = KubernetesGenerator()

    def execute(self, task: Any) -> dict[str, Any]:
        task_title = getattr(task, "title", "")
        task_tags = getattr(task, "tags", [])

        action = "dockerfile"
        if (
            "compose" in str(task_title).lower()
            or "docker-compose" in task_tags
            or "compose" in task_tags
        ):
            action = "compose"
        elif (
            "kubernetes" in str(task_title).lower()
            or "k8s" in str(task_title).lower()
            or "k8s" in task_tags
        ):
            action = "kubernetes"
        elif "multi" in str(task_title).lower() and "stage" in str(task_title).lower():
            action = "multi_stage"

        metadata = {}
        if hasattr(task, "metadata"):
            metadata = task.metadata

        stack = metadata.get("stack", "python")

        if action == "dockerfile":
            content = self.dockerfile_gen.generate(stack, metadata.get("options", {}))
            return {
                "action": "dockerfile",
                "stack": stack,
                "content": content,
                "_confidence": 0.90,
            }
        elif action == "multi_stage":
            content = self.dockerfile_gen.optimize_multi_stage(stack)
            return {
                "action": "multi_stage_dockerfile",
                "stack": stack,
                "content": content,
                "_confidence": 0.88,
            }
        elif action == "compose":
            services = metadata.get("services", [{"name": "app", "build": ".", "port": 8000}])
            content = self.compose_gen.generate(services)
            if metadata.get("network"):
                content = self.compose_gen.add_network(content, metadata["network"])
            if metadata.get("volume"):
                content = self.compose_gen.add_volume(content, metadata["volume"])
            return {
                "action": "compose",
                "content": content,
                "_confidence": 0.88,
            }
        elif action == "kubernetes":
            name = metadata.get("name", "app")
            image = metadata.get("image", f"{name}:latest")
            replicas = metadata.get("replicas", 1)
            host = metadata.get("host", f"{name}.example.com")

            deployment = self.k8s_gen.generate_deployment(name, image, replicas)
            service = self.k8s_gen.generate_service(name, metadata.get("port", 80))
            ingress = self.k8s_gen.generate_ingress(host)

            return {
                "action": "kubernetes",
                "resources": {
                    "deployment": deployment,
                    "service": service,
                    "ingress": ingress,
                },
                "_confidence": 0.88,
            }

        return {
            "action": "dockerfile",
            "content": self.dockerfile_gen.generate("python"),
            "_confidence": 0.70,
        }
