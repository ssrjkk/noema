"""API Gateway module — routing, middleware, auth, rate limiting, circuit breaker."""

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GatewayRoute:
    path: str = "/"
    method: str = "GET"
    upstream: str = ""
    middleware: list[str] = field(default_factory=list)
    rate_limit: int = 0
    timeout: int = 30


@dataclass
class GatewayMiddleware:
    name: str = ""
    type: str = "logging"  # auth, rate_limit, cors, logging, circuit_breaker, retry
    config: dict[str, Any] = field(default_factory=dict)


BUILTIN_MIDDLEWARE: dict[str, dict[str, Any]] = {
    "auth_jwt": {
        "type": "auth",
        "config": {
            "header": "Authorization",
            "prefix": "Bearer",
            "secret_env": "JWT_SECRET",
            "algorithms": ["HS256"],
        },
    },
    "rate_limit_basic": {
        "type": "rate_limit",
        "config": {
            "requests_per_second": 100,
            "burst": 200,
            "key": "client_ip",
        },
    },
    "cors_default": {
        "type": "cors",
        "config": {
            "allowed_origins": ["*"],
            "allowed_methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allowed_headers": ["Content-Type", "Authorization"],
            "max_age": 3600,
        },
    },
    "circuit_breaker": {
        "type": "circuit_breaker",
        "config": {
            "failure_threshold": 5,
            "recovery_timeout": 30,
            "half_open_requests": 3,
        },
    },
    "retry_policy": {
        "type": "retry",
        "config": {
            "max_retries": 3,
            "retry_on": [502, 503, 504],
            "backoff_multiplier": 2,
            "initial_delay_ms": 100,
        },
    },
    "access_logging": {
        "type": "logging",
        "config": {
            "format": "json",
            "include_headers": True,
            "include_body": False,
        },
    },
}


class GatewayConfig:
    def __init__(self) -> None:
        self._routes: list[GatewayRoute] = []
        self._middleware: list[GatewayMiddleware] = []

    def add_route(self, route: GatewayRoute) -> "GatewayConfig":
        self._routes.append(route)
        return self

    def add_middleware(self, middleware: GatewayMiddleware) -> "GatewayConfig":
        self._middleware.append(middleware)
        return self

    def generate_nginx_config(self) -> str:
        lines = [
            "worker_processes auto;",
            "events {",
            "    worker_connections 1024;",
            "}",
            "",
            "http {",
            "    upstream backend {",
        ]

        upstreams = set()
        for route in self._routes:
            if route.upstream and route.upstream not in upstreams:
                upstreams.add(route.upstream)
                host, port = (
                    route.upstream.rsplit(":", 1)
                    if ":" in route.upstream
                    else (route.upstream, "80")
                )
                lines.append(f"        server {host}:{port};")

        if not upstreams:
            lines.append("        server 127.0.0.1:8000;")

        lines.extend(
            [
                "    }",
                "",
                "    log_format json_combined '{",
                '        "time": "$time_iso8601",',
                '        "remote_addr": "$remote_addr",',
                '        "request": "$request",',
                '        "status": "$status",',
                '        "body_bytes_sent": "$body_bytes_sent",',
                '        "request_time": "$request_time",',
                '        "upstream_response_time": "$upstream_response_time"',
                "    }';",
                "",
                "    access_log /var/log/nginx/access.log json_combined;",
                "",
                "    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=100r/s;",
                "",
                "    server {",
                "        listen 80;",
                "        server_name _;",
                "",
                "        client_max_body_size 10m;",
                "        proxy_read_timeout 30s;",
                "        proxy_connect_timeout 10s;",
                "",
            ]
        )

        for mw in self._middleware:
            if mw.type == "cors":
                origins = mw.config.get("allowed_origins", ["*"])
                methods = mw.config.get("allowed_methods", ["GET", "POST", "PUT", "DELETE"])
                headers = mw.config.get("allowed_headers", ["Content-Type", "Authorization"])
                lines.append("        # CORS middleware")
                lines.append(
                    f"        add_header Access-Control-Allow-Origin '{' '.join(origins)}' always;"
                )
                lines.append(
                    f"        add_header Access-Control-Allow-Methods '{', '.join(methods)}' always;"
                )
                lines.append(
                    f"        add_header Access-Control-Allow-Headers '{', '.join(headers)}' always;"
                )
                lines.append("")

        for route in self._routes:
            mw_lines = []
            for mw_name in route.middleware:
                for mw in self._middleware:
                    if mw.name == mw_name:
                        if mw.type == "rate_limit":
                            rps = mw.config.get("requests_per_second", 100)
                            mw_lines.append(
                                f"            limit_req zone=api_limit burst={rps // 10} nodelay;"
                            )
                        elif mw.type == "auth":
                            mw_lines.append(
                                '            if ($http_authorization !~ "^Bearer ") { return 401; }'
                            )

            lines.append(f"        location {route.path} {{")
            if route.rate_limit > 0:
                lines.append(
                    f"            limit_req zone=api_limit burst={route.rate_limit // 10} nodelay;"
                )
            for ml in mw_lines:
                lines.append(ml)
            lines.append("            proxy_pass http://backend;")
            lines.append("            proxy_set_header Host $host;")
            lines.append("            proxy_set_header X-Real-IP $remote_addr;")
            lines.append("            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;")
            lines.append(f"            proxy_read_timeout {route.timeout}s;")
            lines.append("        }")
            lines.append("")

        lines.extend(
            [
                "    }",
                "}",
            ]
        )

        return "\n".join(lines)

    def generate_envoy_config(self) -> str:
        routes_config = []
        for route in self._routes:
            route_entry = {
                "match": {"path": route.path},
                "route": {
                    "cluster": "backend_cluster",
                    "timeout": f"{route.timeout}s",
                },
            }
            if route.method != "GET":
                route_entry["match"]["method"] = route.method
            routes_config.append(route_entry)

        config = {
            "static_resources": {
                "listeners": [
                    {
                        "name": "gateway_listener",
                        "address": {"socket_address": {"address": "0.0.0.0", "port_value": 8080}},
                        "filter_chains": [
                            {
                                "filters": [
                                    {
                                        "name": "envoy.filters.network.http_connection_manager",
                                        "typed_config": {
                                            "@type": "type.googleapis.com/envoy.extensions.filters.network.http_connection_manager.v3.HttpConnectionManager",
                                            "stat_prefix": "gateway",
                                            "codec_type": "AUTO",
                                            "route_config": {
                                                "name": "local_route",
                                                "virtual_hosts": [
                                                    {
                                                        "name": "gateway",
                                                        "domains": ["*"],
                                                        "routes": routes_config
                                                        if routes_config
                                                        else [
                                                            {
                                                                "match": {"prefix": "/"},
                                                                "route": {
                                                                    "cluster": "backend_cluster"
                                                                },
                                                            }
                                                        ],
                                                    }
                                                ],
                                            },
                                            "http_filters": [
                                                {
                                                    "name": "envoy.filters.http.router",
                                                    "typed_config": {},
                                                }
                                            ],
                                        },
                                    }
                                ],
                            }
                        ],
                    }
                ],
                "clusters": [
                    {
                        "name": "backend_cluster",
                        "type": "EDS",
                        "lb_policy": "ROUND_ROBIN",
                    }
                ],
            },
        }

        return json.dumps(config, indent=2)

    def to_code(self, framework: str = "fastapi") -> str:
        fw = framework.lower()
        if fw == "fastapi":
            return self._to_fastapi()
        elif fw == "express":
            return self._to_express()
        elif fw == "kong":
            return self._to_kong()
        return self._to_fastapi()

    def _to_fastapi(self) -> str:
        imports = [
            "from fastapi import FastAPI, Request, Response, HTTPException",
            "from fastapi.middleware.cors import CORSMiddleware",
            "from starlette.middleware.base import BaseHTTPMiddleware",
            "import time",
            "import logging",
            "from collections import defaultdict",
            "from datetime import datetime",
            "",
            "",
            'app = FastAPI(title="API Gateway", version="1.0.0")',
            "",
        ]

        has_cors = any(mw.type == "cors" for mw in self._middleware)
        if has_cors:
            cors_mw = next(mw for mw in self._middleware if mw.type == "cors")
            origins = cors_mw.config.get("allowed_origins", ["*"])
            imports.append(
                f'app.add_middleware(CORSMiddleware, allow_origins={origins}, allow_methods=["*"], allow_headers=["*"])'
            )
            imports.append("")

        has_rate_limit = any(mw.type == "rate_limit" for mw in self._middleware)
        if has_rate_limit:
            imports.extend(
                [
                    "",
                    "class RateLimitMiddleware(BaseHTTPMiddleware):",
                    "    def __init__(self, app, rate_per_second=100):",
                    "        super().__init__(app)",
                    "        self.rate = rate_per_second",
                    "        self.requests = defaultdict(list)",
                    "",
                    "    async def dispatch(self, request: Request, call_next):",
                    '        client_ip = request.client.host if request.client else "unknown"',
                    "        now = time.time()",
                    "        self.requests[client_ip] = [t for t in self.requests[client_ip] if now - t < 1.0]",
                    "        if len(self.requests[client_ip]) >= self.rate:",
                    '            return Response(content="Rate limit exceeded", status_code=429)',
                    "        self.requests[client_ip].append(now)",
                    "        return await call_next(request)",
                    "",
                    "",
                    "app.add_middleware(RateLimitMiddleware, rate_per_second=100)",
                ]
            )

        has_auth = any(mw.type == "auth" for mw in self._middleware)
        if has_auth:
            imports.extend(
                [
                    "",
                    "class AuthMiddleware(BaseHTTPMiddleware):",
                    "    async def dispatch(self, request: Request, call_next):",
                    '        auth = request.headers.get("Authorization", "")',
                    '        if not auth.startswith("Bearer "):',
                    '            return Response(content="Unauthorized", status_code=401)',
                    "        return await call_next(request)",
                    "",
                    "",
                    "app.add_middleware(AuthMiddleware)",
                ]
            )

        imports.append("")
        imports.append("")

        for route in self._routes:
            method = route.method.lower()
            path = route.path.replace("{", "{").replace("}", "}")
            func_name = (
                path.strip("/").replace("/", "_").replace("{", "").replace("}", "") or "root"
            )
            imports.extend(
                [
                    f'@app.{method}("{route.path}")',
                    f"async def {func_name or 'root'}():",
                    f'    """Route: {route.method} {route.path} -> {route.upstream}"""',
                    f"    # Proxy to upstream: {route.upstream}",
                    f'    return {{"proxy_to": "{route.upstream}", "path": "{route.path}"}}',
                    "",
                ]
            )

        return "\n".join(imports)

    def _to_express(self) -> str:
        lines = [
            "const express = require('express');",
            "const { createProxyMiddleware } = require('http-proxy-middleware');",
            "const rateLimit = require('express-rate-limit');",
            "const cors = require('cors');",
            "",
            "const app = express();",
            "app.use(cors());",
            "",
            "const limiter = rateLimit({",
            "    windowMs: 60 * 1000,",
            "    max: 100,",
            "    message: 'Too many requests',",
            "});",
            "app.use(limiter);",
            "",
        ]

        for route in self._routes:
            upstream = route.upstream or "http://localhost:8000"
            lines.extend(
                [
                    f"app.use('{route.path}', createProxyMiddleware({{",
                    f"    target: '{upstream}',",
                    "    changeOrigin: true,",
                    f"    pathRewrite: {{ '^/{route.path.lstrip('/')}': '' }},",
                    "}));",
                    "",
                ]
            )

        lines.extend(
            [
                "const PORT = process.env.PORT || 8080;",
                "app.listen(PORT, () => console.log(`Gateway on port ${PORT}`));",
            ]
        )

        return "\n".join(lines)

    def _to_kong(self) -> str:
        services = []
        for route in self._routes:
            svc_name = route.path.strip("/").replace("/", "_") or "default"
            upstream = route.upstream or "http://localhost:8000"
            services.append(
                {
                    "name": svc_name,
                    "url": upstream,
                    "routes": [
                        {
                            "name": f"{svc_name}_route",
                            "paths": [route.path],
                            "methods": [route.method],
                            "strip_path": True,
                        }
                    ],
                    "plugins": [
                        {"name": mw.type, "config": mw.config}
                        for mw in self._middleware
                        if mw.name in route.middleware
                    ],
                }
            )

        return json.dumps({"services": services}, indent=2)


class GatewayModule:
    NAME = "gateway"
    DESCRIPTION = "API Gateway: routing, middleware, auth, rate limiting, circuit breaker"

    def __init__(self) -> None:
        self.config = GatewayConfig()

    def execute(self, task: Any) -> dict[str, Any]:
        task_title = getattr(task, "title", "")
        task_tags = getattr(task, "tags", [])

        action = "nginx"
        if "envoy" in str(task_title).lower() or "envoy" in task_tags:
            action = "envoy"
        elif "fastapi" in str(task_title).lower() or "python" in task_tags:
            action = "fastapi"
        elif "express" in str(task_title).lower() or "node" in task_tags:
            action = "express"
        elif "kong" in str(task_title).lower() or "kong" in task_tags:
            action = "kong"
        elif "middleware" in str(task_title).lower() or "middleware" in task_tags:
            action = "middleware_list"

        metadata = {}
        if hasattr(task, "metadata"):
            metadata = task.metadata

        self.config = GatewayConfig()

        routes_data = metadata.get(
            "routes",
            [
                {"path": "/api/v1/users", "method": "GET", "upstream": "user-service:8000"},
                {"path": "/api/v1/orders", "method": "POST", "upstream": "order-service:8000"},
                {"path": "/api/v1/products", "method": "GET", "upstream": "product-service:8000"},
                {"path": "/health", "method": "GET", "upstream": "health-service:8000"},
            ],
        )

        for rd in routes_data:
            self.config.add_route(
                GatewayRoute(
                    path=rd.get("path", "/"),
                    method=rd.get("method", "GET"),
                    upstream=rd.get("upstream", "localhost:8000"),
                    middleware=rd.get("middleware", []),
                    rate_limit=rd.get("rate_limit", 0),
                    timeout=rd.get("timeout", 30),
                )
            )

        middleware_data = metadata.get("middleware", ["cors_default", "rate_limit_basic"])
        for mw_name in middleware_data:
            if isinstance(mw_name, dict):
                self.config.add_middleware(
                    GatewayMiddleware(
                        name=mw_name.get("name", "custom"),
                        type=mw_name.get("type", "logging"),
                        config=mw_name.get("config", {}),
                    )
                )
            elif mw_name in BUILTIN_MIDDLEWARE:
                mw_def = BUILTIN_MIDDLEWARE[mw_name]
                self.config.add_middleware(
                    GatewayMiddleware(
                        name=mw_name,
                        type=mw_def["type"],
                        config=mw_def["config"],
                    )
                )

        if action == "nginx":
            content = self.config.generate_nginx_config()
            return {
                "action": "nginx_config",
                "content": content,
                "routes_count": len(self.config._routes),
                "_confidence": 0.88,
            }
        elif action == "envoy":
            content = self.config.generate_envoy_config()
            return {
                "action": "envoy_config",
                "content": content,
                "_confidence": 0.82,
            }
        elif action == "fastapi":
            content = self.config.to_code("fastapi")
            return {
                "action": "fastapi_gateway",
                "content": content,
                "_confidence": 0.85,
            }
        elif action == "express":
            content = self.config.to_code("express")
            return {
                "action": "express_gateway",
                "content": content,
                "_confidence": 0.85,
            }
        elif action == "kong":
            content = self.config.to_code("kong")
            return {
                "action": "kong_config",
                "content": content,
                "_confidence": 0.80,
            }
        elif action == "middleware_list":
            return {
                "action": "middleware_list",
                "builtin_middleware": {
                    name: {"type": mw["type"], "config": mw["config"]}
                    for name, mw in BUILTIN_MIDDLEWARE.items()
                },
                "configured": [
                    {"name": mw.name, "type": mw.type, "config": mw.config}
                    for mw in self.config._middleware
                ],
                "_confidence": 0.90,
            }

        content = self.config.generate_nginx_config()
        return {
            "action": "nginx_config",
            "content": content,
            "_confidence": 0.75,
        }
