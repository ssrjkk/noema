"""Infrastructure as Code module — Terraform, Pulumi generation."""

from typing import Any

AWS_TEMPLATES: dict[str, dict[str, Any]] = {
    "vpc": {
        "type": "aws_vpc",
        "config": {
            "cidr_block": "10.0.0.0/16",
            "enable_dns_hostnames": True,
            "enable_dns_support": True,
            "tags": {"Name": "${var.project}-vpc"},
        },
    },
    "subnet_public": {
        "type": "aws_subnet",
        "config": {
            "vpc_id": "${aws_vpc.main.id}",
            "cidr_block": "10.0.1.0/24",
            "map_public_ip_on_launch": True,
            "tags": {"Name": "${var.project}-public-subnet"},
        },
    },
    "subnet_private": {
        "type": "aws_subnet",
        "config": {
            "vpc_id": "${aws_vpc.main.id}",
            "cidr_block": "10.0.2.0/24",
            "tags": {"Name": "${var.project}-private-subnet"},
        },
    },
    "ecs_cluster": {
        "type": "aws_ecs_cluster",
        "config": {
            "name": "${var.project}-cluster",
            "settings": [{"name": "containerInsights", "value": "enabled"}],
        },
    },
    "ecs_service": {
        "type": "aws_ecs_service",
        "config": {
            "name": "${var.project}-service",
            "cluster": "${aws_ecs_cluster.main.id}",
            "task_definition": "${aws_ecs_task_definition.main.arn}",
            "desired_count": 2,
            "launch_type": "FARGATE",
            "network_configuration": {
                "subnets": ["${aws_subnet.public.id}"],
                "security_groups": ["${aws_security_group.main.id}"],
            },
        },
    },
    "rds": {
        "type": "aws_db_instance",
        "config": {
            "identifier": "${var.project}-db",
            "engine": "postgres",
            "engine_version": "15.4",
            "instance_class": "db.t3.micro",
            "allocated_storage": 20,
            "db_name": "appdb",
            "username": "${var.db_username}",
            "password": "${var.db_password}",
            "skip_final_snapshot": True,
        },
    },
    "s3": {
        "type": "aws_s3_bucket",
        "config": {
            "bucket": "${var.project}-bucket",
            "tags": {"Name": "${var.project}-bucket"},
        },
    },
    "lambda": {
        "type": "aws_lambda_function",
        "config": {
            "function_name": "${var.project}-function",
            "runtime": "python3.12",
            "handler": "index.handler",
            "filename": "lambda.zip",
            "memory_size": 128,
            "timeout": 30,
        },
    },
    "iam_role": {
        "type": "aws_iam_role",
        "config": {
            "name": "${var.project}-role",
            "assume_role_policy": {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Action": "sts:AssumeRole",
                        "Effect": "Allow",
                        "Principal": {"Service": "lambda.amazonaws.com"},
                    }
                ],
            },
        },
    },
}

GCP_TEMPLATES = {
    "gke": {
        "type": "google_container_cluster",
        "config": {
            "name": "${var.project}-gke",
            "location": "us-central1",
            "initial_node_count": 2,
            "min_node_count": 1,
            "max_node_count": 5,
        },
    },
    "cloud_run": {
        "type": "google_cloud_run_service",
        "config": {
            "name": "${var.project}-service",
            "location": "us-central1",
            "template": {
                "spec": {"containers": [{"image": "${var.image}"}]},
            },
        },
    },
}

AZURE_TEMPLATES = {
    "aks": {
        "type": "azurerm_kubernetes_cluster",
        "config": {
            "name": "${var.project}-aks",
            "location": "East US",
            "dns_prefix": "${var.project}",
            "default_node_pool": {"node_count": 2, "vm_size": "Standard_D2_v2"},
        },
    },
}


class TerraformGenerator:
    def __init__(self) -> None:
        self._resources: list[str] = []
        self._variables: list[str] = []
        self._outputs: list[str] = []
        self._providers: list[str] = []

    def _reset(self) -> None:
        self._resources = []
        self._variables = []
        self._outputs = []
        self._providers = []

    def generate_provider(self, name: str, config: dict[str, Any] | None = None) -> str:
        config = config or {}
        lines = [f'provider "{name}" {{']
        for key, value in config.items():
            if isinstance(value, str):
                lines.append(f'  {key} = "{value}"')
            elif isinstance(value, bool):
                lines.append(f"  {key} = {'true' if value else 'false'}")
            elif isinstance(value, (int, float)):
                lines.append(f"  {key} = {value}")
        lines.append("}")
        result = "\n".join(lines)
        self._providers.append(result)
        return result

    def generate_resource(
        self, resource_type: str, name: str, config: dict[str, Any] | None = None
    ) -> str:
        config = config or {}
        lines = [f'resource "{resource_type}" "{name}" {{']
        lines.extend(self._dict_to_hcl(config, indent=1))
        lines.append("}")
        result = "\n".join(lines)
        self._resources.append(result)
        return result

    def generate_variable(
        self,
        name: str,
        var_type: str = "string",
        default: Any = None,
        description: str = "",
        sensitive: bool = False,
    ) -> str:
        lines = [f'variable "{name}" {{']
        lines.append(f"  type = {var_type}")
        if default is not None:
            if isinstance(default, str):
                lines.append(f'  default     = "{default}"')
            elif isinstance(default, bool):
                lines.append(f"  default     = {'true' if default else 'false'}")
            elif isinstance(default, (list, dict)):
                lines.append(f"  default     = {default}")
            else:
                lines.append(f"  default     = {default}")
        if description:
            lines.append(f'  description = "{description}"')
        if sensitive:
            lines.append("  sensitive   = true")
        lines.append("}")
        result = "\n".join(lines)
        self._variables.append(result)
        return result

    def generate_output(self, name: str, value: str, description: str = "") -> str:
        lines = [f'output "{name}" {{']
        lines.append(f"  value = {value}")
        if description:
            lines.append(f'  description = "{description}"')
        lines.append("}")
        result = "\n".join(lines)
        self._outputs.append(result)
        return result

    def generate_full_config(
        self,
        provider: str,
        resources: dict[str, dict[str, Any]],
        variables: dict[str, Any] | None = None,
    ) -> str:
        self._reset()
        sections = []

        sections.append("# Generated by Noema Terraform Module")
        sections.append("")

        provider_config: dict[str, Any] = {}
        if provider == "aws":
            provider_config = {"region": "us-east-1"}
        elif provider == "gcp":
            provider_config = {"project": "${var.project}", "region": "us-central1"}
        elif provider == "azure":
            provider_config = {"features": {}}

        sections.append(self.generate_provider(provider, provider_config))
        sections.append("")

        if variables:
            for name, details in variables.items():
                if isinstance(details, dict):
                    sections.append(
                        self.generate_variable(
                            name,
                            details.get("type", "string"),
                            details.get("default"),
                            details.get("description", ""),
                            details.get("sensitive", False),
                        )
                    )
                else:
                    sections.append(self.generate_variable(name, "string", str(details)))
                sections.append("")

        for res_name, res_config in resources.items():
            res_type = res_config.get("resource_type", f"{provider}_{res_name}")
            res_body = {k: v for k, v in res_config.items() if k != "resource_type"}
            sections.append(self.generate_resource(res_type, res_name, res_body))
            sections.append("")

        return "\n".join(sections)

    def _dict_to_hcl(self, d: dict[str, Any], indent: int = 1) -> list[str]:
        lines = []
        prefix = "  " * indent
        for key, value in d.items():
            if isinstance(value, dict):
                lines.append(f"{prefix}{key} {{")
                lines.extend(self._dict_to_hcl(value, indent + 1))
                lines.append(f"{prefix}}}")
            elif isinstance(value, list):
                if value and isinstance(value[0], dict):
                    for item in value:
                        lines.append(f"{prefix}{key} {{")
                        lines.extend(self._dict_to_hcl(item, indent + 1))
                        lines.append(f"{prefix}}}")
                else:
                    lines.append(f"{prefix}{key} = {value}")
            elif isinstance(value, str):
                lines.append(f'{prefix}{key} = "{value}"')
            elif isinstance(value, bool):
                lines.append(f"{prefix}{key} = {'true' if value else 'false'}")
            elif value is None:
                lines.append(f"{prefix}{key} = null")
            else:
                lines.append(f"{prefix}{key} = {value}")
        return lines


class PulumiGenerator:
    def __init__(self) -> None:
        self._stacks: dict[str, dict[str, Any]] = {}

    def generate_stack(self, name: str, language: str = "python") -> str:
        lang_templates = {
            "python": f"""import pulumi
import pulumi_aws as aws

# Stack: {name}
project = pulumi.get_project()

# Configure your resources here
# Example:
# vpc = aws.ec2.Vpc("{name}-vpc",
#     cidr_block="10.0.0.0/16",
#     tags={{"Name": f"{{project}}-vpc"}})

# pulumi.export("vpc_id", vpc.id)
""",
            "typescript": f"""import * as pulumi from "@pulumi/pulumi";
import * as aws from "@pulumi/aws";

// Stack: {name}
const project = pulumi.getProject();

// Configure your resources here
// const vpc = new aws.ec2.Vpc("{name}-vpc", {{
//     cidrBlock: "10.0.0.0/16",
//     tags: {{ Name: `${{project}}-vpc` }}
// }});

// export const vpcId = vpc.id;
""",
            "go": f"""package main

import (
    "github.com/pulumi/pulumi/sdk/v3/go/pulumi"
)

func main() {{
    pulumi.Run(func(ctx *pulumi.Context) error {{
        // Stack: {name}
        // Configure your resources here
        return nil
    }})
}}
""",
        }
        return lang_templates.get(language, lang_templates["python"])

    def generate_resource(
        self, resource_type: str, name: str, config: dict[str, Any] | None = None
    ) -> str:
        config = config or {}
        config_str = ", ".join(f"{k}={repr(v)}" for k, v in config.items())
        return f'resource = pulumi.Resource("{resource_type}", "{name}", {config_str})'


class TerraformModule:
    NAME = "terraform"
    DESCRIPTION = "Infrastructure as Code: Terraform and Pulumi generation with pre-built templates"

    def __init__(self) -> None:
        self.tf_gen = TerraformGenerator()
        self.pulumi_gen = PulumiGenerator()

    def execute(self, task: Any) -> dict[str, Any]:
        task_title = getattr(task, "title", "")
        task_tags = getattr(task, "tags", [])

        action = "terraform"
        if "pulumi" in str(task_title).lower() or "pulumi" in task_tags:
            action = "pulumi"
        elif "template" in str(task_title).lower() or "template" in task_tags:
            action = "template"

        metadata = {}
        if hasattr(task, "metadata"):
            metadata = task.metadata

        provider = metadata.get("provider", "aws")
        project = metadata.get("project", "myproject")

        if action == "terraform":
            template_name = metadata.get("template", "")
            if template_name and provider == "aws" and template_name in AWS_TEMPLATES:
                tpl = AWS_TEMPLATES[template_name]
                tf_code = self.tf_gen.generate_provider(provider, {"region": "us-east-1"})
                tf_code += "\n\n"
                tf_code += self.tf_gen.generate_resource(
                    tpl["type"], template_name, dict(tpl["config"])
                )
                return {
                    "action": "terraform",
                    "template": template_name,
                    "provider": provider,
                    "content": tf_code,
                    "_confidence": 0.88,
                }

            resources = metadata.get("resources", {})
            variables = metadata.get(
                "variables",
                {
                    "project": {
                        "type": "string",
                        "default": project,
                        "description": "Project name",
                    },
                    "environment": {
                        "type": "string",
                        "default": "production",
                        "description": "Environment",
                    },
                },
            )

            if not resources:
                resources = {
                    "main_vpc": {
                        "resource_type": f"{provider}_vpc",
                        "cidr_block": "10.0.0.0/16",
                        "enable_dns_hostnames": True,
                    },
                    "main_subnet": {
                        "resource_type": f"{provider}_subnet",
                        "cidr_block": "10.0.1.0/24",
                    },
                }

            tf_code = self.tf_gen.generate_full_config(provider, resources, variables)
            return {
                "action": "terraform",
                "provider": provider,
                "content": tf_code,
                "available_templates": {
                    "aws": list(AWS_TEMPLATES.keys()),
                    "gcp": list(GCP_TEMPLATES.keys()),
                    "azure": list(AZURE_TEMPLATES.keys()),
                },
                "_confidence": 0.85,
            }

        elif action == "pulumi":
            language = str(metadata.get("language", "python"))
            stack_name = str(metadata.get("stack_name", project))
            stack_code = self.pulumi_gen.generate_stack(stack_name, language)
            return {
                "action": "pulumi",
                "language": language,
                "stack_name": stack_name,
                "content": stack_code,
                "_confidence": 0.80,
            }

        elif action == "template":
            all_templates = {
                "aws": AWS_TEMPLATES,
                "gcp": GCP_TEMPLATES,
                "azure": AZURE_TEMPLATES,
            }
            return {
                "action": "templates",
                "templates": {
                    provider: {
                        name: {"type": t["type"], "config": t["config"]}
                        for name, t in tmpls.items()
                    }
                    for provider, tmpls in all_templates.items()
                },
                "_confidence": 0.90,
            }

        return {
            "action": "terraform",
            "message": "No specific action matched. Generating default Terraform config.",
            "content": self.tf_gen.generate_full_config(provider, {}),
            "_confidence": 0.50,
        }
