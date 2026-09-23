"""Tests for noema/modules/terraform/kernel.py — Terraform/Pulumi HCL generation."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from noema.modules.terraform.kernel import (
    AWS_TEMPLATES,
    AZURE_TEMPLATES,
    GCP_TEMPLATES,
    PulumiGenerator,
    TerraformGenerator,
    TerraformModule,
)


@pytest.fixture
def tf() -> TerraformGenerator:
    return TerraformGenerator()


@pytest.fixture
def pulumi() -> PulumiGenerator:
    return PulumiGenerator()


def _task(title: str = "", tags: list[str] | None = None, metadata: dict[str, Any] | None = None):
    """A task shaped like the ones the module registry hands to ``execute``."""
    ns = SimpleNamespace(title=title, tags=tags or [])
    if metadata is not None:
        ns.metadata = metadata
    return ns


# ---------------------------------------------------------------------------
# TerraformGenerator — provider
# ---------------------------------------------------------------------------


class TestGenerateProvider:
    def test_renders_supported_value_types(self, tf):
        out = tf.generate_provider(
            "aws",
            {
                "region": "us-east-1",
                "skip_credentials_validation": True,
                "s3_use_acls": False,
                "max_retries": 3,
                "retry_seconds": 1.5,
            },
        )
        assert out.splitlines() == [
            'provider "aws" {',
            '  region = "us-east-1"',
            "  skip_credentials_validation = true",
            "  s3_use_acls = false",
            "  max_retries = 3",
            "  retry_seconds = 1.5",
            "}",
        ]

    def test_skips_values_without_an_hcl_literal_form(self, tf):
        out = tf.generate_provider("azurerm", {"features": {"key_vault": []}})
        assert out == 'provider "azurerm" {\n}'

    def test_empty_config_yields_bare_block(self, tf):
        assert tf.generate_provider("google") == 'provider "google" {\n}'

    def test_result_is_recorded_on_the_generator(self, tf):
        block = tf.generate_provider("aws", {"region": "eu-west-2"})
        assert tf._providers == [block]


# ---------------------------------------------------------------------------
# TerraformGenerator — resource / _dict_to_hcl
# ---------------------------------------------------------------------------


class TestGenerateResource:
    def test_renders_every_hcl_value_shape(self, tf):
        out = tf.generate_resource(
            "aws_instance",
            "web",
            {
                "ami": "ami-123",
                "monitoring": True,
                "instance-initiated_shutdown_behavior": None,
                "count": 2,
                "tags": {"Name": "web", "Team": "platform"},
                "security_groups": ["ssh", "http"],
                "ebs_block_device": [{"device_name": "/dev/sda1", "volume_size": 30}],
            },
        )
        assert out.splitlines() == [
            'resource "aws_instance" "web" {',
            '  ami = "ami-123"',
            "  monitoring = true",
            "  instance-initiated_shutdown_behavior = null",
            "  count = 2",
            "  tags {",
            '    Name = "web"',
            '    Team = "platform"',
            "  }",
            "  security_groups = ['ssh', 'http']",
            "  ebs_block_device {",
            '    device_name = "/dev/sda1"',
            "    volume_size = 30",
            "  }",
            "}",
        ]

    def test_empty_config_yields_bare_block(self, tf):
        assert tf.generate_resource("aws_s3_bucket", "assets") == (
            'resource "aws_s3_bucket" "assets" {\n}'
        )

    def test_nested_lists_of_scalars_stay_inline(self, tf):
        out = tf.generate_resource("aws_x", "y", {"cidrs": ["10.0.0.0/16"]})
        assert "  cidrs = ['10.0.0.0/16']" in out

    def test_result_is_recorded_on_the_generator(self, tf):
        block = tf.generate_resource("aws_vpc", "main", {"cidr_block": "10.0.0.0/16"})
        assert tf._resources == [block]


# ---------------------------------------------------------------------------
# TerraformGenerator — variable / output
# ---------------------------------------------------------------------------


class TestGenerateVariable:
    def test_string_default_with_description_and_sensitivity(self, tf):
        out = tf.generate_variable(
            "db_password", "string", "hunter2", "Database password", sensitive=True
        )
        assert out.splitlines() == [
            'variable "db_password" {',
            "  type = string",
            '  default     = "hunter2"',
            '  description = "Database password"',
            "  sensitive   = true",
            "}",
        ]

    @pytest.mark.parametrize(
        ("default", "rendered"),
        [
            (True, "  default     = true"),
            (False, "  default     = false"),
            (["a", "b"], "  default     = ['a', 'b']"),
            ({"k": "v"}, "  default     = {'k': 'v'}"),
            (42, "  default     = 42"),
        ],
    )
    def test_default_value_shapes(self, tf, default, rendered):
        out = tf.generate_variable("v", "any", default)
        assert rendered in out

    def test_no_default_emits_only_the_type(self, tf):
        assert tf.generate_variable("region", "string") == (
            'variable "region" {\n  type = string\n}'
        )

    def test_result_is_recorded_on_the_generator(self, tf):
        block = tf.generate_variable("project")
        assert tf._variables == [block]


class TestGenerateOutput:
    def test_with_description(self, tf):
        out = tf.generate_output("vpc_id", "aws_vpc.main.id", "The VPC id")
        assert out.splitlines() == [
            'output "vpc_id" {',
            "  value = aws_vpc.main.id",
            '  description = "The VPC id"',
            "}",
        ]

    def test_without_description(self, tf):
        assert tf.generate_output("id", "aws_vpc.main.id") == (
            'output "id" {\n  value = aws_vpc.main.id\n}'
        )

    def test_result_is_recorded_on_the_generator(self, tf):
        block = tf.generate_output("id", "x")
        assert tf._outputs == [block]


class TestReset:
    def test_clears_every_accumulated_section(self, tf):
        tf.generate_provider("aws")
        tf.generate_resource("aws_vpc", "main")
        tf.generate_variable("project")
        tf.generate_output("id", "x")
        assert (tf._providers, tf._resources, tf._variables, tf._outputs) != ([], [], [], [])

        tf._reset()

        assert tf._providers == []
        assert tf._resources == []
        assert tf._variables == []
        assert tf._outputs == []


# ---------------------------------------------------------------------------
# TerraformGenerator — generate_full_config
# ---------------------------------------------------------------------------


class TestGenerateFullConfig:
    def test_assembles_provider_variables_and_resources_for_aws(self, tf):
        out = tf.generate_full_config(
            "aws",
            {
                "main": {"resource_type": "aws_vpc", "cidr_block": "10.0.0.0/16"},
                "bucket": {"acl": "private"},
            },
            {
                "project": {"type": "string", "default": "noema", "description": "Name"},
                "region": "us-east-1",
            },
        )
        lines = out.splitlines()
        assert lines[0] == "# Generated by Noema Terraform Module"
        assert 'provider "aws" {' in out
        assert '  region = "us-east-1"' in out
        assert 'variable "project" {' in out
        assert '  default     = "noema"' in out
        assert 'variable "region" {' in out
        assert "  type = string" in out
        assert '  default     = "us-east-1"' in out
        assert 'resource "aws_vpc" "main" {' in out
        # resource_type absent → derived from "<provider>_<name>"
        assert 'resource "aws_bucket" "bucket" {' in out
        assert out.index('variable "project"') < out.index('variable "region"')
        assert out.index('variable "region"') < out.index('resource "aws_vpc"')

    @pytest.mark.parametrize(
        ("provider", "provider_block"),
        [
            (
                "gcp",
                'provider "gcp" {\n  project = "${var.project}"\n  region = "us-central1"\n}',
            ),
            ("azure", 'provider "azure" {\n}'),
            ("kubernetes", 'provider "kubernetes" {\n}'),
        ],
    )
    def test_provider_blocks_per_cloud(self, tf, provider, provider_block):
        out = tf.generate_full_config(provider, {})
        assert provider_block in out

    def test_azure_provider_block_has_no_body(self, tf):
        out = tf.generate_full_config("azure", {"rg": {"location": "eastus"}})
        assert 'provider "azure" {\n}\n' in out
        assert 'resource "azure_rg" "rg" {' in out

    def test_without_variables_section(self, tf):
        out = tf.generate_full_config("aws", {}, variables=None)
        assert "variable" not in out
        assert 'provider "aws" {' in out


# ---------------------------------------------------------------------------
# PulumiGenerator
# ---------------------------------------------------------------------------


class TestPulumiGenerator:
    def test_python_stack_template(self, pulumi):
        out = pulumi.generate_stack("prod")
        assert "import pulumi" in out
        assert "# Stack: prod" in out
        assert '# vpc = aws.ec2.Vpc("prod-vpc",' in out

    def test_typescript_stack_template(self, pulumi):
        out = pulumi.generate_stack("prod", "typescript")
        assert 'import * as pulumi from "@pulumi/pulumi";' in out
        assert "// Stack: prod" in out

    def test_go_stack_template(self, pulumi):
        out = pulumi.generate_stack("prod", "go")
        assert "package main" in out
        assert "pulumi.Run(func(ctx *pulumi.Context) error {" in out

    def test_unknown_language_falls_back_to_python(self, pulumi):
        assert pulumi.generate_stack("prod", "rust") == pulumi.generate_stack("prod")

    def test_generate_resource_call(self, pulumi):
        out = pulumi.generate_resource("aws:ec2/vpc:Vpc", "main", {"cidrBlock": "10.0.0.0/16"})
        assert out == (
            'resource = pulumi.Resource("aws:ec2/vpc:Vpc", "main", cidrBlock=\'10.0.0.0/16\')'
        )

    def test_generate_resource_without_config(self, pulumi):
        assert pulumi.generate_resource("aws:s3/bucket:Bucket", "assets") == (
            'resource = pulumi.Resource("aws:s3/bucket:Bucket", "assets", )'
        )


# ---------------------------------------------------------------------------
# TerraformModule.execute — dispatch
# ---------------------------------------------------------------------------


class TestModuleDispatch:
    def test_metadata_defaults_when_the_task_has_none(self):
        module = TerraformModule()
        result = module.execute(SimpleNamespace(title="build infra", tags=[]))
        assert result["action"] == "terraform"
        assert result["provider"] == "aws"
        assert result["_confidence"] == 0.85
        assert 'resource "aws_vpc" "main_vpc" {' in result["content"]

    def test_object_without_title_or_tags_uses_defaults(self):
        module = TerraformModule()
        result = module.execute(object())
        assert result["action"] == "terraform"
        assert "10.0.1.0/24" in result["content"]

    def test_pulumi_wins_over_template_in_the_title(self):
        module = TerraformModule()
        result = module.execute(_task("pulumi template stack"))
        assert result["action"] == "pulumi"
        assert result["language"] == "python"
        assert result["stack_name"] == "myproject"

    def test_template_action_selected_by_tag(self):
        module = TerraformModule()
        result = module.execute(_task(tags=["template"]))
        assert result["action"] == "templates"
        assert result["_confidence"] == 0.90
        assert set(result["templates"]) == {"aws", "gcp", "azure"}

    def test_template_action_lists_every_provider(self):
        module = TerraformModule()
        result = module.execute(_task("list templates"))
        templates = result["templates"]
        assert templates["aws"]["vpc"]["type"] == AWS_TEMPLATES["vpc"]["type"]
        assert set(templates["gcp"]) == set(GCP_TEMPLATES)
        assert set(templates["azure"]) == set(AZURE_TEMPLATES)
        assert result["_confidence"] == 0.90


class TestModuleTerraformPath:
    def test_known_aws_template_short_circuits_the_full_config(self):
        module = TerraformModule()
        result = module.execute(
            _task("terraform", metadata={"provider": "aws", "template": "lambda"})
        )
        assert result["action"] == "terraform"
        assert result["template"] == "lambda"
        assert result["provider"] == "aws"
        assert result["_confidence"] == 0.88
        assert 'provider "aws" {\n  region = "us-east-1"\n}' in result["content"]
        assert 'resource "aws_lambda_function" "lambda" {' in result["content"]
        assert '  runtime = "python3.12"' in result["content"]
        assert "available_templates" not in result

    def test_known_template_for_a_non_aws_provider_falls_through(self):
        module = TerraformModule()
        result = module.execute(_task("terraform", metadata={"provider": "gcp", "template": "gke"}))
        assert result["action"] == "terraform"
        assert "template" not in result
        assert result["_confidence"] == 0.85
        assert 'provider "gcp" {' in result["content"]

    def test_unknown_template_name_falls_through(self):
        module = TerraformModule()
        result = module.execute(
            _task("terraform", metadata={"provider": "aws", "template": "nope"})
        )
        assert "template" not in result
        assert result["available_templates"]["aws"] == list(AWS_TEMPLATES)
        assert result["available_templates"]["gcp"] == list(GCP_TEMPLATES)
        assert result["available_templates"]["azure"] == list(AZURE_TEMPLATES)

    def test_explicit_resources_and_variables_are_used_verbatim(self):
        module = TerraformModule()
        result = module.execute(
            _task(
                "terraform",
                metadata={
                    "provider": "aws",
                    "project": "acme",
                    "resources": {"queue": {"resource_type": "aws_sqs_queue", "name": "jobs"}},
                    "variables": {"queue_name": {"type": "string", "default": "jobs"}},
                },
            )
        )
        content = result["content"]
        assert 'resource "aws_sqs_queue" "queue" {' in content
        assert '  name = "jobs"' in content
        assert 'variable "queue_name" {' in content
        assert "Project name" not in content

    def test_empty_resources_dict_receives_the_vpc_subnet_defaults(self):
        module = TerraformModule()
        result = module.execute(
            _task("terraform", metadata={"provider": "gcp", "resources": {}, "variables": {}})
        )
        content = result["content"]
        assert 'resource "gcp_vpc" "main_vpc" {' in content
        assert "  enable_dns_hostnames = true" in content
        assert 'resource "gcp_subnet" "main_subnet" {' in content
        assert "variable" not in content

    def test_project_name_flows_into_the_default_variables(self):
        module = TerraformModule()
        result = module.execute(_task("terraform", metadata={"project": "noema-prod"}))
        assert '  default     = "noema-prod"' in result["content"]
        assert '  description = "Environment"' in result["content"]


class TestModulePulumiPath:
    def test_language_and_stack_name_come_from_metadata(self):
        module = TerraformModule()
        result = module.execute(
            _task(
                "provision with pulumi",
                metadata={"language": "go", "stack_name": "core-infra", "project": "acme"},
            )
        )
        assert result["action"] == "pulumi"
        assert result["language"] == "go"
        assert result["stack_name"] == "core-infra"
        assert result["_confidence"] == 0.80
        assert "// Stack: core-infra" in result["content"]

    def test_stack_name_defaults_to_the_project(self):
        module = TerraformModule()
        result = module.execute(_task(tags=["pulumi"], metadata={"project": "acme"}))
        assert result["stack_name"] == "acme"
        assert "# Stack: acme" in result["content"]

    def test_non_string_metadata_values_are_coerced(self):
        module = TerraformModule()
        result = module.execute(_task(tags=["pulumi"], metadata={"language": 7, "stack_name": 12}))
        assert result["language"] == "7"
        assert result["stack_name"] == "12"
