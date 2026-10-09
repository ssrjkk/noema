"""Live demo — prove every module produces real output."""

# 1. Security Scanner
from noema.modules.security_scanner.kernel import SecurityScanner

s = SecurityScanner()
code = """
password = "admin123"
query = f"SELECT * FROM users WHERE id = {user_id}"
subprocess.call(user_input, shell=True)
api_key = "sk-1234567890abcdefghijklmnopqrstuvwxyz"
"""
result = s.scan_code(code, "python")
print("=== SECURITY SCANNER ===")
print(f"Score: {result.score}/100")
for v in result.vulnerabilities:
    print(f"  [{v.severity}] {v.category}: {v.description} (line {v.line})")
print()

# 2. Quality Analyzer
from noema.modules.quality.kernel import CodeAnalyzer

qa = CodeAnalyzer()
test_code = """
def complex_function(data):
    result = []
    for item in data:
        if item.get('active'):
            if item.get('role') == 'admin':
                if item.get('age') > 18:
                    for sub in item.get('subs', []):
                        if sub.get('verified'):
                            result.append(sub)
    return result

class GodClass:
    def method1(self): pass
    def method2(self): pass
    def method3(self): pass
    def method4(self): pass
    def method5(self): pass
    def method6(self): pass
    def method7(self): pass
    def method8(self): pass
    def method9(self): pass
    def method10(self): pass
    def method11(self): pass
    def method12(self): pass
"""
report = qa.analyze(test_code)
print("=== CODE QUALITY ===")
print(f"Grade: {report.grade}")
print(
    f"Cyclomatic: {report.metrics.get('cyclomatic_complexity')}, Cognitive: {report.metrics.get('cognitive_complexity')}"
)
print(f"Smells: {len(report.smells)}")
for sm in report.smells[:3]:
    print(f"  - {sm.get('type')}: {sm.get('description', '')}")
print()

# 3. Database Schema
from noema.modules.database.kernel import SchemaDesigner

db = SchemaDesigner()
tables = db.design_schema(["user", "auth", "ecommerce"])
print("=== DATABASE SCHEMA ===")
for t in tables[:5]:
    cols = [c.name for c in t.columns[:5]]
    print(f"  {t.name}: {cols}")
print()

# 4. Dockerfile
from noema.modules.containers.kernel import DockerfileGenerator

dg = DockerfileGenerator()
dockerfile = dg.generate("python")
print("=== DOCKERFILE (python) ===")
print(dockerfile[:500])
print()

# 5. Terraform
from noema.modules.terraform.kernel import TerraformGenerator

tg = TerraformGenerator()
hcl = tg.generate_full_config(
    "aws", {"vpc": {"cidr_block": "10.0.0.0/16"}, "subnet": {"cidr_block": "10.0.1.0/24"}}
)
print("=== TERRAFORM HCL ===")
print(hcl[:400])
print()

# 6. i18n
from noema.modules.i18n.kernel import TranslationStore

ts = TranslationStore()
ts.add("greeting", "en", "Hello {name}!")
ts.add("greeting", "ru", "Привет, {name}!")
ts.add("greeting", "de", "Hallo {name}!")
print("=== I18N ===")
for lang in ["en", "ru", "de"]:
    print(f"  {lang}: {ts.get('greeting', lang, name='World')}")
print()

# 7. Cache
from noema.modules.caching.kernel import LRUCache

cache = LRUCache(max_size=3, default_ttl=60)
cache.set("user:1", {"name": "Alice"})
cache.set("user:2", {"name": "Bob"})
cache.set("user:3", {"name": "Charlie"})
cache.set("user:4", {"name": "Diana"})
print("=== CACHE ===")
print(f"  user:1 (evicted) = {cache.get('user:1')}")
print(f"  user:4 (cached) = {cache.get('user:4')}")
print()

# 8. Data Pipeline -> Airflow
from noema.modules.data_pipeline.kernel import DataPipeline, PipelineStep

dp = DataPipeline(name="etl_pipeline")
dp.add_step(PipelineStep(name="extract", type="extract", config={"source": "s3://bucket/data"}))
dp.add_step(
    PipelineStep(name="transform", type="transform", config={"ops": ["filter", "aggregate"]})
)
dp.add_step(PipelineStep(name="load", type="load", config={"target": "warehouse"}))
dag = dp.to_airflow_dag()
print("=== AIRFLOW DAG ===")
print(dag[:500])
print()

# 9. Auth RBAC
from noema.modules.auth.kernel import RBAC

rbac = RBAC()
rbac.assign_role("alice", "admin")
rbac.assign_role("bob", "user")
print("=== AUTH RBAC ===")
print(f"  alice can manage_users: {rbac.has_permission('alice', 'manage_users')}")
print(f"  bob can manage_users: {rbac.has_permission('bob', 'manage_users')}")
print(f"  bob can read: {rbac.has_permission('bob', 'read')}")
print(f"  alice permissions: {rbac.get_user_permissions('alice')}")
print()

# 10. GraphQL
from noema.modules.graphql.kernel import GraphQLField, GraphQLTypeDef, build_full_schema

user_type = GraphQLTypeDef(
    name="User",
    fields=[
        GraphQLField(name="id", field_type="ID", nullable=False),
        GraphQLField(name="name", field_type="String", nullable=False),
        GraphQLField(name="email", field_type="String", nullable=False),
    ],
)
post_type = GraphQLTypeDef(
    name="Post",
    fields=[
        GraphQLField(name="id", field_type="ID", nullable=False),
        GraphQLField(name="title", field_type="String", nullable=False),
    ],
)
schema = build_full_schema([user_type, post_type])
print("=== GRAPHQL SCHEMA ===")
print(schema[:500])
print()

# 11. ML Ops
from noema.modules.ml_ops.kernel import MLPipeline

ml = MLPipeline()
suggestion = ml.suggest_model("image_classification", "pytorch")
print("=== ML MODEL SUGGESTION ===")
print("  Task: image_classification")
print(f"  Recommended: {suggestion.get('recommendation')}")
print(f"  Models: {suggestion.get('suggested_models')}")
print()

# 12. Performance Profiler
from noema.modules.performance.kernel import Profiler

pf = Profiler()
perf_code = """
import requests
for i in range(10000):
    result = requests.get("https://api.example.com/data/" + str(i))
    data = result.json()
    output = ""
    for item in data:
        output += str(item)
"""
analysis = pf.analyze_code_performance(perf_code)
print("=== PERFORMANCE PROFILER ===")
print(f"  Lines: {analysis.get('total_lines')}, Issues: {analysis.get('issues_found')}")
print(f"  Score: {analysis.get('performance_score')}")
for b in analysis.get("bottlenecks", []):
    print(f"  [{b['severity']}] {b['type']}: {b['suggestion']}")
print()

print("=== ALL DEMOS COMPLETE ===")
