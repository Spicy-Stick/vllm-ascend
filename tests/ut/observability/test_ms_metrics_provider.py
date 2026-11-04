# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import importlib.util
import sys
from dataclasses import dataclass
from importlib.metadata import EntryPoint
from pathlib import Path
from types import ModuleType

import yaml

_REPOSITORY_ROOT = Path(__file__).parents[3]
_PACKAGE_ROOT = _REPOSITORY_ROOT / "vllm_ascend" / "observability" / "ms_metrics"
_HANDLER_PACKAGE = "vllm_ascend.observability.ms_metrics.handlers"
_COMMON_MODULE = f"{_HANDLER_PACKAGE}.common"


def _load_all_provider_configs(config_paths):
    configs = []
    for config_path in config_paths:
        configs.extend(yaml.safe_load(Path(config_path).read_text(encoding="utf-8")))
    return configs


def _load_source(module_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(module_name, _PACKAGE_ROOT / filename)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_provider_api(monkeypatch, **attributes):
    package = ModuleType("ms_service_metric")
    provider_api = ModuleType("ms_service_metric.provider_api")
    for name, value in attributes.items():
        setattr(provider_api, name, value)
    monkeypatch.setitem(sys.modules, "ms_service_metric", package)
    monkeypatch.setitem(sys.modules, "ms_service_metric.provider_api", provider_api)


def _load_feature_handlers(monkeypatch, module_name: str, feature: str):
    """Load ``handlers/<feature>.py`` freshly with ``handlers/common.py`` seeded.

    The common module is registered under its real absolute name so the
    feature module's import resolves without importing the whole package.
    """
    common = _load_source(f"{module_name}_common", "handlers/common.py")
    monkeypatch.setitem(sys.modules, _COMMON_MODULE, common)
    return _load_source(module_name, f"handlers/{feature}.py")


def test_get_metric_provider_returns_packaged_yaml(monkeypatch):
    @dataclass
    class FakeMetricProvider:
        name: str
        config_paths: tuple[str, ...]
        priority: int
        owned_symbol_prefixes: tuple[str, ...]
        framework_package: str | None = None
        handler_module_prefixes: tuple[str, ...] = ()
        ownership_mode: str = "overlay"

    _install_provider_api(monkeypatch, MetricProvider=FakeMetricProvider)
    provider_module = _load_source("test_vllm_ascend_metric_provider", "provider.py")

    provider = provider_module.get_metric_provider()

    assert provider.name == "vllm-ascend"
    assert provider.framework_package == "vllm_ascend"
    assert provider.ownership_mode == "overlay"
    assert provider.owned_symbol_prefixes == ("vllm_ascend.",)
    assert provider.handler_module_prefixes == ("vllm_ascend.observability.ms_metrics.",)
    assert provider.config_paths == tuple(sorted(provider.config_paths))
    assert [Path(path).name for path in provider.config_paths] == [
        "base_metrics.yaml",
    ]
    assert all(Path(path).is_file() for path in provider.config_paths)
    config = _load_all_provider_configs(provider.config_paths)
    assert len(config) == 7
    assert all(item["symbol"].startswith("vllm_ascend.") for item in config)
    assert all("id" not in item for item in config)
    assert all(
        item.get("handler", "").startswith(
            (
                "ms_service_metric.provider_handlers:",
                "vllm_ascend.observability.ms_metrics.handlers:",
            )
        )
        for item in config
    )


def test_setup_registers_provider_entry_point_and_yaml_package_data():
    setup_path = _REPOSITORY_ROOT / "setup.py"
    syntax_tree = ast.parse(setup_path.read_text(encoding="utf-8"))
    setup_call = next(
        node
        for node in ast.walk(syntax_tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "setup"
    )
    keywords = {keyword.arg: keyword.value for keyword in setup_call.keywords}
    entry_points = ast.literal_eval(keywords["entry_points"])
    package_data = ast.literal_eval(keywords["package_data"])

    assert entry_points["ms_service_metric.providers"] == [
        "vllm-ascend = vllm_ascend.observability.ms_metrics:get_metric_provider"
    ]
    assert package_data["vllm_ascend.observability.ms_metrics"] == ["config/*.yaml"]


def test_provider_entry_point_loads_without_metric_core(monkeypatch):
    package = ModuleType("vllm_ascend")
    package.__path__ = [str(_REPOSITORY_ROOT / "vllm_ascend")]
    monkeypatch.setitem(sys.modules, "vllm_ascend", package)
    monkeypatch.delitem(sys.modules, "ms_service_metric", raising=False)
    monkeypatch.delitem(sys.modules, "ms_service_metric.provider_api", raising=False)

    entry_point = EntryPoint(
        name="vllm-ascend",
        value="vllm_ascend.observability.ms_metrics:get_metric_provider",
        group="ms_service_metric.providers",
    )

    assert callable(entry_point.load())


def test_provider_yaml_symbols_exist_in_current_vllm_ascend_source():
    config_paths = sorted((_PACKAGE_ROOT / "config").glob("*.yaml"))
    assert len(config_paths) == 1
    config = _load_all_provider_configs(config_paths)

    for item in config:
        module_name, attribute_path = item["symbol"].split(":", 1)
        source_path = _REPOSITORY_ROOT / Path(*module_name.split(".")).with_suffix(".py")
        assert source_path.is_file(), f"Missing symbol module: {item['symbol']}"

        syntax_tree = ast.parse(source_path.read_text(encoding="utf-8"))
        owner_name, method_name = attribute_path.split(".", 1)
        owner = next(
            (node for node in syntax_tree.body if isinstance(node, ast.ClassDef) and node.name == owner_name),
            None,
        )
        assert owner is not None, f"Missing symbol owner: {item['symbol']}"
        assert any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name
            for node in owner.body
        ), f"Missing symbol method: {item['symbol']}"
