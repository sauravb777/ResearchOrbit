from pathlib import Path
import zipfile
def test_release_packager_excludes_runtime_and_build_artifacts(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("package_release", Path("scripts/package_release.py"))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    build_release = module.build_release
    root = tmp_path / "project"
    root.mkdir()
    (root / "pyproject.toml").write_text('[project]\nversion = "99.0.0"\n', encoding="utf-8")
    (root / "README.md").write_text("ok", encoding="utf-8")
    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "bad.pyc").write_bytes(b"bad")
    (root / ".pytest_cache").mkdir()
    (root / ".pytest_cache" / "bad").write_text("bad", encoding="utf-8")
    (root / "build").mkdir()
    (root / "build" / "bad.whl").write_bytes(b"bad")
    (root / "pkg.egg-info").mkdir()
    (root / "pkg.egg-info" / "PKG-INFO").write_text("bad", encoding="utf-8")
    (root / "src").mkdir()
    (root / "src" / "app.py").write_text("ok", encoding="utf-8")
    output = tmp_path / "release.zip"
    version, count = build_release(root, output)
    assert version == "99.0.0"
    assert count == 3
    with zipfile.ZipFile(output) as zf:
        names = set(zf.namelist())
    assert names == {"README.md", "pyproject.toml", "src/app.py"}
def test_release_packager_is_byte_deterministic(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("package_release_deterministic", Path("scripts/package_release.py"))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    root = tmp_path / "project"
    root.mkdir()
    (root / "pyproject.toml").write_text('[project]\nversion = "99.0.0"\n', encoding="utf-8")
    (root / "README.md").write_text("stable", encoding="utf-8")
    first = tmp_path / "one.zip"
    second = tmp_path / "two.zip"
    module.build_release(root, first)
    module.build_release(root, second)
    assert first.read_bytes() == second.read_bytes()
