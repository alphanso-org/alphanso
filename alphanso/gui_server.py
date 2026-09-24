"""Local web interface for building and running ALPHANSO calculations."""

from __future__ import annotations

import ast
import json
import math
import mimetypes
import threading
import time
import webbrowser
from copy import deepcopy
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.metadata import PackageNotFoundError, metadata, version
from importlib.resources import files
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import yaml
from platformdirs import user_data_dir

from .configuration import (
    CALCULATION_TYPES,
    common_config_defaults,
    configuration_schema,
    validate_common_config,
)
from .transport import Transport
from .utils import matdef_to_zaids


ASSET_ROOT = Path(__file__).with_name("gui_assets")
MAX_REQUEST_BYTES = 1_000_000
MAX_SAVED_CONFIGURATIONS = 50
GEOMETRIES = set(CALCULATION_TYPES)
DISTRIBUTION_NAME = __package__.split(".", 1)[0]
_SAVED_CONFIGURATIONS_LOCK = threading.Lock()


def _package_version() -> str:
    try:
        return version(DISTRIBUTION_NAME)
    except PackageNotFoundError:
        return "development"


def package_info() -> dict[str, Any]:
    """Read product identity directly from the installed distribution metadata."""
    try:
        package_metadata = metadata(DISTRIBUTION_NAME)
    except PackageNotFoundError:
        return {
            "name": DISTRIBUTION_NAME,
            "display_name": DISTRIBUTION_NAME.upper(),
            "version": _package_version(),
            "summary": "",
            "urls": {},
        }

    urls: dict[str, str] = {}
    for entry in package_metadata.get_all("Project-URL") or ():
        label, separator, url = entry.partition(",")
        if separator:
            urls[label.strip().lower().replace(" ", "_")] = url.strip()
    name = package_metadata.get("Name", DISTRIBUTION_NAME)
    return {
        "name": name,
        "display_name": name.upper(),
        "version": package_metadata.get("Version", _package_version()),
        "summary": package_metadata.get("Summary", ""),
        "urls": urls,
    }


def _gui_display_name(product: dict[str, Any] | None = None) -> str:
    """Build the visible product name from installed distribution metadata."""
    product = package_info() if product is None else product
    base_name = str(
        product.get("display_name") or product.get("name") or "ALPHANSO"
    ).strip()
    return base_name[:-4].rstrip() if base_name.upper().endswith(" GUI") else base_name


def _json_safe(value: Any) -> Any:
    """Convert API payload values to strict JSON-safe built-in types."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _saved_configurations_path() -> Path:
    """Return the cross-platform application-data path for saved configs."""
    return (
        Path(user_data_dir(DISTRIBUTION_NAME, appauthor=False))
        / "saved-configurations.json"
    )


def saved_configuration_catalog(path: Path | None = None) -> list[dict[str, Any]]:
    """Load valid saved configurations without failing GUI startup."""
    target = _saved_configurations_path() if path is None else Path(path)
    with _SAVED_CONFIGURATIONS_LOCK:
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError, UnicodeError):
            return []
    if not isinstance(payload, list):
        return []
    return [
        _json_safe(entry)
        for entry in payload
        if isinstance(entry, dict) and isinstance(entry.get("config"), dict)
    ][:MAX_SAVED_CONFIGURATIONS]


def write_saved_configuration_catalog(
    configurations: Any,
    path: Path | None = None,
) -> list[dict[str, Any]]:
    """Atomically persist the GUI configuration library."""
    if not isinstance(configurations, list):
        raise ValueError("Saved configurations must be a list.")
    cleaned: list[dict[str, Any]] = []
    for entry in configurations[:MAX_SAVED_CONFIGURATIONS]:
        if not isinstance(entry, dict) or not isinstance(entry.get("config"), dict):
            raise ValueError("Each saved configuration must contain a config object.")
        cleaned.append(_json_safe(entry))

    target = _saved_configurations_path() if path is None else Path(path)
    temporary = target.with_name(f".{target.name}.tmp")
    with _SAVED_CONFIGURATIONS_LOCK:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(
            json.dumps(cleaned, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)
    return cleaned


def _example_description(config: dict[str, Any], kind: str) -> str:
    geometry = CALCULATION_TYPES.get(config.get("calc_type"), {})
    description = geometry.get("description", "ALPHANSO calculation configuration.")
    return f"{description} Loaded from the bundled {kind.lower()} example."


def _example_entry(
    *, identifier: str, source: str, kind: str, config: dict[str, Any]
) -> dict[str, Any]:
    return {
        "id": identifier,
        "source": source,
        "kind": kind,
        "description": _example_description(config, kind),
        "config": _json_safe(config),
    }


def example_catalog() -> list[dict[str, Any]]:
    """Discover every example bundled in the ``example_usage`` package."""
    root = files("example_usage")
    examples: list[dict[str, Any]] = []

    for resource in sorted(root.iterdir(), key=lambda item: item.name):
        if resource.name.endswith((".yaml", ".yml")):
            config = yaml.safe_load(resource.read_text(encoding="utf-8"))
            if not isinstance(config, dict):
                continue
            identifier = f"yaml-{resource.name.removeprefix('example_config_').rsplit('.', 1)[0]}"
            examples.append(_example_entry(
                identifier=identifier,
                source=f"example_usage/{resource.name}",
                kind="YAML",
                config=config,
            ))

    scripts = sorted(
        (
            resource
            for resource in root.iterdir()
            if resource.name.endswith(".py") and resource.name != "__init__.py"
        ),
        key=lambda item: item.name,
    )
    for script in scripts:
        tree = ast.parse(script.read_text(encoding="utf-8"), filename=script.name)
        for node in tree.body:
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
            ):
                variable = node.targets[0].id
                value = node.value
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                variable = node.target.id
                value = node.value
            else:
                continue
            if not variable.endswith("_config") or value is None:
                continue
            try:
                config = ast.literal_eval(value)
            except (TypeError, ValueError):
                continue
            if not isinstance(config, dict):
                continue
            variable_slug = variable.removesuffix("_config").replace("_", "-")
            script_slug = script.name.removesuffix(".py").removeprefix("example_")
            identifier = f"python-{variable_slug}"
            if script.name != "example_script.py":
                identifier = f"python-{script_slug}-{variable_slug}"
            examples.append(_example_entry(
                identifier=identifier,
                source=f"example_usage/{script.name}::{variable}",
                kind="Python API",
                config=config,
            ))

    return examples


def workspace_defaults(examples: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Build editor state from canonical defaults and the bundled examples."""
    examples = example_catalog() if examples is None else examples
    by_geometry: dict[str, dict[str, Any]] = {}
    for example in examples:
        config = example.get("config", {})
        geometry = config.get("calc_type")
        if geometry in CALCULATION_TYPES and geometry not in by_geometry:
            by_geometry[geometry] = deepcopy(config)

    preferred_geometry = "homogeneous" if "homogeneous" in by_geometry else next(
        iter(CALCULATION_TYPES)
    )
    preferred = by_geometry.get(preferred_geometry, {})
    beam = by_geometry.get("beam", {})
    homogeneous = by_geometry.get("homogeneous", {})
    interface = by_geometry.get("interface", {})
    sandwich = by_geometry.get("sandwich", {})

    defaults: dict[str, Any] = {
        "name": preferred.get("name", "Untitled calculation"),
        "calc_type": preferred_geometry,
        **common_config_defaults(),
        "beam_matdef": deepcopy(beam.get("matdef", {})),
        "homogeneous_matdef": deepcopy(homogeneous.get("matdef", {})),
        "source_matdef": deepcopy(
            interface.get("source_matdef", sandwich.get("source_matdef", {}))
        ),
        "source_density": interface.get(
            "source_density", sandwich.get("source_density", 1.0)
        ),
        "target_matdef": deepcopy(
            interface.get("target_matdef", sandwich.get("target_matdef", {}))
        ),
        "intermediate_layers": deepcopy(sandwich.get("intermediate_layers", [])),
        "beam_mode": "spectrum" if "beam_intensities" in beam else "mono",
        "beam_energy": beam.get("beam_energy", 1.0),
        "beam_intensities": deepcopy(
            beam.get("beam_intensities", [[beam.get("beam_energy", 1.0), 1.0]])
        ),
    }
    defaults.update(deepcopy(preferred))
    if preferred_geometry == "homogeneous" and "matdef" in preferred:
        defaults["homogeneous_matdef"] = deepcopy(preferred["matdef"])
    return defaults


def _material_for_transport(material: dict[Any, Any]) -> dict[Any, Any]:
    """Restore numeric ZAID keys stringified by JSON object encoding."""
    return {
        int(key) if isinstance(key, str) and key.isdigit() else key: value
        for key, value in material.items()
    }


def config_for_transport(config: dict[str, Any]) -> dict[str, Any]:
    """Copy a GUI payload and normalize every material for the Python API."""
    prepared = dict(config)
    for key in ("matdef", "source_matdef", "target_matdef"):
        material = prepared.get(key)
        if isinstance(material, dict):
            prepared[key] = _material_for_transport(material)

    layers = prepared.get("intermediate_layers")
    if isinstance(layers, list):
        prepared["intermediate_layers"] = [
            {
                **layer,
                "matdef": _material_for_transport(layer["matdef"]),
            }
            if isinstance(layer, dict) and isinstance(layer.get("matdef"), dict)
            else layer
            for layer in layers
        ]
    return prepared


def _validate_material(material: Any, label: str, errors: list[str]) -> None:
    if not isinstance(material, dict) or not material:
        errors.append(f"{label} must contain at least one isotope.")
        return

    for isotope, fraction in material.items():
        if not isinstance(isotope, str) or not isotope.strip():
            errors.append(f"{label} contains an invalid isotope name.")
            continue
        if isinstance(fraction, bool) or not isinstance(fraction, (int, float)):
            errors.append(f"{label}: {isotope} must have a numeric mass fraction.")
        elif not math.isfinite(float(fraction)) or float(fraction) <= 0:
            errors.append(f"{label}: {isotope} must have a positive mass fraction.")

    if errors:
        return

    total = sum(float(value) for value in material.values())
    if not math.isclose(total, 1.0, rel_tol=1e-5, abs_tol=1e-6):
        errors.append(f"{label} mass fractions must sum to 1.000 (currently {total:.6g}).")

    try:
        mass_fractions, _ = matdef_to_zaids(_material_for_transport(material))
        if not mass_fractions:
            errors.append(
                f"{label} does not contain a recognized isotope, natural element, or ZAID."
            )
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"{label} could not be resolved: {exc}")


def validate_config(config: Any) -> list[str]:
    """Return user-facing validation errors for a GUI calculation payload."""
    if not isinstance(config, dict):
        return ["Calculation configuration must be an object."]

    errors: list[str] = []
    calc_type = config.get("calc_type")
    if calc_type not in GEOMETRIES:
        errors.append("Choose a valid calculation geometry.")
        return errors

    if calc_type in {"beam", "homogeneous"}:
        label = "Target material" if calc_type == "beam" else "Source mixture"
        _validate_material(config.get("matdef"), label, errors)
    else:
        _validate_material(config.get("source_matdef"), "Source material", errors)
        _validate_material(config.get("target_matdef"), "Target material", errors)

        density = config.get("source_density")
        if isinstance(density, bool) or not isinstance(density, (int, float)) or density <= 0:
            errors.append("Source density must be greater than zero.")

    if calc_type == "beam":
        beam_energy = config.get("beam_energy")
        beam_intensities = config.get("beam_intensities")
        if beam_intensities is None:
            if (isinstance(beam_energy, bool)
                    or not isinstance(beam_energy, (int, float))
                    or beam_energy <= 0):
                errors.append("Beam energy must be greater than zero.")
        elif not isinstance(beam_intensities, list) or not beam_intensities:
            errors.append("Beam intensities must contain at least one [energy, intensity] pair.")
        else:
            for index, pair in enumerate(beam_intensities, start=1):
                if (not isinstance(pair, list) or len(pair) != 2
                        or any(isinstance(value, bool) or not isinstance(value, (int, float))
                               for value in pair)):
                    errors.append(f"Beam intensity row {index} must be [energy, intensity].")
                    continue
                if pair[0] <= 0 or pair[1] <= 0:
                    errors.append(f"Beam intensity row {index} values must be positive.")

    if calc_type == "sandwich":
        layers = config.get("intermediate_layers")
        if not isinstance(layers, list) or not layers:
            errors.append("Add at least one intermediate layer.")
        else:
            for index, layer in enumerate(layers, start=1):
                if not isinstance(layer, dict):
                    errors.append(f"Layer {index} is invalid.")
                    continue
                _validate_material(layer.get("matdef"), f"Layer {index}", errors)
                for key, title in (("density", "density"), ("thickness", "thickness")):
                    value = layer.get(key)
                    if (isinstance(value, bool)
                            or not isinstance(value, (int, float))
                            or value <= 0):
                        errors.append(f"Layer {index} {title} must be greater than zero.")

    errors.extend(validate_common_config(config))
    return errors


def isotope_catalog() -> list[dict[str, Any]]:
    """Build the searchable isotope and natural-element material catalog."""
    data_dir = Path(__file__).with_name("data") / "atomic_data"
    with (data_dir / "atomic_data.json").open(encoding="utf-8") as stream:
        atomic_data = json.load(stream)
    with (data_dir / "element_symbols.json").open(encoding="utf-8") as stream:
        symbols = json.load(stream)

    catalog = []
    for atomic_number_text, element in atomic_data["elements"].items():
        atomic_number = int(atomic_number_text)
        symbol = element.get("symbol") or symbols.get(str(atomic_number))
        has_natural_abundance = any(
            abundance is not None and float(abundance) > 0
            for abundance in (
                isotope.get("abundance")
                for isotope in element.get("isotopes", {}).values()
            )
        )
        if symbol and has_natural_abundance:
            catalog.append({
                "label": symbol,
                "z": atomic_number,
                "a": 0,
                "abundance": 1.0,
                "natural": True,
            })

    for isotope in atomic_data["isotopes"].values():
        mass_number = int(isotope["mass_number"])
        atomic_number = int(isotope["atomic_number"])
        if mass_number <= 0:
            continue
        symbol = symbols.get(str(atomic_number))
        if not symbol:
            continue
        catalog.append({
            "label": f"{symbol}-{mass_number}",
            "z": atomic_number,
            "a": mass_number,
            "abundance": isotope.get("abundance"),
            "natural": False,
        })
    return sorted(catalog, key=lambda item: (item["z"], item["a"]))


def _bootstrap_payload() -> dict[str, Any]:
    examples = example_catalog()
    package = package_info()
    return {
        "package": package,
        "configuration": configuration_schema(),
        "workspace_defaults": workspace_defaults(examples),
        "isotopes": isotope_catalog(),
        "examples": examples,
    }


class _DesktopAPI:
    """Small native-only bridge for operating-system file dialogs."""

    def __init__(self, folder_dialog: Any):
        self._folder_dialog = folder_dialog
        self._window: Any = None

    def _bind_window(self, window: Any) -> None:
        self._window = window

    def choose_directory(self, current: str = "") -> str | None:
        if self._window is None:
            return None
        directory = current if isinstance(current, str) and Path(current).is_dir() else ""
        selected = self._window.create_file_dialog(
            self._folder_dialog,
            directory=directory,
        )
        return str(selected[0]) if selected else None


class AlphansoGUIHandler(BaseHTTPRequestHandler):
    """Serve GUI assets and a small same-origin calculation API."""

    server_version = _gui_display_name()

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[gui] {self.address_string()} - {format % args}")

    def _send_headers(self, status: HTTPStatus, content_type: str, length: int) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    def _send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(_json_safe(payload), allow_nan=False).encode("utf-8")
        self._send_headers(status, "application/json; charset=utf-8", len(body))
        self.wfile.write(body)

    def _send_asset(self, requested_path: str) -> None:
        relative = "index.html" if requested_path in {"", "/"} else unquote(requested_path.lstrip("/"))
        candidate = (ASSET_ROOT / relative).resolve()
        try:
            candidate.relative_to(ASSET_ROOT.resolve())
        except ValueError:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if not candidate.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        body = candidate.read_bytes()
        content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type in {"application/javascript", "application/json"}:
            content_type += "; charset=utf-8"
        self._send_headers(HTTPStatus.OK, content_type, len(body))
        self.wfile.write(body)

    def _read_json(self) -> Any:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0 or content_length > MAX_REQUEST_BYTES:
            raise ValueError("Request body is empty or too large.")
        return json.loads(self.rfile.read(content_length).decode("utf-8"))

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = urlparse(self.path).path
        if path == "/api/bootstrap":
            payload = _bootstrap_payload()
            payload["saved_configurations"] = saved_configuration_catalog(
                self.server.saved_configurations_path
            )
            self._send_json(payload)
        elif path == "/api/health":
            self._send_json({"status": "ready", "version": _package_version()})
        elif path.startswith("/api/"):
            self._send_json({"error": "API endpoint not found."}, HTTPStatus.NOT_FOUND)
        else:
            self._send_asset(path)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = urlparse(self.path).path
        try:
            payload = self._read_json()
            if path == "/api/saved-configurations":
                configurations = (
                    payload.get("configurations")
                    if isinstance(payload, dict)
                    else None
                )
                saved = write_saved_configuration_catalog(
                    configurations,
                    self.server.saved_configurations_path,
                )
                self._send_json({"configurations": saved})
                return

            if path == "/api/run":
                config = payload.get("config") if isinstance(payload, dict) else None
                errors = validate_config(config)
                if errors:
                    self._send_json({"error": "Configuration needs attention.", "details": errors}, HTTPStatus.UNPROCESSABLE_ENTITY)
                    return

                started = time.perf_counter()
                result = Transport.calculate(config_for_transport(config))
                elapsed = time.perf_counter() - started
                self._send_json({"result": result, "elapsed_seconds": elapsed})
                return

            if path == "/api/import":
                source = payload.get("yaml") if isinstance(payload, dict) else None
                if not isinstance(source, str) or not source.strip():
                    raise ValueError("Choose a non-empty YAML configuration.")
                config = yaml.safe_load(source)
                if isinstance(config, list):
                    if len(config) != 1:
                        raise ValueError("The GUI can import one calculation at a time.")
                    config = config[0]
                if not isinstance(config, dict):
                    raise ValueError("The YAML file must contain a calculation object.")
                self._send_json({"config": config, "warnings": validate_config(config)})
                return

            self._send_json({"error": "API endpoint not found."}, HTTPStatus.NOT_FOUND)
        except json.JSONDecodeError:
            self._send_json({"error": "Request body is not valid JSON."}, HTTPStatus.BAD_REQUEST)
        except (ValueError, yaml.YAMLError) as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:  # Keep calculation failures inside the GUI.
            self._send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)


class AlphansoGUIServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        request_handler: type[BaseHTTPRequestHandler],
        *,
        saved_configurations_path: Path | None = None,
    ) -> None:
        self.saved_configurations_path = (
            _saved_configurations_path()
            if saved_configurations_path is None
            else Path(saved_configurations_path)
        )
        super().__init__(server_address, request_handler)


def _server_url(server: AlphansoGUIServer, host: str) -> str:
    actual_port = server.server_address[1]
    display_host = "localhost" if host in {"127.0.0.1", "0.0.0.0", "::"} else host
    return f"http://{display_host}:{actual_port}"


def launch_gui(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    """Launch the local ALPHANSO browser interface and block until interrupted."""
    server = AlphansoGUIServer((host, port), AlphansoGUIHandler)
    url = _server_url(server, host)
    print(f"{_gui_display_name()} is running at {url}")
    print("Press Ctrl+C to stop.")
    if open_browser:
        threading.Timer(0.35, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"\nStopping {_gui_display_name()}.")
    finally:
        server.server_close()


def launch_desktop(
    host: str = "127.0.0.1",
    port: int = 0,
    renderer: str | None = None,
) -> None:
    """Launch ALPHANSO in a native desktop window backed by the local API."""
    try:
        import webview
    except ImportError as exc:
        raise RuntimeError(
            'Desktop support is not installed. Run: pip install "alphanso[desktop]"'
        ) from exc

    server = AlphansoGUIServer((host, port), AlphansoGUIHandler)
    url = _server_url(server, host)
    server_thread = threading.Thread(
        target=server.serve_forever,
        name="alphanso-desktop-server",
        daemon=True,
    )
    server_thread.start()

    storage_path = Path(user_data_dir("alphanso", appauthor=False)) / "desktop"
    storage_path.mkdir(parents=True, exist_ok=True)
    product = package_info()
    window_title = _gui_display_name(product)

    webview.settings["ALLOW_DOWNLOADS"] = True
    desktop_api = _DesktopAPI(webview.FileDialog.FOLDER)
    window = webview.create_window(
        window_title,
        url=url,
        width=1380,
        height=900,
        min_size=(960, 680),
        background_color="#f5f6f6",
        text_select=True,
        js_api=desktop_api,
    )
    desktop_api._bind_window(window)
    print(f"Starting {window_title}.")
    try:
        # Cocoa, WinForms/WebView2, or GTK/Qt owns the main thread here.
        webview.start(
            gui=renderer,
            debug=False,
            private_mode=False,
            storage_path=str(storage_path),
        )
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2)
