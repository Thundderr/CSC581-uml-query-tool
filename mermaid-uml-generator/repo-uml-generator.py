"""
Prototype: sample public GitHub repos by language+license, extract simple UML class diagrams
for Python and Java files, emit Mermaid .mmd files and metadata.json.

Usage:
  - set GITHUB_TOKEN env var
  - python repo-uml-generator.py --language Python --license MIT --per_page 20

Caveats:
  - This is a prototype: lightweight AST extraction; will miss advanced Java constructs or dynamic Python attributes.
  - Always verify license terms beyond the SPDX id for your intended use.
"""

import os
import sys
import json
import tempfile
import shutil
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Any
import requests
import ast

# Optional: javalang for Java parsing
try:
    import javalang
except Exception:
    javalang = None

# ------------ Configuration / default permissive SPDX ids (edit as needed) ------------
# These are SPDX identifiers considered permissive for this prototype.
# Adjust this set if you want to allow more/less licenses.
PERMISSIVE_SPX = {
    "MIT",
    "Apache-2.0",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "ISC",
    "CC0-1.0"
}

GITHUB_API = "https://api.github.com"
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    print("ERROR: set GITHUB_TOKEN environment variable with a GitHub personal access token.", file=sys.stderr)
    sys.exit(1)
HEADERS = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}

# Helpers: GitHub API
def search_repositories(language: str, license_query: str, per_page: int = 10, page: int = 1) -> List[Dict[str, Any]]:
    """
    Uses GitHub search API to find repos matching language and license query.
    Example q: "language:Python license:mit"
    """
    q = f"language:{language} license:{license_query}"
    url = f"{GITHUB_API}/search/repositories"
    params = {"q": q, "sort": "stars", "order": "desc", "per_page": per_page, "page": page}
    resp = requests.get(url, headers=HEADERS, params=params)
    resp.raise_for_status()
    data = resp.json()
    return data.get("items", [])

def get_repo_license_spdx(owner: str, repo: str) -> Dict[str, Any]:
    """Return license metadata for a repo via GitHub API: {'spdx_id':..., 'name':..., 'url':...}"""
    url = f"{GITHUB_API}/repos/{owner}/{repo}/license"
    resp = requests.get(url, headers=HEADERS)
    if resp.status_code == 404:
        return {}
    resp.raise_for_status()
    return resp.json().get("license", {})

# Extraction: Python AST
def extract_python_classes_from_text(src_text: str):
    """Return a list of classes with name, attributes (simple assigns), methods (names), bases."""
    classes = []
    try:
        tree = ast.parse(src_text)
    except Exception:
        return classes
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            attrs = []
            methods = []
            for n in node.body:
                if isinstance(n, ast.Assign):
                    # simple single-target assignment
                    for t in n.targets:
                        if isinstance(t, ast.Name):
                            attrs.append(t.id)
                elif isinstance(n, ast.AnnAssign):
                    target = n.target
                    if isinstance(target, ast.Name):
                        attrs.append(target.id)
                elif isinstance(n, ast.FunctionDef):
                    methods.append(n.name)
            bases = []
            for b in node.bases:
                if isinstance(b, ast.Name):
                    bases.append(b.id)
                elif isinstance(b, ast.Attribute):
                    bases.append(b.attr)
            classes.append({"name": node.name, "attrs": attrs, "methods": methods, "bases": bases})
    return classes

# Extraction: Java via javalang (if available)
def extract_java_classes_from_text(src_text: str):
    """Return simple Java class descriptions using javalang if installed."""
    classes = []
    if javalang is None:
        return classes
    try:
        tree = javalang.parse.parse(src_text)
    except Exception:
        return classes
    # traverse type_declarations
    for path, node in tree.filter(javalang.tree.TypeDeclaration):
        if isinstance(node, javalang.tree.ClassDeclaration) or isinstance(node, javalang.tree.InterfaceDeclaration):
            name = node.name
            methods = [m.name for m in node.methods] if getattr(node, "methods", None) else []
            fields = []
            for f in getattr(node, "fields", []) or []:
                for decl in f.declarators:
                    fields.append(decl.name)
            bases = []
            if getattr(node, "extends", None):
                if isinstance(node.extends, list):
                    bases = [ext.name for ext in node.extends]
                else:
                    try:
                        bases = [node.extends.name]
                    except Exception:
                        bases = []
            classes.append({"name": name, "attrs": fields, "methods": methods, "bases": bases})
    return classes

# Mermaid emitter
def classes_to_mermaid_text(classes: List[Dict[str, Any]]) -> str:
    lines = ["classDiagram"]
    for c in classes:
        lines.append(f"    class {c['name']} {{")
        for a in c.get("attrs", []):
            lines.append(f"      +{a}")
        for m in c.get("methods", []):
            lines.append(f"      +{m}()")
        lines.append("    }")
    # inheritance
    for c in classes:
        for b in c.get("bases", []):
            if b:
                lines.append(f"    {b} <|-- {c['name']}")
    return "\n".join(lines)

# Repo processing pipeline
def process_repository(clone_url: str, owner: str, repo_name: str, out_dir: Path, language: str):
    tmpdir = tempfile.mkdtemp(prefix="repo_")
    try:
        # clone shallow
        subprocess.check_call(["git", "clone", "--depth", "1", clone_url, tmpdir], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"Failed to clone {clone_url}: {e}", file=sys.stderr)
        shutil.rmtree(tmpdir, ignore_errors=True)
        return
    project_path = Path(tmpdir)
    results = []
    # Walk files; simple heuristics: analyze .py and .java files
    for p in project_path.rglob("*"):
        if p.is_file():
            try:
                if p.suffix.lower() == ".py" and language.lower() == "python":
                    text = p.read_text(errors="ignore")
                    classes = extract_python_classes_from_text(text)
                    if classes:
                        mermaid = classes_to_mermaid_text(classes)
                        relpath = p.relative_to(project_path)
                        stem = f"{owner}_{repo_name}_{relpath.as_posix().replace('/', '_')}"
                        out_mmd = out_dir / f"{stem}.mmd"
                        out_mmd.write_text(mermaid, encoding="utf-8")
                        # optionally render with mmdc if available
                        try:
                            svg_out = out_dir / f"{stem}.svg"
                            subprocess.run(["mmdc", "-p", "puppeteer-config.json", "-i", str(out_mmd), "-o", str(svg_out)], check=False)
                        except FileNotFoundError:
                            pass
                        results.append({"source": str(relpath), "classes": classes, "mmd": str(out_mmd.name)})
                elif p.suffix.lower() == ".java" and language.lower() == "java":
                    text = p.read_text(errors="ignore")
                    classes = extract_java_classes_from_text(text)
                    if classes:
                        mermaid = classes_to_mermaid_text(classes)
                        relpath = p.relative_to(project_path)
                        stem = f"{owner}_{repo_name}_{relpath.as_posix().replace('/', '_')}"
                        out_mmd = out_dir / f"{stem}.mmd"
                        out_mmd.write_text(mermaid, encoding="utf-8")
                        try:
                            svg_out = out_dir / f"{stem}.svg"
                            subprocess.run(["mmdc", "-i", str(out_mmd), "-o", str(svg_out)], check=False)
                        except FileNotFoundError:
                            pass
                        results.append({"source": str(relpath), "classes": classes, "mmd": str(out_mmd.name)})
            except Exception as e:
                # keep going on file parse errors
                print(f"Error parsing {p}: {e}", file=sys.stderr)
                continue
    shutil.rmtree(tmpdir, ignore_errors=True)
    return results

# Main CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--language", required=True, help="Language (Python or Java)")
    ap.add_argument("--license", required=True, help="License token for search (e.g., mit, apache-2.0)")
    ap.add_argument("--per_page", type=int, default=10, help="Repos per page")
    ap.add_argument("--out", default="dataset_out", help="Output directory")
    args = ap.parse_args()

    language = args.language
    license_query = args.license
    per_page = args.per_page
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    repos = search_repositories(language, license_query, per_page=per_page)
    print(f"Found {len(repos)} repositories from search (language={language}, license={license_query}).")

    manifest = []
    for r in repos:
        full_name = r["full_name"]
        owner, repo_name = full_name.split("/")
        clone_url = r["clone_url"]
        print(f"Checking repo: {full_name} ...")
        license_meta = get_repo_license_spdx(owner, repo_name)
        spdx_id = license_meta.get("spdx_id") or license_meta.get("key") or license_meta.get("name")
        print(f"  Detected license spdx_id={spdx_id}")
        if spdx_id not in PERMISSIVE_SPX:
            print(f"  Skipping {full_name} because license {spdx_id} not in allowed list.")
            continue
        # process & extract
        repo_out = out_root / f"{owner}__{repo_name}"
        repo_out.mkdir(parents=True, exist_ok=True)
        results = process_repository(clone_url, owner, repo_name, repo_out, language)
        if not results:
            print(f"  No classes found for {full_name}.")
            continue
        # save metadata
        metadata = {
            "repo": full_name,
            "clone_url": clone_url,
            "license": license_meta,
            "commit": r.get("default_branch"),
            "results_count": len(results),
            "language": language
        }
        with open(repo_out / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        manifest.append(metadata)
    # write top-level manifest
    with open(out_root / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print("Done. Output at:", out_root)

if __name__ == "__main__":
    main()