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
from collections import deque, defaultdict
import re
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

def safe_mermaid_id(qname: str) -> str:
    # Mermaid identifiers should be simple; keep stable + deterministic
    return re.sub(r"[^A-Za-z0-9_]", "_", qname)

def connected_components(adj: dict[str, set[str]]) -> list[list[str]]:
    seen = set()
    comps = []
    for n in adj.keys():
        if n in seen:
            continue
        q = deque([n])
        seen.add(n)
        comp = []
        while q:
            u = q.popleft()
            comp.append(u)
            for v in adj.get(u, ()):
                if v not in seen:
                    seen.add(v)
                    q.append(v)
        comps.append(comp)
    return comps

def split_component_bfs(comp_nodes: list[str], adj: dict[str, set[str]], max_nodes: int) -> list[list[str]]:
    """Split a large component into BFS chunks capped at max_nodes."""
    comp_set = set(comp_nodes)
    remaining = set(comp_nodes)
    chunks = []

    while remaining:
        seed = next(iter(remaining))
        q = deque([seed])
        chunk = []
        visited_local = {seed}
        remaining.remove(seed)

        while q and len(chunk) < max_nodes:
            u = q.popleft()
            chunk.append(u)
            for v in adj.get(u, ()):
                if v in comp_set and v in remaining and v not in visited_local:
                    visited_local.add(v)
                    remaining.remove(v)
                    q.append(v)

        chunks.append(chunk)

    return chunks

def _collect_type_names(type_node) -> set[str]:
    """Extract bare type names from annotations like User, list[User], Optional[User], pkg.User (best-effort)."""
    names = set()

    def walk(n):
        if n is None:
            return
        # Name: User
        if isinstance(n, ast.Name):
            names.add(n.id)
        # Attribute: pkg.User -> 'User' (we keep only attr to avoid overlinking)
        elif isinstance(n, ast.Attribute):
            names.add(n.attr)
        # Subscript: list[User], Optional[User]
        elif isinstance(n, ast.Subscript):
            walk(n.value)
            walk(n.slice)
        # Tuple: Union[A, B]
        elif isinstance(n, ast.Tuple):
            for e in n.elts:
                walk(e)
        # BinOp in some py versions for | unions
        elif isinstance(n, ast.BinOp):
            walk(n.left); walk(n.right)
        # Constant/Str etc: ignore
        else:
            for child in ast.iter_child_nodes(n):
                walk(child)

    walk(type_node)
    return names

def extract_python_classes(src_text: str, module_qname: str) -> list[dict]:
    """
    Returns list of class dicts:
      {
        "qname": "pkg.mod.Class",
        "label": "Class",
        "attrs": [...],
        "methods": [...],
        "bases": ["Base", ...]  (unqualified names; later we map within module/repo)
        "uses": set(["OtherClass", ...]) (unqualified names)
        "lang": "python",
        "group": module_qname
      }
    """
    out = []
    try:
        tree = ast.parse(src_text)
    except Exception:
        return out

    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue

        attrs, methods, bases, uses = [], [], [], set()

        # bases
        for b in node.bases:
            if isinstance(b, ast.Name):
                bases.append(b.id)
                uses.add(b.id)
            elif isinstance(b, ast.Attribute):
                bases.append(b.attr)
                uses.add(b.attr)

        for n in node.body:
            # class attributes: x = 1 OR x: Type
            if isinstance(n, ast.Assign):
                for t in n.targets:
                    if isinstance(t, ast.Name):
                        attrs.append(t.id)
            elif isinstance(n, ast.AnnAssign):
                if isinstance(n.target, ast.Name):
                    attrs.append(n.target.id)
                # type hint "uses"
                uses |= _collect_type_names(n.annotation)

            # methods: def foo(self, x: User) -> Account
            elif isinstance(n, ast.FunctionDef):
                methods.append(n.name)
                # args annotations
                for a in n.args.args + n.args.kwonlyargs:
                    if a.annotation is not None:
                        uses |= _collect_type_names(a.annotation)
                # return annotation
                if n.returns is not None:
                    uses |= _collect_type_names(n.returns)

        label = node.name
        qname = f"{module_qname}.{label}" if module_qname else label

        out.append({
            "qname": qname,
            "label": label,
            "attrs": attrs,
            "methods": methods,
            "bases": bases,
            "uses": uses,
            "lang": "python",
            "group": module_qname,
        })

    return out

def _java_type_name(t) -> str | None:
    if t is None:
        return None
    # javalang has ReferenceType/BasicType; for UML we care about reference names
    try:
        return t.name
    except Exception:
        return None

def extract_java_classes(src_text: str) -> list[dict]:
    out = []
    if javalang is None:
        return out

    try:
        tree = javalang.parse.parse(src_text)
    except Exception:
        return out

    pkg = ""
    try:
        if tree.package:
            pkg = tree.package.name
    except Exception:
        pkg = ""

    for _, node in tree.filter(javalang.tree.TypeDeclaration):
        if not isinstance(node, (javalang.tree.ClassDeclaration, javalang.tree.InterfaceDeclaration)):
            continue

        label = node.name
        qname = f"{pkg}.{label}" if pkg else label

        attrs, methods, bases, uses = [], [], [], set()

        # extends
        ext = getattr(node, "extends", None)
        if ext is not None:
            # class extends SingleType, interface extends list
            if isinstance(ext, list):
                for e in ext:
                    if getattr(e, "name", None):
                        bases.append(e.name); uses.add(e.name)
            else:
                if getattr(ext, "name", None):
                    bases.append(ext.name); uses.add(ext.name)

        # implements
        impl = getattr(node, "implements", None)
        if impl:
            for i in impl:
                if getattr(i, "name", None):
                    uses.add(i.name)

        # fields
        for f in getattr(node, "fields", []) or []:
            tname = _java_type_name(getattr(f, "type", None))
            if tname:
                uses.add(tname)
            for decl in getattr(f, "declarators", []) or []:
                attrs.append(decl.name)

        # methods
        for m in getattr(node, "methods", []) or []:
            methods.append(m.name)
            rt = _java_type_name(getattr(m, "return_type", None))
            if rt:
                uses.add(rt)
            for p in getattr(m, "parameters", []) or []:
                tn = _java_type_name(getattr(p, "type", None))
                if tn:
                    uses.add(tn)

        out.append({
            "qname": qname,
            "label": label,
            "attrs": attrs,
            "methods": methods,
            "bases": bases,
            "uses": uses,
            "lang": "java",
            "group": pkg,
        })

    return out

def build_class_index(classes: list[dict]) -> tuple[dict[str, dict], dict[str, list[str]]]:
    """
    Returns:
      idx: qname -> class dict
      name_to_qnames: simple label -> list of qnames (for best-effort resolution)
    """
    idx = {}
    name_to_qnames = defaultdict(list)
    for c in classes:
        idx[c["qname"]] = c
        name_to_qnames[c["label"]].append(c["qname"])
    return idx, name_to_qnames

def resolve_unqualified(name: str, current_group: str, name_to_qnames: dict[str, list[str]]) -> str | None:
    """
    Best-effort resolution:
      - If exactly one class has this simple name, use it.
      - Else prefer same 'group' (module for python / package for java) if available.
      - Else give up (avoid incorrect edges).
    """
    cands = name_to_qnames.get(name, [])
    if not cands:
        return None
    if len(cands) == 1:
        return cands[0]
    # prefer same group prefix match
    if current_group:
        for q in cands:
            if q.startswith(current_group + ".") or q == current_group:
                return q
    return None

def build_graph_with_edges(classes: list[dict],
                           add_external_base_stubs: bool = True,
                           add_external_type_stubs: bool = False):
    """
    Returns:
      idx: qname -> class dict (including created stubs)
      adj: undirected adjacency for components/chunking
      edges: list of (src_qname, dst_qname, kind) directed edges
    """
    idx, name_to_qnames = build_class_index(classes)
    adj = {q: set() for q in idx.keys()}
    edges: list[tuple[str, str, str]] = []

    def ensure_stub(qname: str, label: str):
        if qname in idx:
            return
        idx[qname] = {
            "qname": qname,
            "label": label,
            "attrs": [],
            "methods": [],
            "bases": [],
            "uses": set(),
            "lang": "external",
            "group": "external",
        }
        adj[qname] = set()

    for q, c in list(idx.items()):
        group = c.get("group", "")

        # ---- inheritance edges ----
        for b in c.get("bases", []) or []:
            qb = resolve_unqualified(b, group, name_to_qnames)
            if qb and qb in adj:
                # base -> child
                edges.append((qb, q, "extends"))
                adj[q].add(qb); adj[qb].add(q)
            elif add_external_base_stubs:
                stub = f"ext:{b}"
                ensure_stub(stub, b)
                edges.append((stub, q, "extends"))
                adj[q].add(stub); adj[stub].add(q)

        # ---- uses edges (type hints / Java types) ----
        for u in (c.get("uses", set()) or set()):
            qu = resolve_unqualified(u, group, name_to_qnames)
            if qu and qu in adj and qu != q:
                edges.append((q, qu, "uses"))
                adj[q].add(qu); adj[qu].add(q)
            elif add_external_type_stubs:
                stub = f"ext:{u}"
                ensure_stub(stub, u)
                edges.append((q, stub, "uses"))
                adj[q].add(stub); adj[stub].add(q)

    return idx, adj, edges

def is_external_stub(qname: str) -> bool:
    return qname.startswith("ext:")

def chunk_stats(chunk_nodes: list[str], edges: list[tuple[str, str, str]]):
    chunk_set = set(chunk_nodes)
    real_nodes = [n for n in chunk_nodes if not is_external_stub(n)]
    real_set = set(real_nodes)

    # count edges that connect two real nodes
    real_real_edges = 0
    for src, dst, kind in edges:
        if src in real_set and dst in real_set:
            if src in chunk_set and dst in chunk_set:
                real_real_edges += 1

    return {
        "total_nodes": len(chunk_nodes),
        "real_nodes": len(real_nodes),
        "real_real_edges": real_real_edges,
    }

def emit_mermaid_chunk(chunk_qnames: list[str],
                       idx: dict[str, dict],
                       edges: list[tuple[str, str, str]],
                       detail_mode: str = "auto",
                       max_uses_edges: int = 120) -> str:
    """
    Emits nodes + edges from the same edge list used for grouping.
    - Always prints extends edges.
    - Prints uses edges only in "full" mode and caps them to avoid hairballs.
    - External stubs render as name-only nodes.
    """
    n = len(chunk_qnames)
    if detail_mode == "auto":
        mode = "names" if n > 55 else "full"
    else:
        mode = detail_mode

    chunk_set = set(chunk_qnames)
    lines = ["classDiagram"]

    # nodes
    for q in chunk_qnames:
        c = idx[q]
        cid = safe_mermaid_id(q)
        label = c["label"]
        is_stub = q.startswith("ext:")

        if mode == "names" or is_stub:
            lines.append(f'    class {cid}["{label}"]')
        else:
            lines.append(f'    class {cid}["{label}"] {{')
            for a in c.get("attrs", []):
                lines.append(f"      +{a}")
            for m in c.get("methods", []):
                lines.append(f"      +{m}()")
            lines.append("    }")

    # ---- EDGE FILTERING ----

    # Collect inheritance pairs (unordered for quick lookup)
    inheritance_pairs = set()
    for src, dst, kind in edges:
        if kind == "extends" and src in chunk_set and dst in chunk_set:
            inheritance_pairs.add((src, dst))
            inheritance_pairs.add((dst, src))  # allow reverse lookup

    uses_count = 0

    for src, dst, kind in edges:
        if src not in chunk_set or dst not in chunk_set:
            continue

        s = safe_mermaid_id(src)
        d = safe_mermaid_id(dst)

        if kind == "extends":
            lines.append(f"    {s} <|-- {d}")

        elif kind == "uses":
            # Skip if inheritance exists between same pair
            if (src, dst) in inheritance_pairs:
                continue

            # Optional: also skip if reverse inheritance exists
            if (dst, src) in inheritance_pairs:
                continue

            # Only include uses in full mode
            if mode == "full" and uses_count < max_uses_edges:
                lines.append(f"    {s} --> {d}")
                uses_count += 1

    return "\n".join(lines)

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
def process_repository_merged(clone_dir: Path, owner: str, repo_name: str, out_dir: Path,
                              include_python=True, include_java=True,
                              max_classes_per_diagram=80,
                              detail_mode="auto",
                              render_svgs=True,
                              puppeteer_cfg_path: str | None = None):
    """
    clone_dir: already cloned repository path
    out_dir: output folder for this repo
    """
    classes = []
    # ---- Collect classes across repo ----
    for p in clone_dir.rglob("*"):
        if not p.is_file():
            continue

        # skip typical vendor/virtualenv/build folders to reduce noise
        parts = set(p.parts)
        if any(x in parts for x in (".git", "__pycache__", ".venv", "venv", "node_modules", "dist", "build", "target")):
            continue

        try:
            if include_python and p.suffix.lower() == ".py":
                # module qname from relative path, e.g. pkg/sub/mod.py -> pkg.sub.mod
                rel = p.relative_to(clone_dir)
                module_qname = ".".join(rel.with_suffix("").parts)
                text = p.read_text(errors="ignore")
                classes.extend(extract_python_classes(text, module_qname))

            elif include_java and p.suffix.lower() == ".java":
                text = p.read_text(errors="ignore")
                classes.extend(extract_java_classes(text))
        except Exception:
            continue
    
    if not classes:
        return {"diagrams": 0, "classes": 0}

    # ---- Dedupe by qname ----
    idx = {}
    for c in classes:
        q = c["qname"]
        if q not in idx:
            idx[q] = c
        else:
            # merge members if duplicates show up
            idx[q]["attrs"] = list(set(idx[q].get("attrs", [])) | set(c.get("attrs", [])))
            idx[q]["methods"] = list(set(idx[q].get("methods", [])) | set(c.get("methods", [])))
            idx[q]["bases"] = list(set(idx[q].get("bases", [])) | set(c.get("bases", [])))
            idx[q]["uses"] = set(idx[q].get("uses", set())) | set(c.get("uses", set()))

    classes = list(idx.values())
    idx, _ = build_class_index(classes)

    # ---- Build graph (with external stubs) + components + chunks ----
    idx, adj, edges = build_graph_with_edges(
        classes,
        add_external_base_stubs=True,
        add_external_type_stubs=False  # keep False to avoid huge noisy graphs
    )
    comps = connected_components(adj)

    diagrams_written = 0
    for comp_i, comp_nodes in enumerate(sorted(comps, key=len, reverse=True), start=1):
        chunks = split_component_bfs(comp_nodes, adj, max_nodes=max_classes_per_diagram)
        for chunk_i, chunk_nodes in enumerate(chunks, start=1):
            mmd_text = emit_mermaid_chunk(chunk_nodes, idx, edges, detail_mode=detail_mode, max_uses_edges=120)
            stem = f"{owner}__{repo_name}__comp{comp_i:04d}__chunk{chunk_i:04d}"

            stats = chunk_stats(chunk_nodes, edges)
            # Skip chunks with only 1 real class
            if stats["real_nodes"] <= 1:
                continue

            mmd_path = out_dir / f"{stem}.mmd"
            mmd_path.write_text(mmd_text, encoding="utf-8")
            diagrams_written += 1

            if render_svgs:
                svg_path = out_dir / f"{stem}.svg"
                cmd = ["mmdc", "-i", str(mmd_path), "-o", str(svg_path), "-b", "white", "-s", "2", "--theme", "default"]
                if puppeteer_cfg_path:
                    cmd.extend(["-p", puppeteer_cfg_path])
                try:
                    subprocess.run(cmd, check=False, capture_output=True)
                except FileNotFoundError:
                    # mmdc not installed; skip rendering
                    pass

    return {"diagrams": diagrams_written, "classes": len(idx)}

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

        # clone
        tmpdir = tempfile.mkdtemp(prefix="repo_")
        try:
            # clone shallow
            subprocess.check_call(["git", "clone", "--depth", "1", clone_url, tmpdir], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f"Failed to clone {clone_url}: {e}", file=sys.stderr)
            shutil.rmtree(tmpdir, ignore_errors=True)
            return
        project_path = Path(tmpdir)


        # Check for puppeteer-config.json - needed for SVG rendering
        if not Path("puppeteer-config.json").is_file():
            print("Missing puppeteer-config.json! Make sure it's in the same directory as this script.")
            return

        results = process_repository_merged(project_path, owner, repo_name, repo_out, language, puppeteer_cfg_path='puppeteer-config.json')
        if not results:
            print(f"  No classes found for {full_name}.")
            continue
        # Save Metadata
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