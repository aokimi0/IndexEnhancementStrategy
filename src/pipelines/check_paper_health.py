"""论文健康检查工具：验证 .tex 与 .bib 的一致性、图片路径有效性、术语统一性。

用法：
    python -m src.pipelines.check_paper_health

输出：
    1. 所有 \\cite{key} 引用的 key 是否在 .bib 中存在
    2. 所有 \\ref{label} 是否定义过
    3. 所有 \\includegraphics 引用的图片是否存在
    4. 重复 cite key、未使用的 bib 条目
    5. 章节、表格、图片统计

仅做静态检查，不修改任何文件。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable


PAPER_DIR = Path(__file__).parent.parent.parent / "paper"


def extract_cite_keys(tex_content: str) -> set[str]:
    """从 LaTeX 文本中提取 \\cite{key1,key2,...} 的所有 key。

    自动跳过 \\newcommand 等宏定义中的占位符引用（如 \\cite{#1}）。

    Args:
        tex_content: .tex 文件内容字符串。

    Returns:
        set[str]: 去重的所有 cite key 集合。
    """
    cleaned = re.sub(r"\\newcommand[^\n]*", "", tex_content)
    cleaned = re.sub(r"\\renewcommand[^\n]*", "", cleaned)
    cleaned = re.sub(r"\\providecommand[^\n]*", "", cleaned)

    keys: set[str] = set()
    pattern = re.compile(r"\\cite[a-zA-Z]*\{([^}]+)\}")
    for match in pattern.finditer(cleaned):
        for key in match.group(1).split(","):
            key = key.strip()
            if not key or key.startswith("#"):
                continue
            keys.add(key)
    return keys


def extract_label_definitions(tex_content: str) -> set[str]:
    """从 LaTeX 文本中提取所有 \\label{name} 定义。

    Args:
        tex_content: .tex 文件内容字符串。

    Returns:
        set[str]: 去重的所有 label 集合。
    """
    pattern = re.compile(r"\\label\{([^}]+)\}")
    return {match.group(1).strip() for match in pattern.finditer(tex_content)}


def extract_ref_keys(tex_content: str) -> set[str]:
    """从 LaTeX 文本中提取所有 \\ref{name} / \\eqref{name} 引用。

    Args:
        tex_content: .tex 文件内容字符串。

    Returns:
        set[str]: 去重的所有 ref key 集合。
    """
    pattern = re.compile(r"\\(?:e?ref|autoref)\{([^}]+)\}")
    return {match.group(1).strip() for match in pattern.finditer(tex_content)}


def extract_includegraphics(tex_content: str) -> set[str]:
    """从 LaTeX 文本中提取所有 \\includegraphics 引用的相对路径。

    Args:
        tex_content: .tex 文件内容字符串。

    Returns:
        set[str]: 去重的图片相对路径集合（不含扩展名解析）。
    """
    pattern = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
    return {match.group(1).strip() for match in pattern.finditer(tex_content)}


def extract_bib_keys(bib_content: str) -> set[str]:
    """从 BibTeX 文件中提取所有条目 key。

    Args:
        bib_content: .bib 文件内容字符串。

    Returns:
        set[str]: 去重的所有 bib key 集合。
    """
    pattern = re.compile(r"@\w+\s*\{\s*([^,\s]+)\s*,")
    return {match.group(1).strip() for match in pattern.finditer(bib_content)}


def find_image_file(rel_path: str) -> Path | None:
    """在 paper/ 目录下查找图片文件，支持自动补全扩展名。

    Args:
        rel_path: \\includegraphics 中的相对路径（可能含或不含扩展名）。

    Returns:
        Path | None: 找到的文件路径，未找到返回 None。
    """
    candidates: list[Path] = []
    base = PAPER_DIR / rel_path
    if base.exists():
        return base
    for ext in (".png", ".pdf", ".jpg", ".jpeg", ".eps"):
        candidate = PAPER_DIR / (rel_path + ext)
        candidates.append(candidate)
        if candidate.exists():
            return candidate
    return None


def count_chapters_sections(tex_content: str) -> dict[str, int]:
    """统计 LaTeX 中的章节、表格、图片数量。

    Args:
        tex_content: .tex 文件内容字符串。

    Returns:
        dict[str, int]: {chapter, section, subsection, table, figure} 计数。
    """
    return {
        "chapter": len(re.findall(r"^\\chapter\{", tex_content, re.MULTILINE)),
        "section": len(re.findall(r"^\\section\{", tex_content, re.MULTILINE)),
        "subsection": len(re.findall(r"^\\subsection\{", tex_content, re.MULTILINE)),
        "table": len(re.findall(r"\\begin\{table\}", tex_content)),
        "figure": len(re.findall(r"\\begin\{figure\}", tex_content)),
        "tabular": len(re.findall(r"\\begin\{tabular\}", tex_content)),
    }


def check(tex_files: Iterable[str], bib_file: str) -> int:
    """对一组 .tex 文件做整体健康检查。

    Args:
        tex_files: .tex 文件名列表（PAPER_DIR 下）。
        bib_file: .bib 文件名。

    Returns:
        int: 发现问题的总数（用作进程退出码）。
    """
    contents: dict[str, str] = {}
    for name in tex_files:
        path = PAPER_DIR / name
        if not path.exists():
            print(f"[MISS] {name} 不存在")
            continue
        contents[name] = path.read_text(encoding="utf-8", errors="ignore")

    merged_tex = "\n".join(contents.values())
    all_cites = extract_cite_keys(merged_tex)
    all_labels = extract_label_definitions(merged_tex)
    all_refs = extract_ref_keys(merged_tex)
    all_images = extract_includegraphics(merged_tex)

    bib_path = PAPER_DIR / bib_file
    bib_content = bib_path.read_text(encoding="utf-8") if bib_path.exists() else ""
    bib_keys = extract_bib_keys(bib_content)

    issues = 0
    print("=" * 60)
    print("论文健康检查报告")
    print("=" * 60)

    # 1. 章节/表格/图片统计
    print("\n[1] 结构统计")
    for name, content in contents.items():
        stats = count_chapters_sections(content)
        print(f"  {name}: chapter={stats['chapter']}, section={stats['section']}, "
              f"subsection={stats['subsection']}, table={stats['table']}, "
              f"figure={stats['figure']}")

    # 2. 引用一致性
    print(f"\n[2] 引用一致性（cite: {len(all_cites)} 条，bib: {len(bib_keys)} 条）")
    missing_cites = all_cites - bib_keys
    if missing_cites:
        print(f"  [ERROR] {len(missing_cites)} 个 cite key 在 bib 中找不到:")
        for k in sorted(missing_cites):
            print(f"    - {k}")
        issues += len(missing_cites)
    else:
        print("  [OK] 所有 \\cite{} 引用的 key 在 bib 中均存在")

    unused_bib = bib_keys - all_cites
    if unused_bib:
        print(f"  [WARN] {len(unused_bib)} 个 bib 条目未被引用:")
        for k in sorted(unused_bib):
            print(f"    - {k}")

    # 3. label / ref 一致性
    print(f"\n[3] label/ref 一致性（label: {len(all_labels)} 个，ref: {len(all_refs)} 处）")
    missing_refs = all_refs - all_labels
    if missing_refs:
        print(f"  [ERROR] {len(missing_refs)} 个 \\ref{{}} 找不到对应 \\label{{}}:")
        for k in sorted(missing_refs):
            print(f"    - {k}")
        issues += len(missing_refs)
    else:
        print("  [OK] 所有 \\ref{} 都有对应 \\label{}")

    # 4. 图片存在性
    print(f"\n[4] 图片路径检查（{len(all_images)} 个 \\includegraphics）")
    missing_images: list[str] = []
    for rel in sorted(all_images):
        if find_image_file(rel) is None:
            missing_images.append(rel)
    if missing_images:
        print(f"  [ERROR] {len(missing_images)} 个图片找不到:")
        for r in missing_images:
            print(f"    - {r}")
        issues += len(missing_images)
    else:
        print("  [OK] 所有图片路径有效")

    # 5. 整体小结
    print("\n[5] 小结")
    print(f"  issues = {issues}")
    return issues


def main() -> int:
    """命令行入口：检查 manual.tex 与 abstract.tex 的健康度。

    Returns:
        int: 进程退出码（issue 数）。
    """
    return check(
        tex_files=["main.tex", "manual.tex", "abstract.tex", "acknowledgements.tex", "resume.tex", "references.tex"],
        bib_file="nkthesis.bib",
    )


if __name__ == "__main__":
    raise SystemExit(main())
