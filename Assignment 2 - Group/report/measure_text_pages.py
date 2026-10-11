"""Recompile the report without material excluded from the four-page limit.

Run with the project's Python environment:
    python measure_text_pages.py

Requires latexmk, pdfinfo and Google Fire. Generated files stay in the
gitignored .text-length directory. The submission source is never changed.
"""

from pathlib import Path
import re
import subprocess

import fire


def remove_comments(source: str) -> str:
    return re.sub(r"(?<!\\)%[^\n]*", "", source)


def remove_plot_commands(source: str) -> str:
    """Remove resultplot's four balanced arguments, including nested braces."""
    pattern: re.Pattern[str] = re.compile(r"\\resultplot\b")
    while match := pattern.search(source):
        end: int = match.end()
        for argument in range(4):
            while end < len(source) and source[end].isspace():
                end += 1
            if end == len(source) or source[end] != "{":
                raise ValueError(f"Missing resultplot argument {argument + 1}")
            depth: int = 1
            end += 1
            while depth:
                if end == len(source):
                    raise ValueError("Unbalanced resultplot argument")
                if source[end] == "\\":
                    end += 2
                    continue
                if source[end] == "{":
                    depth += 1
                elif source[end] == "}":
                    depth -= 1
                end += 1
        source = source[:match.start()] + source[end:]
    return source


def prose_only(source: str) -> str:
    source = remove_comments(source)
    for environment in ("figure", "figure*", "table", "table*", "longtable", "thebibliography"):
        name: str = re.escape(environment)
        pattern: str = rf"\\begin\{{{name}\}}.*?\\end\{{{name}\}}"
        source = re.sub(pattern, "", source, flags=re.DOTALL)
    source = remove_plot_commands(source)
    source = re.sub(r"\\bibliography(?:style)?\{[^}]*\}", "", source)
    source = re.sub(r"\\(?:clearpage|newpage|onecolumn|twocolumn)\b", "", source)
    # Freeze original reference numbers, without creating new labels here.
    source = re.sub(r"(?m)^[ \t]*\\label\{[^}]*\}[ \t]*\n", "", source)
    return re.sub(r"\\label\{[^}]*\}", "", source)


def compile_report(report: Path, source: Path, output: Path) -> None:
    command: list[str] = ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error"]
    if output != report:
        command.append(f"-outdir={output}")
    command.append(str(source))
    result: subprocess.CompletedProcess[str] = subprocess.run(
        command, cwd=report, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    log: Path = report / ".text-length" / f"{source.stem}-build.log"
    log.write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f"Compilation failed; see {log}")


def measure(source: str = "main.tex") -> None:
    """Count main text and all prose, including appendix prose, separately."""
    path: Path = Path(source).resolve()
    report: Path = path.parent
    output: Path = report / ".text-length"
    output.mkdir(exist_ok=True)
    (output / ".gitignore").write_text("*\n")
    compile_report(report, path, report)
    original: str = remove_comments(path.read_text())
    preamble: str
    document: str
    preamble, document = original.split(r"\begin{document}", 1)
    document = document.split(r"\end{document}", 1)[0]
    if r"\input{" in document or r"\include{" in document:
        raise ValueError("Inline external TeX sources before measuring this document")
    aux: str = (report / f"{path.stem}.aux").read_text()
    references: str = "\n".join(
        line for line in aux.splitlines()
        if line.startswith((r"\newlabel{", r"\bibcite{"))
    )
    variants: dict[str, str] = {
        "main-text": document.split(r"\bibliographystyle", 1)[0],
        "all-text": document,
    }
    summary: list[str] = [
        "Text-only page measurement, unchanged ISBI font, spacing and margins.",
        "Excludes figures, tables, their captions, and references.",
        "Includes title, abstract, equations, headings and contribution statements.",
        "Main-text excludes appendices; all-text includes appendix prose and headings.",
        "This is a recompilation check, not an official ruling on the course policy.",
        "",
    ]
    for name, body in variants.items():
        tex: Path = output / f"{name}.tex"
        tex.write_text(
            preamble + "\n\\makeatletter\n" + references + "\n\\makeatother\n"
            + "\\begin{document}\n" + prose_only(body) + "\n\\end{document}\n"
        )
        compile_report(report, tex, output)
        log: str = tex.with_suffix(".log").read_text()
        if "undefined" in log.lower():
            raise RuntimeError(f"Unresolved references could change the count; inspect {tex.with_suffix('.log')}")
        info: subprocess.CompletedProcess[str] = subprocess.run(
            ["pdfinfo", str(tex.with_suffix(".pdf"))], check=True,
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        pages: re.Match[str] | None = re.search(r"(?m)^Pages:\s+(\d+)", info.stdout)
        if pages is None:
            raise RuntimeError("pdfinfo did not report a page count")
        count: int = int(pages.group(1))
        summary.append(f"{name}: {count} pages; {'within' if count <= 4 else 'over'} the four-page limit.")
        summary.append(f"Preview: {tex.with_suffix('.pdf')}")
    text: str = "\n".join(summary) + "\n"
    (output / "summary.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    fire.Fire(measure)
