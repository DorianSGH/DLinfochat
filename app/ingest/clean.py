"""Turn MediaWiki parser HTML into sectioned plain text.

The rules are generic (block vs inline elements, tables, headings), plus a short list of
deadlock.wiki-specific selectors for page chrome that would otherwise pollute search.
"""

import re
from dataclasses import dataclass

from bs4 import BeautifulSoup, Comment, NavigableString, Tag

# Widgets and chrome with nothing useful for answering questions.
DROP_SELECTORS = ", ".join([
    "style", "script", "link", "noscript", "audio", "video", "figure", "button",
    ".mw-editsection", ".toc", ".reference", ".references", ".mw-references-wrap",
    ".navbox", ".page-tabs", ".heronav-grid", ".gallery", ".model-placeholder",
    ".sb-gate",          # Street Brawl item-draft grid: hundreds of bare item names
    ".tabber__header",   # tab buttons; each panel gets its label as a line instead
    ".hide", "[style*='display:none']", "[style*='display: none']",
])
# Whole sections to skip. Update history would surface outdated stat values.
DROP_SECTIONS = {"update history", "3d model", "3d models", "gallery", "references",
                 "navigation", "see also"}
# Containers that read best as a single line, e.g. a stat value next to its label.
ONE_LINE = ".ability-headerattr, .ability-mainbox, .ability-altbox, .ability-upgrade"
# Icons whose alt text is the only label for a number (e.g. the cooldown clock icon).
LABEL_ICONS = ".ability-headerattr img[alt]"
# Inline elements that are visually separate items (interaction tags under an ability).
AS_BLOCK = ".ability-hint"
# UI hints that leak into text.
NOISE_LINES = {"Click to reveal more info.", "Show All"}

BLOCK_TAGS = {
    "address", "article", "aside", "blockquote", "center", "dd", "details", "div", "dl",
    "dt", "fieldset", "figcaption", "footer", "form", "header", "hr", "li", "main", "nav",
    "ol", "p", "pre", "section", "summary", "table", "tbody", "td", "tfoot", "th", "thead",
    "tr", "ul",
}
HEADING_LEVELS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}
# Marks a heading inside the flat text so it can be split into sections afterwards.
# \x00 can't appear in wiki text, so it can't collide with content.
HEADING_MARK = "\x00"
WHITESPACE = re.compile(r"\s+")


@dataclass
class Section:
    heading: str  # path like "Abilities > Siphon Life"; "" for the intro
    text: str


def html_to_sections(html: str) -> list[Section]:
    soup = BeautifulSoup(html, "lxml")
    root = soup.select_one(".mw-parser-output") or soup.body or soup
    _preprocess(soup, root)
    text = _render(root, inline=False)
    return _split_sections(text)


def _preprocess(soup: BeautifulSoup, root: Tag) -> None:
    for comment in root.find_all(string=lambda s: isinstance(s, Comment)):
        comment.extract()

    # Label tabber panels before their tab buttons are removed; drop "Show All" panels,
    # which just repeat the other tabs.
    for panel in root.select(".tabber__panel"):
        label_el = root.find(id=panel.get("aria-labelledby", ""))
        label = label_el.get_text(strip=True) if label_el else ""
        if label == "Show All":
            panel.decompose()
        elif label:
            label_p = soup.new_tag("p")
            label_p.string = f"{label}:"
            panel.insert(0, label_p)

    for el in root.select(DROP_SELECTORS):
        el.decompose()

    for img in root.select(LABEL_ICONS):
        img.replace_with(f"{img['alt']} ")

    # "<icon> 1" -> "Upgrade (1 AP):", so upgrade tiers stay readable.
    for header in root.select(".ability-upgrade-header"):
        header.replace_with(f"Upgrade ({header.get_text(strip=True)} AP): ")

    for el in root.select(ONE_LINE):
        el["data-one-line"] = "1"

    for el in root.select(AS_BLOCK):
        el.name = "div"


def _render(node: Tag, inline: bool) -> str:
    """Render to text. Block elements become line breaks, unless `inline`, where they
    become spaces. Inline elements are concatenated as-is, so "8" + "m" stays "8m"."""
    parts: list[str] = []
    for child in node.children:
        if isinstance(child, NavigableString):
            # HTML treats any whitespace run (newlines included) as one space; source
            # formatting newlines must not become line breaks.
            parts.append(WHITESPACE.sub(" ", str(child)))
            continue
        if not isinstance(child, Tag):
            continue
        name = child.name
        if name in HEADING_LEVELS and not inline:
            title = _clean_line(child.get_text(" "))
            parts.append(f"\n{HEADING_MARK}{HEADING_LEVELS[name]}{HEADING_MARK}{title}\n")
        elif name == "br":
            parts.append(" " if inline else "\n")
        elif name == "tr" and not inline and not child.find("table"):
            cells = [_clean_line(_render(c, inline=True)) for c in child.find_all(["th", "td"], recursive=False)]
            parts.append("\n" + " | ".join(c for c in cells if c) + "\n")
        elif name in BLOCK_TAGS or name in HEADING_LEVELS:
            if inline:
                parts.append(" " + _render(child, inline=True) + " ")
            elif child.get("data-one-line"):
                parts.append("\n" + _render(child, inline=True) + "\n")
            else:
                parts.append("\n" + ("- " if name == "li" else ""))
                parts.append(_render(child, inline=False))
                parts.append("\n")
        elif name == "img":
            continue
        else:
            parts.append(_render(child, inline))
    return "".join(parts)


def _clean_line(line: str) -> str:
    line = " ".join(line.split())
    # Infobox rows come out as "Health | : | 800" or "Health: | 800"; make them "Health: 800".
    line = re.sub(r"\s*\|\s*:\s*\|\s*", ": ", line)
    line = re.sub(r":\s*\|\s*", ": ", line)
    # Per-boon scaling sits in its own span right after the value: "800+49" -> "800 +49".
    return re.sub(r"(?<=[\w%.])\+(?=\d)", " +", line)


def _split_sections(text: str) -> list[Section]:
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []  # (level, title) of the current heading path
    lines: list[str] = []
    skip_below: int | None = None  # level of a dropped section we're inside

    def flush() -> None:
        if skip_below is None:
            body = _tidy(lines, stack[-1][1] if stack else "")
            if body:
                sections.append(Section(" > ".join(t for _, t in stack), body))
        lines.clear()

    for raw in text.split("\n"):
        if raw.startswith(HEADING_MARK):
            _, level_str, title = raw.split(HEADING_MARK, 2)
            level = int(level_str)
            flush()
            if skip_below is not None and level <= skip_below:
                skip_below = None
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            if skip_below is None and title.lower() in DROP_SECTIONS:
                skip_below = level
        else:
            lines.append(raw)
    flush()
    return sections


def _tidy(lines: list[str], heading: str) -> str:
    out: list[str] = []
    for raw in lines:
        line = _clean_line(raw)
        # Skip empties, UI hints, and lines with no letters/digits (decorative quote marks).
        if not line or line in NOISE_LINES or not any(ch.isalnum() for ch in line):
            continue
        if out and out[-1] == line:  # repeated line, e.g. a card title under its heading
            continue
        if not out and line == heading:
            continue
        out.append(line)
    return "\n".join(out)
