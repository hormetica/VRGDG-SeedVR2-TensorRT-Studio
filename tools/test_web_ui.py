"""Dependency-free markup contracts for the Studio's two browser workflows.

Run with: python -m unittest tools.test_web_ui
"""

from __future__ import annotations

import re
import unittest
from collections import Counter
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path


WEB = Path(__file__).resolve().parents[1] / "web"
VOID_ELEMENTS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}


@dataclass
class Element:
    tag: str
    attrs: dict[str, str | None]
    ancestors: tuple[Element, ...] = field(repr=False)


class Markup(HTMLParser):
    def __init__(self, text: str):
        super().__init__()
        self.elements: list[Element] = []
        self.stack: list[Element] = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        element = Element(tag, dict(attrs), tuple(self.stack))
        self.elements.append(element)
        if tag not in VOID_ELEMENTS:
            self.stack.append(element)

    def handle_endtag(self, tag):
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index].tag == tag:
                del self.stack[index:]
                break


class WebUiContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.markup = Markup((WEB / "index.html").read_text(encoding="utf-8"))
        cls.by_id = {element.attrs["id"]: element for element in cls.markup.elements if "id" in element.attrs}

    def test_ids_are_unique(self):
        counts = Counter(element.attrs["id"] for element in self.markup.elements if "id" in element.attrs)
        self.assertEqual([key for key, count in counts.items() if count > 1], [])

    def test_static_javascript_targets_exist(self):
        for filename, pattern in (("app.js", r"\$\(['\"]#([\w-]+)['\"]\)"), ("images.js", r"\$\(['\"]([\w-]+)['\"]\)")):
            source = (WEB / filename).read_text(encoding="utf-8")
            for target in re.findall(pattern, source):
                with self.subTest(file=filename, target=target):
                    # The updater creates its own dialog and controls at runtime.
                    if target in {"apply-update", "update-message", "update-version"}:
                        continue
                    self.assertIn(target, self.by_id)

    def test_form_controls_have_accessible_labels(self):
        labeled_ids = {element.attrs["for"] for element in self.markup.elements if element.tag == "label" and "for" in element.attrs}
        for element in self.markup.elements:
            if element.tag not in {"input", "select", "progress"}:
                continue
            with self.subTest(control=element.attrs.get("id")):
                self.assertTrue(
                    element.attrs.get("id") in labeled_ids
                    or element.attrs.get("aria-label")
                    or element.attrs.get("aria-labelledby")
                    or any(parent.tag == "label" for parent in element.ancestors)
                )

    def test_workflows_have_independent_controls_and_shared_exit(self):
        for workflow in ("video", "image"):
            workspace = self.by_id[f"{workflow}-workspace"]
            self.assertEqual(workspace.tag, "main")
            self.assertIn("aria-label", workspace.attrs)
        self.assertIn("hidden", self.by_id["image-workspace"].attrs)
        self.assertNotIn("hidden", self.by_id["video-workspace"].attrs)
        self.assertFalse(any(parent.tag == "main" for parent in self.by_id["exit-studio"].ancestors))

    def test_essential_settings_are_visible_and_advanced_settings_are_collapsed(self):
        for target in ("output-preset", "crop-policy", "backend", "model", "image-scale", "image-model"):
            with self.subTest(control=target):
                self.assertFalse(any(parent.tag == "details" for parent in self.by_id[target].ancestors))
        for target in ("batch-size", "seed", "color", "attention", "chunked-render", "skin-finishing", "image-seed"):
            with self.subTest(control=target):
                self.assertTrue(any(parent.tag == "details" and "open" not in parent.attrs for parent in self.by_id[target].ancestors))

    def test_modes_expose_their_selected_state(self):
        for attribute in ("data-mode", "data-image-mode"):
            buttons = [element for element in self.markup.elements if attribute in element.attrs]
            self.assertEqual({element.attrs[attribute] for element in buttons}, {"original", "restored", "compare", "side"})
            self.assertEqual(sum(element.attrs.get("aria-pressed") == "true" for element in buttons), 1)

    def test_empty_workspaces_do_not_offer_unavailable_render_actions(self):
        for target in ("render-preview", "render-full", "image-preview", "image-start"):
            with self.subTest(control=target):
                self.assertIn("disabled", self.by_id[target].attrs)
        for target in ("render-progress", "image-progress", "image-pagination"):
            with self.subTest(control=target):
                self.assertIn("hidden", self.by_id[target].attrs)


if __name__ == "__main__":
    unittest.main()
