"""The user guide keeps up with the code.

    python tests/test_docs.py

A setting nobody documented is a setting nobody uses correctly, and a guide
that names a command or file the code no longer has sends people down a dead
end. These checks fail the moment the two drift.
"""
import os
import re
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from snnsearch.runconfig import DEFAULTS
from snnsearch.cli import build_parser
from snnsearch.hardware import _LIMIT_KEYS

with open(os.path.join(ROOT, "GUIDE.md"), encoding="utf-8") as _fh:
    GUIDE = _fh.read()


def section(title):
    """The text of one '### title' block of the config reference."""
    m = re.search(rf"^### {title}\n(.*?)(?=^### |^## |\Z)", GUIDE, re.S | re.M)
    return m.group(1) if m else ""


class GuideCoversTheCode(unittest.TestCase):
    def test_every_config_setting_is_in_the_reference(self):
        for sect, keys in DEFAULTS.items():
            if not isinstance(keys, dict) or sect in ("architecture", "train", "limits"):
                continue
            body = section(sect)
            self.assertTrue(body, f"no '### {sect}' section in GUIDE.md")
            for key in keys:
                with self.subTest(setting=f"{sect}.{key}"):
                    self.assertIn(f"`{key}`", body)
        for key in _LIMIT_KEYS:
            self.assertIn(f"`{key}`", section("limits"), key)

    def test_every_command_is_described(self):
        sub = next(a for a in build_parser()._actions if a.dest == "mode")
        for mode in sub.choices:
            with self.subTest(mode=mode):
                self.assertIn(f"main.py {mode}", GUIDE)

    def test_every_script_and_file_it_names_exists(self):
        for path in set(re.findall(r"(?:tools|tests|configs|examples|snnsearch)/[\w/]+\.(?:py|yaml|js|md)", GUIDE)):
            with self.subTest(path=path):
                self.assertTrue(os.path.isfile(os.path.join(ROOT, path)), path)

    def test_every_test_file_is_listed_for_developers(self):
        listed = set(re.findall(r"tests/(\w+\.(?:py|js))", GUIDE))
        for name in os.listdir(os.path.join(ROOT, "tests")):
            if re.match(r"(test_\w+|smoke_test)\.(py|js)$", name):
                with self.subTest(test=name):
                    self.assertIn(name, listed)

    def test_readme_points_to_the_guide(self):
        with open(os.path.join(ROOT, "README.md"), encoding="utf-8") as fh:
            readme = fh.read()
        self.assertIn("GUIDE.md", readme)


if __name__ == "__main__":
    unittest.main(verbosity=2)
