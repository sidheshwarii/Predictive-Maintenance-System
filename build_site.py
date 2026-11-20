"""Inlines model/industrial_model.json into website/template.html -> pms-website.html (one self-contained file)."""
import json
t = open("website/template.html", encoding="utf-8").read()
m = open("model/industrial_model.json").read()
open("pms-website.html", "w", encoding="utf-8").write(t.replace("/*__INDUSTRIAL_MODEL__*/null", m))
print("built pms-website.html")
