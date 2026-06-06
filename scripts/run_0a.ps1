$ErrorActionPreference = "Stop"

python -m src.wb_microjepa.train --config configs/wb_microjepa_0a.yaml
python -m src.wb_microjepa.evaluate runs/wb_microjepa_0a
python -m src.wb_microjepa.report runs/wb_microjepa_0a
python -m src.wb_microjepa.inspect_run runs/wb_microjepa_0a

Write-Host "Done. Open runs/wb_microjepa_0a/report.html"
