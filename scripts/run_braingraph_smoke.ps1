$ErrorActionPreference = "Stop"

python -m src.wb_microjepa.train --config configs/wb_microjepa_0b_braingraph_smoke.yaml
python -m src.wb_microjepa.report runs/wb_microjepa_0b_braingraph_smoke
python -m src.wb_microjepa.inspect_run runs/wb_microjepa_0b_braingraph_smoke

Write-Host "Done. Open runs/wb_microjepa_0b_braingraph_smoke/report.html"
