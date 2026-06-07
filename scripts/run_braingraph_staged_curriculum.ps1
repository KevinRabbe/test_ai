$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
$config = Join-Path $repoRoot "configs\wb_microjepa_0j_braingraph_staged_curriculum.yaml"
$runDir = Join-Path $repoRoot "runs\wb_microjepa_0j_braingraph_staged_curriculum"
$summaryOut = Join-Path $repoRoot "runs\wb_microjepa_0j_braingraph_staged_curriculum_compare.csv"

& $python -m src.wb_microjepa.train --config $config
& $python -m src.wb_microjepa.evaluate $runDir
& $python -m src.wb_microjepa.report $runDir
& $python -m src.wb_microjepa.analyze_cases $runDir

& $python -m src.wb_microjepa.compare_frontier `
  $runDir `
  (Join-Path $repoRoot "runs\wb_microjepa_0c_braingraph_delta") `
  (Join-Path $repoRoot "runs\wb_microjepa_0e_braingraph_worldmod_s025") `
  (Join-Path $repoRoot "runs\wb_microjepa_0h_braingraph_action_basis") `
  (Join-Path $repoRoot "runs\wb_microjepa_0i_braingraph_range_curriculum") `
  --output $summaryOut

Write-Host "Done. Summary saved to $summaryOut"
