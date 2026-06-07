$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
$baseConfig = Join-Path $repoRoot "configs\wb_microjepa_0d_braingraph_worldmod.yaml"
$tmpConfigDir = Join-Path $repoRoot "runs\_sweep_configs"
$referenceRun = Join-Path $repoRoot "runs\wb_microjepa_0d_braingraph_worldmod"
$summaryOut = Join-Path $repoRoot "runs\wb_microjepa_0e_worldmod_sweep_summary.csv"

if (-not (Test-Path $referenceRun)) {
  throw "Missing reference run: $referenceRun"
}

New-Item -ItemType Directory -Force -Path $tmpConfigDir | Out-Null

function Format-StrengthTag([double]$strength) {
  return ('{0:000}' -f [int]([math]::Round($strength * 100, 0)))
}

$strengths = @(0.10, 0.25, 0.75, 1.00)
$runDirs = @($referenceRun)

foreach ($strength in $strengths) {
  $tag = Format-StrengthTag $strength
  $runName = "wb_microjepa_0e_braingraph_worldmod_s$tag"
  $configPath = Join-Path $tmpConfigDir "$runName.yaml"
  $runDir = Join-Path $repoRoot "runs\$runName"

  $cfg = Get-Content $baseConfig -Raw
  $cfg = $cfg -replace 'name: wb_microjepa_0d_braingraph_worldmod', "name: $runName"
  $strengthText = $strength.ToString('0.00', [System.Globalization.CultureInfo]::InvariantCulture)
  $cfg = $cfg -replace 'world_modulation_strength: 0\.5', "world_modulation_strength: $strengthText"
  Set-Content -Path $configPath -Value $cfg -Encoding UTF8

  & $python -m src.wb_microjepa.train --config $configPath
  & $python -m src.wb_microjepa.evaluate $runDir
  & $python -m src.wb_microjepa.report $runDir
  & $python -m src.wb_microjepa.analyze_cases $runDir

  $runDirs += $runDir
}

& $python -m src.wb_microjepa.compare_worldmod_sweep $runDirs --output $summaryOut

Write-Host "Done. Summary saved to $summaryOut"
