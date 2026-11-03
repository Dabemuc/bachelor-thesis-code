# Tensoren (*.npz) aus results/ vom Windows-PC per scp auf den Mac kopieren.
# Code und meta.json/CSV kommen weiter per git; *.npz sind gitignored (zu gross fuer den Kern).
#
#   powershell -ExecutionPolicy Bypass -File scripts\push_results_to_mac.ps1            # alle fehlenden
#   powershell -ExecutionPolicy Bypass -File scripts\push_results_to_mac.ps1 results\<lauf> [...]
#
# Kopiert nur *.npz, die auf dem Mac noch fehlen (gleicher relativer Pfad unter results/).
# Bereits vorhandene Dateien werden NICHT ueberschrieben (Rohdaten werden nie veraendert).
# Voraussetzung: Mac -> Systemeinstellungen -> Teilen -> Entfernte Anmeldung (ssh) an.
# Mac-Ziel per Umgebungsvariable aenderbar: $env:MAC_HOST, $env:MAC_DIR
#
# Nur ASCII in dieser Datei: Windows PowerShell 5.1 liest UTF-8 ohne BOM als ANSI.

param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Runs)

$ErrorActionPreference = "Stop"
$MacHost = if ($env:MAC_HOST) { $env:MAC_HOST } else { "dbenner@roosevelt.local" }
$MacDir  = if ($env:MAC_DIR)  { $env:MAC_DIR }  else { "hm-inf/sem7/Bachelorarbeit/bachelor-thesis-code" }  # relativ zu ~
$Root    = Split-Path -Parent $PSScriptRoot
$Results = Join-Path $Root "results"

# Lauf-Ordner bestimmen
if ($Runs -and $Runs.Count -gt 0) {
    $dirs = foreach ($r in $Runs) {
        $p = if ([System.IO.Path]::IsPathRooted($r)) { $r } else { Join-Path $Root $r }
        if (-not (Test-Path $p -PathType Container)) { throw "Kein Ordner: $p" }
        Get-Item $p
    }
} else {
    $dirs = Get-ChildItem $Results -Directory | Where-Object { Get-ChildItem $_.FullName -Filter *.npz -File }
}

# Lokale Dateien: results/<lauf>/<datei>.npz (relativ, mit /)
$local = foreach ($d in $dirs) {
    Get-ChildItem $d.FullName -Filter *.npz -File | ForEach-Object { "$($d.Name)/$($_.Name)" }
}
if (-not $local) { Write-Host "Keine *.npz gefunden."; exit 0 }

# Was liegt schon auf dem Mac? (ein ssh-Aufruf; find statt Glob, zsh bricht bei 0 Treffern ab)
$remote = @(ssh $MacHost "cd ~/$MacDir/results 2>/dev/null && find . -name '*.npz' | sed 's|^\./||'")
if ($LASTEXITCODE -ne 0) { throw "ssh $MacHost fehlgeschlagen (Exit $LASTEXITCODE)" }
$have = @{}; foreach ($f in $remote) { $have[$f.Trim()] = $true }

$todo = @($local | Where-Object { -not $have.ContainsKey($_) })
Write-Host ("{0} *.npz lokal, {1} schon auf dem Mac, {2} zu kopieren -> {3}:~/{4}/results/" -f `
    $local.Count, ($local.Count - $todo.Count), $todo.Count, $MacHost, $MacDir)
if ($todo.Count -eq 0) { exit 0 }

# Zielordner in einem ssh-Aufruf anlegen, dann je Lauf ein scp
$runDirs = $todo | ForEach-Object { $_.Split("/")[0] } | Sort-Object -Unique
$mk = ($runDirs | ForEach-Object { "'results/$_'" }) -join " "
ssh $MacHost "cd ~/$MacDir && mkdir -p $mk"
if ($LASTEXITCODE -ne 0) { throw "mkdir auf dem Mac fehlgeschlagen" }

foreach ($rd in $runDirs) {
    $files = $todo | Where-Object { $_.Split("/")[0] -eq $rd } | ForEach-Object { Join-Path $Results $_ }
    Write-Host "-> $rd ($($files.Count) Datei(en))"
    scp -q @files "${MacHost}:~/$MacDir/results/$rd/"
    if ($LASTEXITCODE -ne 0) { throw "scp fuer $rd fehlgeschlagen" }
}
Write-Host "fertig."
