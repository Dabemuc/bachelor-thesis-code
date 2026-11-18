# Lauf-Ergebnisse aus results/ vom Windows-PC auf den Mac bringen - als EIN tar-Archiv.
# Committet wird auf dem Mac, nicht auf dem PC (sonst kollidieren git und Kopie auf dem Mac).
#
# Direkt (PC -> Mac per ssh; 3 Verbindungen insgesamt):
#   powershell -ExecutionPolicy Bypass -File scripts\push_results_to_mac.ps1
# Ueber den Pi (wenn der Mac vom PC aus nicht erreichbar ist, 18.11.2026):
#   1. Mac:  scripts/relay_from_pi.sh manifest      (Liste der Dateien auf dem Mac -> Pi)
#   2. PC:   powershell -ExecutionPolicy Bypass -File scripts\push_results_to_mac.ps1 -Via lee.local
#   3. Mac:  scripts/relay_from_pi.sh pull          (Archiv vom Pi holen, auspacken, auf dem Pi loeschen)
# Optionen: -NpzOnly (nur *.npz), Lauf-Ordner als Argumente (sonst alle mit *.npz).
#
# Uebertragen werden nur Dateien, die auf dem Mac fehlen; vorhandene werden nie ueberschrieben
# (tar -k). Emulator-Ordner erst mit consistency.npz (zuletzt geschrieben); Ordner mit Schreib-
# zugriff in den letzten 5 Minuten werden ausgelassen.
# Mac-Ziel: $env:MAC_HOST, $env:MAC_DIR. Nur ASCII: Windows PowerShell 5.1 liest UTF-8 ohne BOM als ANSI.
#
# Warum ein Archiv: Bis 18.11. ein scp je Ordner (dutzende Verbindungen). Danach nahm der Mac vom PC
# keine ssh-Verbindungen mehr an (Port 22 Timeout, vom Pi weiter erreichbar) - vermutlich eine
# Sicherheitssoftware, die viele Verbindungen als Angriff wertet (Vermutung).

param(
    [switch]$NpzOnly,
    [string]$Via = "",
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$Runs
)

$ErrorActionPreference = "Stop"
$MacHost = if ($env:MAC_HOST) { $env:MAC_HOST } else { "dbenner@roosevelt.local" }
$MacDir  = if ($env:MAC_DIR)  { $env:MAC_DIR }  else { "hm-inf/sem7/Bachelorarbeit/bachelor-thesis-code" }  # relativ zu ~
$Root    = Split-Path -Parent $PSScriptRoot
$Results = Join-Path $Root "results"

# --- Lauf-Ordner bestimmen ---------------------------------------------------------------------
if ($Runs -and $Runs.Count -gt 0) {
    $dirs = foreach ($r in $Runs) {
        $p = if ([System.IO.Path]::IsPathRooted($r)) { $r } else { Join-Path $Root $r }
        if (-not (Test-Path $p -PathType Container)) { throw "Kein Ordner: $p" }
        Get-Item $p
    }
} else {
    $dirs = Get-ChildItem $Results -Directory | Where-Object { Get-ChildItem $_.FullName -Filter *.npz -File }
}
$Pattern = if ($NpzOnly) { "*.npz" } else { "*" }
$cutoff = (Get-Date).AddMinutes(-5)
$dirs = @($dirs | Where-Object {
    if ($_.Name -match '_emu_' -and -not (Test-Path (Join-Path $_.FullName "consistency.npz")) `
        -and (Get-ChildItem $_.FullName -File | Where-Object { $_.LastWriteTime -gt (Get-Date "2026-11-17 18:00") })) {
        Write-Host "uebersprungen (Emulator-Block noch nicht fertig): $($_.Name)"; return $false
    }
    $recent = Get-ChildItem $_.FullName -File | Where-Object { $_.LastWriteTime -gt $cutoff }
    if ($recent) { Write-Host "uebersprungen (wird noch geschrieben): $($_.Name)"; $false } else { $true }
})
$local = foreach ($d in $dirs) {
    Get-ChildItem $d.FullName -Filter $Pattern -File | ForEach-Object { "$($d.Name)/$($_.Name)" }
}
if (-not $local) { Write-Host "Keine Dateien gefunden."; exit 0 }

# --- Was liegt schon auf dem Mac? --------------------------------------------------------------
if ($Via) {
    $man = Join-Path $env:TEMP "mac_manifest.txt"
    scp -q "${Via}:~/relay/mac_manifest.txt" $man
    if ($LASTEXITCODE -ne 0) { throw "Manifest fehlt auf $Via - zuerst auf dem Mac: scripts/relay_from_pi.sh manifest" }
    $remote = Get-Content $man
} else {
    $remote = @(ssh $MacHost "cd ~/$MacDir/results 2>/dev/null && find . -type f | sed 's|^\./||'")
    if ($LASTEXITCODE -ne 0) { throw "ssh $MacHost fehlgeschlagen (Exit $LASTEXITCODE)" }
}
$have = @{}; foreach ($f in $remote) { $have[$f.Trim()] = $true }
$todo = @($local | Where-Object { -not $have.ContainsKey($_) })
$bytes = ($todo | ForEach-Object { (Get-Item (Join-Path $Results $_)).Length } | Measure-Object -Sum).Sum
Write-Host ("{0} Dateien lokal, {1} schon auf dem Mac, {2} zu uebertragen ({3:N2} GB) in {4} Ordnern" -f `
    $local.Count, ($local.Count - $todo.Count), $todo.Count, ($bytes / 1GB), (($todo | ForEach-Object { $_.Split("/")[0] } | Sort-Object -Unique).Count))
if ($todo.Count -eq 0) { exit 0 }

# --- Ein Archiv bauen ---------------------------------------------------------------------------
$list = Join-Path $env:TEMP "results_relay_list.txt"
$tarPath = Join-Path $env:TEMP "results_relay.tar"
$todo | ForEach-Object { "results/$_" } | Set-Content -Encoding ascii $list
if (Test-Path $tarPath) { Remove-Item $tarPath }
Push-Location $Root
try { tar -cf $tarPath -T $list; if ($LASTEXITCODE -ne 0) { throw "tar fehlgeschlagen" } } finally { Pop-Location }
Write-Host ("Archiv: {0:N2} GB" -f ((Get-Item $tarPath).Length / 1GB))

# --- Uebertragen ------------------------------------------------------------------------------
if ($Via) {
    ssh $Via "mkdir -p ~/relay && df -h ~ | tail -1"
    if ($LASTEXITCODE -ne 0) { throw "ssh $Via fehlgeschlagen" }
    scp $tarPath "${Via}:~/relay/results_relay.tar"
    if ($LASTEXITCODE -ne 0) { throw "scp nach $Via fehlgeschlagen (Archiv bleibt in $tarPath)" }
    Write-Host "Liegt auf $Via in ~/relay/. Jetzt auf dem Mac: scripts/relay_from_pi.sh pull"
} else {
    scp $tarPath "${MacHost}:~/$MacDir/results_relay.tar"
    if ($LASTEXITCODE -ne 0) { throw "scp fehlgeschlagen (Archiv bleibt in $tarPath)" }
    ssh $MacHost "cd ~/$MacDir && tar -xkf results_relay.tar && rm results_relay.tar"
    if ($LASTEXITCODE -ne 0) { throw "Auspacken auf dem Mac fehlgeschlagen" }
    Write-Host "Auf dem Mac ausgepackt."
}
Remove-Item $tarPath
Write-Host "fertig."
