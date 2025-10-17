$translationsDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if ((Get-Location).Path -ne $translationsDir) {
    Set-Location $translationsDir
}

Get-ChildItem -File -Filter "*.po" | ForEach-Object {
    $poFile = $_.Name
    $lang = $poFile -replace "\.po$"

    $langDir = "$lang\LC_MESSAGES"
    if (-not (Test-Path -Path $langDir)) {
        New-Item -ItemType Directory -Path $langDir -Force
    }

    Write-Host "Compiling language '$lang' .po file into .mo file"
    & msgfmt $poFile -o "$lang\LC_MESSAGES\lada.mo"
}
