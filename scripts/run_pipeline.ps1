$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$UserJavaHome = [Environment]::GetEnvironmentVariable("JAVA_HOME", "User")
if ($UserJavaHome) {
    $env:JAVA_HOME = $UserJavaHome
    $env:PATH = "$env:JAVA_HOME\bin;$env:PATH"
}

$LocalHadoop = Join-Path $ProjectRoot ".tools\hadoop"
if (Test-Path (Join-Path $LocalHadoop "bin\winutils.exe")) {
    $env:HADOOP_HOME = $LocalHadoop
    $env:PATH = "$env:HADOOP_HOME\bin;$env:PATH"
}

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Missing virtual environment: $Python"
}
& $Python -m src.pipeline @args

