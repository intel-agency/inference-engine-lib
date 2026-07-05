#!/usr/bin/env pwsh

param(
    [Parameter(Mandatory = $false)]
    [string]$Version = "1.1.0-local.2",

    [Parameter(Mandatory = $false)]
    [string]$Source = "http://localhost:5555/v3/index.json",

    [Parameter(Mandatory = $false)]
    [string]$ApiKey = "local"
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$Project = Join-Path $Root "InferenceEngine.Core" "InferenceEngine.Core.csproj"
$OutDir = Join-Path $Root "InferenceEngine.Core" "bin" "Debug"
$Package = Join-Path $OutDir "InferenceEngine.Core.$Version.nupkg"

Write-Host "Packing InferenceEngine.Core v$Version..." -ForegroundColor Cyan
dotnet pack $Project -p:Version=$Version 
if ($LASTEXITCODE -ne 0) { throw "Pack failed." }

if (-not (Test-Path $Package)) {
    throw "Package not found: $Package"
}

Write-Host "Pushing to $Source ..." -ForegroundColor Cyan
dotnet nuget push $Package -s $Source -k $ApiKey --allow-insecure-connections
if ($LASTEXITCODE -ne 0) { throw "Push failed." }

Write-Host "Done. InferenceEngine.Core.$Version published to local BaGet." -ForegroundColor Green
