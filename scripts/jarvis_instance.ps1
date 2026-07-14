param(
    [ValidateSet("gpus", "create", "list", "get", "ssh", "destroy", "pause", "resume")]
    [string]$Action = "gpus",
    [string]$MachineId = "",
    [int]$DiskGb = 200,
    [string]$GpuName = "RTX-PRO6000",
    [int]$NumGpus = 1,
    [string]$Region = "",
    [string]$Template = "pytorch",
    [string]$Label = "hobbylm-1b-jarvis-shakedown"
)

$ErrorActionPreference = "Stop"
$keyName = "JL" + "_API_KEY"
$apiKey = [Environment]::GetEnvironmentVariable($keyName)

if (-not $apiKey) {
    throw "Jarvis Labs API key is not set. Set `$env:JL_API_KEY in the shell environment (or load it from .env); do not write it to repo files."
}

$jl = (Get-Command jl -ErrorAction SilentlyContinue).Source
if (-not $jl) {
    throw "jl CLI not found. Install with: uv tool install jarvislabs (or: python -m pip install jarvislabs)"
}

# jl reads JL_API_KEY from the environment automatically (CLI arg > JL_API_KEY > config file).
# It is already set in $env: for this process via the check above, so no --token flag is needed.

switch ($Action) {
    "gpus" {
        # Confirm the exact GPU identifier string before creating an instance.
        # Docs list this GPU as "RTX PRO 6000 Blackwell" / "RTX 6000 Pro" but don't
        # publish the --gpu flag value anywhere; read it off this listing.
        & $jl gpus --json
    }
    "create" {
        $createArgs = @("create", "--gpu", $GpuName, "--storage", $DiskGb, "--num-gpus", $NumGpus,
                        "--template", $Template, "--name", $Label, "--json")
        if ($Region) { $createArgs += @("--region", $Region) }
        & $jl @createArgs
    }
    "list" {
        & $jl list --json
    }
    "get" {
        if (-not $MachineId) { throw "Pass -MachineId." }
        & $jl get $MachineId --json
    }
    "ssh" {
        if (-not $MachineId) { throw "Pass -MachineId." }
        & $jl ssh $MachineId
    }
    "pause" {
        if (-not $MachineId) { throw "Pass -MachineId." }
        & $jl pause $MachineId --yes
    }
    "resume" {
        if (-not $MachineId) { throw "Pass -MachineId." }
        & $jl resume $MachineId --yes
    }
    "destroy" {
        if (-not $MachineId) { throw "Pass -MachineId." }
        & $jl destroy $MachineId --yes
    }
}
