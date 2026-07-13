param(
    [ValidateSet("search", "create", "ssh-url", "ssh", "destroy")]
    [string]$Action = "search",
    [int]$OfferId = 0,
    [int]$InstanceId = 0,
    [int]$DiskGb = 500,
    [int]$GpuCount = 4,
    [string]$GpuName = "H100_SXM",
    [double]$MinReliability = 0.995,
    [int]$MinDurationDays = 30,
    [string]$Image = "vastai/pytorch",
    [string]$Label = "hobbylm-1b-shakedown",
    [string]$OnstartCmd = "python -m pip install --upgrade pip && python -m pip install huggingface-hub tqdm tiktoken numpy"
)

$ErrorActionPreference = "Stop"
$keyName = "VAST" + "_API_KEY"
$apiKey = [Environment]::GetEnvironmentVariable($keyName)

if (-not $apiKey) {
    throw "Vast API key is not set. Set it in the shell environment; do not write it to repo files."
}

$vastai = (Get-Command vastai -ErrorAction SilentlyContinue).Source
if (-not $vastai) {
    $vastai = Join-Path $env:LOCALAPPDATA "Programs\Python\Python313\Scripts\vastai.exe"
}
if (-not (Test-Path $vastai)) {
    throw "vastai CLI not found. Install with: python -m pip install vastai"
}

# Keep Vast CLI config/cache out of the repo and out of the user's home directory.
$env:XDG_CONFIG_HOME = Join-Path $env:TEMP "vastai-codex-config"
$env:XDG_CACHE_HOME = Join-Path $env:TEMP "vastai-codex-cache"

$query = "datacenter=True verified=True rentable=True reliability>=$MinReliability num_gpus=$GpuCount gpu_name=$GpuName disk_space>=$DiskGb duration>=$MinDurationDays"

switch ($Action) {
    "search" {
        & $vastai --api-key $apiKey --raw --full search offers $query --type on-demand --storage $DiskGb --order "dph,total_flops-"
    }
    "create" {
        if ($OfferId -le 0) { throw "Pass -OfferId from the search output." }
        $createJson = & $vastai --api-key $apiKey --raw create instance $OfferId --image $Image --disk $DiskGb --ssh --direct --label $Label --onstart-cmd $OnstartCmd --cancel-unavail
        try {
            $created = $createJson | ConvertFrom-Json
            [pscustomobject]@{
                success = $created.success
                new_contract = $created.new_contract
            } | ConvertTo-Json
        }
        catch {
            $createJson
        }
    }
    "ssh-url" {
        if ($InstanceId -le 0) { throw "Pass -InstanceId." }
        & $vastai --api-key $apiKey ssh-url $InstanceId
    }
    "ssh" {
        if ($InstanceId -le 0) { throw "Pass -InstanceId." }
        $url = (& $vastai --api-key $apiKey ssh-url $InstanceId).Trim()
        ssh $url
    }
    "destroy" {
        if ($InstanceId -le 0) { throw "Pass -InstanceId." }
        & $vastai --api-key $apiKey --raw destroy instance $InstanceId --yes
    }
}
