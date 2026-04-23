param(
    [switch]$RunAll,
    [int]$Step = 0,
    [switch]$DryRun = $false,
    [string]$RepoRoot = ".",
    [string]$MainBranch = "codex/stressid-stepwise-fix",
    [string]$RemoteHost = "aspire2a.nus",
    [string]$RemoteScratch = "/home/users/nus/e1553307/scratch/smartstress_jobs",
    [double]$AccuracyDropTolerance = 0.01
)

$ErrorActionPreference = "Stop"

function Invoke-Cmd {
    param(
        [Parameter(Mandatory = $true)][string]$Command,
        [string]$WorkingDirectory = $null
    )
    if ($WorkingDirectory) {
        Push-Location $WorkingDirectory
    }
    try {
        Write-Host ">> $Command"
        Invoke-Expression $Command
        if ($LASTEXITCODE -ne 0) {
            throw "Command failed with exit code ${LASTEXITCODE}: $Command"
        }
    } finally {
        if ($WorkingDirectory) {
            Pop-Location
        }
    }
}

function Read-JsonFile {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (-not (Test-Path $Path)) {
        return $null
    }
    return Get-Content $Path -Raw | ConvertFrom-Json
}

function Write-JsonFile {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)]$Object
    )
    $dir = Split-Path -Parent $Path
    if (-not [string]::IsNullOrEmpty($dir)) {
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
    }
    $Object | ConvertTo-Json -Depth 50 | Set-Content -Path $Path -Encoding UTF8
}

function Build-PbsFromTemplate {
    param(
        [Parameter(Mandatory = $true)][string]$TemplatePath,
        [Parameter(Mandatory = $true)][string]$OutputPath,
        [Parameter(Mandatory = $true)][hashtable]$Replacements
    )
    $content = Get-Content $TemplatePath -Raw
    foreach ($k in $Replacements.Keys) {
        $content = $content.Replace($k, [string]$Replacements[$k])
    }
    $content | Set-Content -Path $OutputPath -Encoding Ascii
}

function Wait-RemoteJob {
    param(
        [Parameter(Mandatory = $true)][string]$RemoteHost,
        [Parameter(Mandatory = $true)][string]$JobId
    )
    while ($true) {
        Start-Sleep -Seconds 20
        $cmd = "qstat -xf $JobId 2>/dev/null | awk -F= '/job_state/{gsub(/ /,"""",$2); print $2; exit}'"
        $state = (& ssh $RemoteHost $cmd).Trim()
        Write-Host "Job $JobId state: $state"
        if ($state -eq "F") {
            break
        }
        if ($state -eq "") {
            Write-Host "State unavailable yet, continue polling..."
        }
    }
}

function Get-StepDefinitions {
    return @(
        @{
            id = 1
            name = "domain_coral"
            modelsRel = "remote_bundle_gpu_20260410/models"
            resultsRel = "remote_bundle_gpu_20260410/results/cross_val_results.json"
            sourceStats = "source_stats_all.json"
            config = @{
                step_name = "step1_domain_coral"
                threshold = 0.5
                enable_coral = $true
                enable_robust_norm = $false
                enable_temp_scaling = $false
                enable_threshold_search = $false
                enable_quality_filter = $false
                robust_clip = 8.0
                quality_z_threshold = 6.0
            }
        },
        @{
            id = 2
            name = "semantic_negative_tighten"
            modelsRel = "remote_bundle_gpu_20260410/models"
            resultsRel = "remote_bundle_gpu_20260410/results/cross_val_results.json"
            sourceStats = "source_stats_neutral_stress.json"
            config = @{
                step_name = "step2_semantic_negative_tighten"
                threshold = 0.5
                enable_coral = $true
                enable_robust_norm = $false
                enable_temp_scaling = $false
                enable_threshold_search = $false
                enable_quality_filter = $false
                robust_clip = 8.0
                quality_z_threshold = 6.0
            }
        },
        @{
            id = 3
            name = "robust_baseline_norm"
            modelsRel = "remote_bundle_gpu_20260410/models"
            resultsRel = "remote_bundle_gpu_20260410/results/cross_val_results.json"
            sourceStats = "source_stats_neutral_stress.json"
            config = @{
                step_name = "step3_robust_baseline_norm"
                threshold = 0.5
                enable_coral = $true
                enable_robust_norm = $true
                enable_temp_scaling = $false
                enable_threshold_search = $false
                enable_quality_filter = $false
                robust_clip = 6.0
                quality_z_threshold = 6.0
            }
        },
        @{
            id = 4
            name = "calibration_threshold"
            modelsRel = "remote_bundle_gpu_20260410/models"
            resultsRel = "remote_bundle_gpu_20260410/results/cross_val_results.json"
            sourceStats = "source_stats_neutral_stress.json"
            config = @{
                step_name = "step4_calibration_threshold"
                threshold = 0.5
                enable_coral = $true
                enable_robust_norm = $true
                enable_temp_scaling = $true
                enable_threshold_search = $true
                enable_quality_filter = $false
                robust_clip = 6.0
                quality_z_threshold = 6.0
            }
        },
        @{
            id = 5
            name = "sampling_protocol_100hz"
            modelsRel = "Models_CrossVal_100Hz"
            resultsRel = "Results_CrossVal_100Hz/cross_val_results.json"
            sourceStats = "source_stats_neutral_stress.json"
            config = @{
                step_name = "step5_sampling_protocol_100hz"
                threshold = 0.5
                enable_coral = $true
                enable_robust_norm = $true
                enable_temp_scaling = $true
                enable_threshold_search = $true
                enable_quality_filter = $false
                robust_clip = 6.0
                quality_z_threshold = 6.0
            }
        },
        @{
            id = 6
            name = "feature_quality_filter"
            modelsRel = "Models_CrossVal_100Hz"
            resultsRel = "Results_CrossVal_100Hz/cross_val_results.json"
            sourceStats = "source_stats_neutral_stress.json"
            config = @{
                step_name = "step6_feature_quality_filter"
                threshold = 0.5
                enable_coral = $true
                enable_robust_norm = $true
                enable_temp_scaling = $true
                enable_threshold_search = $true
                enable_quality_filter = $true
                robust_clip = 6.0
                quality_z_threshold = 5.5
            }
        }
    )
}

function Ensure-ChampionMetrics {
    param(
        [Parameter(Mandatory = $true)][string]$ChampionPath,
        [Parameter(Mandatory = $true)][string]$BaselinePath
    )
    if (Test-Path $ChampionPath) {
        return
    }
    $base = Read-JsonFile -Path $BaselinePath
    if ($null -eq $base) {
        throw "Baseline metrics file not found: $BaselinePath"
    }
    $obj = @{
        source = $BaselinePath
        accuracy = [double]$base.average_metrics.accuracy
        precision = [double]$base.average_metrics.precision
        recall = [double]$base.average_metrics.recall
        f1 = [double]$base.average_metrics.f1
    }
    Write-JsonFile -Path $ChampionPath -Object $obj
}

function Invoke-Step {
    param(
        [Parameter(Mandatory = $true)]$StepDef,
        [Parameter(Mandatory = $true)][string]$RepoRoot,
        [Parameter(Mandatory = $true)][string]$MainBranch,
        [Parameter(Mandatory = $true)][string]$RemoteHost,
        [Parameter(Mandatory = $true)][string]$RemoteScratch,
        [Parameter(Mandatory = $true)][double]$AccuracyDropTolerance,
        [switch]$DryRun
    )
    $stepId = [int]$StepDef.id
    $stepName = [string]$StepDef.name
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $runtimeRoot = Join-Path $RepoRoot "Results_StressID_Compare/stepwise_runtime"
    $stepDir = Join-Path $RepoRoot ("Results_StressID_Compare/step_{0}_{1}" -f $stepId, $stepName)
    New-Item -ItemType Directory -Force -Path $runtimeRoot, $stepDir | Out-Null

    $championPath = Join-Path $runtimeRoot "champion_metrics.json"
    $baselinePath = Join-Path $RepoRoot "Results_StressID_Compare/stressid_crossval_eval.json"
    Ensure-ChampionMetrics -ChampionPath $championPath -BaselinePath $baselinePath
    $champion = Read-JsonFile -Path $championPath

    # step backup tag
    $tagName = "backup/step{0}/before" -f $stepId
    $tagExists = "$(& git -C $RepoRoot tag --list $tagName)".Trim()
    if (-not $tagExists) {
        Invoke-Cmd -Command "git -C `"$RepoRoot`" tag $tagName"
    }

    # trial branch
    $trialBranch = "trial/step{0}-{1}" -f $stepId, $stepName
    $trialExists = "$(& git -C $RepoRoot branch --list $trialBranch)".Trim()
    if ($trialExists) {
        Invoke-Cmd -Command "git -C `"$RepoRoot`" branch -D $trialBranch"
    }
    Invoke-Cmd -Command "git -C `"$RepoRoot`" checkout $MainBranch"
    Invoke-Cmd -Command "git -C `"$RepoRoot`" checkout -b $trialBranch"

    # commit step config on trial branch
    $cfgDir = Join-Path $RepoRoot "scripts/step_configs"
    New-Item -ItemType Directory -Force -Path $cfgDir | Out-Null
    $cfgPath = Join-Path $cfgDir "active_config.json"
    $cfgArchivePath = Join-Path $cfgDir ("step_{0}_{1}.json" -f $stepId, $stepName)
    Write-JsonFile -Path $cfgPath -Object $StepDef.config
    Write-JsonFile -Path $cfgArchivePath -Object $StepDef.config
    Invoke-Cmd -Command "git -C `"$RepoRoot`" add scripts/step_configs/active_config.json scripts/step_configs/step_${stepId}_${stepName}.json"
    Invoke-Cmd -Command "git -C `"$RepoRoot`" commit -m `"step${stepId}: candidate ${stepName}`""

    # prepare runtime assets
    $prepScript = Join-Path $RepoRoot "scripts/prepare_stepwise_assets.py"
    Invoke-Cmd -Command "python `"$prepScript`" --repo-root `"$RepoRoot`" --runtime-dir `"$runtimeRoot`""

    # build local bundle
    $bundleRoot = Join-Path $runtimeRoot ("bundle_step{0}_{1}_{2}" -f $stepId, $stepName, $timestamp)
    $bundleDir = Join-Path $bundleRoot "remote_bundle_gpu_20260410"
    $bundleData = Join-Path $bundleDir "data"
    $bundleModels = Join-Path $bundleDir "models"
    $bundleResults = Join-Path $bundleDir "results"
    $bundleConfig = Join-Path $bundleDir "config"
    New-Item -ItemType Directory -Force -Path $bundleData, $bundleModels, $bundleResults, $bundleConfig | Out-Null

    Copy-Item -Path (Join-Path $RepoRoot "remote_bundle_gpu_20260410/evaluate_stressid_stepwise.py") -Destination (Join-Path $bundleDir "evaluate_stressid_stepwise.py") -Force
    Copy-Item -Path (Join-Path $runtimeRoot "data/STRESSID_TEST.json") -Destination (Join-Path $bundleData "STRESSID_TEST.json") -Force
    Copy-Item -Path (Join-Path $runtimeRoot "data/STRESSID_CALIB.json") -Destination (Join-Path $bundleData "STRESSID_CALIB.json") -Force
    Copy-Item -Path (Join-Path $runtimeRoot ("config/{0}" -f $StepDef.sourceStats)) -Destination (Join-Path $bundleConfig "source_stats.json") -Force
    Copy-Item -Path $cfgPath -Destination (Join-Path $bundleConfig "step_config.json") -Force

    Copy-Item -Path (Join-Path $RepoRoot $StepDef.resultsRel) -Destination (Join-Path $bundleResults "cross_val_results.json") -Force
    $modelsPath = Join-Path $RepoRoot $StepDef.modelsRel
    Copy-Item -Path (Join-Path $modelsPath "*") -Destination $bundleModels -Recurse -Force

    $pbsTemplate = Join-Path $RepoRoot "remote_bundle_gpu_20260410/run_gpu_container.template.pbs"
    $outputJsonName = "stressid_eval_step${stepId}.json"
    $remoteRunRoot = "$RemoteScratch/step${stepId}_${stepName}_$timestamp"
    Build-PbsFromTemplate -TemplatePath $pbsTemplate -OutputPath (Join-Path $bundleDir "run_gpu_container.pbs") -Replacements @{
        "__JOB_NAME__" = "stressid_step${stepId}"
        "__PBS_OUT_PATH__" = "$remoteRunRoot/pbs.out"
        "__RUN_ROOT__" = $remoteRunRoot
        "__OUTPUT_JSON__" = $outputJsonName
        "__STEP_NAME__" = "step${stepId}_${stepName}"
        "__THRESHOLD__" = "0.5"
    }

    $jobId = "DRYRUN"
    if (-not $DryRun) {
        Invoke-Cmd -Command "ssh $RemoteHost `"mkdir -p $remoteRunRoot`""
        $bundleTar = Join-Path $runtimeRoot ("bundle_step{0}_{1}_{2}.tar.gz" -f $stepId, $stepName, $timestamp)
        Invoke-Cmd -Command "tar -C `"$bundleRoot`" -czf `"$bundleTar`" remote_bundle_gpu_20260410"
        Invoke-Cmd -Command "scp `"$bundleTar`" ${RemoteHost}:$remoteRunRoot/remote_bundle_gpu_20260410.tar.gz"
        Invoke-Cmd -Command "ssh $RemoteHost `"cd $remoteRunRoot && tar -xzf remote_bundle_gpu_20260410.tar.gz`""
        $submitOutRaw = & ssh $RemoteHost "cd $remoteRunRoot/remote_bundle_gpu_20260410 && qsub run_gpu_container.pbs"
        $submitOut = "$submitOutRaw".Trim()
        if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($submitOut)) {
            throw "Failed to submit qsub job for step $stepId"
        }
        $jobId = $submitOut.Split(" ")[0]
        Wait-RemoteJob -RemoteHost $RemoteHost -JobId $jobId

        $remoteEval = "$remoteRunRoot/remote_bundle_gpu_20260410/results/$outputJsonName"
        $remotePbsOut = "$remoteRunRoot/pbs.out"
        Copy-Item -Path (Join-Path $bundleDir "run_gpu_container.pbs") -Destination (Join-Path $stepDir "run_gpu_container.pbs") -Force
        Invoke-Cmd -Command "scp ${RemoteHost}:$remoteEval `"$stepDir/eval_report.json`""
        Invoke-Cmd -Command "scp ${RemoteHost}:$remotePbsOut `"$stepDir/pbs.out`""
    } else {
        Copy-Item -Path (Join-Path $bundleDir "run_gpu_container.pbs") -Destination (Join-Path $stepDir "run_gpu_container.pbs") -Force
    }

    # decision
    $candidateAcc = $champion.accuracy
    $candidateF1 = $champion.f1
    $candidateMetrics = @{
        accuracy = $candidateAcc
        precision = $champion.precision
        recall = $champion.recall
        f1 = $candidateF1
    }
    if (-not $DryRun) {
        $report = Read-JsonFile -Path (Join-Path $stepDir "eval_report.json")
        $candidateMetrics = @{
            accuracy = [double]$report.average_metrics.accuracy
            precision = [double]$report.average_metrics.precision
            recall = [double]$report.average_metrics.recall
            f1 = [double]$report.average_metrics.f1
        }
        $candidateAcc = $candidateMetrics.accuracy
        $candidateF1 = $candidateMetrics.f1
    }

    $pass = ($candidateF1 -gt [double]$champion.f1) -and ($candidateAcc -ge ([double]$champion.accuracy - $AccuracyDropTolerance))
    $decision = @{
        step_id = $stepId
        step_name = $stepName
        timestamp = $timestamp
        dry_run = [bool]$DryRun
        remote_host = $RemoteHost
        remote_run_root = $remoteRunRoot
        job_id = $jobId
        rule = @{
            f1_must_improve = $true
            max_accuracy_drop = $AccuracyDropTolerance
        }
        champion_before = $champion
        candidate_metrics = $candidateMetrics
        accepted = [bool]$pass
    }
    Write-JsonFile -Path (Join-Path $stepDir "decision.json") -Object $decision

    $worklogPath = Join-Path $RepoRoot ".codex/WORKLOG.md"
    $statusLine = if ($pass) { "ACCEPT" } else { "REJECT" }

    # merge or reject
    if ($pass) {
        Invoke-Cmd -Command "git -C `"$RepoRoot`" checkout $MainBranch"
        Invoke-Cmd -Command "git -C `"$RepoRoot`" cherry-pick $trialBranch~1..$trialBranch"
        Invoke-Cmd -Command "git -C `"$RepoRoot`" branch -D $trialBranch"
        Add-Content -Path $worklogPath -Value ("- Step {0} ({1}) remote run {2}: F1 {3:N6}, Acc {4:N6}, decision={5}, job={6}" -f $stepId, $stepName, $timestamp, $candidateF1, $candidateAcc, $statusLine, $jobId)
        Invoke-Cmd -Command "git -C `"$RepoRoot`" add .codex/WORKLOG.md"
        $championAfter = @{
            source = "step${stepId}_${stepName}"
            accuracy = $candidateMetrics.accuracy
            precision = $candidateMetrics.precision
            recall = $candidateMetrics.recall
            f1 = $candidateMetrics.f1
        }
        Write-JsonFile -Path $championPath -Object $championAfter
        Invoke-Cmd -Command "git -C `"$RepoRoot`" commit --allow-empty -m `"step${stepId}: accept ${stepName} (f1=${candidateF1:N6}, acc=${candidateAcc:N6})`""
    } else {
        Invoke-Cmd -Command "git -C `"$RepoRoot`" checkout $MainBranch"
        Invoke-Cmd -Command "git -C `"$RepoRoot`" branch -D $trialBranch"
        Add-Content -Path $worklogPath -Value ("- Step {0} ({1}) remote run {2}: F1 {3:N6}, Acc {4:N6}, decision={5}, job={6}" -f $stepId, $stepName, $timestamp, $candidateF1, $candidateAcc, $statusLine, $jobId)
        Invoke-Cmd -Command "git -C `"$RepoRoot`" add .codex/WORKLOG.md"
        Invoke-Cmd -Command "git -C `"$RepoRoot`" commit --allow-empty -m `"step${stepId}: reject ${stepName} (f1=${candidateF1:N6}, acc=${candidateAcc:N6})`""
    }
}

$repo = (Resolve-Path $RepoRoot).Path
$steps = Get-StepDefinitions

if (-not $RunAll -and $Step -le 0) {
    throw "Use -RunAll or provide a positive -Step value."
}

if ($RunAll) {
    $selected = $steps
} else {
    $selected = $steps | Where-Object { $_.id -eq $Step }
    if (-not $selected) {
        throw "Step $Step is not defined."
    }
}

Invoke-Cmd -Command "git -C `"$repo`" checkout $MainBranch"

foreach ($s in $selected) {
    Invoke-Step -StepDef $s -RepoRoot $repo -MainBranch $MainBranch -RemoteHost $RemoteHost -RemoteScratch $RemoteScratch -AccuracyDropTolerance $AccuracyDropTolerance -DryRun:$DryRun
}

Write-Host "All requested steps completed."
