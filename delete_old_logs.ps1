$cutoffDate = "20260317"

$rootDir = "C:\Users\tizia\PycharmProjects\MasterThesis"

Get-ChildItem -Path $rootDir -Recurse -Filter "*.txt" | ForEach-Object {
    if ($_ -match "_(\d{8})_\d{6}\.txt$") {
        $fileDate = $matches[1]
        if ([int]$fileDate -lt [int]$cutoffDate) {
            Write-Host "Deleting $($_.FullName)"
            Remove-Item $_.FullName
        }
    }
}