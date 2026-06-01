$sent = Import-Csv 'd:\brave download\antigravity project\set\fear_greed_index.csv'
$sentDates = $sent | ForEach-Object { [datetime]::ParseExact($_.date, 'yyyy-MM-dd', $null).Date } | Sort-Object -Unique
$trades = Import-Csv 'd:\brave download\antigravity project\set\historical_data.csv'
$tradeDates = $trades | ForEach-Object { [datetime]::ParseExact($_.'Timestamp IST', 'dd-MM-yyyy HH:mm', $null).Date } | Sort-Object -Unique
$common = $tradeDates | Where-Object { $sentDates -contains $_ }
Write-Host ('sent range: {0} to {1}' -f $sentDates[0].ToString('yyyy-MM-dd'), $sentDates[-1].ToString('yyyy-MM-dd'))
Write-Host ('trade range: {0} to {1}' -f $tradeDates[0].ToString('yyyy-MM-dd'), $tradeDates[-1].ToString('yyyy-MM-dd'))
Write-Host ('unique trade dates: {0}' -f $tradeDates.Count)
Write-Host ('common mapping dates: {0}' -f $common.Count)
Write-Host ('first 5 trade dates:')
$tradeDates | Select-Object -First 5 | ForEach-Object { Write-Host $_.ToString('yyyy-MM-dd') }
Write-Host ('last 5 trade dates:')
$tradeDates | Select-Object -Last 5 | ForEach-Object { Write-Host $_.ToString('yyyy-MM-dd') }