# Un comando para el dia de vuelo, desde PowerShell.
#
#   .\scripts\banco_embedded\volar.ps1 192.168.1.121:8300 pi@192.168.1.125 pi@192.168.1.126
#
# Existe porque la palabra "bash" dentro de PowerShell abre WSL, que vive en otro sistema de
# archivos y no tiene las claves de las placas. Hay que llamar al bash de Git por su ruta entera,
# y eso es justo lo que se escribe mal cuando uno tiene prisa.
$bash = "C:\Program Files\Git\bin\bash.exe"
if (-not (Test-Path $bash)) {
    Write-Host "No encuentro el bash de Git en $bash."
    Write-Host "Si Git esta en otro sitio, editar esta ruta o abrir Git Bash y usar volar.sh."
    exit 1
}
$aqui = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = Split-Path -Parent (Split-Path -Parent $aqui)
Push-Location $repo
try   { & $bash "scripts/banco_embedded/volar.sh" @args }
finally { Pop-Location }
