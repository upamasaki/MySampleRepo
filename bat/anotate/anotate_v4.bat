
@echo off

set LABEL_OK=1
set LABEL_NG=3
set LABEL_MISS=5

set DIR_OK=OK
set DIR_NG=NG
set DIR_MISS=MISS



md %DIR_NG%
md %DIR_OK%	
md %DIR_MISS%

setlocal enabledelayedexpansion
for %%a in (*.jpg) do (

  echo %%a
  echo %%a >> log.txt

  %%a

  set /p INPUT_LABEL="ƒ‰ƒxƒ‹‚ð“ü—Í‚µ‚Ä‚­‚¾‚³‚¢>>>"
  echo ƒ‰ƒxƒ‹ : !INPUT_LABEL! 
)
endlocal

pause