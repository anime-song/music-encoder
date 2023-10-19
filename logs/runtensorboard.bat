
if not "%~0"=="%~dp0.\%~nx0" (
     start /min cmd /c,"%~dp0.\%~nx0" %*
     exit
)


start tensorboard --logdir ./ --bind --port 6006
ngrok http 6006