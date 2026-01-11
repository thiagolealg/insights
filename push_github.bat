@echo off
echo ==========================================
echo       ENVIANDO PROJETO PARA GITHUB
echo ==========================================
echo.
echo Executando: git push --force -u origin main
echo.
git push --force -u origin main
echo.
echo ==========================================
if %errorlevel% neq 0 (
    echo ERRO: Falha ao enviar. Verifique seu login.
) else (
    echo SUCESSO: Projeto enviado corretamente!
)
echo ==========================================
pause
