@echo off
echo Stopping Neo4j...
taskkill /F /IM java.exe /FI "WINDOWTITLE eq Neo4j*" 2>nul
echo Neo4j stopped.
