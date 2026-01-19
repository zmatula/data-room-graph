@echo off
set JAVA_HOME=C:\Users\zdmat\code-projects\data-room-graph\neo4j\jdk-17.0.2
set PATH=%JAVA_HOME%\bin;%PATH%
echo JAVA_HOME is set to: %JAVA_HOME%
echo Checking Java version:
java -version
echo.
echo Setting Neo4j initial password...
call "C:\Users\zdmat\code-projects\data-room-graph\neo4j\neo4j-community-5.26.0\bin\neo4j-admin.bat" dbms set-initial-password dataroom123
echo Done.
