@echo off
set JAVA_HOME=C:\Users\zdmat\code-projects\data-room-graph\neo4j\jdk-17.0.2
set PATH=%JAVA_HOME%\bin;%PATH%
echo Starting Neo4j...
call "C:\Users\zdmat\code-projects\data-room-graph\neo4j\neo4j-community-5.26.0\bin\neo4j.bat" console
