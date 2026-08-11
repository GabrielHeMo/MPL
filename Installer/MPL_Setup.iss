#define MyAppName "MPL Optimizer"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "Gabriel Hernandez"
#define MyAppExeName "MPL.exe"

[Setup]
AppId={{8F02E61A-709C-467B-A7FD-ACF61C43C879}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}

DefaultDirName={autopf}\MPL Optimizer
DefaultGroupName=MPL Optimizer

OutputDir=C:\Users\gabo1\source\repos\Asop\Installer\Output
OutputBaseFilename=MPL_Setup_1.0.0

Compression=lzma
SolidCompression=yes
WizardStyle=modern

ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

UninstallDisplayName=MPL Optimizer
PrivilegesRequired=admin

[Files]
Source: "C:\Users\gabo1\source\repos\Asop\Asop\bin\Release\net10.0-windows\win-x64\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional icons:"

[Icons]
Name: "{autoprograms}\MPL Optimizer"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\MPL Optimizer"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch MPL Optimizer"; Flags: nowait postinstall skipifsilent