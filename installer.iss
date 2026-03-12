; W@Home Hive — Inno Setup Installer Script
; One-click installer: WHome.exe (GUI) + WHome.scr (screensaver)
;
; Build:
;   "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer.iss

[Setup]
AppName=W@Home Hive
AppVersion=1.0.2
AppPublisher=Akataleptos
AppPublisherURL=https://akataleptos.com
AppSupportURL=https://wathome.akataleptos.com
DefaultDirName={autopf}\WHome
DefaultGroupName=W@Home Hive
OutputDir=dist
OutputBaseFilename=WHome-Setup
Compression=lzma2
SolidCompression=yes
PrivilegesRequired=admin
UninstallDisplayIcon={app}\WHome.exe
WizardStyle=modern
DisableProgramGroupPage=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Files]
Source: "dist\WHome.exe"; DestDir: "{app}"; Flags: ignoreversion; AfterInstall: CopyExeToScr

[Code]
procedure CopyExeToScr();
begin
  CopyFile(ExpandConstant('{app}\WHome.exe'), ExpandConstant('{app}\WHome.scr'), False);
end;

[Icons]
Name: "{group}\W@Home Hive"; Filename: "{app}\WHome.exe"
Name: "{group}\Uninstall W@Home"; Filename: "{uninstallexe}"
Name: "{autodesktop}\W@Home Hive"; Filename: "{app}\WHome.exe"; Tasks: desktopicon
Name: "{autostartup}\W@Home Hive"; Filename: "{app}\WHome.exe"; Tasks: autostart

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Shortcuts:"
Name: "autostart"; Description: "Start W@Home when Windows starts"; GroupDescription: "Startup:"
Name: "screensaver"; Description: "Set W@Home as the default screensaver"; GroupDescription: "Screensaver:"

[Registry]
; Set WHome as the active screensaver (only if user checked the task)
Root: HKCU; Subkey: "Control Panel\Desktop"; ValueType: string; ValueName: "SCRNSAVE.EXE"; ValueData: "{app}\WHome.scr"; Flags: uninsdeletevalue; Tasks: screensaver
Root: HKCU; Subkey: "Control Panel\Desktop"; ValueType: string; ValueName: "ScreenSaveActive"; ValueData: "1"; Tasks: screensaver
Root: HKCU; Subkey: "Control Panel\Desktop"; ValueType: string; ValueName: "ScreenSaveTimeOut"; ValueData: "300"; Tasks: screensaver
; Clean up autostart registry key on uninstall
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; ValueType: none; ValueName: "WHome"; Flags: uninsdeletevalue

[UninstallRun]
; Kill screensaver and GUI before uninstall so files aren't locked
Filename: "taskkill"; Parameters: "/F /IM WHome.scr"; Flags: runhidden; RunOnceId: "KillScr"
Filename: "taskkill"; Parameters: "/F /IM WHome.exe"; Flags: runhidden; RunOnceId: "KillGui"

[UninstallDelete]
Type: files; Name: "{app}\WHome.scr"
Type: files; Name: "{app}\worker_config.json"
Type: files; Name: "{app}\checkpoint.json"
Type: files; Name: "{app}\compute_status.json"
Type: dirifempty; Name: "{app}"
; Clean up persistent config in %LOCALAPPDATA%\WHome
Type: files; Name: "{localappdata}\WHome\worker_config.json"
Type: files; Name: "{localappdata}\WHome\checkpoint.json"
Type: files; Name: "{localappdata}\WHome\compute_status.json"
Type: dirifempty; Name: "{localappdata}\WHome"

[Run]
Filename: "{app}\WHome.exe"; Description: "Launch W@Home Hive"; Flags: nowait postinstall skipifsilent
